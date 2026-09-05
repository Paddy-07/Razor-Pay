"""
model_service.py

This is NOT a re-implementation or approximation. It is your actual
hmm/model_v3.py and bayesian/risk_model.py logic, refactored from
"top-level script that runs once and exits" into "function you can
call repeatedly from an API" — so the live dashboard is driven by the
same trained HMM and the same Bayesian fusion math you already
evaluated in final_model_comparison.csv.

At startup (once, when the API boots) this module:
  1. Loads behavioral_sequences_v2.csv
  2. Recreates the exact same entity-level 80/20 train/test split
     (same random seed = 42, so this is the SAME split your evaluation
     used)
  3. Scales + log-transforms features exactly as model_v3.py does
  4. Trains a 3-state GaussianHMM on the training entities
  5. Labels the 3 hidden states NORMAL / PROBING / ACTIVE_ABUSE using
     the same risk-ordering heuristic
  6. Scores the held-out test entities to reproduce hmm_v3_results.csv
     in memory
  7. Uses that scored test set to estimate the same Bayesian evidence
     likelihoods risk_model.py estimates (P(evidence | fraud) etc.)

After startup, `score_sequence()` takes a live sequence of feature
windows (whatever the dashboard sliders are set to) and runs it
through the SAME fitted HMM + SAME Bayesian fusion formula.

The one honest simplification: a brand-new sequence typed into the
dashboard has no real prior history, so trajectory fields that depend
on "what happened before this session" (probing_history,
active_history, recent_active_count, persistence) are computed fresh
from only the windows you've added in this session — exactly the same
formula, just starting from a clean slate the way a brand-new account
would.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ============================================================
# Configuration — copied from hmm/model_v3.py
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
DATA_FILE = BACKEND_DIR.parent / "data" / "behavioral_sequences_v2.csv"

FEATURES = [
    "transaction_count_window",
    "amount_total_window",
    "amount_mean_window",
    "amount_max_window",
    "amount_std_window",
    "time_since_previous_transaction",
    "unique_devices_window",
    "device_entity_count",
    "unique_payment_emails_window",
    "unique_receiver_emails_window",
    "unique_addresses_window",
    "unique_products_window",
    "night_activity_rate",
    "amount_change_ratio",
]

LOG_FEATURES = [
    "transaction_count_window",
    "amount_total_window",
    "amount_mean_window",
    "amount_max_window",
    "amount_std_window",
    "time_since_previous_transaction",
    "unique_devices_window",
    "device_entity_count",
    "unique_payment_emails_window",
    "unique_receiver_emails_window",
    "unique_addresses_window",
    "unique_products_window",
]

EVIDENCE_COLUMNS = [
    "rapid_activity",
    "multiple_device_evidence",
    "payment_identity_evidence",
    "address_evidence",
    "shared_device_evidence",
]

# Sensible defaults for features the dashboard doesn't expose as sliders,
# based on the dataset's overall medians. The dashboard can override any
# of these via the "advanced" fields.
DEFAULT_WINDOW = {
    "transaction_count_window": 1,
    "amount_total_window": 50.0,
    "amount_mean_window": 50.0,
    "amount_max_window": 50.0,
    "amount_std_window": 0.0,
    "time_since_previous_transaction": 999999.0,
    "unique_devices_window": 1,
    "device_entity_count": 1,
    "unique_payment_emails_window": 1,
    "unique_receiver_emails_window": 1,
    "unique_addresses_window": 1,
    "unique_products_window": 1,
    "night_activity_rate": 0.0,
    "amount_change_ratio": 0.0,
}


class ModelService:
    """Holds the one, shared, trained model + Bayesian parameters."""

    def __init__(self) -> None:
        self.ready = False
        self.error: str | None = None
        self.stats: dict[str, Any] = {}

    # --------------------------------------------------------------
    def train(self) -> None:
        t0 = time.time()

        df = pd.read_csv(DATA_FILE)
        df = df.sort_values(["entity_id", "time_bin"]).reset_index(drop=True)

        X = df[FEATURES].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        for column in LOG_FEATURES:
            X[column] = np.log1p(np.maximum(X[column], 0))

        entities = df["entity_id"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(42)
        rng.shuffle(entities)
        split_index = int(len(entities) * 0.80)
        train_entities = set(entities[:split_index])
        test_entities = set(entities[split_index:])

        train_mask = df["entity_id"].isin(train_entities)
        test_mask = df["entity_id"].isin(test_entities)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.loc[train_mask])
        X_test = scaler.transform(X.loc[test_mask])

        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        # ---- Build training sequences (same as model_v3.py) ----
        sequences: list[np.ndarray] = []
        for _, group in train_df.groupby("entity_id", sort=False):
            positions = group.index.to_numpy()
            if len(positions) < 2:
                continue
            sequences.append(X_train[positions])

        X_train_hmm = np.vstack(sequences)
        lengths = [len(s) for s in sequences]

        model = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=100,
            tol=0.01,
            random_state=42,
            init_params="stmc",
        )
        model.fit(X_train_hmm, lengths=lengths)

        # ---- State interpretation (same heuristic) ----
        state_scores = []
        for state in range(3):
            m = model.means_[state]
            risk_score = m[0] + m[1] + m[6] + m[7] + m[8] + m[9] + m[10]
            state_scores.append((state, risk_score))
        state_scores.sort(key=lambda x: x[1])

        state_mapping = {
            state_scores[0][0]: "NORMAL",
            state_scores[1][0]: "PROBING",
            state_scores[2][0]: "ACTIVE_ABUSE",
        }
        normal_state = [s for s, n in state_mapping.items() if n == "NORMAL"][0]
        probing_state = [s for s, n in state_mapping.items() if n == "PROBING"][0]
        active_state = [s for s, n in state_mapping.items() if n == "ACTIVE_ABUSE"][0]

        # ---- Score the held-out test entities (same loop as model_v3.py) ----
        results = []
        for entity, group in test_df.groupby("entity_id", sort=False):
            positions = group.index.to_numpy()
            sequence = X_test[positions]

            posterior = model.predict_proba(sequence)
            states = model.predict(sequence)

            for i, position in enumerate(positions):
                state = states[i]
                active_probability = float(posterior[i, active_state])
                previous_states = states[:i]

                probing_history = int(np.sum(previous_states == probing_state))
                recent_states = states[max(0, i - 2):i]
                recent_active_count = int(np.sum(recent_states == active_state))
                transitioned_to_active = int(
                    i > 0 and state == active_state and states[i - 1] != active_state
                )

                persistence = 1
                j = i - 1
                while j >= 0 and states[j] == state:
                    persistence += 1
                    j -= 1

                trajectory_score = (
                    0.45 * active_probability
                    + 0.20 * min(probing_history / 3, 1)
                    + 0.15 * min(recent_active_count / 2, 1)
                    + 0.15 * transitioned_to_active
                    + 0.05 * min(persistence / 3, 1)
                )
                trajectory_score = float(np.clip(trajectory_score, 0, 1))

                row = test_df.iloc[position].to_dict()
                row["state_name"] = state_mapping[state]
                row["active_abuse_probability"] = active_probability
                row["trajectory_score"] = trajectory_score
                row["transitioned_to_active"] = transitioned_to_active
                results.append(row)

        results_df = pd.DataFrame(results)

        # ---- Bayesian evidence + likelihoods (same as risk_model.py) ----
        results_df["hmm_evidence"] = results_df["active_abuse_probability"].clip(0, 1)
        results_df["trajectory_evidence"] = results_df["trajectory_score"].clip(0, 1)
        results_df["rapid_activity"] = (
            results_df["time_since_previous_transaction"] <= 60
        ).astype(int)
        results_df["multiple_device_evidence"] = (
            results_df["unique_devices_window"] >= 3
        ).astype(int)
        results_df["payment_identity_evidence"] = (
            results_df["unique_payment_emails_window"] >= 2
        ).astype(int)
        results_df["address_evidence"] = (
            results_df["unique_addresses_window"] >= 2
        ).astype(int)
        results_df["shared_device_evidence"] = (
            results_df["device_entity_count"] >= 3
        ).astype(int)

        fraud_prior = results_df["isFraud"].mean()
        legitimate_prior = 1 - fraud_prior

        def estimate_likelihood(evidence_column, smoothing=1.0):
            fraud_group = results_df[results_df["isFraud"] == 1]
            legit_group = results_df[results_df["isFraud"] == 0]

            evidence_fraud = fraud_group[evidence_column].sum() + smoothing
            total_fraud = len(fraud_group) + 2 * smoothing
            evidence_legit = legit_group[evidence_column].sum() + smoothing
            total_legit = len(legit_group) + 2 * smoothing

            return evidence_fraud / total_fraud, evidence_legit / total_legit

        likelihoods = {}
        for column in EVIDENCE_COLUMNS:
            p_fraud, p_legit = estimate_likelihood(column)
            likelihoods[column] = {"fraud": p_fraud, "legitimate": p_legit}

        # ---- Save everything the API needs ----
        self.scaler = scaler
        self.model = model
        self.state_mapping = state_mapping
        self.normal_state = normal_state
        self.probing_state = probing_state
        self.active_state = active_state
        self.fraud_prior = float(fraud_prior)
        self.legitimate_prior = float(legitimate_prior)
        self.likelihoods = likelihoods

        self.stats = {
            "train_entities": len(train_entities),
            "test_entities": len(test_entities),
            "test_observations": len(results_df),
            "fraud_prior": float(fraud_prior),
            "hmm_converged": bool(model.monitor_.converged),
            "hmm_iterations": int(model.monitor_.iter),
            "train_seconds": round(time.time() - t0, 1),
            "state_mapping": {int(k): v for k, v in state_mapping.items()},
        }
        self.ready = True

    # --------------------------------------------------------------
    def _bayesian_probability(self, row: dict) -> float:
        fraud_odds = self.fraud_prior / self.legitimate_prior

        hmm_probability = float(np.clip(row["hmm_evidence"], 0.001, 0.999))
        fraud_odds *= (hmm_probability / (1 - hmm_probability)) ** 0.50

        trajectory_probability = float(np.clip(row["trajectory_evidence"], 0.001, 0.999))
        fraud_odds *= (trajectory_probability / (1 - trajectory_probability)) ** 0.75

        for column in EVIDENCE_COLUMNS:
            present = row[column] == 1
            likelihood = self.likelihoods[column]
            if present:
                p_fraud = likelihood["fraud"]
                p_legit = likelihood["legitimate"]
            else:
                p_fraud = 1 - likelihood["fraud"]
                p_legit = 1 - likelihood["legitimate"]
            fraud_odds *= p_fraud / max(p_legit, 1e-9)

        probability = fraud_odds / (1 + fraud_odds)
        return float(np.clip(probability, 0, 1))

    @staticmethod
    def _classify_risk(probability: float) -> str:
        if probability >= 0.70:
            return "HIGH"
        if probability >= 0.30:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _explain(row: dict) -> str:
        reasons = []
        if row["state_name"] == "ACTIVE_ABUSE":
            reasons.append("ACTIVE_ABUSE behavioral state")
        if row["transitioned_to_active"] == 1:
            reasons.append("transition into ACTIVE_ABUSE")
        if row["rapid_activity"] == 1:
            reasons.append("rapid transaction activity")
        if row["multiple_device_evidence"] == 1:
            reasons.append("multiple devices")
        if row["payment_identity_evidence"] == 1:
            reasons.append("multiple payment identities")
        if row["address_evidence"] == 1:
            reasons.append("multiple addresses")
        if row["shared_device_evidence"] == 1:
            reasons.append("shared device behavior")
        if not reasons:
            reasons.append("no strong behavioral anomaly")
        return " | ".join(reasons)

    # --------------------------------------------------------------
    def score_sequence(self, windows: list[dict]) -> list[dict]:
        """
        windows: a list of feature dicts, one per simulated 6-hour window,
        in chronological order (this is one "session" — think of it as one
        entity's behavior unfolding step by step). Missing fields fall back
        to DEFAULT_WINDOW.

        Returns one result dict per window, using the SAME trained HMM and
        SAME Bayesian fusion as the rest of the project.
        """
        if not self.ready:
            raise RuntimeError("Model is not trained yet.")

        filled = []
        for w in windows:
            merged = dict(DEFAULT_WINDOW)
            merged.update(w)
            filled.append(merged)

        raw_df = pd.DataFrame(filled)[FEATURES].copy()
        raw_df = raw_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        transformed = raw_df.copy()
        for column in LOG_FEATURES:
            transformed[column] = np.log1p(np.maximum(transformed[column], 0))

        X = self.scaler.transform(transformed)

        posterior = self.model.predict_proba(X)
        states = self.model.predict(X)

        outputs = []
        for i in range(len(filled)):
            state = states[i]
            active_probability = float(posterior[i, self.active_state])
            previous_states = states[:i]

            probing_history = int(np.sum(previous_states == self.probing_state))
            recent_states = states[max(0, i - 2):i]
            recent_active_count = int(np.sum(recent_states == self.active_state))
            transitioned_to_active = int(
                i > 0 and state == self.active_state and states[i - 1] != self.active_state
            )

            persistence = 1
            j = i - 1
            while j >= 0 and states[j] == state:
                persistence += 1
                j -= 1

            trajectory_score = (
                0.45 * active_probability
                + 0.20 * min(probing_history / 3, 1)
                + 0.15 * min(recent_active_count / 2, 1)
                + 0.15 * transitioned_to_active
                + 0.05 * min(persistence / 3, 1)
            )
            trajectory_score = float(np.clip(trajectory_score, 0, 1))

            raw = filled[i]
            row = {
                "state_name": self.state_mapping[state],
                "active_abuse_probability": active_probability,
                "trajectory_score": trajectory_score,
                "transitioned_to_active": transitioned_to_active,
                "hmm_evidence": active_probability,
                "trajectory_evidence": trajectory_score,
                "rapid_activity": int(raw["time_since_previous_transaction"] <= 60),
                "multiple_device_evidence": int(raw["unique_devices_window"] >= 3),
                "payment_identity_evidence": int(raw["unique_payment_emails_window"] >= 2),
                "address_evidence": int(raw["unique_addresses_window"] >= 2),
                "shared_device_evidence": int(raw["device_entity_count"] >= 3),
            }

            bayesian_probability = self._bayesian_probability(row)
            risk_level = self._classify_risk(bayesian_probability)
            explanation = self._explain(row)
            decision = {"LOW": "ALLOW", "MEDIUM": "REVIEW", "HIGH": "BLOCK"}[risk_level]

            outputs.append(
                {
                    "step": i + 1,
                    "state_name": row["state_name"],
                    "active_abuse_probability": round(active_probability, 4),
                    "trajectory_score": round(trajectory_score, 4),
                    "bayesian_fraud_probability": round(bayesian_probability, 4),
                    "bayesian_risk_level": risk_level,
                    "risk_explanation": explanation,
                    "decision": decision,
                }
            )

        return outputs


service = ModelService()
