from pathlib import Path

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavioral_sequences_v2.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hmm_v3_results.csv"
)


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


# ============================================================
# Load
# ============================================================

print("Loading behavioral sequences V2...")

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["entity_id", "time_bin"]
).reset_index(drop=True)

print(f"Rows     : {len(df):,}")
print(f"Entities : {df['entity_id'].nunique():,}")


# ============================================================
# Feature preparation
# ============================================================

X = df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


for column in LOG_FEATURES:
    X[column] = np.log1p(
        np.maximum(X[column], 0)
    )


# ============================================================
# Entity-level split
# ============================================================

print("\nCreating entity-level split...")

entities = (
    df["entity_id"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(42)

rng.shuffle(entities)

split_index = int(
    len(entities) * 0.80
)

train_entities = set(
    entities[:split_index]
)

test_entities = set(
    entities[split_index:]
)

train_mask = df["entity_id"].isin(
    train_entities
)

test_mask = df["entity_id"].isin(
    test_entities
)

print(
    f"Training entities : "
    f"{len(train_entities):,}"
)

print(
    f"Testing entities  : "
    f"{len(test_entities):,}"
)


# ============================================================
# Scaling
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X.loc[train_mask]
)

X_test = scaler.transform(
    X.loc[test_mask]
)


# ============================================================
# Build HMM training sequences
# ============================================================

train_df = df.loc[train_mask].copy()

test_df = df.loc[test_mask].copy()

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)


def build_training_sequences(
    dataframe,
    transformed_features
):

    sequences = []

    for _, group in dataframe.groupby(
        "entity_id",
        sort=False
    ):

        positions = group.index.to_numpy()

        # HMM sequences require at least 2 observations.
        if len(positions) < 2:
            continue

        sequences.append(
            transformed_features[positions]
        )

    return sequences


train_sequences = build_training_sequences(
    train_df,
    X_train
)


X_train_hmm = np.vstack(
    train_sequences
)

lengths = [
    len(sequence)
    for sequence in train_sequences
]


print("\nTraining sequence statistics:")

print(
    f"Training sequences      : "
    f"{len(train_sequences):,}"
)

print(
    f"Training observations   : "
    f"{len(X_train_hmm):,}"
)


# ============================================================
# Train HMM
# ============================================================

print("\nTraining HMM V3...")

model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=100,
    tol=0.01,
    random_state=42,
    init_params="stmc",
)

model.fit(
    X_train_hmm,
    lengths=lengths
)


print("\nHMM training complete.")

print(
    f"Converged    : "
    f"{model.monitor_.converged}"
)

print(
    f"Iterations   : "
    f"{model.monitor_.iter}"
)

print(
    f"Log likelihood: "
    f"{model.monitor_.history[-1]:.4f}"
)


# ============================================================
# State interpretation
# ============================================================

state_scores = []

for state in range(3):

    mean_vector = model.means_[state]

    risk_score = (
        mean_vector[0]
        + mean_vector[1]
        + mean_vector[6]
        + mean_vector[7]
        + mean_vector[8]
        + mean_vector[9]
        + mean_vector[10]
    )

    state_scores.append(
        (state, risk_score)
    )


state_scores.sort(
    key=lambda x: x[1]
)


state_mapping = {
    state_scores[0][0]: "NORMAL",
    state_scores[1][0]: "PROBING",
    state_scores[2][0]: "ACTIVE_ABUSE",
}


normal_state = [
    state
    for state, name in state_mapping.items()
    if name == "NORMAL"
][0]

probing_state = [
    state
    for state, name in state_mapping.items()
    if name == "PROBING"
][0]

active_state = [
    state
    for state, name in state_mapping.items()
    if name == "ACTIVE_ABUSE"
][0]


print("\nHidden state interpretation:")

for state, score in state_scores:

    print(
        f"State {state} -> "
        f"{state_mapping[state]} "
        f"(risk={score:.4f})"
    )


# ============================================================
# Score ALL test observations
# ============================================================

print("\nScoring test observations...")

results = []

test_work = test_df.copy()

# The test dataframe is already ordered by entity and time.
for entity, group in test_work.groupby(
    "entity_id",
    sort=False
):

    positions = group.index.to_numpy()

    sequence = X_test[positions]

    # --------------------------------------------------------
    # Single observation entities
    # --------------------------------------------------------

    if len(sequence) == 1:

        posterior = model.predict_proba(
            sequence
        )

        states = model.predict(
            sequence
        )

    else:

        posterior = model.predict_proba(
            sequence
        )

        states = model.predict(
            sequence
        )

    # --------------------------------------------------------
    # Generate trajectory information
    # --------------------------------------------------------

    for i, position in enumerate(positions):

        state = states[i]

        active_probability = float(
            posterior[i, active_state]
        )

        previous_states = states[:i]

        probing_history = int(
            np.sum(
                previous_states
                == probing_state
            )
        )

        active_history = int(
            np.sum(
                previous_states
                == active_state
            )
        )

        # Did the entity just move into ACTIVE_ABUSE?
        transitioned_to_active = int(
            i > 0
            and state == active_state
            and states[i - 1] != active_state
        )

        # ----------------------------------------------------
        # Count recent active states
        # ----------------------------------------------------

        recent_start = max(
            0,
            i - 2
        )

        recent_states = states[
            recent_start:i
        ]

        recent_active_count = int(
            np.sum(
                recent_states
                == active_state
            )
        )

        # ----------------------------------------------------
        # Persistence of current state
        # ----------------------------------------------------

        persistence = 1

        j = i - 1

        while (
            j >= 0
            and states[j] == state
        ):
            persistence += 1
            j -= 1

        # ----------------------------------------------------
        # Trajectory score
        # ----------------------------------------------------

        trajectory_score = (
            0.45 * active_probability
            + 0.20 * min(
                probing_history / 3,
                1
            )
            + 0.15 * min(
                recent_active_count / 2,
                1
            )
            + 0.15 * transitioned_to_active
            + 0.05 * min(
                persistence / 3,
                1
            )
        )

        trajectory_score = float(
            np.clip(
                trajectory_score,
                0,
                1
            )
        )

        row = test_df.iloc[position].copy()

        row["hidden_state"] = state

        row["state_name"] = (
            state_mapping[state]
        )

        row["active_abuse_probability"] = (
            active_probability
        )

        row["probing_history"] = (
            probing_history
        )

        row["active_history"] = (
            active_history
        )

        row["recent_active_count"] = (
            recent_active_count
        )

        row["transitioned_to_active"] = (
            transitioned_to_active
        )

        row["state_persistence"] = (
            persistence
        )

        row["trajectory_score"] = (
            trajectory_score
        )

        results.append(row)


results = pd.DataFrame(results)


print(
    f"Generated results: "
    f"{len(results):,}"
)


# ============================================================
# Threshold analysis
# ============================================================

print("\nThreshold analysis:")

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


threshold_results = []


y_true = results["isFraud"].astype(int)

for threshold in thresholds:

    predictions = (
        results["trajectory_score"]
        >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    print(
        f"{threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# Select best F1 threshold
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


results["hmm_prediction"] = (
    results["trajectory_score"]
    >= best_threshold
).astype(int)


# ============================================================
# Final evaluation
# ============================================================

y_pred = results[
    "hmm_prediction"
].astype(int)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
)


print("\n" + "=" * 60)
print("HMM V3 TRAJECTORY RESULTS")
print("=" * 60)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Precision : "
    f"{precision:.4f}"
)

print(
    f"Recall    : "
    f"{recall:.4f}"
)

print(
    f"F1 Score  : "
    f"{f1:.4f}"
)


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud",
        ],
        zero_division=0,
    )
)


# ============================================================
# Confusion matrix
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# ============================================================
# State analysis
# ============================================================

print("\nFraud rate by hidden state:")

state_analysis = (
    results
    .groupby("state_name")["isFraud"]
    .agg(
        observations="count",
        fraud_count="sum",
        fraud_rate="mean",
    )
)

state_analysis["fraud_rate"] *= 100

print(state_analysis)


# ============================================================
# Transition analysis
# ============================================================

print(
    "\nFraud rate by transition into ACTIVE_ABUSE:"
)

transition_analysis = (
    results
    .groupby(
        "transitioned_to_active"
    )["isFraud"]
    .agg(
        observations="count",
        fraud_count="sum",
        fraud_rate="mean",
    )
)

transition_analysis["fraud_rate"] *= 100

print(transition_analysis)


# ============================================================
# Save
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")

print(OUTPUT_FILE)

print("\nDone.")