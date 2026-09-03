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
    / "hmm_v2_results.csv"
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

print(f"Rows    : {len(df):,}")
print(f"Entities: {df['entity_id'].nunique():,}")


# ============================================================
# Prepare features
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
# Entity-level train/test split
# ============================================================

print("\nCreating entity-level split...")

entities = df["entity_id"].drop_duplicates().to_numpy()

rng = np.random.default_rng(42)
rng.shuffle(entities)

split_index = int(len(entities) * 0.80)

train_entities = set(
    entities[:split_index]
)

test_entities = set(
    entities[split_index:]
)

train_mask = df["entity_id"].isin(train_entities)
test_mask = df["entity_id"].isin(test_entities)

print(f"Training entities: {len(train_entities):,}")
print(f"Testing entities : {len(test_entities):,}")


# ============================================================
# Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X.loc[train_mask]
)

X_test = scaler.transform(
    X.loc[test_mask]
)


# ============================================================
# Build sequences
# ============================================================

def build_sequences(dataframe, transformed_X):

    sequences = []
    indices = []

    positions = dataframe.index.to_numpy()

    temp = dataframe.copy()
    temp["_position"] = np.arange(len(temp))

    for entity, group in temp.groupby(
        "entity_id",
        sort=False
    ):

        sequence_positions = group["_position"].to_numpy()

        if len(sequence_positions) < 2:
            continue

        sequences.append(
            transformed_X[sequence_positions]
        )

        indices.append(
            group.index.to_numpy()
        )

    return sequences, indices


train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_sequences, _ = build_sequences(
    train_df,
    X_train
)

test_sequences, test_indices = build_sequences(
    test_df,
    X_test
)


# ============================================================
# Combine training sequences
# ============================================================

X_train_hmm = np.vstack(train_sequences)

lengths = [
    len(sequence)
    for sequence in train_sequences
]


print("\nHMM sequence statistics:")

print(
    f"Training sequences : {len(train_sequences):,}"
)

print(
    f"Training observations: {len(X_train_hmm):,}"
)


# ============================================================
# Train HMM
# ============================================================

print("\nTraining HMM V2...")

model = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=200,
    tol=0.001,
    random_state=42,
)

model.fit(
    X_train_hmm,
    lengths=lengths
)

print("\nHMM training complete.")

print(
    f"Converged: {model.monitor_.converged}"
)

print(
    f"Iterations: {model.monitor_.iter}"
)

print(
    f"Log likelihood: {model.monitor_.history[-1]:.4f}"
)


# ============================================================
# Determine state meanings
# ============================================================

state_scores = []

for state in range(3):

    mean_vector = model.means_[state]

    # Higher values in these dimensions represent
    # stronger behavioral activity.
    risk_score = (
        mean_vector[0] +
        mean_vector[1] +
        mean_vector[6] +
        mean_vector[7] +
        mean_vector[8] +
        mean_vector[9] +
        mean_vector[10]
    )

    state_scores.append(
        (state, risk_score)
    )


state_scores = sorted(
    state_scores,
    key=lambda x: x[1]
)


state_mapping = {
    state_scores[0][0]: "NORMAL",
    state_scores[1][0]: "PROBING",
    state_scores[2][0]: "ACTIVE_ABUSE",
}


print("\nHidden state interpretation:")

for state, score in state_scores:

    print(
        f"State {state} -> "
        f"{state_mapping[state]} "
        f"(risk score={score:.4f})"
    )


# ============================================================
# Score individual sequences
# ============================================================

all_results = []


for sequence, original_indices in zip(
    test_sequences,
    test_indices
):

    posterior = model.predict_proba(
        sequence
    )

    states = model.predict(
        sequence
    )

    for i, state in enumerate(states):

        state_name = state_mapping[state]

        # Posterior probability of ACTIVE_ABUSE
        active_state = [
            s
            for s, name in state_mapping.items()
            if name == "ACTIVE_ABUSE"
        ][0]

        active_probability = posterior[
            i,
            active_state
        ]

        # ----------------------------------------------------
        # Trajectory information
        # ----------------------------------------------------

        previous_states = states[:i]

        probing_count = np.sum(
            previous_states ==
            [
                s for s, name in state_mapping.items()
                if name == "PROBING"
            ][0]
        )

        active_count = np.sum(
            previous_states ==
            active_state
        )

        transitioned_to_active = (
            i > 0
            and states[i] == active_state
            and states[i - 1] != active_state
        )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        persistence = 0

        if i > 0:

            current_state = states[i]

            j = i - 1

            while (
                j >= 0
                and states[j] == current_state
            ):
                persistence += 1
                j -= 1

        # ----------------------------------------------------
        # Trajectory risk
        # ----------------------------------------------------

        trajectory_score = (
            0.50 * active_probability
            + 0.20 * min(probing_count / 3, 1)
            + 0.15 * min(active_count / 3, 1)
            + 0.15 * int(transitioned_to_active)
        )

        trajectory_score = float(
            np.clip(
                trajectory_score,
                0,
                1
            )
        )

        row = test_df.iloc[
            original_indices[i]
        ].copy()

        row["hidden_state"] = state
        row["state_name"] = state_name
        row["active_abuse_probability"] = active_probability
        row["probing_history"] = probing_count
        row["active_history"] = active_count
        row["transitioned_to_active"] = int(
            transitioned_to_active
        )
        row["state_persistence"] = persistence
        row["trajectory_score"] = trajectory_score

        all_results.append(row)


# ============================================================
# Results dataframe
# ============================================================

results = pd.DataFrame(
    all_results
)

print(
    f"\nGenerated results: "
    f"{len(results):,}"
)


# ============================================================
# Prediction
# ============================================================

# Initial threshold.
# We will tune this later.

results["hmm_prediction"] = (
    results["trajectory_score"] >= 0.50
).astype(int)


# ============================================================
# Evaluation
# ============================================================

y_true = results["isFraud"].astype(int)

y_pred = results["hmm_prediction"].astype(int)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
)


print("\n" + "=" * 60)
print("HMM V2 TRAJECTORY RESULTS")
print("=" * 60)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
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
            "Fraud"
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
# Trajectory analysis
# ============================================================

print("\nFraud rate by trajectory transition:")

transition_analysis = (
    results
    .groupby("transitioned_to_active")["isFraud"]
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