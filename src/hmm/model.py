import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from hmmlearn.hmm import GaussianHMM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavioral_sequences.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HMM FEATURES
# ============================================================

FEATURES = [

    "transaction_count_window",

    "amount_total_window",

    "amount_mean_window",

    "amount_max_window",

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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HMM FRAUD TRAJECTORY MODEL")
print("=" * 70)

print("\nLoading behavioral sequences...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Entities: {df['entity_id'].nunique():,}"
)


# ============================================================
# CLEAN FEATURES
# ============================================================

print("\nCleaning HMM features...")

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    df[feature] = (
        df[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ============================================================
# LOG TRANSFORM SKEWED FEATURES
# ============================================================

log_features = [

    "transaction_count_window",

    "amount_total_window",

    "amount_mean_window",

    "amount_max_window",

    "time_since_previous_transaction",

    "unique_devices_window",

    "device_entity_count",

    "unique_payment_emails_window",

    "unique_receiver_emails_window",

    "unique_addresses_window",

    "unique_products_window",
]


for feature in log_features:

    df[feature] = np.log1p(
        df[feature].clip(lower=0)
    )


# ============================================================
# SORT SEQUENCES
# ============================================================

df = df.sort_values(
    ["entity_id", "time_bin"]
).reset_index(drop=True)


# ============================================================
# ENTITY-LEVEL TRAIN/TEST SPLIT
# ============================================================

print("\nCreating entity-level split...")

entities = (
    df["entity_id"]
    .drop_duplicates()
    .values
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


train_df = df[
    df["entity_id"].isin(train_entities)
].copy()

test_df = df[
    df["entity_id"].isin(test_entities)
].copy()


print(
    f"Training entities: {len(train_entities):,}"
)

print(
    f"Testing entities: {len(test_entities):,}"
)

print(
    f"Training observations: {len(train_df):,}"
)

print(
    f"Testing observations: {len(test_df):,}"
)


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(
    train_df[FEATURES]
)

X_test = scaler.transform(
    test_df[FEATURES]
)


# ============================================================
# BUILD SEQUENCE LENGTHS
# ============================================================

train_lengths = (
    train_df
    .groupby("entity_id", sort=False)
    .size()
    .tolist()
)

test_lengths = (
    test_df
    .groupby("entity_id", sort=False)
    .size()
    .tolist()
)


# ============================================================
# TRAIN HMM
# ============================================================

print("\nTraining 3-state Gaussian HMM...")

hmm = GaussianHMM(
    n_components=3,
    covariance_type="diag",
    n_iter=150,
    tol=0.001,
    random_state=42,
    verbose=True
)


hmm.fit(
    X_train,
    lengths=train_lengths
)


print("\nHMM training completed.")

print(
    f"Converged: {hmm.monitor_.converged}"
)

print(
    f"Iterations: {hmm.monitor_.iter}"
)

print(
    f"Final log likelihood: "
    f"{hmm.monitor_.history[-1]:.2f}"
)


# ============================================================
# STATE POSTERIORS
# ============================================================

print("\nCalculating hidden-state probabilities...")

train_posteriors = hmm.predict_proba(
    X_train,
    lengths=train_lengths
)

test_posteriors = hmm.predict_proba(
    X_test,
    lengths=test_lengths
)


# ============================================================
# STATE LABELING
# ============================================================

print("\nInterpreting hidden states...")


# Calculate average standardized feature level
# for each state.

state_means = hmm.means_

# Higher behavioral intensity = higher risk.
#
# We use selected risk-oriented dimensions rather
# than isFraud to assign semantic state names.

risk_features = [
    "transaction_count_window",
    "amount_total_window",
    "unique_devices_window",
    "device_entity_count",
    "unique_payment_emails_window",
    "unique_addresses_window",
]


risk_indices = [
    FEATURES.index(feature)
    for feature in risk_features
]


state_risk_scores = (
    state_means[:, risk_indices]
    .mean(axis=1)
)


state_order = np.argsort(
    state_risk_scores
)


normal_state = state_order[0]

probing_state = state_order[1]

abuse_state = state_order[2]


print("\nState interpretation:")

print(
    f"State {normal_state} → NORMAL"
)

print(
    f"State {probing_state} → PROBING"
)

print(
    f"State {abuse_state} → ACTIVE_ABUSE"
)


# ============================================================
# ADD POSTERIORS TO DATAFRAME
# ============================================================

def add_posteriors(
    data,
    posteriors
):

    data = data.copy()

    data["normal_probability"] = (
        posteriors[:, normal_state]
    )

    data["probing_probability"] = (
        posteriors[:, probing_state]
    )

    data["active_abuse_probability"] = (
        posteriors[:, abuse_state]
    )

    data["hmm_state"] = (
        posteriors.argmax(axis=1)
    )

    data["risk_probability"] = (
        data["active_abuse_probability"]
    )

    return data


train_result = add_posteriors(
    train_df,
    train_posteriors
)

test_result = add_posteriors(
    test_df,
    test_posteriors
)


# ============================================================
# SEMANTIC STATE NAME
# ============================================================

def state_name(state):

    if state == normal_state:
        return "NORMAL"

    if state == probing_state:
        return "PROBING"

    return "ACTIVE_ABUSE"


train_result["state_name"] = (
    train_result["hmm_state"]
    .apply(state_name)
)

test_result["state_name"] = (
    test_result["hmm_state"]
    .apply(state_name)
)


# ============================================================
# SAVE RESULTS
# ============================================================

train_output = (
    OUTPUT_DIR
    / "hmm_train_results.csv"
)

test_output = (
    OUTPUT_DIR
    / "hmm_test_results.csv"
)

train_result.to_csv(
    train_output,
    index=False
)

test_result.to_csv(
    test_output,
    index=False
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("HMM EVALUATION")
print("=" * 70)


# We deliberately use a threshold rather than simply
# taking the most likely hidden state.

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]


y_true = test_result["isFraud"].values

print(
    "\nThreshold performance:"
)

print(
    "\nThreshold | Precision | Recall | F1"
)

print(
    "-" * 45
)


for threshold in thresholds:

    y_pred = (
        test_result["risk_probability"]
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

threshold = 0.50

y_pred = (
    test_result["risk_probability"]
    >= threshold
).astype(int)


print("\n" + "=" * 70)
print("DEFAULT THRESHOLD RESULTS")
print("=" * 70)

print(
    "\nClassification report:\n"
)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraud"
        ],
        zero_division=0
    )
)


print(
    "Confusion matrix:"
)

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# ============================================================
# STATE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("HIDDEN STATE DISTRIBUTION")
print("=" * 70)

print(
    test_result["state_name"]
    .value_counts()
)


# ============================================================
# FRAUD RATE BY STATE
# ============================================================

print("\n" + "=" * 70)
print("FRAUD RATE BY HIDDEN STATE")
print("=" * 70)

state_fraud = (
    test_result
    .groupby("state_name")["isFraud"]
    .agg(
        observations="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
)

state_fraud["fraud_rate"] *= 100

print(
    state_fraud.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# TOP RISKY ENTITIES
# ============================================================

print("\n" + "=" * 70)
print("TOP RISK ENTITIES")
print("=" * 70)

entity_risk = (
    test_result
    .groupby("entity_id")
    .agg(
        max_risk=(
            "risk_probability",
            "max"
        ),

        average_risk=(
            "risk_probability",
            "mean"
        ),

        observations=(
            "time_bin",
            "count"
        ),

        fraud_observations=(
            "isFraud",
            "sum"
        )
    )
    .sort_values(
        "max_risk",
        ascending=False
    )
)


print(
    entity_risk.head(20).to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE ENTITY RISK
# ============================================================

entity_output = (
    OUTPUT_DIR
    / "hmm_entity_risk.csv"
)

entity_risk.to_csv(
    entity_output
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("HMM PIPELINE COMPLETE")
print("=" * 70)

print(
    f"\nTraining results:"
    f"\n{train_output}"
)

print(
    f"\nTesting results:"
    f"\n{test_output}"
)

print(
    f"\nEntity risk:"
    f"\n{entity_output}"
)