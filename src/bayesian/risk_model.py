from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hmm_v3_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "bayesian_risk_results.csv"
)


# ============================================================
# Load HMM results
# ============================================================

print("Loading HMM V3 results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows     : {len(df):,}")
print(f"Entities : {df['entity_id'].nunique():,}")


# ============================================================
# Convert behavioral signals into evidence
# ============================================================

print("\nBuilding Bayesian evidence variables...")


# ------------------------------------------------------------
# Evidence 1: HMM active-abuse probability
# ------------------------------------------------------------

df["hmm_evidence"] = (
    df["active_abuse_probability"]
    .clip(0, 1)
)


# ------------------------------------------------------------
# Evidence 2: Trajectory risk
# ------------------------------------------------------------

df["trajectory_evidence"] = (
    df["trajectory_score"]
    .clip(0, 1)
)


# ------------------------------------------------------------
# Evidence 3: Rapid activity
# ------------------------------------------------------------

df["rapid_activity"] = (
    df["time_since_previous_transaction"] <= 60
).astype(int)


# ------------------------------------------------------------
# Evidence 4: Multiple devices
# ------------------------------------------------------------

df["multiple_device_evidence"] = (
    df["unique_devices_window"] >= 3
).astype(int)


# ------------------------------------------------------------
# Evidence 5: Multiple payment identities
# ------------------------------------------------------------

df["payment_identity_evidence"] = (
    df["unique_payment_emails_window"] >= 2
).astype(int)


# ------------------------------------------------------------
# Evidence 6: Multiple addresses
# ------------------------------------------------------------

df["address_evidence"] = (
    df["unique_addresses_window"] >= 2
).astype(int)


# ------------------------------------------------------------
# Evidence 7: Shared device
# ------------------------------------------------------------

df["shared_device_evidence"] = (
    df["device_entity_count"] >= 3
).astype(int)


# ============================================================
# Prior probability
# ============================================================

# Dataset-level fraud prior.

fraud_prior = df["isFraud"].mean()

legitimate_prior = 1 - fraud_prior

print(
    f"\nFraud prior: "
    f"{fraud_prior:.6f}"
)


# ============================================================
# Evidence likelihoods
# ============================================================

print("\nEstimating evidence likelihoods...")


def estimate_likelihood(
    evidence_column,
    target_column="isFraud",
    smoothing=1.0
):
    """
    Estimate:

        P(evidence | fraud)
        P(evidence | legitimate)

    Laplace smoothing prevents zero probabilities.
    """

    fraud_group = df[
        df[target_column] == 1
    ]

    legitimate_group = df[
        df[target_column] == 0
    ]

    evidence_fraud = (
        fraud_group[evidence_column]
        .sum()
        + smoothing
    )

    total_fraud = (
        len(fraud_group)
        + 2 * smoothing
    )

    evidence_legitimate = (
        legitimate_group[evidence_column]
        .sum()
        + smoothing
    )

    total_legitimate = (
        len(legitimate_group)
        + 2 * smoothing
    )

    p_e_given_fraud = (
        evidence_fraud
        / total_fraud
    )

    p_e_given_legitimate = (
        evidence_legitimate
        / total_legitimate
    )

    return (
        p_e_given_fraud,
        p_e_given_legitimate
    )


evidence_columns = [
    "rapid_activity",
    "multiple_device_evidence",
    "payment_identity_evidence",
    "address_evidence",
    "shared_device_evidence",
]


likelihoods = {}


for column in evidence_columns:

    p_fraud, p_legitimate = (
        estimate_likelihood(column)
    )

    likelihoods[column] = {
        "fraud": p_fraud,
        "legitimate": p_legitimate,
    }

    print(
        f"{column}: "
        f"P(E|Fraud)={p_fraud:.4f}, "
        f"P(E|Legit)={p_legitimate:.4f}"
    )


# ============================================================
# Bayesian evidence fusion
# ============================================================

def bayesian_probability(row):

    # Start with prior odds.

    fraud_odds = (
        fraud_prior
        / legitimate_prior
    )

    # --------------------------------------------------------
    # HMM evidence
    # --------------------------------------------------------

    hmm_probability = np.clip(
        row["hmm_evidence"],
        0.001,
        0.999
    )

    hmm_odds = (
        hmm_probability
        / (1 - hmm_probability)
    )

    # Weight HMM evidence.
    fraud_odds *= (
        hmm_odds ** 0.50
    )

    # --------------------------------------------------------
    # Trajectory evidence
    # --------------------------------------------------------

    trajectory_probability = np.clip(
        row["trajectory_evidence"],
        0.001,
        0.999
    )

    trajectory_odds = (
        trajectory_probability
        / (1 - trajectory_probability)
    )

    fraud_odds *= (
        trajectory_odds ** 0.75
    )

    # --------------------------------------------------------
    # Binary behavioral evidence
    # --------------------------------------------------------

    for column in evidence_columns:

        evidence_present = (
            row[column] == 1
        )

        likelihood = likelihoods[column]

        if evidence_present:

            p_fraud = likelihood["fraud"]

            p_legitimate = (
                likelihood["legitimate"]
            )

        else:

            p_fraud = (
                1 - likelihood["fraud"]
            )

            p_legitimate = (
                1 - likelihood["legitimate"]
            )

        likelihood_ratio = (
            p_fraud
            / max(p_legitimate, 1e-9)
        )

        fraud_odds *= likelihood_ratio

    # --------------------------------------------------------
    # Convert odds → probability
    # --------------------------------------------------------

    probability = (
        fraud_odds
        / (1 + fraud_odds)
    )

    return float(
        np.clip(
            probability,
            0,
            1
        )
    )


print("\nCalculating Bayesian risk...")

df["bayesian_fraud_probability"] = (
    df.apply(
        bayesian_probability,
        axis=1
    )
)


# ============================================================
# Risk classification
# ============================================================

def classify_risk(probability):

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


df["bayesian_risk_level"] = (
    df["bayesian_fraud_probability"]
    .apply(classify_risk)
)


# ============================================================
# Explanation generation
# ============================================================

def generate_explanation(row):

    reasons = []

    if row["state_name"] == "ACTIVE_ABUSE":
        reasons.append(
            "ACTIVE_ABUSE behavioral state"
        )

    if row["transitioned_to_active"] == 1:
        reasons.append(
            "transition into ACTIVE_ABUSE"
        )

    if row["rapid_activity"] == 1:
        reasons.append(
            "rapid transaction activity"
        )

    if row["multiple_device_evidence"] == 1:
        reasons.append(
            "multiple devices"
        )

    if row["payment_identity_evidence"] == 1:
        reasons.append(
            "multiple payment identities"
        )

    if row["address_evidence"] == 1:
        reasons.append(
            "multiple addresses"
        )

    if row["shared_device_evidence"] == 1:
        reasons.append(
            "shared device behavior"
        )

    if not reasons:
        reasons.append(
            "no strong behavioral anomaly"
        )

    return " | ".join(reasons)


df["risk_explanation"] = (
    df.apply(
        generate_explanation,
        axis=1
    )
)


# ============================================================
# Evaluation
# ============================================================

print("\n" + "=" * 60)
print("BAYESIAN RISK RESULTS")
print("=" * 60)


# Use 0.50 only as an initial decision threshold.
df["bayesian_prediction"] = (
    df["bayesian_fraud_probability"]
    >= 0.50
).astype(int)


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


y_true = df["isFraud"].astype(int)

y_pred = df["bayesian_prediction"].astype(int)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
)


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
# Risk-level analysis
# ============================================================

print("\nFraud rate by Bayesian risk level:")

risk_analysis = (
    df
    .groupby(
        "bayesian_risk_level"
    )["isFraud"]
    .agg(
        observations="count",
        fraud_count="sum",
        fraud_rate="mean",
    )
)

risk_analysis["fraud_rate"] *= 100

print(risk_analysis)


# ============================================================
# Average probabilities by actual class
# ============================================================

print(
    "\nAverage predicted fraud probability:"
)

print(
    df.groupby("isFraud")[
        "bayesian_fraud_probability"
    ].mean()
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")

print(OUTPUT_FILE)

print("\nDone.")