from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "processed" / "behavioral_sequences_v2.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "baseline_rule_results.csv"


# ============================================================
# Load data
# ============================================================

print("Loading behavioral sequences...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# Transparent fraud rules
# ============================================================

def calculate_rule_score(row):
    """
    Each rule contributes a fixed number of points.

    The purpose is NOT to create the best possible model.
    The purpose is to create a transparent and auditable
    baseline that we can compare against ML models.
    """

    score = 0
    triggered_rules = []

    # --------------------------------------------------------
    # Rule 1: High transaction volume
    # --------------------------------------------------------

    if row["transaction_count_window"] >= 10:
        score += 20
        triggered_rules.append("HIGH_TRANSACTION_VOLUME")

    # --------------------------------------------------------
    # Rule 2: High total transaction amount
    # --------------------------------------------------------

    if row["amount_total_window"] >= 1000:
        score += 15
        triggered_rules.append("HIGH_AMOUNT_VOLUME")

    # --------------------------------------------------------
    # Rule 3: Many devices associated with entity
    # --------------------------------------------------------

    if row["unique_devices_window"] >= 3:
        score += 15
        triggered_rules.append("MULTIPLE_DEVICES")

    # --------------------------------------------------------
    # Rule 4: Device shared across entities
    # --------------------------------------------------------

    if row["device_entity_count"] >= 3:
        score += 20
        triggered_rules.append("SHARED_DEVICE")

    # --------------------------------------------------------
    # Rule 5: Multiple payment emails
    # --------------------------------------------------------

    if row["unique_payment_emails_window"] >= 2:
        score += 10
        triggered_rules.append("MULTIPLE_PAYMENT_EMAILS")

    # --------------------------------------------------------
    # Rule 6: Multiple receiver emails
    # --------------------------------------------------------

    if row["unique_receiver_emails_window"] >= 2:
        score += 10
        triggered_rules.append("MULTIPLE_RECEIVER_EMAILS")

    # --------------------------------------------------------
    # Rule 7: Multiple addresses
    # --------------------------------------------------------

    if row["unique_addresses_window"] >= 2:
        score += 10
        triggered_rules.append("MULTIPLE_ADDRESSES")

    # --------------------------------------------------------
    # Rule 8: Multiple products
    # --------------------------------------------------------

    if row["unique_products_window"] >= 3:
        score += 5
        triggered_rules.append("MULTIPLE_PRODUCTS")

    # --------------------------------------------------------
    # Rule 9: Rapid transaction activity
    # --------------------------------------------------------

    if row["time_since_previous_transaction"] <= 60:
        score += 20
        triggered_rules.append("RAPID_TRANSACTIONS")

    # --------------------------------------------------------
    # Rule 10: Large amount change
    # --------------------------------------------------------

    if abs(row["amount_change_ratio"]) >= 2:
        score += 10
        triggered_rules.append("LARGE_AMOUNT_CHANGE")

    # --------------------------------------------------------
    # Rule 11: High night activity
    # --------------------------------------------------------

    if row["night_activity_rate"] >= 0.75:
        score += 5
        triggered_rules.append("HIGH_NIGHT_ACTIVITY")

    return score, "|".join(triggered_rules)


# ============================================================
# Apply rules
# ============================================================

print("\nApplying fraud rules...")

results = df.apply(
    calculate_rule_score,
    axis=1,
    result_type="expand"
)

results.columns = ["rule_score", "triggered_rules"]

df["rule_score"] = results["rule_score"]
df["triggered_rules"] = results["triggered_rules"]


# ============================================================
# Convert score into risk level
# ============================================================

def risk_level(score):

    if score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_level"] = df["rule_score"].apply(risk_level)


# ============================================================
# Binary prediction
# ============================================================

# 50+ points = suspicious
df["rule_prediction"] = (df["rule_score"] >= 50).astype(int)


# ============================================================
# Evaluation
# ============================================================

y_true = df["isFraud"]
y_pred = df["rule_prediction"]

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

print("\n" + "=" * 60)
print("RULE-BASED BASELINE RESULTS")
print("=" * 60)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)


# ============================================================
# Risk-level analysis
# ============================================================

print("\nFraud rate by risk level:")

risk_analysis = (
    df.groupby("risk_level")["isFraud"]
    .agg(["count", "sum", "mean"])
    .rename(
        columns={
            "count": "observations",
            "sum": "fraud_count",
            "mean": "fraud_rate",
        }
    )
)

risk_analysis["fraud_rate"] *= 100

print(risk_analysis.sort_index())


# ============================================================
# Rule frequency analysis
# ============================================================

print("\nMost frequently triggered rules:")

rule_counts = {}

for rules in df["triggered_rules"]:

    if not rules:
        continue

    for rule in rules.split("|"):
        rule_counts[rule] = rule_counts.get(rule, 0) + 1


rule_frequency = (
    pd.Series(rule_counts)
    .sort_values(ascending=False)
    .to_frame("trigger_count")
)

print(rule_frequency)


# ============================================================
# Save results
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nDone.")