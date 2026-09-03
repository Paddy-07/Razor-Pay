from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
    / "logistic_results.csv"
)


# ============================================================
# Features
# ============================================================

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


# ============================================================
# Load data
# ============================================================

print("Loading behavioral sequences V2...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Entities: {df['entity_id'].nunique():,}")


# ============================================================
# Prepare features
# ============================================================

X = df[FEATURES].copy()
y = df["isFraud"].astype(int)


# Handle invalid values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# Log transform highly skewed features
# ============================================================

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


for column in LOG_FEATURES:
    X[column] = np.log1p(
        np.maximum(X[column], 0)
    )


# ============================================================
# Entity-level train/test split
# ============================================================

print("\nCreating entity-level train/test split...")

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


X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]


print(f"Training entities : {len(train_entities):,}")
print(f"Testing entities  : {len(test_entities):,}")

print(f"Training rows     : {len(X_train):,}")
print(f"Testing rows      : {len(X_test):,}")


# ============================================================
# Standardization
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# Logistic Regression
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    solver="lbfgs",
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# Predictions
# ============================================================

probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# Evaluation
# ============================================================

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
)


print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
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
        y_test,
        predictions,
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
        y_test,
        predictions
    )
)


# ============================================================
# Feature coefficients
# ============================================================

print("\nFeature coefficients:")

coefficients = pd.DataFrame(
    {
        "feature": FEATURES,
        "coefficient": model.coef_[0],
    }
)

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False,
)

print(
    coefficients[
        [
            "feature",
            "coefficient",
        ]
    ].to_string(index=False)
)


# ============================================================
# Save results
# ============================================================

output = df.loc[test_mask].copy()

output["fraud_probability"] = probabilities

output["logistic_prediction"] = predictions

output.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nResults saved to:")

print(OUTPUT_FILE)

print("\nDone.")