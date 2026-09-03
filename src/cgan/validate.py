from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REAL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavioral_sequences_v2.csv"
)

SYNTHETIC_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "synthetic_fraud_sequences_v2.csv"
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


# ============================================================
# Load
# ============================================================

print("Loading real fraud data...")

real = pd.read_csv(
    REAL_FILE
)

real = real[
    real["isFraud"] == 1
].copy()

print(
    f"Real fraud observations: "
    f"{len(real):,}"
)


print("\nLoading synthetic fraud data...")

synthetic = pd.read_csv(
    SYNTHETIC_FILE
)

print(
    f"Synthetic observations: "
    f"{len(synthetic):,}"
)


# ============================================================
# Distribution comparison
# ============================================================

print("\n" + "=" * 80)
print("REAL VS SYNTHETIC FRAUD DISTRIBUTIONS")
print("=" * 80)

comparison = []


for feature in FEATURES:

    real_values = (
        real[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    synthetic_values = (
        synthetic[feature]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    real_mean = real_values.mean()
    synthetic_mean = synthetic_values.mean()

    real_median = real_values.median()
    synthetic_median = synthetic_values.median()

    real_std = real_values.std()
    synthetic_std = synthetic_values.std()

    mean_difference = abs(
        real_mean - synthetic_mean
    ) / max(
        abs(real_mean),
        1e-9
    )

    median_difference = abs(
        real_median - synthetic_median
    ) / max(
        abs(real_median),
        1e-9
    )

    comparison.append(
        {
            "feature": feature,
            "real_mean": real_mean,
            "synthetic_mean": synthetic_mean,
            "real_median": real_median,
            "synthetic_median": synthetic_median,
            "real_std": real_std,
            "synthetic_std": synthetic_std,
            "mean_relative_difference": mean_difference,
            "median_relative_difference": median_difference,
        }
    )


comparison_df = pd.DataFrame(
    comparison
)


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# Overall similarity
# ============================================================

mean_similarity = (
    1
    - comparison_df[
        "mean_relative_difference"
    ].clip(0, 1).mean()
)

median_similarity = (
    1
    - comparison_df[
        "median_relative_difference"
    ].clip(0, 1).mean()
)


print("\n" + "=" * 80)
print("SIMILARITY SUMMARY")
print("=" * 80)

print(
    f"Mean-based similarity   : "
    f"{mean_similarity:.4f}"
)

print(
    f"Median-based similarity : "
    f"{median_similarity:.4f}"
)


# ============================================================
# Train a synthetic-vs-real discriminator
# ============================================================

print("\nTesting whether synthetic fraud is distinguishable...")

real_sample = real[FEATURES].copy()

synthetic_sample = synthetic[FEATURES].copy()

# Equalize sample sizes

sample_size = min(
    len(real_sample),
    len(synthetic_sample)
)

rng = np.random.default_rng(42)

real_sample = real_sample.sample(
    sample_size,
    random_state=42
)

synthetic_sample = synthetic_sample.sample(
    sample_size,
    random_state=42
)


X = pd.concat(
    [
        real_sample,
        synthetic_sample,
    ],
    ignore_index=True
)

y = np.concatenate(
    [
        np.zeros(sample_size),
        np.ones(sample_size),
    ]
)


X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


# Random forest tries to determine whether
# an observation is REAL or SYNTHETIC.

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)


classifier.fit(
    X,
    y
)


probabilities = classifier.predict_proba(
    X
)[:, 1]


auc = roc_auc_score(
    y,
    probabilities
)


print(
    f"\nReal-vs-synthetic AUC: "
    f"{auc:.4f}"
)


# ============================================================
# Interpretation
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if auc < 0.60:

    print(
        "Synthetic fraud is difficult to distinguish "
        "from real fraud."
    )

elif auc < 0.75:

    print(
        "Synthetic fraud is reasonably similar "
        "to real fraud, but some differences remain."
    )

else:

    print(
        "Synthetic fraud is clearly distinguishable "
        "from real fraud. The generator needs improvement."
    )


# ============================================================
# Save comparison
# ============================================================

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cgan_validation.csv"
)

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nValidation results saved to:")

print(OUTPUT_FILE)

print("\nDone.")