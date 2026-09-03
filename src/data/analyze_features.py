import pandas as pd
from pathlib import Path


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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BEHAVIORAL FEATURE ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Entities: {df['entity_id'].nunique():,}")
print(f"Time windows: {df['time_bin'].nunique():,}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

counts = df["isFraud"].value_counts()

print(counts)

print("\nPercentages:")
print(
    df["isFraud"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# FEATURE LIST
# ============================================================

feature_columns = [
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
# COMPARE LEGITIMATE VS FRAUD
# ============================================================

print("\n" + "=" * 70)
print("LEGITIMATE VS FRAUD FEATURE COMPARISON")
print("=" * 70)

comparison = df.groupby("isFraud")[feature_columns].median().T

comparison.columns = [
    "Legitimate",
    "Fraud"
]

comparison["Fraud_vs_Legit_Ratio"] = (
    comparison["Fraud"]
    / comparison["Legitimate"].replace(0, float("nan"))
)


print(
    comparison.to_string(
        float_format=lambda x: f"{x:,.3f}"
    )
)


# ============================================================
# MEAN COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MEAN VALUES")
print("=" * 70)

mean_comparison = (
    df.groupby("isFraud")[feature_columns]
    .mean()
    .T
)

mean_comparison.columns = [
    "Legitimate",
    "Fraud"
]

print(
    mean_comparison.to_string(
        float_format=lambda x: f"{x:,.3f}"
    )
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = (
    df.isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

for column, percentage in missing.items():

    if percentage > 0:

        print(
            f"{column:35} "
            f"{percentage:6.2f}%"
        )


# ============================================================
# FEATURE CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION WITH FRAUD")
print("=" * 70)

correlations = (
    df[feature_columns + ["isFraud"]]
    .corr()["isFraud"]
    .drop("isFraud")
    .abs()
    .sort_values(ascending=False)
)

for feature, correlation in correlations.items():

    print(
        f"{feature:35} "
        f"{correlation:.4f}"
    )


# ============================================================
# FRAUD BY TIME WINDOW
# ============================================================

print("\n" + "=" * 70)
print("FRAUD RATE OVER TIME")
print("=" * 70)

time_stats = (
    df.groupby("time_bin")["isFraud"]
    .agg(
        transactions="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
)

time_stats["fraud_rate"] *= 100

print(
    time_stats.head(20).to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# ENTITY TRAJECTORY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("ENTITY TRAJECTORY STATISTICS")
print("=" * 70)

entity_stats = (
    df.groupby("entity_id")
    .agg(
        observations=("time_bin", "count"),
        fraud_observations=("isFraud", "sum"),
        first_window=("time_bin", "min"),
        last_window=("time_bin", "max")
    )
)

print(
    f"\nAverage observations per entity: "
    f"{entity_stats['observations'].mean():.2f}"
)

print(
    f"Median observations per entity: "
    f"{entity_stats['observations'].median():.2f}"
)

print(
    f"Entities with at least one fraud observation: "
    f"{(entity_stats['fraud_observations'] > 0).sum():,}"
)

print(
    f"Entities with multiple fraud observations: "
    f"{(entity_stats['fraud_observations'] > 1).sum():,}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "feature_analysis.csv"
)

comparison.to_csv(output_file)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"\nFeature comparison saved to:")
print(output_file)