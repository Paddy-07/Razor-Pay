from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "behavioral_sequences_v2.csv"

TIME_WINDOW_SECONDS = 6 * 60 * 60


# ============================================================
# Columns
# ============================================================

TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]

IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
]


# ============================================================
# Load data
# ============================================================

print("Loading transaction data...")

transactions = pd.read_csv(
    RAW_DIR / "train_transaction.csv",
    usecols=TRANSACTION_COLUMNS
)

print(f"Transactions: {len(transactions):,}")

print("Loading identity data...")

identity = pd.read_csv(
    RAW_DIR / "train_identity.csv",
    usecols=IDENTITY_COLUMNS
)

print(f"Identity records: {len(identity):,}")


# ============================================================
# Merge
# ============================================================

print("\nMerging transaction and identity data...")

df = transactions.merge(
    identity,
    on="TransactionID",
    how="left"
)

print(f"Merged rows: {len(df):,}")


# ============================================================
# Basic cleaning
# ============================================================

numeric_columns = [
    "TransactionAmt",
    "TransactionDT",
    "card1",
    "card2",
    "card3",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_columns = [
    "ProductCD",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

for col in categorical_columns:
    df[col] = df[col].fillna("UNKNOWN")


# ============================================================
# Sort chronologically
# ============================================================

df = df.sort_values("TransactionDT").reset_index(drop=True)


# ============================================================
# Entity construction
# ============================================================

# IEEE-CIS does not provide a direct account ID.
# We use a card fingerprint as a behavioral entity proxy.

card_columns = [
    "card1",
    "card2",
    "card3",
    "card5",
    "card6",
]

df["entity_id"] = (
    df[card_columns]
    .fillna(-1)
    .astype(str)
    .agg("_".join, axis=1)
)


# ============================================================
# Time windows
# ============================================================

df["time_bin"] = (
    df["TransactionDT"] // TIME_WINDOW_SECONDS
).astype(int)


# ============================================================
# Helper features
# ============================================================

df["amount_change_ratio"] = (
    df.groupby("entity_id")["TransactionAmt"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

df["time_since_previous_transaction"] = (
    df.groupby("entity_id")["TransactionDT"]
    .diff()
    .fillna(999999)
)


# Relative-cycle activity indicator.
# TransactionDT is not real-world local time.

relative_hour = (
    (df["TransactionDT"] % 86400) // 3600
)

df["night_activity"] = (
    (relative_hour < 6) |
    (relative_hour >= 22)
).astype(int)


# ============================================================
# Device sharing — PAST ONLY
# ============================================================

print("Calculating past-only device sharing...")

df["device_entity_count"] = 1

valid_device = (
    df["DeviceInfo"].notna() &
    (df["DeviceInfo"] != "UNKNOWN")
)

device_entities = {}

for idx in df.index:

    device = df.at[idx, "DeviceInfo"]

    if not valid_device.loc[idx]:
        df.at[idx, "device_entity_count"] = 0
        continue

    entity = df.at[idx, "entity_id"]

    previous_entities = device_entities.get(device, set())

    df.at[idx, "device_entity_count"] = len(previous_entities)

    if device not in device_entities:
        device_entities[device] = set()

    device_entities[device].add(entity)


# ============================================================
# Window-level aggregation
# ============================================================

print("Building behavioral windows...")

group_columns = [
    "entity_id",
    "time_bin",
]

grouped = df.groupby(group_columns, sort=False)


behavior = grouped.agg(
    isFraud=("isFraud", "max"),

    transaction_count_window=(
        "TransactionID",
        "count"
    ),

    amount_total_window=(
        "TransactionAmt",
        "sum"
    ),

    amount_mean_window=(
        "TransactionAmt",
        "mean"
    ),

    amount_max_window=(
        "TransactionAmt",
        "max"
    ),

    amount_std_window=(
        "TransactionAmt",
        "std"
    ),

    time_since_previous_transaction=(
        "time_since_previous_transaction",
        "min"
    ),

    unique_devices_window=(
        "DeviceInfo",
        "nunique"
    ),

    device_entity_count=(
        "device_entity_count",
        "max"
    ),

    unique_payment_emails_window=(
        "P_emaildomain",
        "nunique"
    ),

    unique_receiver_emails_window=(
        "R_emaildomain",
        "nunique"
    ),

    unique_addresses_window=(
        "addr1",
        "nunique"
    ),

    unique_products_window=(
        "ProductCD",
        "nunique"
    ),

    night_activity_rate=(
        "night_activity",
        "mean"
    ),

    amount_change_ratio=(
        "amount_change_ratio",
        "mean"
    ),
).reset_index()


# ============================================================
# Cleaning aggregated features
# ============================================================

behavior["amount_std_window"] = (
    behavior["amount_std_window"]
    .fillna(0)
)

behavior = behavior.replace(
    [np.inf, -np.inf],
    np.nan
)

behavior = behavior.fillna(0)


# ============================================================
# Sort final behavioral sequences
# ============================================================

behavior = behavior.sort_values(
    ["entity_id", "time_bin"]
).reset_index(drop=True)


# ============================================================
# Save
# ============================================================

print("\nSaving corrected behavioral dataset...")

behavior.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING V2 COMPLETE")
print("=" * 60)

print(f"Behavioral observations : {len(behavior):,}")
print(
    f"Unique entities          : "
    f"{behavior['entity_id'].nunique():,}"
)
print(
    f"Time windows             : "
    f"{behavior['time_bin'].nunique():,}"
)
print(
    f"Fraud observations       : "
    f"{behavior['isFraud'].sum():,}"
)

print("\nFraud rate:")

print(
    f"{behavior['isFraud'].mean() * 100:.2f}%"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDone.")