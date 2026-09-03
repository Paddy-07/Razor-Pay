import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_TRANSACTION = RAW_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DIR / "train_identity.csv"


# ============================================================
# SETTINGS
# ============================================================

# 6-hour behavioral windows
TIME_WINDOW_SECONDS = 6 * 60 * 60


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FRAUD TRAJECTORY PREPROCESSING")
print("=" * 70)

print("\nLoading transaction data...")

transaction_columns = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",

    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",

    "addr1",
    "addr2",

    "dist1",
    "dist2",

    "P_emaildomain",
    "R_emaildomain",

    "C1",
    "C2",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
]


transaction = pd.read_csv(
    TRAIN_TRANSACTION,
    usecols=transaction_columns
)

print(f"Transactions loaded: {len(transaction):,}")


# ============================================================
# LOAD IDENTITY INFORMATION
# ============================================================

print("\nLoading identity data...")

identity_columns = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
]

identity = pd.read_csv(
    TRAIN_IDENTITY,
    usecols=identity_columns
)

print(f"Identity records loaded: {len(identity):,}")


# ============================================================
# MERGE TRANSACTION + IDENTITY
# ============================================================

print("\nMerging transaction and identity information...")

df = transaction.merge(
    identity,
    on="TransactionID",
    how="left"
)

print(f"Merged rows: {len(df):,}")


# ============================================================
# BASIC CLEANING
# ============================================================

print("\nCleaning data...")

# Sort chronologically
df = df.sort_values("TransactionDT").reset_index(drop=True)


# Numeric values
df["TransactionAmt"] = pd.to_numeric(
    df["TransactionAmt"],
    errors="coerce"
)

df["TransactionAmt"] = df["TransactionAmt"].fillna(
    df["TransactionAmt"].median()
)


# ------------------------------------------------------------
# Categorical values
# ------------------------------------------------------------

categorical_columns = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

for column in categorical_columns:

    if column in df.columns:
        df[column] = (
            df[column]
            .fillna("UNKNOWN")
            .astype(str)
        )


# ============================================================
# CREATE BEHAVIORAL ENTITY
# ============================================================

print("\nCreating behavioral entities...")

# IEEE-CIS does not provide a direct account ID.
#
# We construct a stable entity using card-related
# attributes available in the dataset.

card_columns = [
    "card1",
    "card2",
    "card3",
    "card5",
    "card6",
]

for column in card_columns:

    df[column] = (
        df[column]
        .fillna(-1)
        .astype(str)
    )


df["entity_id"] = (
    df["card1"] + "_" +
    df["card2"] + "_" +
    df["card3"] + "_" +
    df["card5"] + "_" +
    df["card6"]
)


# ============================================================
# DEVICE NORMALIZATION
# ============================================================

df["DeviceInfo"] = (
    df["DeviceInfo"]
    .fillna("UNKNOWN")
    .astype(str)
)


df["DeviceType"] = (
    df["DeviceType"]
    .fillna("UNKNOWN")
    .astype(str)
)


# ============================================================
# TIME FEATURES
# ============================================================

print("\nCreating temporal features...")

# TransactionDT is measured in seconds from an arbitrary
# reference point in the IEEE-CIS dataset.

df["time_bin"] = (
    df["TransactionDT"] // TIME_WINDOW_SECONDS
).astype(int)


# Relative time inside the 6-hour window

df["time_in_window"] = (
    df["TransactionDT"] % TIME_WINDOW_SECONDS
)


# ============================================================
# TRANSACTION VELOCITY
# ============================================================

print("Creating velocity features...")

# Number of transactions made by an entity
# inside each time window.

group = df.groupby(
    ["entity_id", "time_bin"]
)


df["transaction_count_window"] = (
    group["TransactionID"]
    .transform("count")
)


# ============================================================
# AMOUNT FEATURES
# ============================================================

df["amount_total_window"] = (
    group["TransactionAmt"]
    .transform("sum")
)


df["amount_mean_window"] = (
    group["TransactionAmt"]
    .transform("mean")
)


df["amount_max_window"] = (
    group["TransactionAmt"]
    .transform("max")
)


df["amount_std_window"] = (
    group["TransactionAmt"]
    .transform("std")
    .fillna(0)
)


# ============================================================
# DEVICE BEHAVIOR
# ============================================================

print("Creating device behavior features...")

df["unique_devices_window"] = (
    group["DeviceInfo"]
    .transform("nunique")
)


# ============================================================
# DEVICE SHARING
# ============================================================

print("Calculating device sharing...")

device_entity_counts = (
    df.groupby("DeviceInfo")["entity_id"]
    .nunique()
    .rename("device_entity_count")
)


df = df.merge(
    device_entity_counts,
    on="DeviceInfo",
    how="left"
)


# ============================================================
# EMAIL BEHAVIOR
# ============================================================

print("Creating email behavior features...")

df["unique_payment_emails_window"] = (
    group["P_emaildomain"]
    .transform("nunique")
)


df["unique_receiver_emails_window"] = (
    group["R_emaildomain"]
    .transform("nunique")
)


# ============================================================
# ADDRESS BEHAVIOR
# ============================================================

df["unique_addresses_window"] = (
    group["addr1"]
    .transform("nunique")
)


# ============================================================
# PRODUCT DIVERSITY
# ============================================================

df["unique_products_window"] = (
    group["ProductCD"]
    .transform("nunique")
)


# ============================================================
# TIME-OF-DAY BEHAVIOR
# ============================================================

# We don't know the real-world timezone because
# TransactionDT is a relative timestamp.
#
# We therefore use the relative position within
# the transaction timeline.

SECONDS_PER_DAY = 24 * 60 * 60

df["time_of_day"] = (
    df["TransactionDT"] % SECONDS_PER_DAY
)


# Night activity indicator.
#
# 00:00 - 06:00 is treated as nighttime.

df["night_activity"] = (
    (df["time_of_day"] < 6 * 60 * 60)
    .astype(int)
)


df["night_activity_rate"] = (
    df.groupby(["entity_id", "time_bin"])["night_activity"]
    .transform("mean")
)


# ============================================================
# ENTITY TRANSACTION INTERVAL
# ============================================================

print("Creating transaction interval features...")

df["previous_transaction_time"] = (
    df.groupby("entity_id")["TransactionDT"]
    .shift(1)
)


df["time_since_previous_transaction"] = (
    df["TransactionDT"]
    - df["previous_transaction_time"]
)


df["time_since_previous_transaction"] = (
    df["time_since_previous_transaction"]
    .fillna(TIME_WINDOW_SECONDS)
)


# ============================================================
# BEHAVIORAL CHANGE
# ============================================================

print("Creating behavioral-change features...")

previous_amount = (
    df.groupby("entity_id")["TransactionAmt"]
    .shift(1)
)


df["amount_change_ratio"] = (
    df["TransactionAmt"]
    / previous_amount.replace(0, np.nan)
)


df["amount_change_ratio"] = (
    df["amount_change_ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(1.0)
)


# ============================================================
# FINAL FEATURE TABLE
# ============================================================

feature_columns = [

    "entity_id",
    "time_bin",

    # Target
    "isFraud",

    # Basic transaction behavior
    "transaction_count_window",
    "amount_total_window",
    "amount_mean_window",
    "amount_max_window",
    "amount_std_window",

    # Velocity
    "time_since_previous_transaction",

    # Device behavior
    "unique_devices_window",
    "device_entity_count",

    # Email behavior
    "unique_payment_emails_window",
    "unique_receiver_emails_window",

    # Address/product behavior
    "unique_addresses_window",
    "unique_products_window",

    # Temporal behavior
    "night_activity_rate",

    # Behavioral drift
    "amount_change_ratio",
]


features = (
    df[feature_columns]
    .drop_duplicates(
        subset=["entity_id", "time_bin"]
    )
    .reset_index(drop=True)
)


# ============================================================
# SORT SEQUENCES
# ============================================================

features = features.sort_values(
    ["entity_id", "time_bin"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output_file = (
    PROCESSED_DIR /
    "behavioral_sequences.csv"
)


features.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"\nBehavioral sequences created: "
    f"{len(features):,}"
)

print(
    f"Unique entities: "
    f"{features['entity_id'].nunique():,}"
)

print(
    f"Time windows: "
    f"{features['time_bin'].nunique():,}"
)

print(
    f"Fraud observations: "
    f"{features['isFraud'].sum():,}"
)

print(
    f"\nSaved to:\n{output_file}"
)

print("\nFeature columns:")

for column in feature_columns:
    print(f"  ✓ {column}")

print("\nDone.")