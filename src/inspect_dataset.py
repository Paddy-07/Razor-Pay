import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"


# --------------------------------------------------
# Load a small sample first
# --------------------------------------------------

print("=" * 70)
print("IEEE-CIS FRAUD DATASET INSPECTION")
print("=" * 70)

print("\nLoading transaction data...")

transaction = pd.read_csv(
    TRAIN_TRANSACTION,
    nrows=10000
)

print("\nTransaction dataset sample loaded.")
print("Rows:", len(transaction))
print("Columns:", len(transaction.columns))


# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRANSACTION COLUMNS")
print("=" * 70)

for i, column in enumerate(transaction.columns, start=1):
    print(f"{i:3}. {column}")


# --------------------------------------------------
# Target distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("FRAUD DISTRIBUTION IN SAMPLE")
print("=" * 70)

print(transaction["isFraud"].value_counts())

print("\nPercentage:")
print(
    transaction["isFraud"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# Important behavioral columns
# --------------------------------------------------

important_columns = [
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
    "C14"
]

print("\n" + "=" * 70)
print("IMPORTANT COLUMNS AVAILABLE")
print("=" * 70)

for column in important_columns:
    if column in transaction.columns:
        print(f"✓ {column}")
    else:
        print(f"✗ {column}")


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP 20 MISSING-VALUE COLUMNS")
print("=" * 70)

missing = (
    transaction.isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .head(20)
)

for column, percentage in missing.items():
    print(f"{column:25} {percentage:6.2f}%")


# --------------------------------------------------
# Identity dataset
# --------------------------------------------------

if TRAIN_IDENTITY.exists():

    print("\n" + "=" * 70)
    print("LOADING IDENTITY DATA")
    print("=" * 70)

    identity = pd.read_csv(
        TRAIN_IDENTITY,
        nrows=10000
    )

    print("Rows:", len(identity))
    print("Columns:", len(identity.columns))

    print("\nIdentity columns:")

    for i, column in enumerate(identity.columns, start=1):
        print(f"{i:3}. {column}")

else:
    print("\nWARNING: train_identity.csv was not found.")


# --------------------------------------------------
# Time range
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRANSACTION TIME")
print("=" * 70)

print("Minimum TransactionDT:", transaction["TransactionDT"].min())
print("Maximum TransactionDT:", transaction["TransactionDT"].max())

print("\nInspection completed.")