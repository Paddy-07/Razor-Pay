from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    / "synthetic_fraud_sequences_v2.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cgan_generator_v2.pt"
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


LATENT_DIM = 32
HIDDEN_DIM = 128
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 0.0002
NUM_SYNTHETIC = 10000

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):

    def __init__(self, feature_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                LATENT_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                feature_dim
            ),
        )

    def forward(self, noise):

        return self.network(noise)


# ============================================================
# Discriminator
# ============================================================

class Discriminator(nn.Module):

    def __init__(self, feature_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                feature_dim,
                HIDDEN_DIM
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                HIDDEN_DIM,
                1
            ),

            nn.Sigmoid(),
        )

    def forward(self, features):

        return self.network(features)


# ============================================================
# Load data
# ============================================================

print("Loading behavioral sequences V2...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Total rows: {len(df):,}"
)

print(
    f"Fraud rows: {df['isFraud'].sum():,}"
)


# ============================================================
# Select REAL fraud observations
# ============================================================

fraud_df = df[
    df["isFraud"] == 1
].copy()

print(
    f"Fraud observations used for CGAN: "
    f"{len(fraud_df):,}"
)


# ============================================================
# Prepare features
# ============================================================

X = fraud_df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


# Log transform skewed variables

for column in LOG_FEATURES:

    X[column] = np.log1p(
        np.maximum(
            X[column],
            0
        )
    )


# ============================================================
# Scale
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)


dataset = TensorDataset(
    X_tensor
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ============================================================
# Create models
# ============================================================

feature_dim = len(FEATURES)

generator = Generator(
    feature_dim
)

discriminator = Discriminator(
    feature_dim
)


optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


criterion = nn.BCELoss()


# ============================================================
# Training
# ============================================================

print("\nTraining Fraud-only GAN...")

for epoch in range(EPOCHS):

    generator_losses = []
    discriminator_losses = []

    for (real_features,) in loader:

        batch_size = (
            real_features.size(0)
        )

        real_labels = torch.ones(
            batch_size,
            1
        )

        fake_labels = torch.zeros(
            batch_size,
            1
        )

        # ----------------------------------------------------
        # Discriminator
        # ----------------------------------------------------

        optimizer_D.zero_grad()

        real_output = discriminator(
            real_features
        )

        real_loss = criterion(
            real_output,
            real_labels
        )

        noise = torch.randn(
            batch_size,
            LATENT_DIM
        )

        fake_features = generator(
            noise
        )

        fake_output = discriminator(
            fake_features.detach()
        )

        fake_loss = criterion(
            fake_output,
            fake_labels
        )

        discriminator_loss = (
            real_loss + fake_loss
        )

        discriminator_loss.backward()

        optimizer_D.step()

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        optimizer_G.zero_grad()

        fake_output = discriminator(
            fake_features
        )

        generator_loss = criterion(
            fake_output,
            real_labels
        )

        generator_loss.backward()

        optimizer_G.step()

        generator_losses.append(
            generator_loss.item()
        )

        discriminator_losses.append(
            discriminator_loss.item()
        )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"G Loss: "
        f"{np.mean(generator_losses):.4f} | "
        f"D Loss: "
        f"{np.mean(discriminator_losses):.4f}"
    )


# ============================================================
# Save generator
# ============================================================

torch.save(
    generator.state_dict(),
    MODEL_FILE
)

print(
    f"\nGenerator saved to:\n"
    f"{MODEL_FILE}"
)


# ============================================================
# Generate synthetic fraud
# ============================================================

print(
    f"\nGenerating "
    f"{NUM_SYNTHETIC:,} synthetic fraud observations..."
)

generator.eval()

with torch.no_grad():

    noise = torch.randn(
        NUM_SYNTHETIC,
        LATENT_DIM
    )

    synthetic_scaled = (
        generator(noise)
        .numpy()
    )


# ============================================================
# Inverse scaling
# ============================================================

synthetic_transformed = (
    scaler.inverse_transform(
        synthetic_scaled
    )
)


synthetic = pd.DataFrame(
    synthetic_transformed,
    columns=FEATURES
)


# ============================================================
# Reverse log transformation
# ============================================================

for column in LOG_FEATURES:

    synthetic[column] = np.expm1(
        synthetic[column]
    )


# ============================================================
# Realistic constraints
# ============================================================

COUNT_FEATURES = [
    "transaction_count_window",
    "unique_devices_window",
    "device_entity_count",
    "unique_payment_emails_window",
    "unique_receiver_emails_window",
    "unique_addresses_window",
    "unique_products_window",
]


for column in COUNT_FEATURES:

    synthetic[column] = (
        synthetic[column]
        .clip(lower=0)
        .round()
    )


# ------------------------------------------------------------
# Use REAL fraud ranges as hard safety limits
# ------------------------------------------------------------

for column in FEATURES:

    real_values = fraud_df[column].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(real_values) == 0:
        continue

    lower = real_values.quantile(0.01)
    upper = real_values.quantile(0.99)

    synthetic[column] = (
        synthetic[column]
        .clip(
            lower=lower,
            upper=upper
        )
    )


# ------------------------------------------------------------
# Specific bounded features
# ------------------------------------------------------------

synthetic["night_activity_rate"] = (
    synthetic["night_activity_rate"]
    .clip(0, 1)
)

synthetic["amount_change_ratio"] = (
    synthetic["amount_change_ratio"]
    .clip(-10, 10)
)


# ============================================================
# Labels
# ============================================================

synthetic["isFraud"] = 1

synthetic["synthetic"] = 1


# ============================================================
# Save
# ============================================================

synthetic.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("CGAN V2 COMPLETE")
print("=" * 60)

print(
    f"Synthetic observations: "
    f"{len(synthetic):,}"
)

print("\nSynthetic fraud statistics:")

print(
    synthetic[FEATURES]
    .describe()
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\nDone.")