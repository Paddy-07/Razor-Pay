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
    / "synthetic_fraud_sequences.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cgan_generator.pt"
)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# Configuration for training
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


# ============================================================
# Hyperparameters
# ============================================================

LATENT_DIM = 32

HIDDEN_DIM = 128

BATCH_SIZE = 256

EPOCHS = 30

LEARNING_RATE = 0.0002

NUM_SYNTHETIC = 10000


# ============================================================
# Generator
# ============================================================

class Generator(nn.Module):

    def __init__(
        self,
        feature_dim,
        latent_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                latent_dim + 1,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                feature_dim
            ),
        )

    def forward(
        self,
        noise,
        condition
    ):

        x = torch.cat(
            [noise, condition],
            dim=1
        )

        return self.network(x)


# ============================================================
# Discriminator
# ============================================================

class Discriminator(nn.Module):

    def __init__(
        self,
        feature_dim,
        hidden_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                feature_dim + 1,
                hidden_dim
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.LeakyReLU(0.2),

            nn.Linear(
                hidden_dim,
                1
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        features,
        condition
    ):

        x = torch.cat(
            [features, condition],
            dim=1
        )

        return self.network(x)


# ============================================================
# Load data
# ============================================================

print("Loading behavioral sequences V2...")

df = pd.read_csv(INPUT_FILE)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# Prepare features
# ============================================================

X = df[FEATURES].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


# Log transform skewed features

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

y = df["isFraud"].astype(int).to_numpy()


# ============================================================
# Train on both classes
# ============================================================

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
).reshape(-1, 1)


dataset = TensorDataset(
    X_tensor,
    y_tensor
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ============================================================
# Models
# ============================================================

feature_dim = len(FEATURES)

generator = Generator(
    feature_dim,
    LATENT_DIM,
    HIDDEN_DIM
)

discriminator = Discriminator(
    feature_dim,
    HIDDEN_DIM
)


# ============================================================
# Optimizers
# ============================================================

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

print("\nTraining Conditional GAN...")

for epoch in range(EPOCHS):

    generator_losses = []
    discriminator_losses = []

    for real_features, condition in loader:

        batch_size = real_features.size(0)

        # ----------------------------------------------------
        # Real / fake labels
        # ----------------------------------------------------

        real_labels = torch.ones(
            batch_size,
            1
        )

        fake_labels = torch.zeros(
            batch_size,
            1
        )

        # ----------------------------------------------------
        # Train discriminator
        # ----------------------------------------------------

        optimizer_D.zero_grad()

        real_output = discriminator(
            real_features,
            condition
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
            noise,
            condition
        )

        fake_output = discriminator(
            fake_features.detach(),
            condition
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
        # Train generator
        # ----------------------------------------------------

        optimizer_G.zero_grad()

        fake_output = discriminator(
            fake_features,
            condition
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

    fraud_condition = torch.ones(
        NUM_SYNTHETIC,
        1
    )

    synthetic_scaled = generator(
        noise,
        fraud_condition
    ).numpy()


# ============================================================
# Convert back to original scale
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

    synthetic[column] = (
        np.expm1(
            synthetic[column]
        )
    )


# ============================================================
# Basic validity constraints
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


NON_NEGATIVE_FEATURES = [
    "amount_total_window",
    "amount_mean_window",
    "amount_max_window",
    "amount_std_window",
    "time_since_previous_transaction",
]


for column in NON_NEGATIVE_FEATURES:

    synthetic[column] = (
        synthetic[column]
        .clip(lower=0)
    )


synthetic["night_activity_rate"] = (
    synthetic["night_activity_rate"]
    .clip(0, 1)
)

synthetic["amount_change_ratio"] = (
    synthetic["amount_change_ratio"]
    .clip(-10, 10)
)


# Add synthetic label

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
print("CGAN SYNTHETIC FRAUD GENERATION COMPLETE")
print("=" * 60)

print(
    f"Synthetic observations: "
    f"{len(synthetic):,}"
)

print("\nSynthetic fraud statistics:")

print(
    synthetic[
        [
            "transaction_count_window",
            "amount_total_window",
            "unique_devices_window",
            "device_entity_count",
            "unique_payment_emails_window",
            "unique_addresses_window",
        ]
    ].describe()
)

print("\nSaved to:")

print(OUTPUT_FILE)

print("\nDone.")