import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import QuantileTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/processed/behavioral_sequences_v2.csv"
OUTPUT_FILE = "data/processed/synthetic_fraud_sequences_v3.csv"
MODEL_FILE = "data/processed/cgan_generator_v3.pt"

RANDOM_STATE = 42
N_SYNTHETIC = 10000

LATENT_DIM = 32
HIDDEN_DIM = 128

EPOCHS = 100
BATCH_SIZE = 256

LR = 0.0001
BETAS = (0.0, 0.9)

CRITIC_STEPS = 5
GRADIENT_PENALTY = 10.0


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


COUNT_FEATURES = [
    "transaction_count_window",
    "unique_devices_window",
    "device_entity_count",
    "unique_payment_emails_window",
    "unique_receiver_emails_window",
    "unique_addresses_window",
    "unique_products_window",
]


BOUNDED_FEATURES = [
    "night_activity_rate",
]


# ============================================================
# SEED
# ============================================================

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("CGAN V3 / WGAN-GP FRAUD GENERATOR")
print("=" * 80)
print(f"Device: {device}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading fraud observations...")

df = pd.read_csv(INPUT_FILE)

fraud = df[df["isFraud"] == 1].copy()

print(f"Real fraud observations: {len(fraud):,}")

X = fraud[FEATURES].copy()

# Remove invalid values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Keep real limits for later post-processing
real_min = X.min()
real_max = X.max()

# Use robust 1%-99% bounds
lower = X.quantile(0.01)
upper = X.quantile(0.99)

# ============================================================
# QUANTILE TRANSFORMATION
# ============================================================

print("\nApplying quantile transformation...")

n_quantiles = min(2000, len(X))

transformer = QuantileTransformer(
    n_quantiles=n_quantiles,
    output_distribution="normal",
    random_state=RANDOM_STATE
)

X_transformed = transformer.fit_transform(X)

X_tensor = torch.tensor(
    X_transformed,
    dtype=torch.float32
).to(device)


# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.network(z)


# ============================================================
# CRITIC
# ============================================================

class Critic(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


generator = Generator(
    LATENT_DIM,
    len(FEATURES),
    HIDDEN_DIM
).to(device)

critic = Critic(
    len(FEATURES),
    HIDDEN_DIM
).to(device)


g_optimizer = optim.Adam(
    generator.parameters(),
    lr=LR,
    betas=BETAS
)

c_optimizer = optim.Adam(
    critic.parameters(),
    lr=LR,
    betas=BETAS
)


# ============================================================
# GRADIENT PENALTY
# ============================================================

def gradient_penalty(real, fake):

    batch_size = real.size(0)

    alpha = torch.rand(
        batch_size,
        1,
        device=device
    )

    alpha = alpha.expand_as(real)

    interpolated = (
        alpha * real +
        (1 - alpha) * fake
    )

    interpolated.requires_grad_(True)

    critic_output = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(
        batch_size,
        -1
    )

    gradient_norm = gradients.norm(
        2,
        dim=1
    )

    penalty = (
        (gradient_norm - 1) ** 2
    ).mean()

    return penalty


# ============================================================
# TRAINING
# ============================================================

print("\nStarting WGAN-GP training...")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Critic steps: {CRITIC_STEPS}")

n_samples = len(X_tensor)

for epoch in range(EPOCHS):

    permutation = torch.randperm(n_samples)

    epoch_g_loss = []
    epoch_c_loss = []

    for start in range(0, n_samples, BATCH_SIZE):

        indices = permutation[
            start:start + BATCH_SIZE
        ]

        real_batch = X_tensor[indices]

        current_batch = real_batch.size(0)

        # ----------------------------------------------------
        # CRITIC
        # ----------------------------------------------------

        for _ in range(CRITIC_STEPS):

            z = torch.randn(
                current_batch,
                LATENT_DIM,
                device=device
            )

            fake_batch = generator(z).detach()

            real_score = critic(real_batch)
            fake_score = critic(fake_batch)

            gp = gradient_penalty(
                real_batch,
                fake_batch
            )

            c_loss = (
                fake_score.mean()
                - real_score.mean()
                + GRADIENT_PENALTY * gp
            )

            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()

        # ----------------------------------------------------
        # GENERATOR
        # ----------------------------------------------------

        z = torch.randn(
            current_batch,
            LATENT_DIM,
            device=device
        )

        fake_batch = generator(z)

        g_loss = -critic(fake_batch).mean()

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        epoch_g_loss.append(g_loss.item())
        epoch_c_loss.append(c_loss.item())

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Critic Loss: {np.mean(epoch_c_loss):.4f} | "
            f"Generator Loss: {np.mean(epoch_g_loss):.4f}"
        )


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

print("\nGenerating synthetic fraud observations...")

generator.eval()

with torch.no_grad():

    z = torch.randn(
        N_SYNTHETIC,
        LATENT_DIM,
        device=device
    )

    synthetic_transformed = generator(z).cpu().numpy()


# ============================================================
# INVERSE TRANSFORM
# ============================================================

synthetic = transformer.inverse_transform(
    synthetic_transformed
)

synthetic_df = pd.DataFrame(
    synthetic,
    columns=FEATURES
)


# ============================================================
# POST PROCESSING
# ============================================================

print("\nApplying behavioral constraints...")

for feature in COUNT_FEATURES:

    synthetic_df[feature] = np.round(
        synthetic_df[feature]
    )

    synthetic_df[feature] = synthetic_df[feature].clip(
        lower=lower[feature],
        upper=upper[feature]
    )

    synthetic_df[feature] = synthetic_df[feature].clip(
        lower=0
    )


# Amount-related features must be non-negative
for feature in [
    "amount_total_window",
    "amount_mean_window",
    "amount_max_window",
    "amount_std_window",
]:

    synthetic_df[feature] = synthetic_df[feature].clip(
        lower=0
    )

    synthetic_df[feature] = synthetic_df[feature].clip(
        lower=lower[feature],
        upper=upper[feature]
    )


# Time gap
synthetic_df[
    "time_since_previous_transaction"
] = synthetic_df[
    "time_since_previous_transaction"
].clip(
    lower=0,
    upper=upper["time_since_previous_transaction"]
)


# Night activity is a rate
synthetic_df[
    "night_activity_rate"
] = synthetic_df[
    "night_activity_rate"
].clip(0, 1)


# Amount change ratio
synthetic_df[
    "amount_change_ratio"
] = synthetic_df[
    "amount_change_ratio"
].clip(
    lower=-10,
    upper=10
)


# Final NaN / infinity cleanup
synthetic_df = synthetic_df.replace(
    [np.inf, -np.inf],
    np.nan
)

synthetic_df = synthetic_df.fillna(
    X.median()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

synthetic_df.to_csv(
    OUTPUT_FILE,
    index=False
)

torch.save(
    {
        "generator_state_dict": generator.state_dict(),
        "transformer": transformer,
        "features": FEATURES
    },
    MODEL_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SYNTHETIC FRAUD SUMMARY")
print("=" * 80)

print(
    synthetic_df[FEATURES].describe().T[
        ["mean", "std", "min", "max"]
    ]
)

print("\nSaved synthetic data:")
print(os.path.abspath(OUTPUT_FILE))

print("\nSaved generator:")
print(os.path.abspath(MODEL_FILE))

print("\nDone.")