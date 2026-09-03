from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "behavioral_sequences_v2.csv"
)

HMM_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hmm_v3_results.csv"
)

BAYESIAN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "bayesian_risk_results.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_model_comparison.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


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
# LOAD DATA
# ============================================================

print("=" * 80)
print("FINAL FAIR MODEL EVALUATION")
print("=" * 80)

print("\nLoading behavioral dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Total observations: {len(df):,}")
print(f"Total entities: {df['entity_id'].nunique():,}")


# ============================================================
# SAME ENTITY-LEVEL SPLIT
# ============================================================

print("\nCreating common entity-level train/test split...")

entities = (
    df["entity_id"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(RANDOM_STATE)

rng.shuffle(entities)

split_index = int(
    len(entities) * 0.80
)

train_entities = set(
    entities[:split_index]
)

test_entities = set(
    entities[split_index:]
)

test_df = df[
    df["entity_id"].isin(test_entities)
].copy()

train_df = df[
    df["entity_id"].isin(train_entities)
].copy()

print(f"Training entities: {len(train_entities):,}")
print(f"Testing entities : {len(test_entities):,}")
print(f"Test observations: {len(test_df):,}")

y_test = test_df["isFraud"].astype(int).to_numpy()


# ============================================================
# METRIC FUNCTION
# ============================================================

results = []


def evaluate_model(name, y_true, y_pred, y_score=None):

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    auc = np.nan

    if y_score is not None:

        try:
            auc = roc_auc_score(
                y_true,
                y_score,
            )
        except ValueError:
            pass

    results.append(
        {
            "model": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": auc,
            "true_negatives": cm[0, 0],
            "false_positives": cm[0, 1],
            "false_negatives": cm[1, 0],
            "true_positives": cm[1, 1],
        }
    )

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    if not np.isnan(auc):
        print(f"ROC-AUC   : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# 1. RULE-BASED MODEL
# ============================================================

print("\n\n[1/4] Evaluating Rule-Based Baseline...")

rule_score = np.zeros(len(test_df))


def add_rule(condition, points):
    global rule_score
    rule_score[condition] += points


add_rule(
    test_df["transaction_count_window"] >= 10,
    20,
)

add_rule(
    test_df["amount_total_window"] >= 1000,
    15,
)

add_rule(
    test_df["unique_devices_window"] >= 3,
    15,
)

add_rule(
    test_df["device_entity_count"] >= 3,
    20,
)

add_rule(
    test_df["unique_payment_emails_window"] >= 2,
    10,
)

add_rule(
    test_df["unique_receiver_emails_window"] >= 2,
    10,
)

add_rule(
    test_df["unique_addresses_window"] >= 2,
    10,
)

add_rule(
    test_df["unique_products_window"] >= 3,
    5,
)

add_rule(
    test_df["time_since_previous_transaction"] <= 60,
    20,
)

add_rule(
    test_df["amount_change_ratio"].abs() >= 2,
    10,
)

add_rule(
    test_df["night_activity_rate"] >= 0.75,
    5,
)


rule_prediction = (
    rule_score >= 50
).astype(int)


evaluate_model(
    "Rule-Based Baseline",
    y_test,
    rule_prediction,
    rule_score,
)


# ============================================================
# 2. LOGISTIC REGRESSION
# ============================================================

print("\n\n[2/4] Evaluating Logistic Regression...")

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()


# Handle invalid values

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan,
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan,
)

X_train = X_train.fillna(
    X_train.median()
)

X_test = X_test.fillna(
    X_train.median()
)


# Log transform skewed positive features

LOG_FEATURES = [
    "transaction_count_window",
    "amount_total_window",
    "amount_mean_window",
    "amount_max_window",
    "amount_std_window",
    "time_since_previous_transaction",
    "device_entity_count",
]


for feature in LOG_FEATURES:

    X_train[feature] = np.log1p(
        X_train[feature].clip(lower=0)
    )

    X_test[feature] = np.log1p(
        X_test[feature].clip(lower=0)
    )


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train_scaled,
    train_df["isFraud"].astype(int),
)


logistic_probability = model.predict_proba(
    X_test_scaled
)[:, 1]

logistic_prediction = (
    logistic_probability >= 0.50
).astype(int)


evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_prediction,
    logistic_probability,
)


# ============================================================
# 3. HMM TRAJECTORY MODEL
# ============================================================

print("\n\n[3/4] Evaluating HMM Trajectory Model...")

if not HMM_FILE.exists():

    print(
        f"HMM results file not found:\n{HMM_FILE}"
    )

else:

    hmm = pd.read_csv(HMM_FILE)

    print(
        f"HMM observations available: {len(hmm):,}"
    )

    print(
        "HMM columns:",
        list(hmm.columns)
    )

    # Keep only common test entities

    hmm_test = hmm[
        hmm["entity_id"].isin(test_entities)
    ].copy()

    print(
        f"HMM test observations: "
        f"{len(hmm_test):,}"
    )


    # Prediction column candidates

    prediction_candidates = [
        "hmm_prediction",
        "trajectory_prediction",
        "prediction",
        "predicted_fraud",
        "fraud_prediction",
    ]


    # Score column candidates

    score_candidates = [
        "trajectory_score",
        "active_abuse_probability",
        "active_probability",
        "fraud_probability",
        "risk_score",
    ]


    pred_col = next(
        (
            col
            for col in prediction_candidates
            if col in hmm_test.columns
        ),
        None,
    )


    score_col = next(
        (
            col
            for col in score_candidates
            if col in hmm_test.columns
        ),
        None,
    )


    if pred_col is None and score_col is None:

        print(
            "Could not automatically identify "
            "HMM prediction/score column."
        )

    else:

        if pred_col is not None:

            hmm_pred = (
                hmm_test[pred_col]
                .astype(int)
                .to_numpy()
            )

        else:

            hmm_pred = (
                hmm_test[score_col] >= 0.60
            ).astype(int).to_numpy()


        if score_col is not None:

            hmm_score = (
                hmm_test[score_col]
                .astype(float)
                .to_numpy()
            )

        else:

            hmm_score = None


        evaluate_model(
            "HMM Trajectory",
            hmm_test["isFraud"].astype(int),
            hmm_pred,
            hmm_score,
        )


# ============================================================
# 4. BAYESIAN RISK
# ============================================================

print("\n\n[4/4] Evaluating Bayesian Risk Model...")

if not BAYESIAN_FILE.exists():

    print(
        f"Bayesian results file not found:\n"
        f"{BAYESIAN_FILE}"
    )

else:

    bayes = pd.read_csv(
        BAYESIAN_FILE
    )

    print(
        f"Bayesian observations available: "
        f"{len(bayes):,}"
    )

    print(
        "Bayesian columns:",
        list(bayes.columns)
    )


    # Keep only common test entities

    bayes_test = bayes[
        bayes["entity_id"].isin(test_entities)
    ].copy()


    # Bayesian prediction candidates

    prediction_candidates = [
        "bayesian_prediction",
        "prediction",
        "predicted_fraud",
        "fraud_prediction",
    ]


    # Bayesian probability candidates

    score_candidates = [
        "bayesian_fraud_probability",
        "risk_score",
        "fraud_probability",
        "predicted_probability",
        "probability",
    ]


    pred_col = next(
        (
            col
            for col in prediction_candidates
            if col in bayes_test.columns
        ),
        None,
    )


    score_col = next(
        (
            col
            for col in score_candidates
            if col in bayes_test.columns
        ),
        None,
    )


    if pred_col is None and score_col is None:

        print(
            "Could not automatically identify "
            "Bayesian prediction/score column."
        )

    else:

        if pred_col is not None:

            bayes_pred = (
                bayes_test[pred_col]
                .astype(int)
                .to_numpy()
            )

        else:

            bayes_pred = (
                bayes_test[score_col] >= 0.50
            ).astype(int).to_numpy()


        if score_col is not None:

            bayes_score = (
                bayes_test[score_col]
                .astype(float)
                .to_numpy()
            )

        else:

            bayes_score = None


        evaluate_model(
            "Bayesian Risk Fusion",
            bayes_test["isFraud"].astype(int),
            bayes_pred,
            bayes_score,
        )


# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)


if len(results_df) > 0:

    print(
        results_df[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )


    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


    print(
        f"\nSaved comparison to:\n{OUTPUT_FILE}"
    )

else:

    print(
        "No model results were generated."
    )


print("\nDone.")