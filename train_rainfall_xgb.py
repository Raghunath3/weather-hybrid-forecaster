import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = "data/train_2019_2021.csv"
VAL_FILE = "data/validation_2022.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading rainfall datasets...")

train = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

validation = pd.read_csv(
    VAL_FILE,
    parse_dates=["timestamp"]
)


# ============================================================
# TARGET
# ============================================================

TARGET = "target_rain_24h"


# ============================================================
# FEATURES
# ============================================================

exclude_columns = [
    "timestamp",
    "target_temp_24h",
    "target_rain_24h"
]

features = [
    col for col in train.columns
    if col not in exclude_columns
]


X_train = train[features]
y_train = train[TARGET].astype(int)

X_val = validation[features]
y_val = validation[TARGET].astype(int)


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print("\nTraining:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nValidation:")
print(y_val.value_counts())
print(y_val.value_counts(normalize=True))


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    f"\nCalculated scale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# XGBOOST CLASSIFIER
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST RAINFALL CLASSIFIER")
print("========================================")

model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

probabilities = model.predict_proba(
    X_val
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    y_val,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_val,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_val,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_val,
    probabilities
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("XGBOOST RAINFALL RESULTS — 2022")
print("========================================")

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        target_names=[
            "No Rain",
            "Rain"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_val,
        predictions
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = validation[
    [
        "timestamp",
        "target_rain_24h"
    ]
].copy()

results[
    "rain_probability"
] = probabilities

results[
    "rain_prediction_0.5"
] = predictions


results.to_csv(
    "results/rainfall_xgb_validation_2022.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "XGBoost"
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1": [
        f1
    ],

    "ROC_AUC": [
        roc_auc
    ]
})


metrics.to_csv(
    "results/rainfall_xgb_validation_metrics.csv",
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
)


importance.to_csv(
    "results/rainfall_xgb_feature_importance.csv",
    index=False
)


print("\n========================================")
print("SAVED")
print("========================================")

print(
    "results/rainfall_xgb_validation_2022.csv"
)

print(
    "results/rainfall_xgb_validation_metrics.csv"
)

print(
    "results/rainfall_xgb_feature_importance.csv"
)

print("\nRainfall XGBoost training complete.")