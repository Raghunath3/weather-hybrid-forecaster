import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = "data/train_2019_2022.csv"
TEST_FILE = "data/test_2023.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading final rainfall datasets...")

train = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

test = pd.read_csv(
    TEST_FILE,
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

X_test = test[features]
y_test = test[TARGET].astype(int)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("FINAL DATASET")
print("========================================")

print(
    f"Training records: {len(train)}"
)

print(
    f"Test records:     {len(test)}"
)

print(
    f"Training period:  {train['timestamp'].min()} "
    f"to {train['timestamp'].max()}"
)

print(
    f"Test period:      {test['timestamp'].min()} "
    f"to {test['timestamp'].max()}"
)


print("\nTraining class distribution:")

print(
    y_train.value_counts()
)

print(
    y_train.value_counts(
        normalize=True
    )
)


print("\n2023 test class distribution:")

print(
    y_test.value_counts()
)

print(
    y_test.value_counts(
        normalize=True
    )
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = (
    y_train == 0
).sum()

positive = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative / positive
)


print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n========================================")
print("TRAINING FINAL XGBOOST MODEL")
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

print("\nGenerating 2023 predictions...")

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# FROZEN THRESHOLD
# ============================================================

THRESHOLD = 0.20

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

f2 = fbeta_score(
    y_test,
    predictions,
    beta=2,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FINAL RAINFALL RESULTS — 2023")
print("========================================")

print(
    f"Threshold : {THRESHOLD:.2f}"
)

print(
    f"Accuracy  : {accuracy:.4f}"
)

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
    f"F2 Score  : {f2:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "No Rain",
            "Rain"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = test[
    [
        "timestamp",
        "target_rain_24h"
    ]
].copy()


results[
    "rain_probability"
] = probabilities


results[
    "rain_prediction"
] = predictions


results[
    "threshold"
] = THRESHOLD


results.to_csv(
    "results/final_rainfall_predictions_2023.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "XGBoost"
    ],

    "Threshold": [
        THRESHOLD
    ],

    "Accuracy": [
        accuracy
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

    "F2": [
        f2
    ],

    "ROC_AUC": [
        roc_auc
    ]
})


metrics.to_csv(
    "results/final_rainfall_metrics_2023.csv",
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

}).sort_values(
    "Importance",
    ascending=False
)


importance.to_csv(
    "results/final_rainfall_feature_importance.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

import os

os.makedirs(
    "models",
    exist_ok=True
)


model.save_model(
    "models/final_rainfall_xgb.json"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("FINAL RAINFALL EVALUATION COMPLETE")
print("========================================")

print("\nSaved:")

print(
    "results/final_rainfall_predictions_2023.csv"
)

print(
    "results/final_rainfall_metrics_2023.csv"
)

print(
    "results/final_rainfall_feature_importance.csv"
)

print(
    "models/final_rainfall_xgb.json"
)