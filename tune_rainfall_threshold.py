import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)


INPUT_FILE = "results/rainfall_xgb_validation_2022.csv"


# ============================================================
# LOAD VALIDATION PREDICTIONS
# ============================================================

df = pd.read_csv(INPUT_FILE)

y_true = df["target_rain_24h"].astype(int)

probabilities = df["rain_probability"]


# ============================================================
# TEST MULTIPLE THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    f2 = fbeta_score(
        y_true,
        predictions,
        beta=2,
        zero_division=0
    )

    results.append({

        "Threshold": round(
            threshold,
            2
        ),

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "F2": f2
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("RAINFALL THRESHOLD ANALYSIS")
print("========================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST THRESHOLDS
# ============================================================

best_f1 = results_df.loc[
    results_df["F1"].idxmax()
]

best_f2 = results_df.loc[
    results_df["F2"].idxmax()
]


print("\n========================================")
print("BEST THRESHOLDS")
print("========================================")

print("\nBest F1 threshold:")

print(
    best_f1.to_string()
)


print("\nBest F2 threshold:")

print(
    best_f2.to_string()
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "results/rainfall_threshold_analysis_2022.csv",
    index=False
)


print("\nSaved:")
print(
    "results/rainfall_threshold_analysis_2022.csv"
)