import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# FILES
# ============================================================

TEST_FILE = "data/test_2023.csv"
XGB_FILE = "results/baseline_predictions.csv"
PERSISTENCE_FILE = "results/persistence_predictions.csv"

OUTPUT_FILE = "results/monthly_performance.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test predictions...")

test = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)

xgb = pd.read_csv(
    XGB_FILE,
    parse_dates=["timestamp"]
)

persistence = pd.read_csv(
    PERSISTENCE_FILE,
    parse_dates=["timestamp"]
)


# ============================================================
# MERGE PREDICTIONS
# ============================================================

df = test[
    [
        "timestamp",
        "target_temp_24h"
    ]
].copy()

df = df.merge(
    xgb[
        [
            "timestamp",
            "xgb_prediction"
        ]
    ],
    on="timestamp",
    how="left"
)

df = df.merge(
    persistence[
        [
            "timestamp",
            "persistence_prediction"
        ]
    ],
    on="timestamp",
    how="left"
)


df["month"] = df["timestamp"].dt.month


# ============================================================
# MONTHLY METRICS
# ============================================================

results = []


for month in range(1, 13):

    monthly = df[
        df["month"] == month
    ]

    y_true = monthly["target_temp_24h"]

    xgb_pred = monthly["xgb_prediction"]

    persistence_pred = monthly[
        "persistence_prediction"
    ]

    xgb_mae = mean_absolute_error(
        y_true,
        xgb_pred
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            xgb_pred
        )
    )

    persistence_mae = mean_absolute_error(
        y_true,
        persistence_pred
    )

    persistence_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            persistence_pred
        )
    )

    results.append({

        "Month": month,

        "Records": len(monthly),

        "XGBoost_MAE": xgb_mae,

        "XGBoost_RMSE": xgb_rmse,

        "Persistence_MAE": persistence_mae,

        "Persistence_RMSE": persistence_rmse
    })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MONTHLY PERFORMANCE ANALYSIS")
print("========================================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# DETERMINE WINNER EACH MONTH
# ============================================================

results_df["Winner"] = np.where(
    results_df["XGBoost_MAE"]
    <
    results_df["Persistence_MAE"],
    "XGBoost",
    "Persistence"
)


print("\n========================================")
print("MONTHLY WINNERS")
print("========================================")

print(
    results_df[
        [
            "Month",
            "XGBoost_MAE",
            "Persistence_MAE",
            "Winner"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nMonthly analysis complete.")