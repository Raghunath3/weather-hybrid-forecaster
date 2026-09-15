import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# FILES
# ============================================================

INPUT_FILE = "results/final_predictions_2023.csv"

OUTPUT_FILE = "results/monthly_final_2023_metrics.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading final 2023 predictions...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print("Records:", len(df))

print(
    "Period:",
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# ============================================================
# MODEL COLUMNS
# ============================================================

models = {
    "Persistence": "persistence_prediction",
    "XGBoost": "xgb_prediction",
    "LSTM": "lstm_prediction",
    "Hybrid": "hybrid_90_10_prediction"
}

actual_column = "actual_temperature_24h"


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_mape(y_true, y_pred):

    mask = y_true != 0

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    -
                    y_pred[mask]
                )
                /
                y_true[mask]
            )
        )
        * 100
    )


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = calculate_mape(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, mape, r2


# ============================================================
# MONTHLY ANALYSIS
# ============================================================

df["month"] = df["timestamp"].dt.month


month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


results = []


for month in range(1, 13):

    monthly = df[
        df["month"] == month
    ]

    if len(monthly) == 0:
        continue

    y_true = monthly[
        actual_column
    ].values

    for model_name, prediction_column in models.items():

        y_pred = monthly[
            prediction_column
        ].values

        mae, rmse, mape, r2 = calculate_metrics(
            y_true,
            y_pred
        )

        results.append({

            "Month_Number": month,

            "Month": month_names[month],

            "Records": len(monthly),

            "Model": model_name,

            "MAE": mae,

            "RMSE": rmse,

            "MAPE": mape,

            "R2": r2
        })


results_df = pd.DataFrame(results)


# ============================================================
# PRINT MONTHLY RESULTS
# ============================================================

print("\n========================================")
print("MONTHLY 2023 PERFORMANCE")
print("========================================")


for month in range(1, 13):

    month_df = results_df[
        results_df["Month_Number"] == month
    ]

    print(
        f"\n--- {month_names[month]} "
        f"({month_df['Records'].iloc[0]} records) ---"
    )

    display_df = month_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "MAPE",
            "R2"
        ]
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# BEST / WORST MONTH FOR EACH MODEL
# ============================================================

print("\n========================================")
print("BEST AND WORST MONTHS")
print("========================================")


for model in models.keys():

    model_df = results_df[
        results_df["Model"] == model
    ]

    best_row = model_df.loc[
        model_df["MAE"].idxmin()
    ]

    worst_row = model_df.loc[
        model_df["MAE"].idxmax()
    ]

    print(
        f"\n{model}:"
    )

    print(
        f"  Best month : "
        f"{best_row['Month']} "
        f"(MAE = {best_row['MAE']:.4f} °C)"
    )

    print(
        f"  Worst month: "
        f"{worst_row['Month']} "
        f"(MAE = {worst_row['MAE']:.4f} °C)"
    )


# ============================================================
# WHICH MODEL WINS EACH MONTH?
# ============================================================

print("\n========================================")
print("BEST MODEL BY MONTH")
print("========================================")


monthly_winners = []


for month in range(1, 13):

    month_df = results_df[
        results_df["Month_Number"] == month
    ]

    winner = month_df.loc[
        month_df["MAE"].idxmin()
    ]

    monthly_winners.append({

        "Month": winner["Month"],

        "Best_Model": winner["Model"],

        "Best_MAE": winner["MAE"]
    })

    print(
        f"{winner['Month']:<12} "
        f"{winner['Model']:<12} "
        f"MAE = {winner['MAE']:.4f}"
    )


# ============================================================
# COUNT MONTHLY WINS
# ============================================================

winner_df = pd.DataFrame(
    monthly_winners
)


win_counts = (
    winner_df["Best_Model"]
    .value_counts()
)


print("\n========================================")
print("MONTHLY WIN COUNT")
print("========================================")


for model in models.keys():

    print(
        f"{model:<12}: "
        f"{win_counts.get(model, 0)} / 12 months"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


winner_df.to_csv(
    "results/monthly_model_winners_2023.csv",
    index=False
)


print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "results/monthly_model_winners_2023.csv"
)