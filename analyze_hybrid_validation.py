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

INPUT_FILE = "results/development_predictions_2022.csv"

OUTPUT_FILE = "results/hybrid_validation_results.csv"


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("Loading 2022 development predictions...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)


# ============================================================
# ALIGN XGBOOST AND LSTM
# ============================================================

print("\nAligning XGBoost and LSTM predictions...")

df = df.dropna(
    subset=[
        "xgb_prediction",
        "lstm_prediction",
        "target_temp_24h"
    ]
).copy()


print(
    "Common validation records:",
    len(df)
)


# ============================================================
# ACTUAL VALUES
# ============================================================

y_true = df["target_temp_24h"].values

xgb_pred = df["xgb_prediction"].values

lstm_pred = df["lstm_prediction"].values


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_mape(y_true, y_pred):

    mask = y_true != 0

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def evaluate(
    name,
    y_true,
    y_pred
):

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

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# ============================================================
# INDIVIDUAL MODEL PERFORMANCE
# ============================================================

print("\n========================================")
print("2022 VALIDATION PERFORMANCE")
print("========================================")


xgb_result = evaluate(
    "XGBoost",
    y_true,
    xgb_pred
)


lstm_result = evaluate(
    "LSTM",
    y_true,
    lstm_pred
)


# ============================================================
# ERROR CORRELATION
# ============================================================

xgb_error = y_true - xgb_pred

lstm_error = y_true - lstm_pred


error_correlation = np.corrcoef(
    xgb_error,
    lstm_error
)[0, 1]


print(
    f"\nXGBoost MAE : {xgb_result['MAE']:.4f}"
)

print(
    f"LSTM MAE    : {lstm_result['MAE']:.4f}"
)

print(
    f"\nError correlation: {error_correlation:.4f}"
)


# ============================================================
# EQUAL-WEIGHT HYBRID
# ============================================================

equal_weight = (
    0.5 * xgb_pred
    +
    0.5 * lstm_pred
)


equal_result = evaluate(
    "Hybrid 50/50",
    y_true,
    equal_weight
)


print("\n========================================")
print("EQUAL-WEIGHT HYBRID")
print("========================================")

print(
    f"MAE  : {equal_result['MAE']:.4f}"
)

print(
    f"RMSE : {equal_result['RMSE']:.4f}"
)

print(
    f"MAPE : {equal_result['MAPE']:.4f}"
)

print(
    f"R2   : {equal_result['R2']:.4f}"
)


# ============================================================
# SEARCH FOR BEST FUSION WEIGHT
# ============================================================

print("\n========================================")
print("SEARCHING FOR BEST FUSION WEIGHT")
print("========================================")


weights = np.arange(
    0.0,
    1.01,
    0.05
)


weight_results = []


for xgb_weight in weights:

    lstm_weight = 1.0 - xgb_weight

    hybrid_prediction = (
        xgb_weight * xgb_pred
        +
        lstm_weight * lstm_pred
    )

    mae = mean_absolute_error(
        y_true,
        hybrid_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            hybrid_prediction
        )
    )

    weight_results.append({

        "XGBoost_Weight": xgb_weight,

        "LSTM_Weight": lstm_weight,

        "MAE": mae,

        "RMSE": rmse
    })


weight_df = pd.DataFrame(
    weight_results
)


# ============================================================
# BEST WEIGHT
# ============================================================

best_row = weight_df.loc[
    weight_df["MAE"].idxmin()
]


best_xgb_weight = best_row[
    "XGBoost_Weight"
]

best_lstm_weight = best_row[
    "LSTM_Weight"
]


best_hybrid_prediction = (
    best_xgb_weight * xgb_pred
    +
    best_lstm_weight * lstm_pred
)


best_result = evaluate(
    "Optimized Hybrid",
    y_true,
    best_hybrid_prediction
)


print(
    "\nBest XGBoost weight:",
    f"{best_xgb_weight:.2f}"
)

print(
    "Best LSTM weight:",
    f"{best_lstm_weight:.2f}"
)


print("\nOptimized Hybrid Performance:")

print(
    f"MAE  : {best_result['MAE']:.4f}"
)

print(
    f"RMSE : {best_result['RMSE']:.4f}"
)

print(
    f"MAPE : {best_result['MAPE']:.4f}"
)

print(
    f"R2   : {best_result['R2']:.4f}"
)


# ============================================================
# COMPARE ALL MODELS
# ============================================================

comparison = pd.DataFrame([
    xgb_result,
    lstm_result,
    equal_result,
    best_result
])


print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE WEIGHT SEARCH
# ============================================================

weight_df.to_csv(
    "results/fusion_weight_search_2022.csv",
    index=False
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE VALIDATION HYBRID PREDICTIONS
# ============================================================

prediction_output = df[
    [
        "timestamp",
        "target_temp_24h"
    ]
].copy()


prediction_output[
    "xgb_prediction"
] = xgb_pred


prediction_output[
    "lstm_prediction"
] = lstm_pred


prediction_output[
    "hybrid_prediction"
] = best_hybrid_prediction


prediction_output.to_csv(
    "results/hybrid_predictions_2022.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("HYBRID VALIDATION ANALYSIS COMPLETE")
print("========================================")

print(
    "\nSaved:"
)

print(
    "results/hybrid_validation_results.csv"
)

print(
    "results/fusion_weight_search_2022.csv"
)

print(
    "results/hybrid_predictions_2022.csv"
)