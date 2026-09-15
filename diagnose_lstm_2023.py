import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


INPUT_FILE = "results/final_predictions_2023.csv"


print("Loading final 2023 predictions...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n========================================")
print("LSTM 2023 DIAGNOSTIC")
print("========================================")

print(
    "Records:",
    len(df)
)

print(
    "Period:",
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)


# ============================================================
# CREATE ERROR
# ============================================================

df["lstm_error"] = (
    df["actual_temperature_24h"]
    -
    df["lstm_prediction"]
)

df["absolute_error"] = (
    np.abs(df["lstm_error"])
)


# ============================================================
# PREDICTION STATISTICS
# ============================================================

print("\n========================================")
print("LSTM PREDICTION STATISTICS")
print("========================================")

print(
    "\nActual temperature:"
)

print(
    df["actual_temperature_24h"].describe()
)


print(
    "\nLSTM prediction:"
)

print(
    df["lstm_prediction"].describe()
)


print(
    "\nLSTM error:"
)

print(
    df["lstm_error"].describe()
)


# ============================================================
# OVERALL BIAS
# ============================================================

print("\n========================================")
print("LSTM BIAS")
print("========================================")

mean_error = df["lstm_error"].mean()

mean_prediction = df[
    "lstm_prediction"
].mean()

mean_actual = df[
    "actual_temperature_24h"
].mean()


print(
    f"Mean actual temperature:     {mean_actual:.4f} °C"
)

print(
    f"Mean LSTM prediction:         {mean_prediction:.4f} °C"
)

print(
    f"Mean prediction error:        {mean_error:.4f} °C"
)


if mean_error < 0:

    print(
        "\nLSTM generally OVERPREDICTS temperature."
    )

elif mean_error > 0:

    print(
        "\nLSTM generally UNDERPREDICTS temperature."
    )

else:

    print(
        "\nLSTM has approximately zero mean bias."
    )


# ============================================================
# MONTHLY BIAS
# ============================================================

df["month"] = df[
    "timestamp"
].dt.month


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


print("\n========================================")
print("MONTHLY LSTM BIAS")
print("========================================")


monthly_results = []


for month in range(1, 13):

    monthly = df[
        df["month"] == month
    ]

    y_true = monthly[
        "actual_temperature_24h"
    ].values

    y_pred = monthly[
        "lstm_prediction"
    ].values

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

    bias = np.mean(
        y_true - y_pred
    )

    actual_mean = np.mean(
        y_true
    )

    prediction_mean = np.mean(
        y_pred
    )

    monthly_results.append({

        "Month": month_names[month],

        "Records": len(monthly),

        "Actual_Mean": actual_mean,

        "LSTM_Mean": prediction_mean,

        "Bias": bias,

        "MAE": mae,

        "RMSE": rmse
    })


monthly_df = pd.DataFrame(
    monthly_results
)


print(
    monthly_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# EXTREME PREDICTIONS
# ============================================================

print("\n========================================")
print("LSTM PREDICTION RANGE")
print("========================================")


print(
    f"Actual minimum: {df['actual_temperature_24h'].min():.2f} °C"
)

print(
    f"Actual maximum: {df['actual_temperature_24h'].max():.2f} °C"
)

print(
    f"LSTM minimum:   {df['lstm_prediction'].min():.2f} °C"
)

print(
    f"LSTM maximum:   {df['lstm_prediction'].max():.2f} °C"
)


# ============================================================
# LARGEST ERRORS
# ============================================================

print("\n========================================")
print("10 LARGEST LSTM ERRORS")
print("========================================")


largest_errors = df.nlargest(
    10,
    "absolute_error"
)[
    [
        "timestamp",
        "actual_temperature_24h",
        "lstm_prediction",
        "lstm_error",
        "absolute_error"
    ]
]


print(
    largest_errors.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# ERROR BY QUARTER
# ============================================================

df["quarter"] = (
    df["timestamp"].dt.quarter
)


print("\n========================================")
print("QUARTERLY LSTM PERFORMANCE")
print("========================================")


quarter_results = []


for quarter in range(1, 5):

    q = df[
        df["quarter"] == quarter
    ]

    y_true = q[
        "actual_temperature_24h"
    ].values

    y_pred = q[
        "lstm_prediction"
    ].values

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

    r2 = r2_score(
        y_true,
        y_pred
    )

    quarter_results.append({

        "Quarter": f"Q{quarter}",

        "Records": len(q),

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2,

        "Bias": np.mean(
            y_true - y_pred
        )
    })


quarter_df = pd.DataFrame(
    quarter_results
)


print(
    quarter_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE DIAGNOSTIC DATA
# ============================================================

monthly_df.to_csv(
    "results/lstm_monthly_diagnostic_2023.csv",
    index=False
)


quarter_df.to_csv(
    "results/lstm_quarterly_diagnostic_2023.csv",
    index=False
)


largest_errors.to_csv(
    "results/lstm_largest_errors_2023.csv",
    index=False
)


print("\n========================================")
print("LSTM DIAGNOSTIC COMPLETE")
print("========================================")

print(
    "\nSaved:"
)

print(
    "results/lstm_monthly_diagnostic_2023.csv"
)

print(
    "results/lstm_quarterly_diagnostic_2023.csv"
)

print(
    "results/lstm_largest_errors_2023.csv"
)