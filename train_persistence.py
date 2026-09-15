import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# FILE PATH
# ============================================================

TEST_FILE = "data/test_2023.csv"

OUTPUT_FILE = "results/persistence_predictions.csv"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading 2023 test dataset...")

test = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)

test = test.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# PERSISTENCE FORECAST
# ============================================================

print("\nCreating 24-hour persistence predictions...")

# Prediction:
# Temperature 24 hours ahead = current temperature

test["persistence_prediction"] = test["temperature"]


# ============================================================
# TARGET
# ============================================================

y_true = test["target_temp_24h"]

y_pred = test["persistence_prediction"]


# ============================================================
# METRICS
# ============================================================

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

mape = np.mean(
    np.abs(
        (y_true - y_pred) / y_true
    )
) * 100

r2 = r2_score(
    y_true,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("PERSISTENCE BASELINE RESULTS")
print("========================================")

print(f"MAE  : {mae:.4f} °C")
print(f"RMSE : {rmse:.4f} °C")
print(f"MAPE : {mape:.4f} %")
print(f"R²   : {r2:.4f}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output = test[
    [
        "timestamp",
        "temperature",
        "target_temp_24h",
        "persistence_prediction"
    ]
].copy()

output.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nPrediction file saved to:")
print(OUTPUT_FILE)

print("\nPersistence baseline complete.")