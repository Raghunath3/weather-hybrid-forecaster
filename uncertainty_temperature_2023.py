import pandas as pd
import numpy as np


# ============================================================
# FILES
# ============================================================

VALIDATION_FILE = "results/development_predictions_2022.csv"
TEST_FILE = "results/final_predictions_2023.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation and final test predictions...")

val = pd.read_csv(
    VALIDATION_FILE,
    parse_dates=["timestamp"]
)

test = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)


# ============================================================
# CHECK COLUMNS
# ============================================================

print("\nValidation columns:")
print(val.columns.tolist())

print("\nTest columns:")
print(test.columns.tolist())


# ============================================================
# VALIDATION DATA
# ============================================================

# Development predictions contain:
# actual_temperature_24h
# xgb_prediction

val_actual = val[
    "target_temp_24h"
].values

val_prediction = val[
    "xgb_prediction"
].values


# ============================================================
# VALIDATION ERRORS
# ============================================================

val_errors = (
    val_actual -
    val_prediction
)

val_abs_errors = np.abs(
    val_errors
)


print("\n========================================")
print("2022 VALIDATION ERROR DISTRIBUTION")
print("========================================")

print(
    f"Validation records: {len(val)}"
)

print(
    f"Mean error: {np.mean(val_errors):.4f} °C"
)

print(
    f"Std error: {np.std(val_errors, ddof=1):.4f} °C"
)

print(
    f"MAE: {np.mean(val_abs_errors):.4f} °C"
)


# ============================================================
# QUANTILE-BASED UNCERTAINTY
# ============================================================

# Absolute validation errors determine
# the radius of the prediction interval.

q50 = np.quantile(
    val_abs_errors,
    0.50
)

q80 = np.quantile(
    val_abs_errors,
    0.80
)

q90 = np.quantile(
    val_abs_errors,
    0.90
)

q95 = np.quantile(
    val_abs_errors,
    0.95
)


print("\n========================================")
print("VALIDATION ERROR QUANTILES")
print("========================================")

print(
    f"50% error radius: {q50:.4f} °C"
)

print(
    f"80% error radius: {q80:.4f} °C"
)

print(
    f"90% error radius: {q90:.4f} °C"
)

print(
    f"95% error radius: {q95:.4f} °C"
)


# ============================================================
# 2023 PREDICTIONS
# ============================================================

test_prediction = test[
    "xgb_prediction"
].values


# ============================================================
# CREATE PREDICTION INTERVALS
# ============================================================

test_results = test[
    [
        "timestamp",
        "actual_temperature_24h",
        "xgb_prediction"
    ]
].copy()


test_results[
    "lower_80"
] = (
    test_prediction - q80
)

test_results[
    "upper_80"
] = (
    test_prediction + q80
)


test_results[
    "lower_90"
] = (
    test_prediction - q90
)

test_results[
    "upper_90"
] = (
    test_prediction + q90
)


test_results[
    "lower_95"
] = (
    test_prediction - q95
)

test_results[
    "upper_95"
] = (
    test_prediction + q95
)


# ============================================================
# COVERAGE TEST
# ============================================================

actual = test_results[
    "actual_temperature_24h"
].values


coverage_80 = np.mean(
    (
        actual >= test_results["lower_80"].values
    )
    &
    (
        actual <= test_results["upper_80"].values
    )
)


coverage_90 = np.mean(
    (
        actual >= test_results["lower_90"].values
    )
    &
    (
        actual <= test_results["upper_90"].values
    )
)


coverage_95 = np.mean(
    (
        actual >= test_results["lower_95"].values
    )
    &
    (
        actual <= test_results["upper_95"].values
    )
)


print("\n========================================")
print("2023 PREDICTION INTERVAL COVERAGE")
print("========================================")

print(
    f"80% interval coverage: "
    f"{coverage_80:.4f} "
    f"({coverage_80 * 100:.2f}%)"
)

print(
    f"90% interval coverage: "
    f"{coverage_90:.4f} "
    f"({coverage_90 * 100:.2f}%)"
)

print(
    f"95% interval coverage: "
    f"{coverage_95:.4f} "
    f"({coverage_95 * 100:.2f}%)"
)


# ============================================================
# INTERVAL WIDTH
# ============================================================

print("\n========================================")
print("AVERAGE INTERVAL WIDTH")
print("========================================")

print(
    f"80% interval width: "
    f"{2 * q80:.4f} °C"
)

print(
    f"90% interval width: "
    f"{2 * q90:.4f} °C"
)

print(
    f"95% interval width: "
    f"{2 * q95:.4f} °C"
)


# ============================================================
# SAVE
# ============================================================

test_results.to_csv(
    "results/xgb_temperature_uncertainty_2023.csv",
    index=False
)


summary = pd.DataFrame({

    "Interval": [
        "80%",
        "90%",
        "95%"
    ],

    "Error_Radius_C": [
        q80,
        q90,
        q95
    ],

    "Interval_Width_C": [
        2 * q80,
        2 * q90,
        2 * q95
    ],

    "Observed_Coverage_2023": [
        coverage_80,
        coverage_90,
        coverage_95
    ]
})


summary.to_csv(
    "results/xgb_temperature_uncertainty_summary_2023.csv",
    index=False
)


print("\n========================================")
print("UNCERTAINTY ANALYSIS COMPLETE")
print("========================================")

print("\nSaved:")

print(
    "results/xgb_temperature_uncertainty_2023.csv"
)

print(
    "results/xgb_temperature_uncertainty_summary_2023.csv"
)