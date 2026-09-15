import pandas as pd
import numpy as np


INPUT_FILE = "results/hybrid_predictions_2022.csv"


print("Loading hybrid predictions...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# CREATE RESIDUAL
# ============================================================

df["residual"] = (
    df["target_temp_24h"]
    -
    df["hybrid_prediction"]
)


# ============================================================
# CHECK ORIGINAL HYBRID
# ============================================================

print("\n========================================")
print("BASIC DATA CHECK")
print("========================================")

print("Rows:", len(df))

print(
    "Start:",
    df["timestamp"].min()
)

print(
    "End:",
    df["timestamp"].max()
)


# ============================================================
# RESIDUAL FEATURES
# ============================================================

df["residual_lag_1"] = (
    df["residual"].shift(1)
)

df["residual_lag_24"] = (
    df["residual"].shift(24)
)


# ============================================================
# CHECK TEMPORAL AVAILABILITY
# ============================================================

print("\n========================================")
print("TEMPORAL LEAKAGE CHECK")
print("========================================")


# At timestamp t, residual_lag_1 should come
# from timestamp t-1.
#
# residual_lag_24 should come from timestamp t-24.

check_1 = (
    df["residual_lag_1"]
    .iloc[1:]
    .reset_index(drop=True)
    ==
    df["residual"]
    .iloc[:-1]
    .reset_index(drop=True)
)


check_24 = (
    df["residual_lag_24"]
    .iloc[24:]
    .reset_index(drop=True)
    ==
    df["residual"]
    .iloc[:-24]
    .reset_index(drop=True)
)


print(
    "Residual lag-1 alignment:",
    bool(check_1.all())
)

print(
    "Residual lag-24 alignment:",
    bool(check_24.all())
)


# ============================================================
# CHECK WHETHER RESIDUAL USES FUTURE TARGET
# ============================================================

print("\n========================================")
print("FUTURE TARGET CHECK")
print("========================================")


# The residual at timestamp t is:
#
# actual temperature at t+24
# minus prediction made for t+24.
#
# Therefore residual(t) is NOT available
# at timestamp t.
#
# It becomes known only after t+24.

print(
    "\nImportant:"
)

print(
    "Residual(t) becomes known only after the"
)

print(
    "actual target at t+24 has occurred."
)


print(
    "\nTherefore residual_lag_1 at time t"
)

print(
    "contains information about the actual"
)

print(
    "temperature at approximately t+23."
)

print(
    "That is NOT available when forecasting"
)

print(
    "temperature at t+24."
)


# ============================================================
# IDENTIFY PROBLEMATIC FEATURES
# ============================================================

print("\n========================================")
print("FEATURE SAFETY")
print("========================================")


safe_features = [
    "hybrid_prediction"
]

unsafe_features = [
    "residual_lag_1",
    "residual_lag_24"
]


print("\nSAFE FEATURES:")

for feature in safe_features:
    print("  ", feature)


print("\nPOTENTIALLY LEAKING FEATURES:")

for feature in unsafe_features:
    print("  ", feature)


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n========================================")
print("AUDIT CONCLUSION")
print("========================================")

print(
    "\nResidual lag features must NOT be used"
)

print(
    "as currently implemented for a true"
)

print(
    "24-hour-ahead forecasting system."
)

print(
    "\nThe 55.77% improvement should therefore"
)

print(
    "NOT be reported as a valid research result."
)

print(
    "\nWe will redesign residual correction using"
)

print(
    "only information genuinely available at"
)

print(
    "forecast time."
)