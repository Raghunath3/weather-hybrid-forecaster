import pandas as pd
import numpy as np


FILE_PATH = "data/engineered_weather_data.csv"


print("Loading engineered Chennai weather dataset...")

df = pd.read_csv(
    FILE_PATH,
    parse_dates=["timestamp"]
)


# ============================================================
# 1. BASIC OVERVIEW
# ============================================================

print("\n========== DATASET OVERVIEW ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. DATE RANGE
# ============================================================

print("\n========== DATE RANGE ==========")

print("Start:", df["timestamp"].min())
print("End:  ", df["timestamp"].max())


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing)

print(
    "\nTotal missing values:",
    missing.sum()
)


# ============================================================
# 4. INFINITE VALUES
# ============================================================

print("\n========== INFINITE VALUES ==========")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

infinite_values = np.isinf(
    df[numeric_columns]
).sum().sum()

print("Total infinite values:", infinite_values)


# ============================================================
# 5. DUPLICATE TIMESTAMPS
# ============================================================

print("\n========== DUPLICATE TIMESTAMPS ==========")

duplicates = df["timestamp"].duplicated().sum()

print("Duplicate timestamps:", duplicates)


# ============================================================
# 6. CHRONOLOGICAL ORDER
# ============================================================

print("\n========== CHRONOLOGICAL ORDER ==========")

is_sorted = df["timestamp"].is_monotonic_increasing

print("Timestamps sorted:", is_sorted)


# ============================================================
# 7. HOURLY CONTINUITY
# ============================================================

print("\n========== HOURLY CONTINUITY ==========")

time_difference = df["timestamp"].diff()

expected_difference = pd.Timedelta(hours=1)

gaps = (
    time_difference > expected_difference
).sum()

print("Hourly gaps:", gaps)


# ============================================================
# 8. TEMPERATURE TARGET VALIDATION
# ============================================================

print("\n========== TEMPERATURE TARGET VALIDATION ==========")

# Independently calculate what the target should be
expected_temperature_target = (
    df["temperature"].shift(-24)
)

# Compare only rows where both values exist
comparison = (
    df["target_temp_24h"].iloc[:-24].reset_index(drop=True)
    ==
    expected_temperature_target.iloc[:-24].reset_index(drop=True)
)

print(
    "Correct temperature targets:",
    comparison.sum()
)

print(
    "Incorrect temperature targets:",
    (~comparison).sum()
)


# ============================================================
# 9. RAINFALL TARGET VALIDATION
# ============================================================

print("\n========== RAINFALL TARGET VALIDATION ==========")

expected_rain_target = (
    (df["rain"].shift(-24) > 0.1)
    .astype(int)
)

rain_comparison = (
    df["target_rain_24h"].iloc[:-24].astype(int).reset_index(drop=True)
    ==
    expected_rain_target.iloc[:-24].reset_index(drop=True)
)

print(
    "Correct rainfall targets:",
    rain_comparison.sum()
)

print(
    "Incorrect rainfall targets:",
    (~rain_comparison).sum()
)


# ============================================================
# 10. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")

print("\nTemperature target:")
print(df["target_temp_24h"].describe())

print("\nRainfall event target:")
print(
    df["target_rain_24h"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 11. FEATURE VALUE CHECK
# ============================================================

print("\n========== FEATURE RANGE CHECK ==========")

print("\nHour:")
print(df["hour"].min(), "to", df["hour"].max())

print("\nMonth:")
print(df["month"].min(), "to", df["month"].max())

print("\nWind direction:")
print(
    df["wind_direction"].min(),
    "to",
    df["wind_direction"].max()
)

print("\nWind direction sin range:")
print(
    df["wind_direction_sin"].min(),
    "to",
    df["wind_direction_sin"].max()
)

print("\nWind direction cos range:")
print(
    df["wind_direction_cos"].min(),
    "to",
    df["wind_direction_cos"].max()
)


# ============================================================
# 12. FINAL
# ============================================================

print("\n========================================")
print("ENGINEERED DATA VALIDATION COMPLETE")
print("========================================")