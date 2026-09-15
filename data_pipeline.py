import pandas as pd
import numpy as np


INPUT_FILE = "data/chennai_weather_2019_2023.csv"
OUTPUT_FILE = "data/engineered_weather_data.csv"


print("Loading Chennai weather dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Original records: {len(df)}")


# ============================================================
# 1. TIME-BASED FEATURES
# ============================================================

print("\nCreating temporal features...")

df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month

# Cyclical representation of hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Cyclical representation of month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


# ============================================================
# 2. WIND DIRECTION FEATURES
# ============================================================

print("Creating wind-direction features...")

df["wind_direction_sin"] = np.sin(
    np.deg2rad(df["wind_direction"])
)

df["wind_direction_cos"] = np.cos(
    np.deg2rad(df["wind_direction"])
)


# ============================================================
# 3. TEMPERATURE LAG FEATURES
# ============================================================

print("Creating temperature lag features...")

temperature_lags = [1, 3, 6, 12, 24]

for lag in temperature_lags:
    df[f"temperature_lag_{lag}h"] = df["temperature"].shift(lag)


# ============================================================
# 4. OTHER WEATHER LAGS
# ============================================================

print("Creating meteorological lag features...")

df["humidity_lag_1h"] = df["humidity"].shift(1)
df["pressure_lag_1h"] = df["pressure"].shift(1)
df["wind_speed_lag_1h"] = df["wind_speed"].shift(1)
df["rain_lag_1h"] = df["rain"].shift(1)


# ============================================================
# 5. ROLLING WEATHER FEATURES
# ============================================================

print("Creating rolling statistics...")

df["temperature_rolling_mean_24h"] = (
    df["temperature"]
    .rolling(window=24)
    .mean()
)

df["temperature_rolling_std_24h"] = (
    df["temperature"]
    .rolling(window=24)
    .std()
)


# ============================================================
# 6. RATE OF CHANGE FEATURES
# ============================================================

print("Creating change features...")

df["temperature_change_1h"] = (
    df["temperature"] - df["temperature"].shift(1)
)

df["pressure_change_1h"] = (
    df["pressure"] - df["pressure"].shift(1)
)


# ============================================================
# 7. TARGET: TEMPERATURE 24 HOURS AHEAD
# ============================================================

print("Creating 24-hour temperature target...")

df["target_temp_24h"] = (
    df["temperature"].shift(-24)
)


# ============================================================
# 8. TARGET: RAINFALL EVENT 24 HOURS AHEAD
# ============================================================

print("Creating 24-hour rainfall-event target...")

future_rain = df["rain"].shift(-24)

df["target_rain_24h"] = np.where(
    future_rain.notna(),
    (future_rain > 0.1).astype(int),
    np.nan
)


# ============================================================
# 9. REMOVE ROWS CREATED BY LAG/TARGET OPERATIONS
# ============================================================

print("\nRemoving rows with insufficient history...")

before = len(df)

df = df.dropna().reset_index(drop=True)

after = len(df)

print(f"Rows removed: {before - after}")
print(f"Final records: {after}")


# ============================================================
# 10. SAVE ENGINEERED DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("FEATURE ENGINEERING COMPLETE")
print("========================================")

print(f"Saved to: {OUTPUT_FILE}")

print(f"\nFinal rows: {len(df)}")
print(f"Final columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget statistics:")

print("\nTemperature target:")
print(df["target_temp_24h"].describe())

print("\nRainfall-event target:")
print(df["target_rain_24h"].value_counts())