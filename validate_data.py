import pandas as pd


FILE_PATH = "data/chennai_weather_2019_2023.csv"


print("Loading Chennai weather dataset...")

df = pd.read_csv(FILE_PATH, parse_dates=["timestamp"])


print("\n========== DATASET OVERVIEW ==========")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


print("\n========== DATE RANGE ==========")

print("Start:", df["timestamp"].min())
print("End:  ", df["timestamp"].max())


print("\n========== MISSING VALUES ==========")

missing = df.isnull().sum()

print(missing)


print("\n========== DUPLICATE TIMESTAMPS ==========")

duplicates = df["timestamp"].duplicated().sum()

print("Duplicate timestamps:", duplicates)


print("\n========== DATA TYPES ==========")

print(df.dtypes)


print("\n========== BASIC STATISTICS ==========")

print(df.describe())


print("\n========== HOURLY CONTINUITY ==========")

df = df.sort_values("timestamp")

time_difference = df["timestamp"].diff()

expected_difference = pd.Timedelta(hours=1)

gaps = (time_difference > expected_difference).sum()

print("Missing hourly gaps:", gaps)


print("\n========== DATASET VALIDATION COMPLETE ==========")