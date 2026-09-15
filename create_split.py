import pandas as pd


INPUT_FILE = "data/engineered_weather_data.csv"

TRAIN_FILE = "data/train_2019_2022.csv"
TEST_FILE = "data/test_2023.csv"


print("Loading engineered Chennai weather dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\nCreating chronological train/test split...")

train = df[df["timestamp"] < "2023-01-01"].copy()

test = df[df["timestamp"] >= "2023-01-01"].copy()


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

print("\n========== TRAINING DATA ==========")

print("Start:", train["timestamp"].min())
print("End:  ", train["timestamp"].max())
print("Records:", len(train))


print("\n========== TESTING DATA ==========")

print("Start:", test["timestamp"].min())
print("End:  ", test["timestamp"].max())
print("Records:", len(test))


# ============================================================
# CHECK FOR OVERLAP
# ============================================================

print("\n========== OVERLAP CHECK ==========")

train_dates = set(train["timestamp"])
test_dates = set(test["timestamp"])

overlap = train_dates.intersection(test_dates)

print("Overlapping timestamps:", len(overlap))


# ============================================================
# CHECK CHRONOLOGICAL ORDER
# ============================================================

print("\n========== CHRONOLOGICAL CHECK ==========")

print(
    "Training sorted:",
    train["timestamp"].is_monotonic_increasing
)

print(
    "Testing sorted:",
    test["timestamp"].is_monotonic_increasing
)


# ============================================================
# SAVE DATASETS
# ============================================================

train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)


print("\n========================================")
print("CHRONOLOGICAL SPLIT COMPLETE")
print("========================================")

print("Training file:", TRAIN_FILE)
print("Testing file: ", TEST_FILE)