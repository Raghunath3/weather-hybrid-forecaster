import pandas as pd


INPUT_FILE = "data/engineered_weather_data.csv"

TRAIN_FILE = "data/train_2019_2021.csv"
VALIDATION_FILE = "data/validation_2022.csv"
TEST_FILE = "data/test_2023.csv"


print("Loading engineered Chennai weather dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train = df[
    df["timestamp"] < "2022-01-01"
].copy()


validation = df[
    (df["timestamp"] >= "2022-01-01") &
    (df["timestamp"] < "2023-01-01")
].copy()


test = df[
    df["timestamp"] >= "2023-01-01"
].copy()


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\n========================================")
print("TRAINING DATA")
print("========================================")

print("Start:", train["timestamp"].min())
print("End:  ", train["timestamp"].max())
print("Records:", len(train))


print("\n========================================")
print("VALIDATION DATA")
print("========================================")

print("Start:", validation["timestamp"].min())
print("End:  ", validation["timestamp"].max())
print("Records:", len(validation))


print("\n========================================")
print("FINAL TEST DATA")
print("========================================")

print("Start:", test["timestamp"].min())
print("End:  ", test["timestamp"].max())
print("Records:", len(test))


# ============================================================
# OVERLAP CHECK
# ============================================================

print("\n========================================")
print("OVERLAP CHECK")
print("========================================")

train_times = set(train["timestamp"])
validation_times = set(validation["timestamp"])
test_times = set(test["timestamp"])


print(
    "Train ∩ Validation:",
    len(train_times.intersection(validation_times))
)

print(
    "Train ∩ Test:",
    len(train_times.intersection(test_times))
)

print(
    "Validation ∩ Test:",
    len(validation_times.intersection(test_times))
)


# ============================================================
# CHRONOLOGICAL ORDER CHECK
# ============================================================

print("\n========================================")
print("CHRONOLOGICAL CHECK")
print("========================================")

print(
    "Training sorted:",
    train["timestamp"].is_monotonic_increasing
)

print(
    "Validation sorted:",
    validation["timestamp"].is_monotonic_increasing
)

print(
    "Testing sorted:",
    test["timestamp"].is_monotonic_increasing
)


# ============================================================
# SAVE
# ============================================================

train.to_csv(
    TRAIN_FILE,
    index=False
)

validation.to_csv(
    VALIDATION_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)


print("\n========================================")
print("VALIDATION SPLIT COMPLETE")
print("========================================")

print("Training file:")
print(TRAIN_FILE)

print("\nValidation file:")
print(VALIDATION_FILE)

print("\nTesting file:")
print(TEST_FILE)