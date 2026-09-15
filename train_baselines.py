import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "data/train_2019_2022.csv"
TEST_FILE = "data/test_2023.csv"

OUTPUT_FILE = "results/baseline_predictions.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training and testing datasets...")

train = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

test = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)

train = train.sort_values("timestamp").reset_index(drop=True)
test = test.sort_values("timestamp").reset_index(drop=True)


print("\n========== DATASET INFORMATION ==========")

print(
    f"Training records: {len(train)}"
)

print(
    f"Testing records:  {len(test)}"
)

print(
    f"Training period: {train['timestamp'].min()} "
    f"to {train['timestamp'].max()}"
)

print(
    f"Testing period:  {test['timestamp'].min()} "
    f"to {test['timestamp'].max()}"
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

TARGET = "target_temp_24h"

DROP_COLUMNS = [
    "timestamp",
    "target_temp_24h",
    "target_rain_24h"
]


X_train = train.drop(
    columns=DROP_COLUMNS
)

y_train = train[TARGET]

X_test = test.drop(
    columns=DROP_COLUMNS
)

y_test = test[TARGET]


print("\n========== FEATURE INFORMATION ==========")

print("Number of input features:", X_train.shape[1])

print("Training shape:", X_train.shape)

print("Testing shape:", X_test.shape)


# ============================================================
# XGBOOST BASELINE
# ============================================================

print("\n")
print("========================================")
print("XGBOOST BASELINE")
print("========================================")

print("Training XGBoost...")


xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# XGBoost does not require feature scaling
xgb_model.fit(
    X_train,
    y_train
)


print("XGBoost training complete.")

xgb_predictions = xgb_model.predict(
    X_test
)


# ============================================================
# LSTM DATA PREPARATION
# ============================================================

print("\n")
print("========================================")
print("LSTM BASELINE")
print("========================================")

print("Preparing LSTM sequences...")


# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 24


def create_sequences(X, y, sequence_length):

    X_sequences = []
    y_sequences = []

    for i in range(
        sequence_length,
        len(X)
    ):

        X_sequences.append(
            X[i - sequence_length:i]
        )

        y_sequences.append(
            y.iloc[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


X_train_lstm, y_train_lstm = create_sequences(
    X_train_scaled,
    y_train,
    SEQUENCE_LENGTH
)


X_test_lstm, y_test_lstm = create_sequences(
    X_test_scaled,
    y_test,
    SEQUENCE_LENGTH
)


print(
    "LSTM training shape:",
    X_train_lstm.shape
)

print(
    "LSTM testing shape:",
    X_test_lstm.shape
)


# ============================================================
# BUILD LSTM
# ============================================================

print("\nBuilding LSTM model...")


lstm_model = Sequential([
    
    Input(
        shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    ),

    LSTM(
        64,
        activation="tanh",
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1
    )
])


lstm_model.compile(
    optimizer="adam",
    loss="mse"
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# TRAIN LSTM
# ============================================================

print("Training LSTM...")

history = lstm_model.fit(
    X_train_lstm,
    y_train_lstm,

    epochs=30,

    batch_size=64,

    validation_split=0.1,

    shuffle=False,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("LSTM training complete.")


# ============================================================
# LSTM PREDICTIONS
# ============================================================

lstm_predictions = (
    lstm_model
    .predict(
        X_test_lstm,
        verbose=0
    )
    .flatten()
)


# ============================================================
# METRICS
# ============================================================

def calculate_mape(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    -
                    y_pred[mask]
                )
                /
                y_true[mask]
            )
        )
        * 100
    )


def evaluate_model(
    model_name,
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

    print("\n----------------------------------------")

    print(model_name)

    print("----------------------------------------")

    print(
        f"MAE  : {mae:.4f} °C"
    )

    print(
        f"RMSE : {rmse:.4f} °C"
    )

    print(
        f"MAPE : {mape:.4f} %"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# ============================================================
# EVALUATE XGBOOST
# ============================================================

xgb_results = evaluate_model(
    "XGBoost Baseline",
    y_test,
    xgb_predictions
)


# ============================================================
# EVALUATE LSTM
# ============================================================

lstm_results = evaluate_model(
    "LSTM Baseline",
    y_test_lstm,
    lstm_predictions
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\nSaving predictions...")


# XGBoost predictions correspond to every test record
xgb_output = test[
    [
        "timestamp",
        "target_temp_24h"
    ]
].copy()

xgb_output[
    "xgb_prediction"
] = xgb_predictions


# LSTM starts after first 24 test observations
lstm_output = test.iloc[
    SEQUENCE_LENGTH:
][
    [
        "timestamp",
        "target_temp_24h"
    ]
].copy()


lstm_output[
    "lstm_prediction"
] = lstm_predictions


# Merge predictions
predictions = pd.merge(
    xgb_output,
    lstm_output[
        [
            "timestamp",
            "lstm_prediction"
        ]
    ],
    on="timestamp",
    how="left"
)


predictions.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([
    xgb_results,
    lstm_results
])


metrics.to_csv(
    "results/baseline_metrics.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("========================================")
print("BASELINE EXPERIMENT COMPLETE")
print("========================================")

print(
    "\nPrediction file:"
)

print(
    OUTPUT_FILE
)

print(
    "\nMetrics file:"
)

print(
    "results/baseline_metrics.csv"
)

print("\nFinal comparison:")

print(
    metrics.to_string(
        index=False
    )
)