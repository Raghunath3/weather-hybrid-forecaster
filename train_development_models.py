import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = "data/train_2019_2021.csv"
VALIDATION_FILE = "data/validation_2022.csv"

OUTPUT_FILE = "results/development_predictions_2022.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading development datasets...")

train = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

validation = pd.read_csv(
    VALIDATION_FILE,
    parse_dates=["timestamp"]
)

train = train.sort_values("timestamp").reset_index(drop=True)
validation = validation.sort_values("timestamp").reset_index(drop=True)


print("\n========================================")
print("DATA INFORMATION")
print("========================================")

print(
    "Training:",
    train["timestamp"].min(),
    "to",
    train["timestamp"].max()
)

print(
    "Validation:",
    validation["timestamp"].min(),
    "to",
    validation["timestamp"].max()
)

print("Training records:", len(train))
print("Validation records:", len(validation))


# ============================================================
# FEATURES
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

X_validation = validation.drop(
    columns=DROP_COLUMNS
)

y_validation = validation[TARGET]


print("\nNumber of features:", X_train.shape[1])


# ============================================================
# XGBOOST
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST")
print("========================================")

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

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_validation
)

print("XGBoost validation predictions generated.")


# ============================================================
# LSTM SCALING
# ============================================================

print("\n========================================")
print("PREPARING LSTM")
print("========================================")

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
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

X_validation_lstm, y_validation_lstm = create_sequences(
    X_validation_scaled,
    y_validation,
    SEQUENCE_LENGTH
)


print(
    "LSTM training shape:",
    X_train_lstm.shape
)

print(
    "LSTM validation shape:",
    X_validation_lstm.shape
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
        activation="tanh"
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(1)
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

lstm_model.fit(
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


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

lstm_predictions = (
    lstm_model
    .predict(
        X_validation_lstm,
        verbose=0
    )
    .flatten()
)


print("LSTM validation predictions generated.")


# ============================================================
# SAVE XGBOOST + LSTM PREDICTIONS
# ============================================================

# XGBoost has predictions for every validation record.
xgb_output = validation[
    [
        "timestamp",
        "target_temp_24h"
    ]
].copy()

xgb_output[
    "xgb_prediction"
] = xgb_predictions


# LSTM begins after the first 24 observations.
lstm_output = validation.iloc[
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


# Merge predictions using timestamp.
output = pd.merge(
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


output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("DEVELOPMENT PREDICTIONS COMPLETE")
print("========================================")

print(
    "Saved to:",
    OUTPUT_FILE
)

print(
    "\nXGBoost predictions:",
    output["xgb_prediction"].notna().sum()
)

print(
    "LSTM predictions:",
    output["lstm_prediction"].notna().sum()
)

print("\nFirst 5 rows:")
print(output.head())