import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# CONFIGURATION
# ============================================================

ENGINEERED_FILE = "data/engineered_weather_data.csv"

TRAIN_FILE = "data/train_2019_2022.csv"

TEST_FILE = "data/test_2023.csv"

PREDICTIONS_FILE = "results/final_predictions_2023.csv"

METRICS_FILE = "results/final_metrics_2023.csv"


# Fixed hybrid weight selected using 2022 validation
XGB_WEIGHT = 0.90
LSTM_WEIGHT = 0.10

SEQUENCE_LENGTH = 24


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # Current weather
    "temperature",
    "humidity",
    "pressure",
    "rain",
    "cloud_cover",
    "wind_speed",
    "wind_direction",

    # Time
    "hour",
    "month",

    # Cyclical time
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # Wind direction
    "wind_direction_sin",
    "wind_direction_cos",

    # Temperature lags
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",

    # Other weather lags
    "humidity_lag_1h",
    "pressure_lag_1h",
    "wind_speed_lag_1h",
    "rain_lag_1h",

    # Rolling statistics
    "temperature_rolling_mean_24h",
    "temperature_rolling_std_24h",

    # Weather changes
    "temperature_change_1h",
    "pressure_change_1h"
]


TARGET = "target_temp_24h"


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_mape(y_true, y_pred):

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

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# ============================================================
# 1. LOAD DATA
# ============================================================

print("========================================")
print("FINAL 2023 EVALUATION")
print("========================================")

print("\nLoading engineered dataset...")

df = pd.read_csv(
    ENGINEERED_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    "Total engineered records:",
    len(df)
)


# ============================================================
# 2. LOAD TRAIN / TEST PERIODS
# ============================================================

train_period = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

test_period = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)


train_start = train_period["timestamp"].min()
train_end = train_period["timestamp"].max()

test_start = test_period["timestamp"].min()
test_end = test_period["timestamp"].max()


print("\nTraining period:")
print(
    train_start,
    "to",
    train_end
)

print(
    "Training records:",
    len(train_period)
)


print("\nFinal test period:")
print(
    test_start,
    "to",
    test_end
)

print(
    "Test records:",
    len(test_period)
)


# ============================================================
# 3. CREATE TRAIN / TEST DATA FROM ENGINEERED DATA
# ============================================================

train_df = df[
    (df["timestamp"] >= train_start)
    &
    (df["timestamp"] <= train_end)
].copy()


test_df = df[
    (df["timestamp"] >= test_start)
    &
    (df["timestamp"] <= test_end)
].copy()


train_df = train_df.sort_values(
    "timestamp"
).reset_index(drop=True)


test_df = test_df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ============================================================
# 4. VERIFY SPLIT
# ============================================================

print("\n========================================")
print("FINAL SPLIT CHECK")
print("========================================")

print(
    "Training rows:",
    len(train_df)
)

print(
    "Test rows:",
    len(test_df)
)

print(
    "Training end:",
    train_df["timestamp"].max()
)

print(
    "Test start:",
    test_df["timestamp"].min()
)


overlap = set(
    train_df["timestamp"]
).intersection(
    set(test_df["timestamp"])
)


print(
    "Timestamp overlap:",
    len(overlap)
)


if len(overlap) != 0:

    raise ValueError(
        "Training and test timestamps overlap."
    )


if train_df["timestamp"].max() >= test_df["timestamp"].min():

    raise ValueError(
        "Chronological split is invalid."
    )


# ============================================================
# 5. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    print("\nMissing features:")

    for feature in missing_features:
        print(
            "  ",
            feature
        )

    raise ValueError(
        "Required features are missing."
    )


print(
    "\nAll 28 forecasting features found."
)


# ============================================================
# 6. PREPARE XGBOOST DATA
# ============================================================

X_train_xgb = train_df[
    FEATURES
]

y_train = train_df[
    TARGET
].values


X_test_xgb = test_df[
    FEATURES
]

y_test = test_df[
    TARGET
].values


print("\n========================================")
print("XGBOOST")
print("========================================")


print(
    "Training XGBoost on 2019-2022..."
)


# ============================================================
# 7. TRAIN XGBOOST
# ============================================================

xgb_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=SEED,

    n_jobs=-1,

    objective="reg:squarederror"
)


xgb_model.fit(
    X_train_xgb,
    y_train
)


print(
    "XGBoost training complete."
)


# ============================================================
# 8. XGBOOST TEST PREDICTIONS
# ============================================================

xgb_predictions = xgb_model.predict(
    X_test_xgb
)


print(
    "XGBoost test predictions:",
    len(xgb_predictions)
)


# ============================================================
# 9. PERSISTENCE BASELINE
# ============================================================

print("\n========================================")
print("PERSISTENCE BASELINE")
print("========================================")


# For a 24-hour-ahead forecast:
#
# predicted temperature at t+24
# =
# observed temperature at t
#
# This is a legitimate naïve forecasting baseline.

persistence_predictions = (
    test_df["temperature"].values
)


print(
    "Persistence predictions:",
    len(persistence_predictions)
)


# ============================================================
# 10. PREPARE LSTM SCALING
# ============================================================

print("\n========================================")
print("LSTM")
print("========================================")


print(
    "Fitting MinMaxScaler on training data only..."
)


scaler = MinMaxScaler()


scaler.fit(
    train_df[FEATURES]
)


# Transform training and test features
# using the scaler fitted ONLY on training data.

train_scaled = scaler.transform(
    train_df[FEATURES]
)

test_scaled = scaler.transform(
    test_df[FEATURES]
)


# ============================================================
# 11. CREATE LSTM TRAINING SEQUENCES
# ============================================================

def create_training_sequences(
    X,
    y,
    sequence_length
):

    X_sequences = []
    y_values = []

    for i in range(
        sequence_length,
        len(X)
    ):

        X_sequences.append(
            X[
                i - sequence_length:i
            ]
        )

        y_values.append(
            y[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_values)
    )


X_lstm_train, y_lstm_train = (
    create_training_sequences(
        train_scaled,
        y_train,
        SEQUENCE_LENGTH
    )
)


print(
    "LSTM training shape:",
    X_lstm_train.shape
)

print(
    "LSTM target shape:",
    y_lstm_train.shape
)


# ============================================================
# 12. CREATE TEST SEQUENCES WITH PREVIOUS 24 HOURS
# ============================================================

# IMPORTANT:
#
# The first 2023 prediction needs the previous
# 24 hours of known weather observations.
#
# Therefore we prepend the final 24 training
# observations to the 2023 test observations.
#
# This does NOT leak future information.
#
# All of these observations were already known
# before the corresponding 2023 forecasts.

test_context = train_scaled[
    -SEQUENCE_LENGTH:
]


combined_test_scaled = np.vstack([
    test_context,
    test_scaled
])


X_lstm_test = []


for i in range(
    SEQUENCE_LENGTH,
    len(combined_test_scaled)
):

    X_lstm_test.append(
        combined_test_scaled[
            i - SEQUENCE_LENGTH:i
        ]
    )


X_lstm_test = np.array(
    X_lstm_test
)


print(
    "LSTM test shape:",
    X_lstm_test.shape
)


if len(X_lstm_test) != len(test_df):

    raise ValueError(
        "LSTM test sequence count does not "
        "match test records."
    )


# ============================================================
# 13. BUILD LSTM MODEL
# ============================================================

print(
    "\nBuilding LSTM model..."
)


lstm_model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            SEQUENCE_LENGTH,
            len(FEATURES)
        )
    ),

    tf.keras.layers.LSTM(
        64,
        activation="tanh"
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1
    )
])


lstm_model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=["mae"]
)


lstm_model.summary()


# ============================================================
# 14. TRAIN LSTM
# ============================================================

print(
    "\nTraining LSTM..."
)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


history = lstm_model.fit(

    X_lstm_train,

    y_lstm_train,

    epochs=30,

    batch_size=64,

    validation_split=0.10,

    shuffle=False,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print(
    "\nLSTM training complete."
)


print(
    "Epochs actually trained:",
    len(history.history["loss"])
)


# ============================================================
# 15. LSTM TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating LSTM 2023 predictions..."
)


lstm_predictions = (
    lstm_model.predict(
        X_lstm_test,
        verbose=0
    )
    .reshape(-1)
)


print(
    "LSTM test predictions:",
    len(lstm_predictions)
)


# ============================================================
# 16. FIXED 90/10 HYBRID
# ============================================================

print("\n========================================")
print("FIXED 90/10 HYBRID")
print("========================================")


print(
    "XGBoost weight:",
    XGB_WEIGHT
)

print(
    "LSTM weight:",
    LSTM_WEIGHT
)


hybrid_predictions = (

    XGB_WEIGHT
    *
    xgb_predictions

    +

    LSTM_WEIGHT
    *
    lstm_predictions
)


# ============================================================
# 17. VERIFY ALL PREDICTIONS
# ============================================================

if not (
    len(y_test)
    ==
    len(xgb_predictions)
    ==
    len(lstm_predictions)
    ==
    len(hybrid_predictions)
    ==
    len(persistence_predictions)
):

    raise ValueError(
        "Prediction lengths do not match."
    )


# ============================================================
# 18. EVALUATE ALL MODELS
# ============================================================

print("\n========================================")
print("FINAL 2023 TEST RESULTS")
print("========================================")


persistence_result = evaluate_model(

    "Persistence",

    y_test,

    persistence_predictions
)


xgb_result = evaluate_model(

    "XGBoost",

    y_test,

    xgb_predictions
)


lstm_result = evaluate_model(

    "LSTM",

    y_test,

    lstm_predictions
)


hybrid_result = evaluate_model(

    "Hybrid 90/10",

    y_test,

    hybrid_predictions
)


results = pd.DataFrame([

    persistence_result,

    xgb_result,

    lstm_result,

    hybrid_result

])


print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 19. CALCULATE IMPROVEMENTS
# ============================================================

xgb_mae = xgb_result["MAE"]

hybrid_mae = hybrid_result["MAE"]

persistence_mae = persistence_result["MAE"]


hybrid_vs_xgb = (

    (
        xgb_mae
        -
        hybrid_mae
    )
    /
    xgb_mae

) * 100


hybrid_vs_persistence = (

    (
        persistence_mae
        -
        hybrid_mae
    )
    /
    persistence_mae

) * 100


print(
    "\nHybrid MAE improvement vs XGBoost:",
    f"{hybrid_vs_xgb:.2f}%"
)


print(
    "Hybrid MAE improvement vs Persistence:",
    f"{hybrid_vs_persistence:.2f}%"
)


# ============================================================
# 20. SAVE METRICS
# ============================================================

results.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# 21. SAVE ALL PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "timestamp":
        test_df["timestamp"],

    "actual_temperature_24h":
        y_test,

    "persistence_prediction":
        persistence_predictions,

    "xgb_prediction":
        xgb_predictions,

    "lstm_prediction":
        lstm_predictions,

    "hybrid_90_10_prediction":
        hybrid_predictions
})


# Errors
predictions[
    "persistence_error"
] = (
    predictions[
        "actual_temperature_24h"
    ]
    -
    predictions[
        "persistence_prediction"
    ]
)


predictions[
    "xgb_error"
] = (
    predictions[
        "actual_temperature_24h"
    ]
    -
    predictions[
        "xgb_prediction"
    ]
)


predictions[
    "lstm_error"
] = (
    predictions[
        "actual_temperature_24h"
    ]
    -
    predictions[
        "lstm_prediction"
    ]
)


predictions[
    "hybrid_error"
] = (
    predictions[
        "actual_temperature_24h"
    ]
    -
    predictions[
        "hybrid_90_10_prediction"
    ]
)


predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# 22. SAVE LSTM MODEL
# ============================================================

lstm_model.save(
    "models/final_lstm_2023.keras"
)


# ============================================================
# 23. SAVE XGBOOST MODEL
# ============================================================

xgb_model.save_model(
    "models/final_xgboost_2023.json"
)


# ============================================================
# 24. FINAL INFORMATION
# ============================================================

print("\n========================================")
print("FINAL EVALUATION COMPLETE")
print("========================================")


print(
    "\nTraining period:"
)

print(
    train_start,
    "to",
    train_end
)


print(
    "\nUntouched test period:"
)

print(
    test_start,
    "to",
    test_end
)


print(
    "\nFixed hybrid:"
)

print(
    "90% XGBoost + 10% LSTM"
)


print(
    "\nResidual correction:"
)

print(
    "NOT USED"
)


print(
    "\nSaved files:"
)

print(
    METRICS_FILE
)

print(
    PREDICTIONS_FILE
)

print(
    "models/final_lstm_2023.keras"
)

print(
    "models/final_xgboost_2023.json"
)


print(
    "\n2023 evaluation is complete."
)