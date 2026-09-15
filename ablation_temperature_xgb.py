import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# FILES
# ============================================================

TRAIN_FILE = "data/train_2019_2021.csv"
VAL_FILE = "data/validation_2022.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

train = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["timestamp"]
)

validation = pd.read_csv(
    VAL_FILE,
    parse_dates=["timestamp"]
)


# ============================================================
# TARGET
# ============================================================

TARGET = "target_temp_24h"


# ============================================================
# ALL FEATURES
# ============================================================

exclude = [
    "timestamp",
    "target_temp_24h",
    "target_rain_24h"
]

all_features = [
    c for c in train.columns
    if c not in exclude
]


# ============================================================
# FEATURE GROUPS
# ============================================================

lag_features = [
    c for c in all_features
    if "lag_" in c
]

rolling_features = [
    c for c in all_features
    if "rolling_" in c
]

temporal_features = [
    "hour",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]

wind_direction_features = [
    "wind_direction_sin",
    "wind_direction_cos"
]

change_features = [
    "temperature_change_1h",
    "pressure_change_1h"
]


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = {}


# Full model
experiments["Full Model"] = all_features


# Remove lag features
experiments["Without Lag Features"] = [
    c for c in all_features
    if c not in lag_features
]


# Remove rolling features
experiments["Without Rolling Features"] = [
    c for c in all_features
    if c not in rolling_features
]


# Remove temporal / seasonal features
experiments["Without Temporal Features"] = [
    c for c in all_features
    if c not in temporal_features
]


# Remove wind direction representation
experiments["Without Wind Direction"] = [
    c for c in all_features
    if c not in wind_direction_features
]


# Remove change features
experiments["Without Change Features"] = [
    c for c in all_features
    if c not in change_features
]


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []


for name, features in experiments.items():

    print("\n========================================")
    print(name)
    print("========================================")

    print(
        f"Number of features: {len(features)}"
    )

    X_train = train[features]

    y_train = train[TARGET]

    X_val = validation[features]

    y_val = validation[TARGET]


    model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_val
    )


    mae = mean_absolute_error(
        y_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    r2 = r2_score(
        y_val,
        predictions
    )


    results.append({

        "Experiment": name,

        "Features": len(features),

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n\n========================================")
print("TEMPERATURE XGBOOST ABLATION RESULTS")
print("========================================")


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CALCULATE CHANGE FROM FULL MODEL
# ============================================================

full_mae = results_df.loc[
    results_df["Experiment"] == "Full Model",
    "MAE"
].iloc[0]


full_rmse = results_df.loc[
    results_df["Experiment"] == "Full Model",
    "RMSE"
].iloc[0]


results_df[
    "MAE_Change_vs_Full_%"
] = (
    (results_df["MAE"] - full_mae)
    / full_mae
) * 100


results_df[
    "RMSE_Change_vs_Full_%"
] = (
    (results_df["RMSE"] - full_rmse)
    / full_rmse
) * 100


print("\n========================================")
print("CHANGE RELATIVE TO FULL MODEL")
print("========================================")


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "results/temperature_xgb_ablation_2022.csv",
    index=False
)


print("\nSaved:")

print(
    "results/temperature_xgb_ablation_2022.csv"
)

print("\nAblation study complete.")