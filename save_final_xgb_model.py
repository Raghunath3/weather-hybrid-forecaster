import pandas as pd
import xgboost as xgb

# Load 2019-2022 training data
train = pd.read_csv("data/train_2019_2022.csv")

# Exact 28 features used by final XGBoost model
features = [
    "temperature",
    "humidity",
    "pressure",
    "rain",
    "cloud_cover",
    "wind_speed",
    "wind_direction",
    "hour",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_direction_sin",
    "wind_direction_cos",
    "temperature_lag_1h",
    "temperature_lag_3h",
    "temperature_lag_6h",
    "temperature_lag_12h",
    "temperature_lag_24h",
    "humidity_lag_1h",
    "pressure_lag_1h",
    "wind_speed_lag_1h",
    "rain_lag_1h",
    "temperature_rolling_mean_24h",
    "temperature_rolling_std_24h",
    "temperature_change_1h",
    "pressure_change_1h"
]

X_train = train[features]
y_train = train["target_temp_24h"]

print("Training XGBoost model...")

model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Saving model...")

model.save_model(
    "models/final_temperature_xgb.json"
)

print("\nModel saved successfully:")
print("models/final_temperature_xgb.json")