from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import requests
import pandas as pd
import numpy as np
import xgboost as xgb

from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Weather Intelligence Decision Support System",
    description=(
        "Explainable machine-learning framework for "
        "24-hour-ahead temperature and rainfall forecasting."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIGURATION
# ============================================================

LATITUDE = 13.0827
LONGITUDE = 80.2707

LOCATION_NAME = "Chennai"

TIMEZONE = "Asia/Kolkata"

TEMPERATURE_MODEL_PATH = (
    "models/final_temperature_xgb.json"
)

RAINFALL_MODEL_PATH = (
    "models/final_rainfall_xgb.json"
)

# 90% empirical interval radius obtained from
# the 2022 validation residual distribution.
#
# 2022 validation:
# 90th percentile absolute error = 1.6113°C
#
# This value was frozen before evaluating 2023.
TEMPERATURE_INTERVAL_RADIUS = 1.6113

# Rainfall decision threshold selected using
# 2022 validation and frozen for final operation.
RAINFALL_THRESHOLD = 0.20


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("Loading temperature model...")

temperature_model = xgb.Booster()

temperature_model.load_model(
    TEMPERATURE_MODEL_PATH
)


print("Loading rainfall model...")

rainfall_model = xgb.Booster()

rainfall_model.load_model(
    RAINFALL_MODEL_PATH
)


print("Models loaded successfully.")


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

# These are the actual global SHAP results calculated
# from the 2023 test-set analysis.

GLOBAL_SHAP_IMPORTANCE = [

    {
        "feature": "temperature",
        "mean_absolute_shap": 1.173333
    },

    {
        "feature": "temperature_lag_24h",
        "mean_absolute_shap": 0.561630
    },

    {
        "feature": "temperature_lag_1h",
        "mean_absolute_shap": 0.450330
    },

    {
        "feature": "hour_cos",
        "mean_absolute_shap": 0.294310
    },

    {
        "feature": "month_cos",
        "mean_absolute_shap": 0.124940
    },

    {
        "feature": "month",
        "mean_absolute_shap": 0.077249
    },

    {
        "feature": "hour",
        "mean_absolute_shap": 0.077069
    },

    {
        "feature": "temperature_rolling_mean_24h",
        "mean_absolute_shap": 0.052971
    },

    {
        "feature": "wind_direction_sin",
        "mean_absolute_shap": 0.051751
    },

    {
        "feature": "temperature_lag_3h",
        "mean_absolute_shap": 0.048065
    }
]


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    try:

        with open(
            "index.html",
            "r",
            encoding="utf-8"
        ) as file:

            return file.read()

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {

        "status": "ok",

        "temperature_model":
            "loaded",

        "rainfall_model":
            "loaded",

        "location":
            LOCATION_NAME,

        "forecast_horizon":
            "24 hours",

        "rainfall_threshold":
            RAINFALL_THRESHOLD
    }


# ============================================================
# CURRENT SYSTEM TIME
# ============================================================

def get_local_now():

    return datetime.now(
        ZoneInfo(TIMEZONE)
    )


# ============================================================
# FETCH WEATHER HISTORY
# ============================================================

def fetch_recent_weather():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "rain",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m"
        ],

        # We only need recent history for
        # the lag/rolling features.
        "past_hours": 48,

        # One future hour is harmless because
        # we explicitly reject timestamps > now.
        "forecast_hours": 1,

        "timezone":
            TIMEZONE
    }


    response = requests.get(
        url,
        params=params,
        timeout=20
    )


    response.raise_for_status()


    payload = response.json()


    if "hourly" not in payload:

        raise RuntimeError(
            "Hourly weather data missing."
        )


    return payload


# ============================================================
# CONVERT WEATHER RESPONSE TO DATAFRAME
# ============================================================

def weather_to_dataframe(payload):

    hourly = payload["hourly"]


    df = pd.DataFrame({

        "timestamp":
            pd.to_datetime(
                hourly["time"]
            ),

        "temperature":
            hourly["temperature_2m"],

        "humidity":
            hourly["relative_humidity_2m"],

        "pressure":
            hourly["surface_pressure"],

        "rain":
            hourly["rain"],

        "cloud_cover":
            hourly["cloud_cover"],

        "wind_speed":
            hourly["wind_speed_10m"],

        "wind_direction":
            hourly["wind_direction_10m"]
    })


    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )


    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    df = df.copy()


    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["month"] = (
        df["timestamp"].dt.month
    )


    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )


    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )


    # --------------------------------------------------------
    # WIND DIRECTION CYCLIC ENCODING
    # --------------------------------------------------------

    df["wind_direction_sin"] = np.sin(
        np.deg2rad(
            df["wind_direction"]
        )
    )

    df["wind_direction_cos"] = np.cos(
        np.deg2rad(
            df["wind_direction"]
        )
    )


    # --------------------------------------------------------
    # TEMPERATURE LAGS
    # --------------------------------------------------------

    for lag in [
        1,
        3,
        6,
        12,
        24
    ]:

        df[
            f"temperature_lag_{lag}h"
        ] = (
            df["temperature"]
            .shift(lag)
        )


    # --------------------------------------------------------
    # OTHER LAGS
    # --------------------------------------------------------

    df[
        "humidity_lag_1h"
    ] = (
        df["humidity"]
        .shift(1)
    )


    df[
        "pressure_lag_1h"
    ] = (
        df["pressure"]
        .shift(1)
    )


    df[
        "wind_speed_lag_1h"
    ] = (
        df["wind_speed"]
        .shift(1)
    )


    df[
        "rain_lag_1h"
    ] = (
        df["rain"]
        .shift(1)
    )


    # --------------------------------------------------------
    # 24-HOUR ROLLING FEATURES
    # --------------------------------------------------------

    df[
        "temperature_rolling_mean_24h"
    ] = (
        df["temperature"]
        .rolling(
            window=24,
            min_periods=24
        )
        .mean()
    )


    df[
        "temperature_rolling_std_24h"
    ] = (
        df["temperature"]
        .rolling(
            window=24,
            min_periods=24
        )
        .std()
    )


    # --------------------------------------------------------
    # CHANGE FEATURES
    # --------------------------------------------------------

    df[
        "temperature_change_1h"
    ] = (
        df["temperature"]
        .diff(1)
    )


    df[
        "pressure_change_1h"
    ] = (
        df["pressure"]
        .diff(1)
    )


    return df


# ============================================================
# SELECT LATEST OBSERVABLE ROW
# ============================================================

def select_forecast_origin(df):

    now = get_local_now()

    # Convert current local time to timezone-naive
    # because Open-Meteo timestamps were requested
    # in Asia/Kolkata.
    now_naive = (
        pd.Timestamp(now)
        .tz_localize(None)
    )


    # CRITICAL LEAKAGE PROTECTION:
    #
    # Only timestamps <= current time are allowed.
    #
    # Therefore no future weather observation can
    # accidentally enter the feature vector.

    available = df[
        df["timestamp"] <= now_naive
    ].copy()


    if available.empty:

        raise RuntimeError(
            "No current/past weather observation available."
        )


    latest_timestamp = (
        available["timestamp"]
        .max()
    )


    row = available[
        available["timestamp"]
        == latest_timestamp
    ].iloc[-1]


    return row, latest_timestamp


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    engineered_df,
    forecast_origin
):

    row = engineered_df[
        engineered_df["timestamp"]
        == forecast_origin
    ]


    if row.empty:

        raise RuntimeError(
            "Forecast-origin feature row not found."
        )


    row = row.iloc[0]


    missing = [
        feature
        for feature in FEATURES
        if pd.isna(row[feature])
    ]


    if missing:

        raise RuntimeError(
            "Missing engineered features: "
            + ", ".join(missing)
        )


    X = pd.DataFrame(
        [
            [
                row[feature]
                for feature in FEATURES
            ]
        ],
        columns=FEATURES
    )


    return X


# ============================================================
# LOCAL SHAP / XGBOOST CONTRIBUTIONS
# ============================================================

def calculate_local_shap(X):

    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURES
    )


    contributions = (
        temperature_model.predict(
            dmatrix,
            pred_contribs=True
        )[0]
    )


    # Last value is the bias/base contribution.
    feature_contributions = (
        contributions[:-1]
    )

    base_value = float(
        contributions[-1]
    )


    local_features = []


    for feature, contribution in zip(
        FEATURES,
        feature_contributions
    ):

        local_features.append({

            "feature":
                feature,

            "shap_value":
                round(
                    float(contribution),
                    6
                ),

            "absolute_shap_value":
                round(
                    abs(
                        float(contribution)
                    ),
                    6
                )
        })


    # Strongest contributors first

    local_features.sort(
        key=lambda item:
            item["absolute_shap_value"],
        reverse=True
    )


    return {

        "base_value":
            round(
                base_value,
                6
            ),

        "top_contributors":
            local_features[:10]
    }


# ============================================================
# TEMPERATURE FORECAST
# ============================================================

def predict_temperature(X):

    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURES
    )


    prediction = float(
        temperature_model.predict(
            dmatrix
        )[0]
    )


    lower = (
        prediction
        - TEMPERATURE_INTERVAL_RADIUS
    )


    upper = (
        prediction
        + TEMPERATURE_INTERVAL_RADIUS
    )


    return {

        "prediction":
            round(
                prediction,
                2
            ),

        "lower_90":
            round(
                lower,
                2
            ),

        "upper_90":
            round(
                upper,
                2
            ),

        "interval_radius":
            TEMPERATURE_INTERVAL_RADIUS,

        "interval_method":
            "Empirical residual-based interval",

        "calibration_period":
            "2022 validation set"
    }


# ============================================================
# RAINFALL FORECAST
# ============================================================

def predict_rainfall(X):

    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURES
    )


    probability = float(
        rainfall_model.predict(
            dmatrix
        )[0]
    )


    probability = np.clip(
        probability,
        0.0,
        1.0
    )


    if probability >= RAINFALL_THRESHOLD:

        classification = (
            "RAIN EXPECTED"
        )

    else:

        classification = (
            "NO RAIN EXPECTED"
        )


    return {

        "probability":
            round(
                probability * 100,
                2
            ),

        "probability_decimal":
            round(
                probability,
                4
            ),

        "threshold":
            RAINFALL_THRESHOLD,

        "classification":
            classification,

        "target_definition":
            "Rainfall > 0.1 mm at t+24h"
    }


# ============================================================
# DECISION SUPPORT
# ============================================================

def generate_alerts(
    current,
    rainfall
):

    alerts = []


    # High rainfall probability

    if rainfall[
        "probability_decimal"
    ] >= 0.50:

        alerts.append({

            "severity":
                "high",

            "type":
                "rainfall",

            "message":
                "High probability of rainfall "
                "at the 24-hour forecast horizon."
        })


    elif rainfall[
        "probability_decimal"
    ] >= RAINFALL_THRESHOLD:

        alerts.append({

            "severity":
                "moderate",

            "type":
                "rainfall",

            "message":
                "Rainfall is expected at the "
                "24-hour forecast horizon."
        })


    # Heat alert

    if current[
        "temperature"
    ] >= 35:

        alerts.append({

            "severity":
                "high",

            "type":
                "temperature",

            "message":
                "Current temperature exceeds "
                "35°C."
        })


    return alerts


# ============================================================
# MAIN FORECAST ENDPOINT
# ============================================================

@app.get("/api/forecast")
async def forecast():

    try:

        # ----------------------------------------------------
        # 1. Retrieve recent weather history
        # ----------------------------------------------------

        payload = (
            fetch_recent_weather()
        )


        # ----------------------------------------------------
        # 2. Convert to dataframe
        # ----------------------------------------------------

        raw_df = (
            weather_to_dataframe(
                payload
            )
        )


        # ----------------------------------------------------
        # 3. Engineer features
        # ----------------------------------------------------

        engineered_df = (
            engineer_features(
                raw_df
            )
        )


        # ----------------------------------------------------
        # 4. Select ONLY latest observable
        # ----------------------------------------------------

        latest_row, forecast_origin = (
            select_forecast_origin(
                engineered_df
            )
        )


        # ----------------------------------------------------
        # 5. Construct model input
        # ----------------------------------------------------

        X = build_model_input(
            engineered_df,
            forecast_origin
        )


        # ----------------------------------------------------
        # 6. Temperature prediction
        # ----------------------------------------------------

        temperature_result = (
            predict_temperature(X)
        )


        # ----------------------------------------------------
        # 7. Rainfall prediction
        # ----------------------------------------------------

        rainfall_result = (
            predict_rainfall(X)
        )


        # ----------------------------------------------------
        # 8. Local SHAP
        # ----------------------------------------------------

        local_shap = (
            calculate_local_shap(X)
        )


        # ----------------------------------------------------
        # 9. Current conditions
        # ----------------------------------------------------

        current = {

            "temperature":
                round(
                    float(
                        latest_row[
                            "temperature"
                        ]
                    ),
                    1
                ),

            "humidity":
                round(
                    float(
                        latest_row[
                            "humidity"
                        ]
                    ),
                    1
                ),

            "pressure":
                round(
                    float(
                        latest_row[
                            "pressure"
                        ]
                    ),
                    1
                ),

            "wind_speed":
                round(
                    float(
                        latest_row[
                            "wind_speed"
                        ]
                    ),
                    1
                ),

            "rain":
                round(
                    float(
                        latest_row[
                            "rain"
                        ]
                    ),
                    2
                ),

            "cloud_cover":
                round(
                    float(
                        latest_row[
                            "cloud_cover"
                        ]
                    ),
                    1
                ),

            "wind_direction":
                round(
                    float(
                        latest_row[
                            "wind_direction"
                        ]
                    ),
                    1
                )
        }


        # ----------------------------------------------------
        # 10. Decision support
        # ----------------------------------------------------

        alerts = generate_alerts(
            current,
            rainfall_result
        )


        # ----------------------------------------------------
        # 11. Response
        # ----------------------------------------------------

        return {

            "system": {

                "name":
                    "Weather Intelligence "
                    "Decision Support System",

                "version":
                    "1.0.0",

                "location":
                    LOCATION_NAME,

                "latitude":
                    LATITUDE,

                "longitude":
                    LONGITUDE,

                "timezone":
                    TIMEZONE,

                "data_source":
                    "Open-Meteo",

                "forecast_origin":
                    forecast_origin.isoformat(),

                "forecast_horizon":
                    "24 hours"
            },


            "current": current,


            "temperature_forecast":
                temperature_result,


            "rainfall_prediction":
                rainfall_result,


            "explainability": {

                "method":
                    "XGBoost TreeSHAP",

                "local":
                    local_shap,

                "global":
                    GLOBAL_SHAP_IMPORTANCE
            },


            "decision_support": {

                "alerts":
                    alerts,

                "alert_count":
                    len(alerts)
            }

        }


    except requests.RequestException as error:

    print("========== WEATHER API ERROR ==========")
    print(repr(error))
    print("=======================================")

    raise HTTPException(
        status_code=503,
        detail=f"Weather data service unavailable: {str(error)}"
    )

except Exception as error:

    print("========== FORECAST ERROR ==========")
    print(repr(error))
    print("====================================")

    raise HTTPException(
        status_code=500,
        detail=f"Forecast processing error: {str(error)}"
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )