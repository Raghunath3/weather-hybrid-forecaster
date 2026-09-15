import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import os


# ============================================================
# FILES
# ============================================================

DATA_FILE = "data/test_2023.csv"
MODEL_FILE = "models/final_temperature_xgb.json"

OUTPUT_DIR = "results/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test data...")

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["timestamp"]
)

print(
    f"Test records: {len(df)}"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

# These are the same 28 features used by the
# final XGBoost temperature model.

feature_columns = [
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


X = df[
    feature_columns
].copy()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading XGBoost model...")

model = xgb.Booster()

model.load_model(
    MODEL_FILE
)


# ============================================================
# TREE SHAP
# ============================================================

print("\nCalculating TreeSHAP values...")

dtest = xgb.DMatrix(
    X,
    feature_names=feature_columns
)

# Native XGBoost prediction contributions
# avoids the SHAP/XGBoost compatibility issue.

shap_values = model.predict(
    dtest,
    pred_contribs=True
)


# Last column is the bias/base contribution.
shap_features = shap_values[
    :, :-1
]


print(
    f"SHAP matrix shape: {shap_features.shape}"
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_features),
    axis=0
)

mean_shap = np.mean(
    shap_features,
    axis=0
)


importance = pd.DataFrame({

    "Feature": feature_columns,

    "Mean_Absolute_SHAP":
        mean_abs_shap,

    "Mean_SHAP":
        mean_shap
})


importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(
    drop=True
)


importance[
    "Importance_Rank"
] = np.arange(
    1,
    len(importance) + 1
)


importance = importance[
    [
        "Importance_Rank",
        "Feature",
        "Mean_Absolute_SHAP",
        "Mean_SHAP"
    ]
]


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print("\n========================================")
print("TOP TEMPERATURE FEATURES")
print("========================================")

print(
    importance.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE IMPORTANCE
# ============================================================

importance.to_csv(
    f"{OUTPUT_DIR}/temperature_shap_feature_importance_2023.csv",
    index=False
)


# ============================================================
# SHAP BAR PLOT
# ============================================================

print("\nCreating SHAP bar plot...")

plt.figure(
    figsize=(10, 8)
)

shap.summary_plot(
    shap_features,
    X,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/temperature_shap_summary_bar_2023.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP BEESWARM
# ============================================================

print("Creating SHAP beeswarm plot...")

plt.figure(
    figsize=(10, 8)
)

shap.summary_plot(
    shap_features,
    X,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/temperature_shap_beeswarm_2023.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FEATURE DIRECTION ANALYSIS
# ============================================================

top_features = importance.head(15)[
    "Feature"
].tolist()


direction_results = []


for feature in top_features:

    index = feature_columns.index(
        feature
    )

    values = shap_features[
        :, index
    ]

    direction_results.append({

        "Feature": feature,

        "Mean_SHAP":
            np.mean(values),

        "Mean_Absolute_SHAP":
            np.mean(np.abs(values)),

        "Positive_SHAP_Fraction":
            np.mean(values > 0),

        "Negative_SHAP_Fraction":
            np.mean(values < 0)
    })


direction_df = pd.DataFrame(
    direction_results
)


direction_df.to_csv(
    f"{OUTPUT_DIR}/temperature_shap_direction_analysis_2023.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("TEMPERATURE SHAP ANALYSIS COMPLETE")
print("========================================")

print("\nSaved:")

print(
    f"{OUTPUT_DIR}/temperature_shap_feature_importance_2023.csv"
)

print(
    f"{OUTPUT_DIR}/temperature_shap_summary_bar_2023.png"
)

print(
    f"{OUTPUT_DIR}/temperature_shap_beeswarm_2023.png"
)

print(
    f"{OUTPUT_DIR}/temperature_shap_direction_analysis_2023.csv"
)