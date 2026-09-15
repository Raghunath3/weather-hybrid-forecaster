import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os


# ============================================================
# FILES
# ============================================================

TEST_FILE = "data/test_2023.csv"
MODEL_FILE = "models/final_rainfall_xgb.json"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading final rainfall model and 2023 data...")

test = pd.read_csv(
    TEST_FILE,
    parse_dates=["timestamp"]
)

exclude_columns = [
    "timestamp",
    "target_temp_24h",
    "target_rain_24h"
]

features = [
    col for col in test.columns
    if col not in exclude_columns
]

X_test = test[features]

print(
    f"2023 test records: {len(X_test)}"
)

print(
    f"Number of features: {len(features)}"
)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

print("\nLoading trained XGBoost model...")

model = xgb.Booster()

model.load_model(
    MODEL_FILE
)


# ============================================================
# CREATE DMATRIX
# ============================================================

print("\nCreating XGBoost DMatrix...")

dtest = xgb.DMatrix(
    X_test,
    feature_names=features
)


# ============================================================
# XGBOOST NATIVE TREE SHAP
# ============================================================

print("\nCalculating native XGBoost SHAP contributions...")

contributions = model.predict(
    dtest,
    pred_contribs=True
)

print(
    f"Contribution matrix shape: {contributions.shape}"
)


# Last column is the SHAP bias/base value.
shap_values = contributions[:, :-1]


print(
    f"SHAP value matrix shape: {shap_values.shape}"
)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print(
    "\n========================================"
)

print(
    "GLOBAL SHAP FEATURE IMPORTANCE"
)

print(
    "========================================"
)


mean_abs_shap = np.abs(
    shap_values
).mean(
    axis=0
)


importance = pd.DataFrame({

    "Feature": features,

    "Mean_Absolute_SHAP":
        mean_abs_shap

}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(
    drop=True
)


importance[
    "Rank"
] = np.arange(
    1,
    len(importance) + 1
)


importance = importance[
    [
        "Rank",
        "Feature",
        "Mean_Absolute_SHAP"
    ]
]


print(
    importance.head(
        15
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "results/shap",
    exist_ok=True
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    "results/shap/rainfall_shap_feature_importance_2023.csv",
    index=False
)


# ============================================================
# SHAP BAR PLOT
# ============================================================

print(
    "\nCreating SHAP summary bar plot..."
)


shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    max_display=15,
    show=False
)


plt.tight_layout()


plt.savefig(
    "results/shap/rainfall_shap_summary_bar_2023.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SHAP BEESWARM
# ============================================================

print(
    "Creating SHAP beeswarm plot..."
)


shap.summary_plot(
    shap_values,
    X_test,
    max_display=15,
    show=False
)


plt.tight_layout()


plt.savefig(
    "results/shap/rainfall_shap_beeswarm_2023.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# SHAP DIRECTION ANALYSIS
# ============================================================

print(
    "\n========================================"
)

print(
    "SHAP DIRECTION ANALYSIS"
)

print(
    "========================================"
)


top_features = importance.head(
    10
)["Feature"].tolist()


direction_results = []


for feature in top_features:

    index = features.index(
        feature
    )

    values = shap_values[
        :,
        index
    ]

    direction_results.append({

        "Feature": feature,

        "Mean_SHAP": np.mean(
            values
        ),

        "Mean_Absolute_SHAP":
            np.mean(
                np.abs(values)
            ),

        "Positive_SHAP_Fraction":
            np.mean(
                values > 0
            ),

        "Negative_SHAP_Fraction":
            np.mean(
                values < 0
            )
    })


direction_df = pd.DataFrame(
    direction_results
)


print(
    direction_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


direction_df.to_csv(
    "results/shap/rainfall_shap_direction_analysis_2023.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n========================================"
)

print(
    "SHAP ANALYSIS COMPLETE"
)

print(
    "========================================"
)

print("\nSaved:")

print(
    "results/shap/rainfall_shap_feature_importance_2023.csv"
)

print(
    "results/shap/rainfall_shap_summary_bar_2023.png"
)

print(
    "results/shap/rainfall_shap_beeswarm_2023.png"
)

print(
    "results/shap/rainfall_shap_direction_analysis_2023.csv"
)