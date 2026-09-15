import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

FILE = "results/final_predictions_2023.csv"

df = pd.read_csv(FILE)


# ============================================================
# MODELS
# ============================================================

models = {
    "Persistence": "persistence_prediction",
    "XGBoost": "xgb_prediction",
    "LSTM": "lstm_prediction",
    "Hybrid": "hybrid_90_10_prediction"
}


actual = df[
    "actual_temperature_24h"
].values


# ============================================================
# ABSOLUTE ERRORS
# ============================================================

errors = {}

for name, column in models.items():

    predictions = df[column].values

    errors[name] = np.abs(
        actual - predictions
    )


# ============================================================
# MAE
# ============================================================

mae = {}

for name in models:

    mae[name] = np.mean(
        errors[name]
    )


# ============================================================
# PAIRWISE COMPARISONS
# ============================================================

pairs = [

    ("Persistence", "XGBoost"),

    ("Persistence", "LSTM"),

    ("Persistence", "Hybrid"),

    ("XGBoost", "LSTM"),

    ("XGBoost", "Hybrid"),

    ("LSTM", "Hybrid")
]


results = []


for baseline, model in pairs:

    baseline_mae = mae[baseline]

    model_mae = mae[model]


    # Positive means model improved over baseline
    improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
    ) * 100


    # Paired absolute-error difference
    difference = (
        errors[baseline]
        -
        errors[model]
    )


    mean_difference = np.mean(
        difference
    )

    std_difference = np.std(
        difference,
        ddof=1
    )


    # Cohen's dz for paired samples
    cohens_dz = (
        mean_difference
        /
        std_difference
    )


    # Probability that model has lower error
    lower_error_fraction = np.mean(
        errors[model]
        <
        errors[baseline]
    )


    results.append({

        "Baseline": baseline,

        "Model": model,

        "Baseline_MAE": baseline_mae,

        "Model_MAE": model_mae,

        "MAE_Improvement_%":
            improvement,

        "Mean_Paired_Error_Difference":
            mean_difference,

        "Cohens_dz":
            cohens_dz,

        "Fraction_Model_Lower_Error":
            lower_error_fraction
    })


# ============================================================
# DISPLAY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n========================================")
print("PRACTICAL IMPROVEMENT & EFFECT SIZE")
print("2023 TEMPERATURE FORECASTING")
print("========================================")


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# INDIVIDUAL MODEL MAE
# ============================================================

print("\n========================================")
print("FINAL MODEL MAE")
print("========================================")


for name in models:

    print(
        f"{name:12s}: {mae[name]:.6f} °C"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "results/effect_size_2023.csv",
    index=False
)


print("\nSaved:")

print(
    "results/effect_size_2023.csv"
)

print(
    "\nEffect-size analysis complete."
)