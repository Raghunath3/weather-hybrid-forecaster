import pandas as pd
import numpy as np

from scipy.stats import wilcoxon


# ============================================================
# LOAD FINAL 2023 PREDICTIONS
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


# ============================================================
# ACTUAL VALUES
# ============================================================

actual = df[
    "actual_temperature_24h"
].values


# ============================================================
# CALCULATE ABSOLUTE ERRORS
# ============================================================

errors = {}

for name, column in models.items():

    predictions = df[column].values

    errors[name] = np.abs(
        actual - predictions
    )


# ============================================================
# MODEL PAIRS
# ============================================================

pairs = [

    ("Persistence", "XGBoost"),

    ("Persistence", "LSTM"),

    ("Persistence", "Hybrid"),

    ("XGBoost", "LSTM"),

    ("XGBoost", "Hybrid"),

    ("LSTM", "Hybrid")
]


# ============================================================
# WILCOXON TEST
# ============================================================

results = []


print("\n========================================")
print("WILCOXON SIGNED-RANK TEST")
print("2023 TEMPERATURE FORECASTING")
print("========================================")


for model_a, model_b in pairs:

    error_a = errors[model_a]

    error_b = errors[model_b]


    difference = (
        error_a - error_b
    )


    # Remove exact ties
    nonzero = (
        difference != 0
    )


    test_a = error_a[nonzero]

    test_b = error_b[nonzero]


    statistic, p_value = wilcoxon(
        test_a,
        test_b,
        alternative="two-sided"
    )


    mean_a = np.mean(
        error_a
    )

    mean_b = np.mean(
        error_b
    )


    median_a = np.median(
        error_a
    )

    median_b = np.median(
        error_b
    )


    if p_value < 0.05:

        significance = "Significant"

    else:

        significance = "Not Significant"


    results.append({

        "Model_A": model_a,

        "Model_B": model_b,

        "MAE_A": mean_a,

        "MAE_B": mean_b,

        "Median_Error_A":
            median_a,

        "Median_Error_B":
            median_b,

        "Wilcoxon_Statistic":
            statistic,

        "p_value":
            p_value,

        "Significance":
            significance
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# BONFERRONI CORRECTION
# ============================================================

print("\n========================================")
print("BONFERRONI-CORRECTED RESULTS")
print("========================================")


number_of_tests = len(
    results_df
)

alpha = 0.05

corrected_alpha = (
    alpha /
    number_of_tests
)


print(
    f"Original alpha: {alpha}"
)

print(
    f"Number of comparisons: "
    f"{number_of_tests}"
)

print(
    f"Bonferroni corrected alpha: "
    f"{corrected_alpha:.6f}"
)


results_df[
    "Bonferroni_Significant"
] = (
    results_df["p_value"]
    < corrected_alpha
)


print(
    results_df[
        [
            "Model_A",
            "Model_B",
            "p_value",
            "Bonferroni_Significant"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "results/statistical_significance_2023.csv",
    index=False
)


print("\nSaved:")

print(
    "results/statistical_significance_2023.csv"
)

print("\nStatistical significance analysis complete.")