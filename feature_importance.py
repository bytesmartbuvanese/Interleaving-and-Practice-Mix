
"""Analyze and export feature importance from the saved correctness model.

The script loads the preprocessing/model pipeline from
``trained_models/best_question_correctness_model.pkl`` and asks the pipeline's
preprocessor for the final transformed feature names. For a model exposing
coefficients, it reports absolute coefficient magnitude and whether each
coefficient is positive, negative, or neutral. For a model exposing tree
feature importances, it reports those values instead. Unsupported model types
raise a ``ValueError`` rather than producing a misleading report.

The importance table is sorted from most to least important, assigned ranks,
and augmented with each feature's percentage of total importance. The script
writes the complete table to
``trained_models/analysis/feature_importance.csv`` and saves a horizontal bar
chart of the top features as ``feature_importance.png`` in the same directory.
It also prints the highest-ranked results and output locations.

Run from the project root with ``python feature_importance.py`` after training
and saving a compatible model pipeline.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "trained_models/best_question_correctness_model.pkl"
OUTPUT_DIR = "trained_models/analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the Trained Pipeline
# ---------------------------------------------------------------------------

print("\nLoading trained model...")

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# ---------------------------------------------------------------------------
# Resolve Transformed Feature Names
# ---------------------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------------------------------------------
# Calculate Feature Importance
# ---------------------------------------------------------------------------

print("\nCalculating feature importance...")

if hasattr(model, "coef_"):

    # Logistic Regression
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Importance": abs(coefficients)
    })

    importance_df["Influence"] = importance_df[
        "Coefficient"
    ].apply(
        lambda x: (
            "Positive"
            if x > 0
            else "Negative"
            if x < 0
            else "Neutral"
        )
    )

elif hasattr(model, "feature_importances_"):

    # Tree-based models
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": None,
        "Importance": model.feature_importances_,
        "Influence": "Tree importance"
    })

else:
    raise ValueError(
        "This model does not support feature importance extraction."
    )

# ---------------------------------------------------------------------------
# Rank Feature Importance
# ---------------------------------------------------------------------------

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

total_importance = importance_df["Importance"].sum()

importance_df["Importance_Percentage"] = (
    importance_df["Importance"] /
    total_importance * 100
)

importance_df.insert(
    0,
    "Rank",
    range(1, len(importance_df) + 1)
)

# ---------------------------------------------------------------------------
# Save Tabular Results
# ---------------------------------------------------------------------------

csv_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    csv_path,
    index=False
)

# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

print(
    importance_df[
        [
            "Rank",
            "Feature",
            "Importance_Percentage",
            "Influence"
        ]
    ].head(20).to_string(index=False)
)

# ---------------------------------------------------------------------------
# Plot the Highest-Ranked Features
# ---------------------------------------------------------------------------

top_features = importance_df.head(20).sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(12, 9))

plt.barh(
    top_features["Feature"],
    top_features["Importance_Percentage"]
)

plt.xlabel("Relative Feature Importance (%)")
plt.ylabel("Features")

plt.title(
    "Top 20 Features Influencing Question Correctness"
)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------------------------
# Report Output Locations
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 65)

print("CSV saved:", csv_path)
print("Plot saved:", plot_path)

print("\nTop 5 influential features:")

for _, row in importance_df.head(5).iterrows():

    print(
        f"{row['Rank']}. {row['Feature']} "
        f"({row['Importance_Percentage']:.2f}%)"
    )
