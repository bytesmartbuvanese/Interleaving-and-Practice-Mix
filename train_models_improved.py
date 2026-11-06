"""Train and evaluate a leakage-aware question-correctness classifier.

The training workflow joins student, question, topic, session, and interaction
records, then constructs historical features using only answers and timestamps
that precede the current attempt. It divides data into training, validation,
and test sets by student, so the same learner does not appear in multiple
partitions. Hyperparameter and decision-threshold selection use validation
students; held-out test students are reserved for the final evaluation.

The model pipeline imputes and scales numeric inputs, imputes and one-hot
encodes categorical inputs, and trains an XGBoost binary classifier. The
script compares tuned-threshold and conventional 0.50-threshold metrics and
saves the final model, model metadata, comparison metrics, and tuning results
under ``trained_models/``.

Inputs are ``data/students.csv``, ``data/questions.csv``, ``data/topics.csv``,
``data/practice_sessions.csv``, and ``data/student_interactions.csv``. Exam
results are intentionally not used. Run from the project root with
``python train_models_improved.py`` after installing the dependencies declared
by the project.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from model_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data"
MODEL_DIR = "trained_models"
RANDOM_STATE = 42
TEST_SIZE = 0.15          # Held-out students; never used for tuning
VALIDATION_SIZE = 0.15    # Held-out students from the remaining pool
N_TUNING_TRIALS = 18

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset Assembly and Leakage-Aware Features
# ---------------------------------------------------------------------------
def read_csv(name):
    """Load a CSV from ``DATA_DIR`` or raise if the dataset is missing."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Put the generated CSV files inside '{DATA_DIR}/'."
        )
    return pd.read_csv(path)


def build_training_frame():
    """Join source datasets and derive features available before each answer.

    Returns model inputs, binary targets, student group labels, and the numeric
    and categorical feature-name lists.
    """
    print("Loading project datasets...")
    students = read_csv("students.csv")
    questions = read_csv("questions.csv")
    topics = read_csv("topics.csv")
    sessions = read_csv("practice_sessions.csv")
    interactions = read_csv("student_interactions.csv")

    # Join only attributes that exist before the answer is submitted.
    df = interactions.merge(students, on="student_id", how="left")
    df = df.merge(questions, on="question_id", how="left")
    df = df.merge(
        sessions[["session_id", "student_id", "session_start_timestamp"]],
        on=["session_id", "student_id"],
        how="left",
        validate="many_to_one",
    )
    topic_columns = [
        c for c in ["subtopic_id", "subject_id", "topic_id"]
        if c in topics.columns
    ]
    topic_lookup = topics[topic_columns].drop_duplicates("subtopic_id")
    df = df.merge(topic_lookup, on="subtopic_id", how="left", validate="many_to_one")

    df["interaction_timestamp"] = pd.to_datetime(
        df["interaction_timestamp"], errors="coerce"
    )
    df["session_start_timestamp"] = pd.to_datetime(
        df["session_start_timestamp"], errors="coerce"
    )
    df = df.dropna(
        subset=["student_id", "interaction_timestamp", "is_correct", "subtopic_id"]
    ).copy()
    df["is_correct"] = df["is_correct"].astype(int)

    # Stable ordering makes every historical feature strictly prior to this row.
    df = df.sort_values(
        ["student_id", "interaction_timestamp", "interaction_id"]
    ).reset_index(drop=True)

    print(f"Merged interactions: {len(df):,}")

    # Historical features: shift(1) prevents the current answer entering its
    # own predictors. Grouping is per student, so held-out students cannot
    # contribute history to training students.
    subtopic_keys = ["student_id", "subtopic_id"]
    subject_keys = ["student_id", "subject_id"]

    subtopic_correct = df.groupby(subtopic_keys, sort=False)["is_correct"]
    subject_correct = df.groupby(subject_keys, sort=False)["is_correct"]

    df["subtopic_previous_attempts"] = subtopic_correct.cumcount()
    df["subtopic_prior_accuracy"] = subtopic_correct.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["subtopic_recent_accuracy_5"] = subtopic_correct.transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    df["subject_previous_attempts"] = subject_correct.cumcount()
    df["subject_prior_accuracy"] = subject_correct.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["subject_recent_accuracy_5"] = subject_correct.transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    response_group = df.groupby(subtopic_keys, sort=False)["response_time_ms"]
    df["subtopic_prior_avg_response_ms"] = response_group.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    previous_time = df.groupby(subtopic_keys, sort=False)[
        "interaction_timestamp"
    ].shift(1)
    df["days_since_subtopic_practice"] = (
        (df["interaction_timestamp"] - previous_time).dt.total_seconds() / 86400
    )

    # Student's previous overall performance, still based only on past answers.
    overall_group = df.groupby("student_id", sort=False)["is_correct"]
    df["student_previous_attempts"] = overall_group.cumcount()
    df["student_prior_accuracy"] = overall_group.transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # These are known at question-selection time or before the answer:
    # profile fields, question-bank metadata, and historical interaction data.
    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    # Keep only fields present in the supplied data. Missing optional metadata
    # is handled by the imputers below.
    for col in numeric_features:
        if col not in df.columns:
            df[col] = np.nan
    for col in categorical_features:
        if col not in df.columns:
            df[col] = np.nan

    feature_columns = numeric_features + categorical_features
    X = df[feature_columns].copy()
    y = df["is_correct"].astype(int)
    groups = df["student_id"].astype(str)

    # Remove invalid labels, if any.
    valid = y.isin([0, 1])
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), \
        groups.loc[valid].reset_index(drop=True), numeric_features, categorical_features


def split_by_student(X, y, groups):
    """Create train, validation, and test partitions with disjoint students."""
    # First reserve the final test students.
    first = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_val_idx, test_idx = next(first.split(X, y, groups))

    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    # Then reserve validation students from the remaining students.
    relative_val_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    second = GroupShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=RANDOM_STATE + 1
    )
    train_rel, val_rel = next(
        second.split(X_train_val, y_train_val, groups_train_val)
    )

    train_idx = np.asarray(train_val_idx)[train_rel]
    val_idx = np.asarray(train_val_idx)[val_rel]

    return train_idx, val_idx, np.asarray(test_idx)


# ---------------------------------------------------------------------------
# Preprocessing and Model Evaluation
# ---------------------------------------------------------------------------
def make_pipeline(numeric_features, categorical_features, params):
    """Build preprocessing and XGBoost steps for the selected feature columns."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    return Pipeline([("preprocessor", prep), ("model", model)])


def metric_row(name, y_true, probabilities, threshold):
    """Calculate classification and probability metrics at a threshold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Model": name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1_Score": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Brier_Score": brier_score_loss(y_true, probabilities),
    }


# ---------------------------------------------------------------------------
# Tuning, Final Evaluation, and Artifact Export
# ---------------------------------------------------------------------------
def main():
    """Tune on validation students, evaluate once, and save the final model."""
    X, y, groups, numeric_features, categorical_features = build_training_frame()
    train_idx, val_idx, test_idx = split_by_student(X, y, groups)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    train_students = set(groups.iloc[train_idx])
    val_students = set(groups.iloc[val_idx])
    test_students = set(groups.iloc[test_idx])
    assert train_students.isdisjoint(val_students)
    assert train_students.isdisjoint(test_students)
    assert val_students.isdisjoint(test_students)

    print("\nStudent-level split")
    print(f"Train: {len(X_train):,} rows / {len(train_students):,} students")
    print(f"Validation: {len(X_val):,} rows / {len(val_students):,} students")
    print(f"Test: {len(X_test):,} rows / {len(test_students):,} students")
    print("\nTuning XGBoost on validation students only...")

    # Search model capacity and regularization. These do not add data fields;
    # they tune how the model learns from practically available inputs.
    search_space = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.015, 0.025, 0.04, 0.06],
        "min_child_weight": [2, 4, 6, 8],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85, 0.95],
        "reg_alpha": [0.0, 0.05, 0.2, 0.5],
        "reg_lambda": [1.0, 2.0, 4.0, 6.0],
        "gamma": [0.0, 0.05, 0.15],
    }
    candidates = list(ParameterSampler(
        search_space, n_iter=N_TUNING_TRIALS, random_state=RANDOM_STATE
    ))

    tuning_rows = []
    best_pipeline = None
    best_params = None
    best_val_accuracy = -1.0
    best_val_auc = -1.0

    for trial, params in enumerate(candidates, start=1):
        pipeline = make_pipeline(numeric_features, categorical_features, params)
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]

        # Select threshold using validation data only. Test set remains untouched.
        thresholds = np.arange(0.30, 0.701, 0.01)
        threshold_scores = [
            (accuracy_score(y_val, (val_prob >= t).astype(int)), t)
            for t in thresholds
        ]
        val_accuracy, best_threshold = max(threshold_scores, key=lambda x: x[0])
        val_auc = roc_auc_score(y_val, val_prob)

        tuning_rows.append({
            "trial": trial,
            "validation_accuracy": val_accuracy,
            "validation_roc_auc": val_auc,
            "threshold": best_threshold,
            **params,
        })
        print(
            f"Trial {trial:02d}/{N_TUNING_TRIALS} | "
            f"val accuracy={val_accuracy:.4f} | val ROC-AUC={val_auc:.4f} | "
            f"threshold={best_threshold:.2f}"
        )

        # Accuracy is the primary objective requested. ROC-AUC breaks ties.
        if (val_accuracy > best_val_accuracy) or (
            np.isclose(val_accuracy, best_val_accuracy) and val_auc > best_val_auc
        ):
            best_val_accuracy = val_accuracy
            best_val_auc = val_auc
            best_pipeline = pipeline
            best_params = params
            selected_threshold = best_threshold

    # Refit the selected configuration on train + validation students.
    # The test students have not participated in fitting or tuning.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    final_pipeline = make_pipeline(
        numeric_features, categorical_features, best_params
    )
    final_pipeline.fit(X_train_val, y_train_val)

    test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = metric_row(
        "Tuned XGBoost", y_test, test_prob, selected_threshold
    )

    # Also report the conventional 0.50 threshold for transparent comparison.
    default_metrics = metric_row("Tuned XGBoost (threshold 0.50)", y_test, test_prob, 0.50)
    comparison = pd.DataFrame([final_metrics, default_metrics])
    comparison.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"), index=False)
    pd.DataFrame(tuning_rows).to_csv(
        os.path.join(MODEL_DIR, "xgboost_tuning_results.csv"), index=False
    )

    model_path = os.path.join(
        MODEL_DIR, "best_question_correctness_model.pkl"
    )
    joblib.dump(final_pipeline, model_path)

    info = {
        "best_model": "XGBoost",
        "selection_metric": "validation_accuracy",
        "validation_accuracy": float(best_val_accuracy),
        "validation_roc_auc": float(best_val_auc),
        "decision_threshold": float(selected_threshold),
        "test_metrics": final_metrics,
        "features_numeric": numeric_features,
        "features_categorical": categorical_features,
        "target": "is_correct",
        "split_method": "GroupShuffleSplit by student",
        "exam_percentage_used": False,
        "future_session_statistics_used": False,
        "best_parameters": best_params,
        "train_students": len(train_students),
        "validation_students": len(val_students),
        "test_students": len(test_students),
    }
    with open(os.path.join(MODEL_DIR, "best_model_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print("\n" + "=" * 65)
    print("FINAL HELD-OUT TEST RESULTS")
    print("=" * 65)
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key:>12}: {value:.4f}")
        else:
            print(f"{key:>12}: {value}")

    test_predictions = (test_prob >= selected_threshold).astype(int)
    print("\nClassification report (held-out students):")
    print(classification_report(y_test, test_predictions, zero_division=0))

    print(f"\nSaved model: {model_path}")
    print(f"Saved comparison: {MODEL_DIR}/model_comparison.csv")
    print(f"Saved tuning results: {MODEL_DIR}/xgboost_tuning_results.csv")
    print(f"Saved metadata: {MODEL_DIR}/best_model_info.json")
    print(
        "\nImportant: compare this held-out test accuracy with your previous "
        "0.7666 only if both runs used the same generated dataset and equivalent "
        "student-level test populations. A result below 85% is still valid; "
        "do not change the split or add unavailable information to inflate it."
    )


if __name__ == "__main__":
    main()
