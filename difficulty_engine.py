
"""Recommend question difficulty levels using mastery and attempt history.

The engine loads the question bank, student interactions, student profiles,
curriculum topics, and the latest BKT mastery records. It validates required
columns, maps text or normalized numeric difficulty values to three difficulty
bands, removes invalid response labels, and builds student/subtopic/subject
history summaries. For each eligible learner and subtopic, it selects
unattempted questions and assigns a target difficulty from the learner's
mastery level.

When the optional trained correctness model is available and its expected
features can be constructed, predicted answer probabilities contribute to the
question ranking. If the model or compatible features are unavailable, the
engine retains its BKT-and-difficulty-based selection. Recommendations are
limited per student and subtopic, then written with a difficulty summary to
``trained_models/difficulty_engine/``.

Required inputs are ``data/questions.csv``,
``data/student_interactions.csv``, ``data/topics.csv``, ``data/students.csv``,
and ``trained_models/bkt/student_subtopic_mastery.csv``. The correctness model
at ``trained_models/best_question_correctness_model.pkl`` is optional.
Run with ``python difficulty_engine.py``; ``--student-id`` limits processing
to one learner and ``--top-n`` sets the per-subtopic recommendation limit.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "trained_models"
BKT_DIR = MODEL_DIR / "bkt"

QUESTIONS_FILE = DATA_DIR / "questions.csv"
INTERACTIONS_FILE = DATA_DIR / "student_interactions.csv"
TOPICS_FILE = DATA_DIR / "topics.csv"
STUDENTS_FILE = DATA_DIR / "students.csv"
MASTERY_FILE = BKT_DIR / "student_subtopic_mastery.csv"

CORRECTNESS_MODEL_FILE = (
    MODEL_DIR / "best_question_correctness_model.pkl"
)

OUTPUT_DIR = MODEL_DIR / "difficulty_engine"

# ---------------------------------------------------------
# DIFFICULTY CONFIGURATION
# ---------------------------------------------------------

DIFFICULTY_MAP = {
    "easy": 1,
    "beginner": 1,
    "low": 1,
    "medium": 2,
    "moderate": 2,
    "intermediate": 2,
    "hard": 3,
    "difficult": 3,
    "advanced": 3,
}

DIFFICULTY_LABELS = {
    1: "Easy",
    2: "Medium",
    3: "Hard",
}


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    """Load required project CSVs and validate columns used by this engine.

    The topic lookup is merged into questions so each question has its
    subject/topic identifiers for history and recommendation features.
    """

    required_files = [
        QUESTIONS_FILE,
        INTERACTIONS_FILE,
        TOPICS_FILE,
        STUDENTS_FILE,
        MASTERY_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}"
            )

    questions = pd.read_csv(QUESTIONS_FILE)
    interactions = pd.read_csv(INTERACTIONS_FILE)
    topics = pd.read_csv(TOPICS_FILE)
    students = pd.read_csv(STUDENTS_FILE)
    mastery = pd.read_csv(MASTERY_FILE)

    required_question_columns = {
        "question_id",
        "subtopic_id",
        "difficulty_level",
    }

    required_interaction_columns = {
        "student_id",
        "question_id",
        "is_correct",
    }

    required_mastery_columns = {
        "student_id",
        "subtopic_id",
        "mastery_probability",
    }
    required_topic_columns = {
        "subtopic_id",
        "subject_id",
        "topic_id",
    }
    required_student_columns = {
        "student_id",
        "grade_level",
        "educational_tier",
        "institution_type",
    }

    for column in required_question_columns:
        if column not in questions.columns:
            raise ValueError(
                f"Missing questions.csv column: {column}"
            )

    for column in required_interaction_columns:
        if column not in interactions.columns:
            raise ValueError(
                f"Missing student_interactions.csv column: {column}"
            )

    for column in required_mastery_columns:
        if column not in mastery.columns:
            raise ValueError(
                f"Missing BKT mastery column: {column}"
            )

    for column in required_topic_columns:
        if column not in topics.columns:
            raise ValueError(
                f"Missing topics.csv column: {column}"
            )

    for column in required_student_columns:
        if column not in students.columns:
            raise ValueError(
                f"Missing students.csv column: {column}"
            )

    topic_lookup = topics[
        ["subtopic_id", "subject_id", "topic_id"]
    ].drop_duplicates("subtopic_id")
    questions = questions.merge(
        topic_lookup,
        on="subtopic_id",
        how="left",
        validate="many_to_one",
    )

    missing_subject = questions["subject_id"].isna()
    if missing_subject.any():
        missing_ids = questions.loc[missing_subject, "subtopic_id"].unique()
        raise ValueError(
            "Questions reference subtopics missing from topics.csv: "
            f"{missing_ids[:10].tolist()}"
        )

    return questions, interactions, mastery, students


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def preprocess_data(questions, interactions, mastery):
    """Normalize difficulty, response, and mastery values for processing.

    Unknown difficulty labels raise ``ValueError``; non-binary interaction
    labels and incomplete mastery rows are excluded.
    """

    questions = questions.copy()
    interactions = interactions.copy()
    mastery = mastery.copy()

    difficulty_text = questions["difficulty_level"].astype(str).str.strip().str.lower()
    difficulty_numeric = difficulty_text.map(DIFFICULTY_MAP)
    normalized_values = pd.to_numeric(difficulty_text, errors="coerce")

    normalized_mask = (
        difficulty_numeric.isna()
        & normalized_values.between(0, 1, inclusive="both")
    )
    difficulty_numeric.loc[normalized_mask] = pd.cut(
        normalized_values.loc[normalized_mask],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=[1, 2, 3],
    ).astype(float)

    questions["difficulty_numeric"] = difficulty_numeric

    if questions["difficulty_numeric"].isna().any():
        unknown = questions.loc[
            questions["difficulty_numeric"].isna(),
            "difficulty_level"
        ].unique()

        raise ValueError(
            "Unknown question difficulty labels: "
            f"{unknown.tolist()}. Use Easy/Medium/Hard labels or numeric "
            "values between 0 and 1."
        )

    interactions["is_correct"] = pd.to_numeric(
        interactions["is_correct"],
        errors="coerce"
    )

    interactions = interactions[
        interactions["is_correct"].isin([0, 1])
    ].copy()

    mastery["mastery_probability"] = pd.to_numeric(
        mastery["mastery_probability"],
        errors="coerce"
    ).clip(0, 1)

    mastery = mastery.dropna(
        subset=["student_id", "subtopic_id", "mastery_probability"]
    )

    return questions, interactions, mastery


# ---------------------------------------------------------
# STUDENT PERFORMANCE HISTORY
# ---------------------------------------------------------

def calculate_student_history(interactions, questions):
    """Aggregate observed history by student, subtopic, subject, and question.

    The returned dictionary contains separate data frames for those four
    grouping levels, including accuracy and attempt-count measures.
    """
    interaction_columns = [
        "interaction_id",
        "student_id",
        "question_id",
        "is_correct",
        "response_time_ms",
        "interaction_timestamp",
    ]
    history = interactions[
        [column for column in interaction_columns if column in interactions]
    ].merge(
        questions[["question_id", "subtopic_id", "subject_id"]],
        on="question_id",
        how="left",
        validate="many_to_one",
    )
    history["interaction_timestamp"] = pd.to_datetime(
        history["interaction_timestamp"], errors="coerce"
    )
    if "response_time_ms" in history:
        history["response_time_ms"] = pd.to_numeric(
            history["response_time_ms"], errors="coerce"
        )
    history = history.sort_values(
        ["interaction_timestamp", "interaction_id"], kind="mergesort"
    )

    def summarize(keys, prefix):
        """Aggregate attempt count, overall accuracy, and the latest five rows."""
        grouped = history.groupby(keys, sort=False, dropna=False)
        summary = grouped["is_correct"].agg(["count", "mean"]).reset_index()
        summary = summary.rename(columns={
            "count": f"{prefix}_previous_attempts",
            "mean": f"{prefix}_prior_accuracy",
        })
        recent = (
            history.groupby(keys, sort=False, dropna=False, group_keys=False)
            .tail(5)
            .groupby(keys, sort=False, dropna=False)["is_correct"]
            .mean()
            .rename(f"{prefix}_recent_accuracy_5")
            .reset_index()
        )
        summary = summary.merge(recent, on=keys, how="left", validate="one_to_one")
        return summary

    subtopic_history = summarize(
        ["student_id", "subtopic_id"], "subtopic"
    )
    response_time = (
        history.groupby(["student_id", "subtopic_id"], sort=False)[
            "response_time_ms"
        ]
        .mean()
        .reset_index(name="subtopic_prior_avg_response_ms")
    )
    last_practice = (
        history.groupby(["student_id", "subtopic_id"], sort=False)[
            "interaction_timestamp"
        ]
        .max()
        .reset_index()
    )
    subtopic_history = subtopic_history.merge(
        response_time,
        on=["student_id", "subtopic_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        last_practice,
        on=["student_id", "subtopic_id"],
        how="left",
        validate="one_to_one",
    )
    subtopic_history["days_since_subtopic_practice"] = (
        pd.Timestamp.now() - subtopic_history["interaction_timestamp"]
    ).dt.total_seconds() / 86400
    subtopic_history = subtopic_history.drop(columns="interaction_timestamp")

    subject_history = summarize(
        ["student_id", "subject_id"], "subject"
    )
    student_history = (
        history.groupby("student_id", sort=False)["is_correct"]
        .agg(
            student_previous_attempts="count",
            student_prior_accuracy="mean",
        )
        .reset_index()
    )
    question_attempts = (
        history.groupby(["student_id", "question_id"], sort=False)
        .size()
        .reset_index(name="previous_question_attempts")
    )
    return {
        "subtopic": subtopic_history,
        "subject": subject_history,
        "student": student_history,
        "question": question_attempts,
    }


# ---------------------------------------------------------
# LOAD CORRECTNESS MODEL
# ---------------------------------------------------------

def load_correctness_model():
    """Load the optional correctness model, or report its absence/failure."""

    if not CORRECTNESS_MODEL_FILE.exists():
        print(
            "Correctness model not found. "
            "Using BKT-based difficulty selection."
        )
        return None

    try:
        model = joblib.load(CORRECTNESS_MODEL_FILE)

        print("Correctness prediction model loaded.")
        return model

    except Exception as error:
        print(f"Could not load correctness model: {error}")
        return None


# ---------------------------------------------------------
# BUILD MODEL FEATURES
# ---------------------------------------------------------

def build_model_features(
    model,
    candidate_data,
    student_history,
    students,
):
    """
    Construct features only when they match the trained model's
    recorded feature names.

    If the model expects unknown features, the engine falls back
    to BKT-based selection rather than inventing feature values.
    """

    if not hasattr(model, "feature_names_in_"):
        return None

    feature_names = list(model.feature_names_in_)

    feature_data = candidate_data.merge(
        students[
            ["student_id", "grade_level", "educational_tier", "institution_type"]
        ].drop_duplicates("student_id"),
        on="student_id",
        how="left",
        validate="many_to_one",
    )
    feature_data = feature_data.merge(
        student_history["subtopic"],
        on=["student_id", "subtopic_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        student_history["subject"],
        on=["student_id", "subject_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        student_history["student"],
        on="student_id",
        how="left",
        validate="many_to_one",
    ).merge(
        student_history["question"],
        on=["student_id", "question_id"],
        how="left",
        validate="many_to_one",
    )
    feature_data["attempt_number"] = (
        feature_data["previous_question_attempts"].fillna(0) + 1
    )
    feature_data["difficulty_level"] = pd.to_numeric(
        feature_data["difficulty_level"], errors="coerce"
    )
    feature_data["estimated_solving_time_sec"] = pd.to_numeric(
        feature_data["estimated_solving_time_sec"], errors="coerce"
    )
    feature_data["marks_allocated"] = pd.to_numeric(
        feature_data["marks_allocated"], errors="coerce"
    )

    missing = [name for name in feature_names if name not in feature_data]
    if missing:
        return None, missing

    return feature_data[feature_names], []


# ---------------------------------------------------------
# PREDICT QUESTION CORRECTNESS
# ---------------------------------------------------------

def predict_correctness(model, candidate_data, student_history, students):
    """Attach model probabilities when compatible inputs are available.

    If no model is supplied, required features are unavailable, or inference
    fails, the candidate rows retain missing predictions for BKT fallback.
    """

    candidate_data = candidate_data.copy()
    candidate_data["predicted_correctness"] = np.nan

    if model is None:
        return candidate_data

    X = build_model_features(
        model,
        candidate_data,
        student_history,
        students,
    )

    if isinstance(X, tuple):
        features, missing = X
        if features is None:
            print(
                "Correctness model features unavailable "
                f"({', '.join(missing)}); using BKT-based selection."
            )
            return candidate_data
        X = features

    if X is None:
        return candidate_data

    try:

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)

            classes = list(model.classes_)

            if 1 not in classes:
                return candidate_data

            positive_index = classes.index(1)

            candidate_data["predicted_correctness"] = (
                probabilities[:, positive_index]
            )

        else:

            candidate_data["predicted_correctness"] = (
                model.predict(X)
            )

    except Exception as error:

        print(
            "Correctness prediction failed. "
            f"Using BKT-based selection. Details: {error}"
        )

    return candidate_data


# ---------------------------------------------------------
# SELECT TARGET DIFFICULTY
# ---------------------------------------------------------

def determine_target_difficulty(mastery_probability):
    """Map a mastery probability to a difficulty band and mastery label."""

    if mastery_probability < 0.40:
        return 1, "Needs Support"

    elif mastery_probability < 0.75:
        return 2, "Developing"

    else:
        return 3, "Mastered"


# ---------------------------------------------------------
# GENERATE RECOMMENDATIONS
# ---------------------------------------------------------

def generate_recommendations(
    questions,
    interactions,
    mastery,
    students,
    student_id=None,
    top_n=10
):
    """Select unattempted questions for each eligible student and subtopic.

    Recommendations are ordered by their difficulty-fit score and limited to
    ``top_n`` rows per student/subtopic.
    """

    student_history = calculate_student_history(interactions, questions)

    if "last_practice_timestamp" in mastery.columns:
        mastery = mastery.copy()
        mastery["last_practice_timestamp"] = pd.to_datetime(
            mastery["last_practice_timestamp"], errors="coerce"
        )
        mastery = (
            mastery.sort_values("last_practice_timestamp", kind="mergesort")
            .drop_duplicates(["student_id", "subtopic_id"], keep="last")
        )

    if student_id is not None:

        mastery = mastery[
            mastery["student_id"].astype(str) == str(student_id)
        ].copy()

        if mastery.empty:
            raise ValueError(
                f"No BKT mastery records found for student {student_id}"
            )

    model = load_correctness_model()

    recommendations = []

    for _, mastery_row in mastery.iterrows():

        current_student = mastery_row["student_id"]
        subtopic_id = mastery_row["subtopic_id"]

        mastery_probability = float(
            mastery_row["mastery_probability"]
        )

        target_difficulty, mastery_level = (
            determine_target_difficulty(mastery_probability)
        )

        candidate_questions = questions[
            questions["subtopic_id"].astype(str)
            == str(subtopic_id)
        ].copy()

        if candidate_questions.empty:
            continue

        # Exclude questions already attempted by this student.
        attempted_ids = interactions.loc[
            interactions["student_id"].astype(str)
            == str(current_student),
            "question_id"
        ].astype(str).unique()

        candidate_questions = candidate_questions[
            ~candidate_questions["question_id"].astype(str)
            .isin(attempted_ids)
        ].copy()

        if candidate_questions.empty:
            continue

        candidate_questions["student_id"] = current_student

        candidate_questions["mastery_probability"] = (
            mastery_probability
        )

        candidate_questions["target_difficulty"] = (
            target_difficulty
        )

        candidate_questions["mastery_level"] = mastery_level

        candidate_questions = predict_correctness(
            model,
            candidate_questions,
            student_history,
            students,
        )

        candidate_questions["difficulty_distance"] = (
            candidate_questions["difficulty_numeric"]
            - target_difficulty
        ).abs()

        # Prefer questions predicted to be challenging but achievable.
        candidate_questions["selection_score"] = (
            1 - (
                candidate_questions["difficulty_distance"] / 2
            )
        )

        has_prediction = (
            candidate_questions["predicted_correctness"].notna()
        )

        candidate_questions.loc[
            has_prediction,
            "selection_score"
        ] = (
            0.50 * candidate_questions.loc[
                has_prediction, "selection_score"
            ]
            +
            0.50 * (
                1 - (
                    candidate_questions.loc[
                        has_prediction,
                        "predicted_correctness"
                    ] - 0.70
                ).abs()
            )
        )

        candidate_questions["recommendation_reason"] = (
            "Selected based on BKT mastery and question difficulty"
        )

        candidate_questions.loc[
            has_prediction,
            "recommendation_reason"
        ] = (
            "Selected using BKT mastery, question difficulty "
            "and predicted correctness"
        )

        recommendations.append(candidate_questions)

    if not recommendations:
        return pd.DataFrame()

    result = pd.concat(
        recommendations,
        ignore_index=True
    )

    result = result.sort_values(
        by=[
            "student_id",
            "subtopic_id",
            "selection_score",
        ],
        ascending=[True, True, False]
    )

    # Select up to top_n questions for each student-subtopic.
    result = (
        result.groupby(
            ["student_id", "subtopic_id"],
            group_keys=False
        )
        .head(top_n)
        .copy()
    )

    result["recommended_difficulty"] = (
        result["difficulty_numeric"]
        .map(DIFFICULTY_LABELS)
    )

    output_columns = [
        "student_id",
        "subtopic_id",
        "question_id",
        "question_text",
        "difficulty_level",
        "recommended_difficulty",
        "mastery_probability",
        "mastery_level",
        "target_difficulty",
        "predicted_correctness",
        "selection_score",
        "recommendation_reason",
    ]

    # Include optional question metadata when available.
    optional_columns = [
        "question_type",
        "concept_tag",
        "estimated_solving_time_sec",
        "marks_allocated",
    ]

    output_columns.extend([
        column for column in optional_columns
        if column in result.columns
    ])

    output_columns = [
        column for column in output_columns
        if column in result.columns
    ]

    return result[output_columns]


# ---------------------------------------------------------
# CREATE SUMMARY
# ---------------------------------------------------------

def create_summary(recommendations):
    """Count recommended questions by student and recommended difficulty."""

    if recommendations.empty:
        return pd.DataFrame()

    summary = (
        recommendations.groupby(
            ["student_id", "recommended_difficulty"]
        )
        .size()
        .reset_index(name="recommended_question_count")
    )

    return summary


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    """Run the adaptive difficulty workflow and write recommendation reports."""

    parser = argparse.ArgumentParser(
        description="Adaptive Question Difficulty Engine"
    )

    parser.add_argument(
        "--student-id",
        type=str,
        default=None,
        help="Generate recommendations for one student"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maximum questions per student-subtopic"
    )

    args = parser.parse_args()

    print("\n========================================")
    print(" ADAPTIVE DIFFICULTY ENGINE")
    print("========================================\n")

    print("Loading datasets...")

    questions, interactions, mastery, students = load_data()

    print(f"Questions loaded: {len(questions):,}")
    print(f"Interactions loaded: {len(interactions):,}")
    print(f"BKT mastery records: {len(mastery):,}")

    print("\nPreprocessing data...")

    questions, interactions, mastery = preprocess_data(
        questions,
        interactions,
        mastery
    )

    print("\nGenerating adaptive difficulty recommendations...")

    recommendations = generate_recommendations(
        questions=questions,
        interactions=interactions,
        mastery=mastery,
        students=students,
        student_id=args.student_id,
        top_n=args.top_n
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    recommendations_path = (
        OUTPUT_DIR / "question_recommendations.csv"
    )

    summary_path = (
        OUTPUT_DIR / "difficulty_summary.csv"
    )

    recommendations.to_csv(
        recommendations_path,
        index=False
    )

    summary = create_summary(recommendations)

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n========================================")
    print(" DIFFICULTY ENGINE COMPLETED")
    print("========================================")

    print(
        f"\nRecommended questions: {len(recommendations):,}"
    )

    print(f"Saved: {recommendations_path}")
    print(f"Saved: {summary_path}")

    if not recommendations.empty:

        print("\nRecommended difficulty distribution:")

        print(
            recommendations[
                "recommended_difficulty"
            ].value_counts()
        )

        print("\nSample recommendations:")

        print(
            recommendations.head(10).to_string(index=False)
        )

    else:

        print(
            "\nNo recommendations generated. "
            "Check that BKT subtopic IDs match question subtopic IDs "
            "and that eligible unattempted questions exist."
        )


if __name__ == "__main__":
    main()
