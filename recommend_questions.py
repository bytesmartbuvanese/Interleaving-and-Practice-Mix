
"""Interactively recommend and save a personalized interleaved practice set.

The script loads the trained correctness model and the student, question,
topic, interaction, session, and subject datasets. It enriches questions with
curriculum identifiers and derives each learner's prior performance by
subtopic, subject, and overall student history. For the selected learner and
subject, it builds the model's expected feature rows, predicts correctness,
and ranks candidate questions by subtopic weakness and proximity to the
configured target challenge level. Previously answered questions are excluded
unless there are too few unseen questions for the requested session size.

The selected sequence is printed with question, topic, difficulty, and
predicted-correctness details. It is also saved as a timestamped recommendation
CSV with a companion JSON file containing session metadata and question order.
The files are written under ``recommendations/``.

The module currently loads its model and datasets and prepares the output
directory during import. Running it as a script prompts for a student ID,
subject ID, and session size, then generates and saves the recommendations.
"""

import os
import json
import uuid
import joblib
import random
import numpy as np
import pandas as pd

from datetime import datetime

from model_features import FEATURE_COLUMNS

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATA_DIR = "data"
MODEL_PATH = "trained_models/best_question_correctness_model.pkl"

STUDENTS_PATH = os.path.join(DATA_DIR, "students.csv")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.csv")
TOPICS_PATH = os.path.join(DATA_DIR, "topics.csv")
INTERACTIONS_PATH = os.path.join(DATA_DIR, "student_interactions.csv")
SESSIONS_PATH = os.path.join(DATA_DIR, "practice_sessions.csv")
SUBJECTS_PATH = os.path.join(DATA_DIR, "subjects.csv")

OUTPUT_DIR = "recommendations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

TARGET_CORRECTNESS = 0.65

# ==========================================================
# 2. LOAD MODEL AND DATASETS
# ==========================================================

print("\nLoading recommendation system...")

model = joblib.load(MODEL_PATH)

students = pd.read_csv(STUDENTS_PATH)
questions = pd.read_csv(QUESTIONS_PATH)
topics = pd.read_csv(TOPICS_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH)
sessions = pd.read_csv(SESSIONS_PATH)

interactions["interaction_timestamp"] = pd.to_datetime(
    interactions["interaction_timestamp"], errors="coerce"
)
interactions = interactions.loc[
    interactions["interaction_timestamp"] <= pd.Timestamp.now()
].copy()
question_attempt_counts = interactions.groupby(
    ["student_id", "question_id"]
).size()

sessions["session_start_timestamp"] = pd.to_datetime(
    sessions["session_start_timestamp"]
)

# Add subject and topic information to questions.

questions = questions.merge(
    topics[
        [
            "subtopic_id",
            "subject_id",
            "topic_id",
            "topic_name",
            "subtopic_name",
            "grade_level"
        ]
    ].drop_duplicates("subtopic_id"),
    on="subtopic_id",
    how="left"
)

# ==========================================================
# 3. STUDENT INFORMATION
# ==========================================================

def get_student(student_id):
    """Return the matching student row or raise when the ID is unknown."""

    student = students[
        students["student_id"] == student_id
    ]

    if student.empty:
        raise ValueError("Student ID not found.")

    return student.iloc[0]


# ==========================================================
# 4. HISTORICAL PERFORMANCE
# ==========================================================

def get_student_history(student_id):
    """Return this student's interactions enriched with question topics."""

    student_interactions = interactions[
        interactions["student_id"] == student_id
    ].copy()

    history = student_interactions.merge(
        questions[
            [
                "question_id",
                "subtopic_id",
                "subject_id",
                "topic_id"
            ]
        ].drop_duplicates("question_id"),
        on="question_id",
        how="left"
    )

    history = history.sort_values(
        ["interaction_timestamp", "interaction_id"]
    )

    return history


def calculate_subtopic_history(history):
    """Summarize prior attempts, accuracy, response time, and recency per subtopic."""

    history_features = {}

    for subtopic_id, group in history.groupby("subtopic_id"):

        group = group.sort_values(
            "interaction_timestamp"
        )

        correctness = group["is_correct"].astype(float)

        history_features[subtopic_id] = {

            "previous_attempts": len(group),

            "historical_accuracy": correctness.mean(),

            "recent_accuracy": correctness.tail(5).mean(),

            "previous_avg_response_time": (
                group["response_time_ms"].mean()
            ),

            "last_practice": (
                group["interaction_timestamp"].max()
            )
        }

    return history_features


# ==========================================================
# 5. BUILD MODEL FEATURES
# ==========================================================

def build_question_features(
    student,
    question,
    history
):
    """Create one model input row from student, question, and history values."""

    subtopic_id = question["subtopic_id"]

    topic_history = history.get(
        subtopic_id,
        {}
    )

    student_attempts = history.get("_student_attempts", 0)
    student_accuracy = history.get("_student_accuracy", np.nan)
    question_attempts = question_attempt_counts.get(
        (student["student_id"], question["question_id"]), 0
    )

    subtopic_attempts = topic_history.get(
        "previous_attempts",
        0
    )

    subtopic_accuracy = topic_history.get(
        "historical_accuracy",
        np.nan
    )

    subtopic_recent_accuracy = topic_history.get(
        "recent_accuracy",
        np.nan
    )

    subtopic_avg_response_time = topic_history.get(
        "previous_avg_response_time",
        np.nan
    )

    last_practice = topic_history.get(
        "last_practice"
    )

    days_since_practice = (
        (datetime.now() - last_practice).total_seconds()
        / 86400
        if last_practice is not None
        else np.nan
    )

    # Subject history is calculated from prior interactions.
    # The function receives this value through its attached
    # subject-level history fields.

    subject_accuracy = history.get(
        "_subject_accuracy",
        0.5
    )

    subject_attempts = history.get(
        "_subject_attempts",
        0
    )

    return {

        "grade_level": student["grade_level"],
        "difficulty_level": question["difficulty_level"],
        "estimated_solving_time_sec": question["estimated_solving_time_sec"],
        "marks_allocated": question["marks_allocated"],
        "attempt_number": question_attempts + 1,
        "subtopic_previous_attempts": subtopic_attempts,
        "subtopic_prior_accuracy": subtopic_accuracy,
        "subtopic_recent_accuracy_5": subtopic_recent_accuracy,
        "subject_previous_attempts": subject_attempts,
        "subject_prior_accuracy": subject_accuracy,
        "subject_recent_accuracy_5": history.get(
            "_subject_recent_accuracy", np.nan
        ),
        "subtopic_prior_avg_response_ms": subtopic_avg_response_time,
        "days_since_subtopic_practice": days_since_practice,
        "student_previous_attempts": student_attempts,
        "student_prior_accuracy": student_accuracy,
        "educational_tier": student["educational_tier"],
        "institution_type": student["institution_type"],
        "subject_id": question["subject_id"],
        "topic_id": question["topic_id"],
        "subtopic_id": question["subtopic_id"],
        "question_type": question["question_type"],
    }


# ==========================================================
# 7. PREDICT QUESTION CORRECTNESS
# ==========================================================

def predict_questions(
    student,
    subject_id,
    history,
    session_size
):
    """Predict correctness for available questions in one subject.

    Previously answered questions are considered only when the unseen question
    pool is smaller than the requested session size.
    """

    subject_questions = questions[
        questions["subject_id"] == subject_id
    ].copy()

    if subject_questions.empty:
        raise ValueError(
            "No questions available for this subject."
        )

    # Do not recommend previously answered questions
    # unless there are insufficient unseen questions.

    answered_ids = set(
        interactions.loc[
            interactions["student_id"] ==
            student["student_id"],
            "question_id"
        ].astype(str)
    )

    unseen = subject_questions[
        ~subject_questions["question_id"].astype(str).isin(
            answered_ids
        )
    ].copy()

    if len(unseen) < session_size:

        print(
            "\nNote: Not enough unanswered questions. "
            "Previously answered questions may be included."
        )

        candidate_questions = subject_questions

    else:

        candidate_questions = unseen

    feature_rows = []

    for _, question in candidate_questions.iterrows():
        features = build_question_features(
            student,
            question,
            history
        )
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    missing_features = set(FEATURE_COLUMNS) - set(feature_df.columns)
    if missing_features:
        raise ValueError(
            "Recommendation features do not match the trained model: "
            f"missing {sorted(missing_features)}."
        )
    feature_df = feature_df.loc[:, FEATURE_COLUMNS]

    probabilities = model.predict_proba(
        feature_df
    )[:, 1]

    candidate_questions = candidate_questions.copy()

    candidate_questions["predicted_correctness"] = (
        probabilities
    )

    return candidate_questions


# ==========================================================
# 8. CALCULATE LEARNING PRIORITY
# ==========================================================

def calculate_learning_priority(
    question,
    history
):
    """Score a question using historical weakness and target challenge fit."""

    subtopic_id = question["subtopic_id"]

    topic_history = history.get(
        subtopic_id,
        {}
    )

    accuracy = topic_history.get(
        "historical_accuracy",
        0.5
    )

    attempts = topic_history.get(
        "previous_attempts",
        0
    )

    # Students with lower historical accuracy receive
    # higher learning priority.

    weakness_score = 1 - accuracy

    # Give unseen subtopics an exploration priority.

    if attempts == 0:
        weakness_score = 0.65

    # Encourage questions near the desired challenge level.

    probability = question["predicted_correctness"]

    challenge_score = max(
        0,
        1 - abs(
            probability - TARGET_CORRECTNESS
        )
    )

    priority = (
        0.60 * weakness_score +
        0.40 * challenge_score
    )

    return priority


# ==========================================================
# 9. INTERLEAVING SELECTION
# ==========================================================

def generate_interleaved_session(
    candidates,
    history,
    session_size
):
    """Select high-priority questions while avoiding adjacent same-subtopic items."""

    candidates = candidates.copy()

    candidates["learning_priority"] = candidates.apply(
        lambda row: calculate_learning_priority(
            row,
            history
        ),
        axis=1
    )

    # Rank questions within each subtopic.

    candidates = candidates.sort_values(
        by="learning_priority",
        ascending=False
    )

    # Keep the best candidate questions from each subtopic.

    subtopic_groups = {}

    for subtopic_id, group in candidates.groupby(
        "subtopic_id"
    ):

        subtopic_groups[subtopic_id] = (
            group.sort_values(
                "learning_priority",
                ascending=False
            ).to_dict("records")
        )

    # Rank subtopics according to their learning priority.

    subtopic_priority = {}

    for subtopic_id, group in subtopic_groups.items():

        subtopic_priority[subtopic_id] = (
            group[0]["learning_priority"]
        )

    ordered_subtopics = sorted(
        subtopic_groups.keys(),
        key=lambda x: subtopic_priority[x],
        reverse=True
    )

    # Round-robin selection:
    # Avoid consecutive questions from the same subtopic.

    recommendations = []

    last_subtopic = None

    while len(recommendations) < session_size:

        available = [
            st for st in ordered_subtopics
            if len(subtopic_groups[st]) > 0
        ]

        if not available:
            break

        available.sort(
            key=lambda st: subtopic_priority[st],
            reverse=True
        )

        # Prefer a different subtopic from the previous question.

        different = [
            st for st in available
            if st != last_subtopic
        ]

        selected_subtopic = (
            different[0]
            if different
            else available[0]
        )

        selected_question = (
            subtopic_groups[selected_subtopic].pop(0)
        )

        recommendations.append(
            selected_question
        )

        last_subtopic = selected_subtopic

    return pd.DataFrame(recommendations)


# ==========================================================
# 10. DISPLAY RECOMMENDATIONS
# ==========================================================

def display_recommendations(
    student,
    subject_id,
    recommendations
):
    """Print question details and the resulting subtopic distribution."""

    subject_name = subjects_name = subjects_lookup.get(
        subject_id,
        subject_id
    )

    print("\n" + "=" * 75)
    print("PERSONALIZED INTERLEAVED PRACTICE SESSION")
    print("=" * 75)

    print("Student:", student["student_id"])
    print("Subject:", subject_name)

    print(
        "Recommended questions:",
        len(recommendations)
    )

    print("\n" + "-" * 75)

    for index, (_, question) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        probability = (
            question["predicted_correctness"] * 100
        )

        print(f"\nQuestion {index}")

        print("Question ID:", question["question_id"])

        print("Topic:", question["topic_name"])

        print("Subtopic:", question["subtopic_name"])

        print("Difficulty:", question["difficulty_level"])

        print("Question:", question["question_text"])

        print(
            "Predicted correctness:",
            f"{probability:.2f}%"
        )

        if probability < 0.50:
            level = "Challenging"

        elif probability < 0.75:
            level = "Moderate"

        else:
            level = "Comfortable"

        print("Predicted challenge:", level)

        print("-" * 50)

    # Summary by subtopic

    print("\nSUBTOPIC DISTRIBUTION")

    distribution = recommendations.groupby(
        "subtopic_name"
    ).size()

    print(distribution.to_string())


# ==========================================================
# 11. SAVE PRACTICE SESSION
# ==========================================================

def save_recommendations(
    student_id,
    subject_id,
    recommendations
):
    """Write a recommendation CSV and companion session metadata JSON."""

    session_id = (
        "REC_" + uuid.uuid4().hex[:10].upper()
    )

    output = recommendations.copy()

    output.insert(
        0,
        "session_id",
        session_id
    )

    output.insert(
        1,
        "student_id",
        student_id
    )

    output.insert(
        2,
        "subject_id",
        subject_id
    )

    output.insert(
        3,
        "recommendation_timestamp",
        datetime.now().isoformat()
    )

    output["predicted_correctness"] = (
        output["predicted_correctness"].round(4)
    )

    output["learning_priority"] = (
        output["learning_priority"].round(4)
    )

    filename = (
        f"{OUTPUT_DIR}/{session_id}_recommendations.csv"
    )

    output.to_csv(
        filename,
        index=False
    )

    # Save the sequence and metadata.

    metadata = {

        "session_id": session_id,

        "student_id": student_id,

        "subject_id": subject_id,

        "question_count": len(output),

        "practice_mode": "Interleaved",

        "question_sequence": (
            output["question_id"].tolist()
        ),

        "created_at": datetime.now().isoformat()

    }

    metadata_path = (
        f"{OUTPUT_DIR}/{session_id}_metadata.json"
    )

    with open(metadata_path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print("\nRecommendation saved successfully.")

    print("CSV:", filename)

    print("Metadata:", metadata_path)


# ==========================================================
# 12. MAIN EXECUTION
# ==========================================================

subjects_lookup = (
    topics[
        ["subject_id"]
    ]
    .drop_duplicates()
    .merge(
        pd.read_csv(SUBJECTS_PATH),
        on="subject_id",
        how="left"
    )
    .set_index("subject_id")["subject_name"]
    .to_dict()
)


def main():
    """Collect interactive choices and create, display, and save a session."""

    print("\n" + "=" * 65)
    print("INTERLEAVING AND PRACTICE MIX")
    print("PERSONALIZED QUESTION RECOMMENDATION ENGINE")
    print("=" * 65)

    # Student input

    student_id = input(
        "\nEnter Student ID (e.g. STU_001): "
    ).strip()

    student = get_student(student_id)

    print("\nAvailable Subjects:")

    for subject_id, subject_name in subjects_lookup.items():

        print(
            f"{subject_id} - {subject_name}"
        )

    subject_id = input(
        "\nEnter Subject ID (e.g. SUBJ_MATH): "
    ).strip()

    if subject_id not in subjects_lookup:
        raise ValueError("Invalid subject ID.")

    session_size_input = input(
        "\nNumber of questions (default 10): "
    ).strip()

    session_size = (
        int(session_size_input)
        if session_size_input
        else 10
    )

    if session_size < 1:
        raise ValueError(
            "Session size must be at least 1."
        )

    # Retrieve historical information.

    student_history_df = get_student_history(student_id)

    history_df = student_history_df[
        student_history_df["subject_id"] == subject_id
    ].copy()
    history = calculate_subtopic_history(
        history_df
    )

    history["_subject_accuracy"] = history_df["is_correct"].mean()
    history["_subject_recent_accuracy"] = (
        history_df.sort_values(
            ["interaction_timestamp", "interaction_id"]
        )["is_correct"].tail(5).mean()
    )
    history["_subject_attempts"] = len(history_df)
    history["_student_accuracy"] = student_history_df["is_correct"].mean()
    history["_student_attempts"] = len(student_history_df)

    # Predict correctness.

    print("\nPredicting question correctness...")

    candidates = predict_questions(
        student,
        subject_id,
        history,
        session_size
    )

    # Generate interleaved practice.

    print("Generating interleaved question sequence...")

    recommendations = generate_interleaved_session(
        candidates,
        history,
        session_size
    )

    if recommendations.empty:
        raise ValueError(
            "Unable to generate recommendations."
        )

    # Display and save.

    display_recommendations(
        student,
        subject_id,
        recommendations
    )

    save_recommendations(
        student_id,
        subject_id,
        recommendations
    )

    print("\nRecommendation process completed.")


if __name__ == "__main__":

    main()
