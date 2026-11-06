
"""Create personalized, interleaved question sequences for practice sessions.

The scheduler combines curriculum topic metadata, BKT mastery probabilities,
and questions previously selected by the difficulty engine. It joins those
records into a per-student question pool, then ranks candidate questions using
mastery weakness and how often the associated topic and subtopic have already
appeared in the sequence. It avoids repeating question IDs and prefers a
different subtopic and topic from the preceding selection whenever alternatives
are available.

For each learner, the selected questions are assigned a sequence order and a
selection reason. A session summary reports question counts, topic/subtopic
coverage and switches, interleaving entropy, average mastery, and average
weakness priority. The CSV outputs are written to
``trained_models/interleaving_scheduler/``.

Inputs are ``data/topics.csv``,
``trained_models/bkt/student_subtopic_mastery.csv``, and
``trained_models/difficulty_engine/question_recommendations.csv``. Run with
``python interleaving_scheduler.py``; ``--student-id`` selects one learner and
``--questions`` controls the requested number of questions per session.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "trained_models"

TOPICS_FILE = DATA_DIR / "topics.csv"
MASTERY_FILE = MODEL_DIR / "bkt" / "student_subtopic_mastery.csv"

RECOMMENDATIONS_FILE = (
    MODEL_DIR
    / "difficulty_engine"
    / "question_recommendations.csv"
)

OUTPUT_DIR = MODEL_DIR / "interleaving_scheduler"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    """Load topic, BKT mastery, and question recommendation CSVs.

    Raises ``FileNotFoundError`` for missing inputs and ``ValueError`` when a
    required column is absent.
    """

    required_files = [
        TOPICS_FILE,
        MASTERY_FILE,
        RECOMMENDATIONS_FILE,
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required file not found: {file}"
            )

    topics = pd.read_csv(TOPICS_FILE)
    mastery = pd.read_csv(MASTERY_FILE)
    recommendations = pd.read_csv(RECOMMENDATIONS_FILE)

    required_topic_columns = {
        "topic_id",
        "subtopic_id",
        "topic_name",
        "subtopic_name",
    }

    required_mastery_columns = {
        "student_id",
        "subtopic_id",
        "mastery_probability",
    }

    required_recommendation_columns = {
        "student_id",
        "subtopic_id",
        "question_id",
    }

    for column in required_topic_columns:
        if column not in topics.columns:
            raise ValueError(
                f"Missing topics.csv column: {column}"
            )

    for column in required_mastery_columns:
        if column not in mastery.columns:
            raise ValueError(
                f"Missing BKT column: {column}"
            )

    for column in required_recommendation_columns:
        if column not in recommendations.columns:
            raise ValueError(
                f"Missing recommendation column: {column}"
            )

    return topics, mastery, recommendations


# ---------------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------------

def preprocess_data(topics, mastery, recommendations):
    """Normalize identifiers and mastery scores and deduplicate questions."""

    topics = topics.copy()
    mastery = mastery.copy()
    recommendations = recommendations.copy()

    for dataframe in [topics, mastery, recommendations]:

        for column in [
            "student_id",
            "subtopic_id",
            "question_id",
        ]:

            if column in dataframe.columns:
                dataframe[column] = (
                    dataframe[column].astype(str)
                )

    mastery["mastery_probability"] = pd.to_numeric(
        mastery["mastery_probability"],
        errors="coerce"
    ).clip(0, 1)

    recommendations = recommendations.drop_duplicates(
        subset=["student_id", "question_id"]
    )

    return topics, mastery, recommendations


# ---------------------------------------------------------
# PREPARE QUESTION POOL
# ---------------------------------------------------------

def prepare_question_pool(
    topics,
    mastery,
    recommendations
):
    """Combine recommended questions with curriculum and mastery metadata."""

    topic_columns = [
        "subtopic_id",
        "topic_id",
        "topic_name",
        "subtopic_name",
    ]

    # Include subject metadata when available.
    if "subject_id" in topics.columns:
        topic_columns.append("subject_id")

    if "subject_name" in topics.columns:
        topic_columns.append("subject_name")

    topic_mapping = topics[
        topic_columns
    ].drop_duplicates(
        subset=["subtopic_id"]
    )

    pool = recommendations.merge(
        topic_mapping,
        on="subtopic_id",
        how="left"
    )

    pool = pool.merge(
        mastery[
            [
                "student_id",
                "subtopic_id",
                "mastery_probability",
            ]
        ],
        on=["student_id", "subtopic_id"],
        how="left",
        suffixes=("", "_bkt")
    )

    # Prefer the mastery value from BKT.
    if "mastery_probability_bkt" in pool.columns:
        pool["mastery_probability"] = (
            pool["mastery_probability_bkt"]
        )
        pool.drop(
            columns=["mastery_probability_bkt"],
            inplace=True
        )

    pool["mastery_probability"] = (
        pool["mastery_probability"].fillna(0.5)
    )

    pool["weakness_priority"] = (
        1 - pool["mastery_probability"]
    )

    pool = pool.dropna(
        subset=["topic_id", "subtopic_id"]
    )

    return pool


# ---------------------------------------------------------
# SELECT INTERLEAVED QUESTIONS
# ---------------------------------------------------------

def create_interleaved_sequence(
    student_questions,
    questions_per_session
):
    """Select a prioritized sequence without repeating question IDs.

    Selection favors low-mastery concepts and topics/subtopics receiving less
    practice, while preferring not to repeat the previous subtopic or topic.
    """

    available = student_questions.copy()

    selected = []

    used_questions = set()
    subtopic_counts = {}
    topic_counts = {}

    previous_subtopic = None
    previous_topic = None

    for sequence_number in range(
        1,
        questions_per_session + 1
    ):

        candidates = available[
            ~available["question_id"].isin(used_questions)
        ].copy()

        if candidates.empty:
            break

        candidates["subtopic_count"] = (
            candidates["subtopic_id"]
            .map(subtopic_counts)
            .fillna(0)
        )

        candidates["topic_count"] = (
            candidates["topic_id"]
            .map(topic_counts)
            .fillna(0)
        )

        # Prefer weak concepts that have received less practice.
        candidates["selection_score"] = (
            0.55 * candidates["weakness_priority"]
            +
            0.25 / (1 + candidates["subtopic_count"])
            +
            0.20 / (1 + candidates["topic_count"])
        )

        # Avoid immediately repeating the same subtopic.
        different_subtopic = candidates[
            candidates["subtopic_id"] != previous_subtopic
        ]

        if not different_subtopic.empty:
            candidates = different_subtopic

        # Prefer changing topics when possible.
        different_topic = candidates[
            candidates["topic_id"] != previous_topic
        ]

        if not different_topic.empty:
            candidates = different_topic

        candidates = candidates.sort_values(
            by=[
                "selection_score",
                "weakness_priority",
            ],
            ascending=False
        )

        chosen = candidates.iloc[0].copy()

        chosen["sequence_order"] = sequence_number

        if previous_subtopic is None:
            reason = "Initial question selected by mastery priority"

        elif chosen["subtopic_id"] != previous_subtopic:
            reason = "Interleaved with a different subtopic"

        else:
            reason = "No alternative subtopic available"

        chosen["selection_reason"] = reason

        selected.append(chosen)

        used_questions.add(chosen["question_id"])

        subtopic = chosen["subtopic_id"]
        topic = chosen["topic_id"]

        subtopic_counts[subtopic] = (
            subtopic_counts.get(subtopic, 0) + 1
        )

        topic_counts[topic] = (
            topic_counts.get(topic, 0) + 1
        )

        previous_subtopic = subtopic
        previous_topic = topic

    if not selected:
        return pd.DataFrame()

    return pd.DataFrame(selected)


# ---------------------------------------------------------
# GENERATE ALL STUDENT SESSIONS
# ---------------------------------------------------------

def generate_sessions(
    pool,
    questions_per_session,
    student_id=None
):
    """Generate a practice sequence per student, optionally for one student."""

    if student_id is not None:

        pool = pool[
            pool["student_id"].astype(str)
            == str(student_id)
        ].copy()

        if pool.empty:
            raise ValueError(
                f"No recommended questions found for {student_id}"
            )

    all_sessions = []

    students = sorted(pool["student_id"].unique())

    for student in students:

        student_questions = pool[
            pool["student_id"] == student
        ].copy()

        sequence = create_interleaved_sequence(
            student_questions,
            questions_per_session
        )

        if sequence.empty:
            continue

        session_id = f"SESSION_{student}"

        sequence.insert(
            0,
            "session_id",
            session_id
        )

        sequence.insert(
            1,
            "student_id",
            student
        )

        all_sessions.append(sequence)

    if not all_sessions:
        return pd.DataFrame()

    return pd.concat(
        all_sessions,
        ignore_index=True
    )


# ---------------------------------------------------------
# CALCULATE SESSION METRICS
# ---------------------------------------------------------

def calculate_entropy(sequence):
    """Calculate Shannon entropy over the sequence's subtopic distribution."""

    counts = sequence["subtopic_id"].value_counts()

    probabilities = counts / counts.sum()

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy)


def create_session_summary(sequences):
    """Summarize topic switching, mastery, priority, and sequence per session."""

    if sequences.empty:
        return pd.DataFrame()

    summaries = []

    for session_id, group in sequences.groupby("session_id"):

        group = group.sort_values("sequence_order")

        subtopic_sequence = (
            group["subtopic_id"].tolist()
        )

        topic_sequence = (
            group["topic_id"].tolist()
        )

        subtopic_switches = sum(
            subtopic_sequence[i] != subtopic_sequence[i - 1]
            for i in range(1, len(subtopic_sequence))
        )

        topic_switches = sum(
            topic_sequence[i] != topic_sequence[i - 1]
            for i in range(1, len(topic_sequence))
        )

        summaries.append({

            "session_id": session_id,

            "student_id": group["student_id"].iloc[0],

            "total_questions": len(group),

            "unique_topics": group["topic_id"].nunique(),

            "unique_subtopics": group["subtopic_id"].nunique(),

            "topic_switch_count": topic_switches,

            "subtopic_switch_count": subtopic_switches,

            "interleaving_entropy": calculate_entropy(group),

            "average_mastery": (
                group["mastery_probability"].mean()
            ),

            "average_weakness_priority": (
                group["weakness_priority"].mean()
            ),

            "question_sequence": ",".join(
                group["question_id"].astype(str)
            ),

            "subtopic_sequence": ",".join(
                group["subtopic_id"].astype(str)
            ),

        })

    return pd.DataFrame(summaries)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    """Run scheduling, persist sequences and summaries, and print examples."""

    parser = argparse.ArgumentParser(
        description="Personalized Interleaving Scheduler"
    )

    parser.add_argument(
        "--student-id",
        type=str,
        default=None
    )

    parser.add_argument(
        "--questions",
        type=int,
        default=10,
        help="Number of questions per practice session"
    )

    args = parser.parse_args()

    if args.questions < 1:
        raise ValueError(
            "Number of questions must be at least 1."
        )

    print("\n========================================")
    print(" PERSONALIZED INTERLEAVING SCHEDULER")
    print("========================================")

    print("\nLoading datasets...")

    topics, mastery, recommendations = load_data()

    print(f"Topics records: {len(topics):,}")
    print(f"Mastery records: {len(mastery):,}")
    print(f"Recommended questions: {len(recommendations):,}")

    print("\nPreprocessing...")

    topics, mastery, recommendations = preprocess_data(
        topics,
        mastery,
        recommendations
    )

    print("\nPreparing question pool...")

    pool = prepare_question_pool(
        topics,
        mastery,
        recommendations
    )

    print(f"Eligible question records: {len(pool):,}")

    print("\nGenerating interleaved sessions...")

    sequences = generate_sessions(
        pool=pool,
        questions_per_session=args.questions,
        student_id=args.student_id
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sequence_file = (
        OUTPUT_DIR / "practice_sequences.csv"
    )

    summary_file = (
        OUTPUT_DIR / "session_summary.csv"
    )

    sequences.to_csv(
        sequence_file,
        index=False
    )

    summary = create_session_summary(sequences)

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n========================================")
    print(" INTERLEAVING SCHEDULER COMPLETED")
    print("========================================")

    print(f"\nGenerated sessions: {len(summary):,}")
    print(f"Questions scheduled: {len(sequences):,}")

    print(f"\nSaved: {sequence_file}")
    print(f"Saved: {summary_file}")

    if not summary.empty:

        print("\nSession summary:")

        print(
            summary.head(10).to_string(index=False)
        )

        print("\nSample practice sequence:")

        display_columns = [
            "student_id",
            "sequence_order",
            "question_id",
            "subtopic_id",
            "topic_id",
            "mastery_probability",
            "selection_reason",
        ]

        print(
            sequences[
                [
                    col for col in display_columns
                    if col in sequences.columns
                ]
            ].head(15).to_string(index=False)
        )

    else:

        print(
            "\nNo sessions generated. Check that question "
            "recommendations match the topic and mastery IDs."
        )


if __name__ == "__main__":
    main()
