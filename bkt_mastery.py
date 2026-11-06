"""Estimate learner mastery for each practiced curriculum subtopic.

Purpose
-------
This module implements Bayesian Knowledge Tracing (BKT) to turn each learner's
ordered correct/incorrect answers into a probability of mastery for the
associated curriculum subtopic. It is an offline data-processing and model-fit
script: it reads the project's CSV datasets, estimates shared BKT parameters,
applies those parameters to valid interaction sequences, and writes analysis
tables and fit metadata. It does not expose a web service or save a reusable
trained estimator.

Processing workflow
-------------------
The script loads student interactions, question-to-subtopic mappings, and
subtopic curriculum metadata from ``data/``. It validates required columns,
joins the metadata, converts timestamps and answers to usable values, discards
rows missing a student ID, subtopic ID, valid timestamp, or binary answer, and
sorts the remaining rows by student, subtopic, timestamp, and interaction
identifier.

Responses are grouped into separate sequences for each student and subtopic.
When at least ten unique students are available, a group-based split reserves
students from parameter fitting; otherwise all available students are used
for fitting. A single set of initial-mastery, learning, guessing, and slipping
probabilities is estimated from the selected sequences with SciPy's optimizer.
The configured defaults are used when there are fewer than ten training
sequences or the optimizer returns a non-finite objective value. The resulting
parameters are then applied to every valid interaction, including interactions
from held-out students. For each answer, the script records mastery before the
answer and updates it using the response evidence followed by the modeled
learning transition.

Inputs
------
The expected input files are:

* ``data/student_interactions.csv``: interaction, student, question, answer,
  timestamp, and session fields.
* ``data/questions.csv``: question identifiers and their ``subtopic_id``.
* ``data/topics.csv``: subtopic identifiers and their subject, topic, and
  display-name metadata.

These files are read with pandas. The interaction table must contain the
columns validated in :func:`prepare_data`; topic metadata must include
``subtopic_id``, ``subject_id``, ``topic_id``, and ``subtopic_name``.

Outputs
-------
The script creates ``trained_models/bkt/`` if needed and writes:

* ``interaction_mastery.csv``: one row per valid response, including mastery
  immediately before and after that response.
* ``student_subtopic_mastery.csv``: each student's latest recorded mastery
  estimate and its ``Needs Support``, ``Developing``, or ``Mastered`` band for
  each practiced subtopic.
* ``subtopic_mastery_summary.csv``: student counts and aggregate mastery
  statistics and band counts for each subtopic.
* ``bkt_parameters.json``: parameter values, fitting status, training and
  held-out counts, processed interaction count, and mastery-band definitions.

Execution and interpretation
----------------------------
Run from the project root with ``python bkt_mastery.py``. The implementation
uses NumPy and pandas for calculations and data preparation, SciPy for
parameter optimization, and scikit-learn's ``GroupShuffleSplit`` to keep
students grouped during the fitting split. These outputs are probabilistic
indicators derived from recorded answers, not verified measures of a learner's
actual knowledge. The module also suppresses Python warnings for the duration
of its process.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and Default Model Parameters
# ---------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = os.path.join("trained_models", "bkt")
RANDOM_STATE = 42

DEFAULT_PARAMS = {
    "p_init": 0.30,
    "p_learn": 0.15,
    "p_guess": 0.20,
    "p_slip": 0.10,
}


# ---------------------------------------------------------------------------
# Interaction Preparation and Parameter Estimation
# ---------------------------------------------------------------------------
def load_csv(filename):
    """Read a project dataset from ``DATA_DIR``.

    Raises
    ------
    FileNotFoundError
        If the requested CSV does not exist.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot find {path}. Put the generated CSV files inside '{DATA_DIR}/'."
        )
    return pd.read_csv(path)


def prepare_data():
    """Join curriculum metadata to valid, ordered student interactions.

    Returns the cleaned interaction frame and per-student/subtopic response
    sequences used for parameter fitting.
    """
    interactions = load_csv("student_interactions.csv")
    questions = load_csv("questions.csv")
    topics = load_csv("topics.csv")

    required = {
        "interaction_id", "session_id", "student_id", "question_id",
        "is_correct", "interaction_timestamp"
    }
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"student_interactions.csv missing columns: {sorted(missing)}")

    df = interactions.merge(
        questions[["question_id", "subtopic_id"]].drop_duplicates("question_id"),
        on="question_id", how="left", validate="many_to_one"
    )
    topic_cols = ["subtopic_id", "subject_id", "topic_id", "subtopic_name"]
    absent = set(topic_cols) - set(topics.columns)
    if absent:
        raise ValueError(f"topics.csv missing columns: {sorted(absent)}")
    df = df.merge(
        topics[topic_cols].drop_duplicates("subtopic_id"),
        on="subtopic_id", how="left", validate="many_to_one"
    )

    df["interaction_timestamp"] = pd.to_datetime(
        df["interaction_timestamp"], errors="coerce"
    )
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
    df = df.dropna(
        subset=["student_id", "subtopic_id", "interaction_timestamp", "is_correct"]
    ).copy()
    df = df[df["is_correct"].isin([0, 1])].copy()
    df["is_correct"] = df["is_correct"].astype(int)
    df["student_id"] = df["student_id"].astype(str)
    df = df.sort_values(
        ["student_id", "subtopic_id", "interaction_timestamp", "interaction_id"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid interaction rows found after joining questions and topics.")

    sequences = []
    for (student, subtopic), group in df.groupby(
        ["student_id", "subtopic_id"], sort=False
    ):
        observations = group["is_correct"].astype(int).tolist()
        if observations:
            sequences.append({
                "student_id": student,
                "subtopic_id": str(subtopic),
                "observations": observations,
            })
    return df, sequences


def decode_parameters(raw):
    """Map unconstrained optimizer values into valid BKT probabilities."""
    sigmoid = 1.0 / (1.0 + np.exp(-raw))
    return {
        "p_init": float(0.01 + 0.98 * sigmoid[0]),
        "p_learn": float(0.001 + 0.799 * sigmoid[1]),
        "p_guess": float(0.001 + 0.489 * sigmoid[2]),
        "p_slip": float(0.001 + 0.489 * sigmoid[3]),
    }


def log_likelihood(sequences, params):
    """Calculate the log likelihood of response sequences under BKT params."""
    p_init = params["p_init"]
    p_learn = params["p_learn"]
    p_guess = params["p_guess"]
    p_slip = params["p_slip"]
    total = 0.0
    eps = 1e-12

    for observations in sequences:
        p_known = p_init
        for answer in observations:
            if answer == 1:
                p_observation = (
                    p_known * (1.0 - p_slip) + (1.0 - p_known) * p_guess
                )
                posterior = (
                    p_known * (1.0 - p_slip)
                ) / max(p_observation, eps)
            else:
                p_observation = (
                    p_known * p_slip + (1.0 - p_known) * (1.0 - p_guess)
                )
                posterior = (
                    p_known * p_slip
                ) / max(p_observation, eps)

            total += np.log(max(p_observation, eps))
            p_known = posterior + (1.0 - posterior) * p_learn

    return total


def fit_parameters(training_sequences):
    """Fit shared BKT parameters, falling back to defaults when needed.

    Returns the parameter mapping and a string describing how it was obtained.
    """
    print("Fitting shared BKT parameters...")
    if len(training_sequences) < 10:
        print("Too few sequences; using standard starting parameters.")
        return DEFAULT_PARAMS.copy(), "defaults_due_to_insufficient_sequences"

    initial = np.array([-0.85, -1.75, -1.40, -2.20])

    def objective(raw):
        """Minimize the regularized negative log likelihood."""
        params = decode_parameters(raw)
        return -log_likelihood(training_sequences, params) + (
            0.02 * np.sum((raw - initial) ** 2)
        )

    result = minimize(
        objective, initial, method="L-BFGS-B", options={"maxiter": 500}
    )
    if not np.isfinite(result.fun):
        print("Optimization failed; using standard starting parameters.")
        return DEFAULT_PARAMS.copy(), "defaults_due_to_optimization_failure"

    params = decode_parameters(result.x)
    return params, "fitted_on_training_students"


# ---------------------------------------------------------------------------
# Mastery Updates and Output Tables
# ---------------------------------------------------------------------------
def update_mastery(prior, answer, params):
    """Update a learner's mastery estimate after one binary response."""
    if answer == 1:
        likelihood_known = 1.0 - params["p_slip"]
        likelihood_unknown = params["p_guess"]
    else:
        likelihood_known = params["p_slip"]
        likelihood_unknown = 1.0 - params["p_guess"]

    numerator = prior * likelihood_known
    denominator = numerator + (1.0 - prior) * likelihood_unknown
    posterior = numerator / denominator if denominator > 0 else prior

    # Learning transition after the observed response.
    return float(np.clip(
        posterior + (1.0 - posterior) * params["p_learn"], 0.0, 1.0
    ))


def create_mastery_outputs(df, params):
    """Build per-interaction, latest-state, and subtopic-summary frames."""
    rows = []
    for (student, subtopic), group in df.groupby(
        ["student_id", "subtopic_id"], sort=False
    ):
        group = group.sort_values(
            ["interaction_timestamp", "interaction_id"]
        )
        mastery = params["p_init"]

        for row in group.itertuples(index=False):
            before = mastery
            mastery = update_mastery(mastery, int(row.is_correct), params)
            rows.append({
                "interaction_id": row.interaction_id,
                "session_id": row.session_id,
                "student_id": str(row.student_id),
                "question_id": row.question_id,
                "subtopic_id": str(row.subtopic_id),
                "subject_id": row.subject_id,
                "topic_id": row.topic_id,
                "subtopic_name": row.subtopic_name,
                "interaction_timestamp": row.interaction_timestamp,
                "is_correct": int(row.is_correct),
                "mastery_before": round(before, 6),
                "mastery_after": round(mastery, 6),
            })

    history = pd.DataFrame(rows)
    latest = (
        history.sort_values(["interaction_timestamp", "interaction_id"])
        .groupby(["student_id", "subtopic_id"], as_index=False)
        .tail(1).copy()
    )
    latest = latest.rename(columns={
        "mastery_after": "mastery_probability",
        "interaction_timestamp": "last_practice_timestamp",
    })
    latest["mastery_level"] = pd.cut(
        latest["mastery_probability"],
        bins=[-0.001, 0.40, 0.70, 1.001],
        labels=["Needs Support", "Developing", "Mastered"],
    ).astype(str)

    summary = latest.groupby(
        ["subject_id", "topic_id", "subtopic_id", "subtopic_name"],
        dropna=False
    ).agg(
        students_practiced=("student_id", "nunique"),
        average_mastery=("mastery_probability", "mean"),
        median_mastery=("mastery_probability", "median"),
        needs_support_count=(
            "mastery_level", lambda x: (x == "Needs Support").sum()
        ),
        developing_count=(
            "mastery_level", lambda x: (x == "Developing").sum()
        ),
        mastered_count=("mastery_level", lambda x: (x == "Mastered").sum()),
    ).reset_index()

    summary["average_mastery"] = summary["average_mastery"].round(4)
    summary["median_mastery"] = summary["median_mastery"].round(4)
    return history, latest, summary


# ---------------------------------------------------------------------------
# Command-Line Workflow
# ---------------------------------------------------------------------------
def main():
    """Fit BKT parameters and write mastery records and metadata."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, sequences = prepare_data()

    # Fit shared parameters on training students only.
    if df["student_id"].nunique() >= 10:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=RANDOM_STATE
        )
        _, heldout_rows = next(
            splitter.split(np.zeros((len(df), 1)), groups=df["student_id"])
        )
        heldout_students = set(df.iloc[heldout_rows]["student_id"])
        training_students = set(df["student_id"]) - heldout_students
        training_sequences = [
            s["observations"] for s in sequences
            if s["student_id"] in training_students
        ]
    else:
        heldout_students = set()
        training_students = set(df["student_id"])
        training_sequences = [s["observations"] for s in sequences]

    params, fit_status = fit_parameters(training_sequences)
    history, latest, summary = create_mastery_outputs(df, params)

    history.to_csv(os.path.join(OUTPUT_DIR, "interaction_mastery.csv"), index=False)
    latest.to_csv(
        os.path.join(OUTPUT_DIR, "student_subtopic_mastery.csv"), index=False
    )
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "subtopic_mastery_summary.csv"), index=False
    )

    metadata = {
        "algorithm": "Bayesian Knowledge Tracing",
        "parameter_fit_status": fit_status,
        "parameters": params,
        "training_student_count": len(training_students),
        "heldout_student_count": len(heldout_students),
        "training_sequence_count": len(training_sequences),
        "interaction_count": len(history),
        "mastery_bands": {
            "Needs Support": "mastery_probability <= 0.40",
            "Developing": "0.40 < mastery_probability <= 0.70",
            "Mastered": "mastery_probability > 0.70",
        },
        "note": (
            "Mastery is a probabilistic estimate, not a verified measure of "
            "actual knowledge. It updates after each recorded answer."
        ),
    }
    with open(
        os.path.join(OUTPUT_DIR, "bkt_parameters.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\nBKT completed.")
    print(f"Interactions processed: {len(history):,}")
    print(f"Student-subtopic states: {len(latest):,}")
    print(f"Subtopics summarized: {len(summary):,}")
    print("\nMastery distribution:")
    print(latest["mastery_level"].value_counts().to_string())
    print("\nSaved files in:", OUTPUT_DIR)
interleaving_and_practice_mix

if __name__ == "__main__":
    main()
