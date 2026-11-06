"""Define the shared input-feature contract for correctness modeling.

``NUMERIC_FEATURES`` lists profile, question, attempt-history, accuracy,
response-time, and practice-recency values. ``CATEGORICAL_FEATURES`` lists
learner, institution, curriculum, and question-type identifiers. The combined
``FEATURE_COLUMNS`` list concatenates the numeric fields followed by the
categorical fields; consumers use this order to select and align prediction
data with the model's training inputs.

This module contains only feature-name constants. It does not load data, train
a model, or perform inference.
"""

# ---------------------------------------------------------------------------
# Numeric Features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "grade_level",
    "difficulty_level",
    "estimated_solving_time_sec",
    "marks_allocated",
    "attempt_number",
    "subtopic_previous_attempts",
    "subtopic_prior_accuracy",
    "subtopic_recent_accuracy_5",
    "subject_previous_attempts",
    "subject_prior_accuracy",
    "subject_recent_accuracy_5",
    "subtopic_prior_avg_response_ms",
    "days_since_subtopic_practice",
    "student_previous_attempts",
    "student_prior_accuracy",
]

# ---------------------------------------------------------------------------
# Categorical Features
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "educational_tier",
    "institution_type",
    "subject_id",
    "topic_id",
    "subtopic_id",
    "question_type",
]

# ---------------------------------------------------------------------------
# Combined Model Feature Contract
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
