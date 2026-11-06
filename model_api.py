"""Serve question-correctness predictions from the trained model over HTTP.

At application startup, the API loads the saved model pipeline and, when
available, the decision threshold recorded in its metadata. The
``QuestionFeatures`` schema accepts the model's numeric and categorical input
features plus optional student/question identifiers; extra request fields are
rejected. The ``/predict`` endpoint handles one question attempt, while
``/predict/batch`` accepts between one and 256 attempts. Both return the
probability of correctness and a threshold-based boolean prediction.

The ``/health`` endpoint reports whether the model has been loaded and the
number of expected features. Inference failures are logged and surfaced as
HTTP 500 responses. The service expects
``trained_models/best_question_correctness_model.pkl`` and optionally reads
``trained_models/best_model_info.json`` for the decision threshold.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Model and Metadata Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "trained_models" / "best_question_correctness_model.pkl"
MODEL_INFO_PATH = ROOT / "trained_models" / "best_model_info.json"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request and Response Schemas
# ---------------------------------------------------------------------------
class QuestionFeatures(BaseModel):
    """Optional model inputs and identifiers for one question attempt."""

    model_config = ConfigDict(extra="forbid")

    student_id: str | None = None
    question_id: str | None = None
    grade_level: float | None = None
    difficulty_level: float | None = None
    estimated_solving_time_sec: float | None = None
    marks_allocated: float | None = None
    attempt_number: float | None = None
    subtopic_previous_attempts: float | None = None
    subtopic_prior_accuracy: float | None = None
    subtopic_recent_accuracy_5: float | None = None
    subject_previous_attempts: float | None = None
    subject_prior_accuracy: float | None = None
    subject_recent_accuracy_5: float | None = None
    subtopic_prior_avg_response_ms: float | None = None
    days_since_subtopic_practice: float | None = None
    student_previous_attempts: float | None = None
    student_prior_accuracy: float | None = None
    educational_tier: str | None = None
    institution_type: str | None = None
    subject_id: str | None = None
    topic_id: str | None = None
    subtopic_id: str | None = None
    question_type: str | None = None


class BatchPredictionRequest(BaseModel):
    """A bounded collection of question attempts for batch inference."""

    predictions: Annotated[
        list[QuestionFeatures],
        Field(min_length=1, max_length=256),
    ]


class PredictionResult(BaseModel):
    """Predicted correctness probability and threshold-based decision."""

    student_id: str | None
    question_id: str | None
    probability_correct: float
    predicted_correct: bool
    decision_threshold: float


class HealthResult(BaseModel):
    """Model availability and feature-count status returned by health checks."""

    status: str
    model_loaded: bool
    feature_count: int


# ---------------------------------------------------------------------------
# Model Lifecycle
# ---------------------------------------------------------------------------
def _load_model() -> tuple[object, list[str], float]:
    """Load the trained pipeline and its configured decision threshold.

    Returns the model, its expected feature names, and a threshold in [0, 1].

    Raises
    ------
    FileNotFoundError
        If the trained model file is missing.
    ValueError
        If the model has no recorded feature names or its threshold is invalid.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("The trained model does not expose its feature names.")

    feature_names = list(model.feature_names_in_)
    if MODEL_INFO_PATH.is_file():
        with MODEL_INFO_PATH.open(encoding="utf-8") as file:
            metadata = json.load(file)
        threshold = float(metadata.get("decision_threshold", 0.5))
    else:
        threshold = 0.5
    if not 0 <= threshold <= 1:
        raise ValueError(f"Model decision threshold is invalid: {threshold}")
    return model, feature_names, threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference artifacts once and attach them to application state."""
    model, feature_names, threshold = _load_model()
    app.state.model = model
    app.state.feature_names = feature_names
    app.state.threshold = threshold
    yield


app = FastAPI(
    title="Interleaving and Practice Mix Model API",
    description="Predict correctness for one or more question attempts.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health and Prediction Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health(request: Request) -> HealthResult:
    """Report whether the model has been loaded into application state."""
    model = getattr(request.app.state, "model", None)
    return HealthResult(
        status="ok" if model is not None else "starting",
        model_loaded=model is not None,
        feature_count=len(getattr(request.app.state, "feature_names", [])),
    )


def _predict(request: Request, rows: list[QuestionFeatures]) -> list[PredictionResult]:
    """Build model-ordered features and return predictions for input rows.

    Model inference errors are logged and surfaced as an HTTP 500 response.
    """
    model = request.app.state.model
    feature_names = request.app.state.feature_names
    threshold = request.app.state.threshold

    frame = pd.DataFrame([
        row.model_dump(include=set(feature_names))
        for row in rows
    ], columns=feature_names)

    try:
        probabilities = model.predict_proba(frame)
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("The trained model has no positive class (1).")
        positive_index = classes.index(1)
        positive_probabilities = probabilities[:, positive_index]
    except Exception as exc:
        logger.exception("Question-correctness inference failed.")
        raise HTTPException(
            status_code=500,
            detail="Question-correctness inference failed.",
        ) from exc

    return [
        PredictionResult(
            student_id=row.student_id,
            question_id=row.question_id,
            probability_correct=float(probability),
            predicted_correct=bool(probability >= threshold),
            decision_threshold=threshold,
        )
        for row, probability in zip(rows, positive_probabilities, strict=True)
    ]


@app.post("/predict", response_model=PredictionResult)
def predict(features: QuestionFeatures, request: Request) -> PredictionResult:
    """Predict correctness for one question attempt."""
    return _predict(request, [features])[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> list[PredictionResult]:
    """Predict correctness for each attempt in a validated batch."""
    return _predict(request, payload.predictions)
