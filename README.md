# Interleaving and Practice Mix

A Python 3.12 project for building data-driven, personalized practice workflows. It estimates learner mastery from past answers, predicts question correctness, recommends appropriately challenging questions, and arranges practice in an interleaved sequence. It also provides curriculum-graph analysis and an HTTP API for model predictions.

> **Research/demo project:** the checked-in sample data is synthetic. Mastery and correctness scores are estimates from recorded interactions, not verified measures of a learner's knowledge. Evaluate on representative, appropriately governed data before using recommendations in a real learning setting.

## Project workflow

The main workflows are separate scripts rather than one end-to-end command:

1. **Generate data (optional):** `synthetic_dataset_generator.py` creates linked synthetic subject, student, curriculum, question, exam, session, and interaction CSVs. It writes files to the current working directory.
2. **Train correctness model:** `train_models_improved.py` assembles features from student profiles, question metadata, and prior interactions, then trains an XGBoost classifier. Historical features use only earlier attempts, and train/validation/test partitions are separated by student. Exam results are deliberately excluded.
3. **Estimate mastery:** `bkt_mastery.py` fits shared Bayesian Knowledge Tracing (BKT) parameters from interaction sequences and writes per-interaction and latest student/subtopic mastery estimates.
4. **Recommend and schedule practice:** `recommend_questions.py` interactively creates a student/subject practice set using the correctness model and historical performance. Alternatively, `difficulty_engine.py` uses BKT mastery (and the model when compatible) to recommend questions, and `interleaving_scheduler.py` turns those recommendations into a session sequence.
5. **Explore curriculum and predictions:** `build_knowledge_graph.py` validates and exports the curriculum graph; `model_api.py` serves the trained correctness model over HTTP. `feature_importance.py` exports model feature rankings and a chart.

The workflows consume and produce CSV/JSON/model files in `data/`, `trained_models/`, and (for the interactive recommender) `recommendations/`. Run dependent steps in order; for example, the difficulty engine needs BKT output, and the scheduler needs difficulty-engine recommendations.

## Requirements and setup

- Python `>=3.12` (also recorded in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

Install the project dependencies from the repository root:

```bash
uv sync
```

The dependencies include pandas, NumPy, scikit-learn, XGBoost, FastAPI, Uvicorn, NetworkX, joblib, and Matplotlib. BKT parameter fitting uses SciPy.

## Data

The checked-in `data/` directory contains these CSV inputs:

| File | Purpose |
| --- | --- |
| `students.csv` | Learner profile and education attributes |
| `subjects.csv` | Subject identifiers and names |
| `topics.csv` | Subject/topic/subtopic hierarchy and prerequisite references |
| `questions.csv` | Question text, subtopic, type, difficulty, and metadata |
| `practice_sessions.csv` | Session mode, question sequence, and session metadata |
| `student_interactions.csv` | Learner answers, correctness, response times, and timestamps |
| `student_exam_results.csv` | Synthetic exam records; not used by the correctness-model training workflow |

To regenerate the synthetic data, run the generator with `data/` as its working directory so its output lands where the project scripts expect it:

```bash
mkdir -p data
(cd data && uv run --project .. python ../synthetic_dataset_generator.py)
```

This replaces the CSVs in `data/`. The generator uses a fixed random seed and creates synthetic records; do not treat them as real learner observations.

## Train the correctness model

From the repository root, with the input CSVs in `data/`:

```bash
uv run python train_models_improved.py
```

The script selects model parameters and a decision threshold using validation students, then evaluates on held-out test students. It saves the fitted pipeline and reports under `trained_models/`, including:

- `best_question_correctness_model.pkl` — preprocessing and XGBoost pipeline
- `best_model_info.json` — feature names, split information, threshold, and metrics
- `model_comparison.csv` and `xgboost_tuning_results.csv` — evaluation and tuning results

For the supplied dataset, the checked-in README previously recorded held-out accuracy of about `0.7566` at threshold `0.47`. Treat that as a dataset- and split-specific result, not a production guarantee; rerun evaluation when data or code changes. Compare results only across equivalent student-level splits and datasets, not against random row splits.

## Estimate BKT mastery

Run this after the data is available:

```bash
uv run python bkt_mastery.py
```

It writes to `trained_models/bkt/`:

- `interaction_mastery.csv` — mastery before and after each valid response
- `student_subtopic_mastery.csv` — each learner's latest subtopic mastery estimate and band
- `subtopic_mastery_summary.csv` — aggregate mastery statistics by subtopic
- `bkt_parameters.json` — fitted/default parameters and fit metadata

BKT fits shared initial-mastery, learning, guessing, and slipping probabilities. With enough learners it reserves students during parameter fitting, then applies the fitted parameters to the valid interaction histories. Mastery is probabilistic and depends on the available response data.

## Recommend questions and build practice sessions

### Interactive recommender

Train the model first, then run:

```bash
uv run python recommend_questions.py
```

Enter a student ID, subject ID, and session size when prompted. The script predicts correctness for candidate questions, prioritizes weak concepts and a target challenge level, and avoids previously answered questions unless there are too few unseen questions. It saves a timestamped recommendation CSV and session metadata JSON under `recommendations/`.

### BKT-based difficulty engine and scheduler

Run the following in sequence after generating BKT mastery:

```bash
uv run python difficulty_engine.py
uv run python interleaving_scheduler.py
```

The difficulty engine recommends unattempted questions by student and subtopic, using mastery to choose an easy/medium/hard target. If a compatible trained correctness model and its features are available, predicted correctness also contributes to ranking; otherwise it falls back to BKT and difficulty. Use `--student-id ID` to limit either command to one learner. The difficulty engine also accepts `--top-n N` (default `10`); the scheduler accepts `--questions N` (default `10`).

The scheduler uses the difficulty engine's recommendation file, topic metadata, and BKT mastery to prioritize weaker and less-practiced concepts while avoiding adjacent repeats when alternatives exist. It writes `practice_sequences.csv` and `session_summary.csv` under `trained_models/interleaving_scheduler/`.

The generated `question_recommendations.csv` can exceed GitHub's per-file limit. The repository stores its snapshot as numbered parts in `trained_models/difficulty_engine/`, each with a header. To reconstruct that snapshot before running the scheduler against it:

```bash
python - <<'PY'
import csv
from pathlib import Path

directory = Path("trained_models/difficulty_engine")
parts = sorted(directory.glob("question_recommendations.part-*.csv"))
with (directory / "question_recommendations.csv").open(
    "w", encoding="utf-8", newline=""
) as output:
    writer = csv.writer(output)
    for index, part in enumerate(parts):
        with part.open(encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader)
            if index == 0:
                writer.writerow(header)
            writer.writerows(reader)
PY
```

Running `difficulty_engine.py` regenerates the full, unsplit recommendations file from project data.

## Build and query the curriculum graph

The knowledge graph is built from subjects, topics, and questions. It models containment and prerequisite relationships, validates identifiers and references, and reports unresolved prerequisites and cycles. Optional student BKT results are queried separately and do not alter the global curriculum graph.

```bash
uv run python build_knowledge_graph.py
```

By default, graph JSON/CSV exports, a summary, and a validation report are written to `trained_models/knowledge_graph/`. Directories can be overridden with `--data-dir`, `--bkt-dir`, and `--output-dir`. The module also exposes graph-query helpers such as `get_subjects`, `get_topics_for_subject`, `get_prerequisites`, and `get_student_mastery`.

## Serve correctness predictions

Train and save the model first, then start the local FastAPI service:

```bash
uv run uvicorn model_api:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs> for interactive API documentation.

- `GET /health` reports whether the model loaded and the expected feature count.
- `POST /predict` predicts one question attempt.
- `POST /predict/batch` predicts 1–256 attempts.

The API reads `trained_models/best_question_correctness_model.pkl` and optionally uses `trained_models/best_model_info.json` for its decision threshold. Prediction inputs follow the feature contract in `model_features.py`; extra fields are rejected, and omitted values are handled by the fitted preprocessing pipeline.

## Feature analysis

After training, run:

```bash
uv run python feature_importance.py
```

The script writes ranked feature importance to `trained_models/analysis/feature_importance.csv` and a top-features chart to `trained_models/analysis/feature_importance.png`.

## Repository layout

```text
.
├── data/                              # Source CSV datasets
├── src/interleaving_and_practice_mix/ # Minimal installable package entry point
├── trained_models/                    # Model, BKT, graph, and workflow outputs
├── bkt_mastery.py                     # Bayesian Knowledge Tracing workflow
├── build_knowledge_graph.py           # Curriculum graph construction/validation
├── difficulty_engine.py               # Mastery-based question recommendation
├── feature_importance.py              # Model interpretation exports
├── interleaving_scheduler.py           # Interleaved session generation
├── model_api.py                       # FastAPI inference service
├── model_features.py                  # Shared training/inference feature schema
├── recommend_questions.py              # Interactive personalized recommender
├── synthetic_dataset_generator.py     # Synthetic CSV generator
└── train_models_improved.py           # Leakage-aware XGBoost training/evaluation
```

The `interleaving-and-practice-mix` console entry point currently prints a basic startup message; the data, training, recommendation, and API workflows are run through the scripts above.
