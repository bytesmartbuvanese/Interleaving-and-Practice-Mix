# Interleaving and Practice Mix

The correctness model uses question metadata, student profile fields, and
student/subject/subtopic performance calculated only from earlier attempts.
The feature names are shared by training and inference in `model_features.py`.
Exam results and future session data are not model inputs. Recommendations skip
previously answered questions when enough unseen questions are available, so
question-specific answer history is not used as a model feature.

## Train and evaluate

Place the generated CSV files in `data/`, then run:

```bash
uv run python train_models_improved.py
```

The script selects parameters and a decision threshold using validation
students, then reports results on a separate held-out student set. Compare
accuracy only against results from the same student-level split and data; a
random row split can look better because the same students appear in both sets.

For the current supplied dataset, the held-out accuracy is about `0.7566`
(threshold `0.47`). This is a measurement, not a guaranteed production
accuracy; evaluate again on representative real student data before deployment.

## Generate recommendations

After training, run:

```bash
uv run python recommend_questions.py
```

The recommender loads datasets from `data/` and the model from
`trained_models/best_question_correctness_model.pkl`. The same feature schema
is used as training; historical features are computed from the student's
recorded prior interactions.

## Build the curriculum knowledge graph

The graph builder uses `subjects.csv`, `topics.csv`, and `questions.csv`.
Prerequisites may reference either topic or subtopic IDs; unresolved references
and prerequisite cycles are preserved in the validation report. Student BKT
mastery is read separately from
`trained_models/bkt/student_subtopic_mastery.csv` and does not modify the global
curriculum graph.

```bash
uv run python build_knowledge_graph.py
```

By default the graph builder writes JSON, node/edge CSVs, a summary, and a
validation report to `trained_models/knowledge_graph/`. Use `--data-dir`,
`--bkt-dir`, and `--output-dir` to select alternate locations. Import
`build_knowledge_graph` to access graph queries and `get_student_mastery()` for
student-specific BKT lookups.

## Large recommendation output

The generated `question_recommendations.csv` exceeds GitHub's per-file limit,
so its checked-in snapshot is stored as numbered CSV parts in
`trained_models/difficulty_engine/`. Each part includes its own header. To
reconstruct the snapshot, run:

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

Running `difficulty_engine.py` also generates the un-split output from the
project data.

## Serve correctness predictions over HTTP

Start the local FastAPI server from the project directory:

```bash
uv run uvicorn model_api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the API. `GET /health` checks
readiness, `POST /predict` predicts one question, and `POST /predict/batch`
accepts up to 256 question feature records. Input feature names match the
trained model; omitted historical numeric or categorical values are handled by
the model's fitted imputers. The response includes correctness probability,
the predicted class, and the model's saved decision threshold.