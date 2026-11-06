"""Build and validate a graph of subjects, topics, subtopics, and questions.

The script reads the curriculum CSV files, checks required columns and
references, and creates a directed graph with ``contains`` and
``prerequisite_of`` relationships. It records problems such as duplicate
identifiers, invalid attributes, unresolved prerequisites, self-prerequisites,
and prerequisite cycles instead of silently treating the input as complete.
Helper functions expose graph lookups for curriculum relationships.

The curriculum graph intentionally does not contain learner-specific state.
Optional Bayesian Knowledge Tracing (BKT) results are loaded from a separate
CSV and queried by ``get_student_mastery`` without mutating the graph.

By default, the script reads ``data/subjects.csv``, ``data/topics.csv``, and
``data/questions.csv``. It optionally reads the BKT mastery file at
``trained_models/bkt/student_subtopic_mastery.csv`` and writes graph JSON/CSV
exports, a summary, and validation findings under
``trained_models/knowledge_graph/``. The data, BKT, and output directories can
be overridden with command-line options. Run with
``uv run python build_knowledge_graph.py`` from the project directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_BKT_DIR = PROJECT_DIR / "trained_models" / "bkt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "trained_models" / "knowledge_graph"

SUBJECT_COLUMNS = ["subject_id", "subject_name", "educational_tier"]
TOPIC_COLUMNS = [
    "subject_id",
    "topic_id",
    "topic_name",
    "subtopic_id",
    "subtopic_name",
    "grade_level",
    "prerequisite_topic_ids",
]
QUESTION_COLUMNS = [
    "question_id",
    "subtopic_id",
    "question_text",
    "question_type",
    "difficulty_level",
    "concept_tag",
    "estimated_solving_time_sec",
    "marks_allocated",
]
MASTERY_COLUMNS = [
    "student_id",
    "subtopic_id",
    "mastery_probability",
    "mastery_level",
    "last_practice_timestamp",
]


def _blank_validation() -> dict[str, list[dict[str, Any]]]:
    """Create the empty report buckets used while validating graph inputs."""
    return {
        "missing_subject_references": [],
        "missing_topic_references": [],
        "missing_subtopic_references": [],
        "questions_without_valid_subtopics": [],
        "unresolved_prerequisites": [],
        "prerequisite_cycles": [],
        "self_prerequisites": [],
        "duplicate_identifiers": [],
        "invalid_attributes": [],
    }


def _read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Read a CSV as strings and ensure that its required headers are present.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not name a file.
    ValueError
        If parsing fails or required columns are absent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV file not found: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read CSV file {path}: {exc}") from exc
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing_columns)}"
        )
    return frame


def load_and_validate_csvs(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load subjects, topics, and questions from the curriculum data folder."""
    data_dir = Path(data_dir)
    subjects = _read_csv(data_dir / "subjects.csv", SUBJECT_COLUMNS)
    topics = _read_csv(data_dir / "topics.csv", TOPIC_COLUMNS)
    questions = _read_csv(data_dir / "questions.csv", QUESTION_COLUMNS)
    return subjects, topics, questions


def _text(value: Any) -> str:
    """Convert a CSV value to stripped text, treating ``None`` as empty."""
    if value is None:
        return ""
    return str(value).strip()


def _number(
    value: Any,
    *,
    source_id: str,
    attribute: str,
    validation: dict[str, list[dict[str, Any]]],
    integer: bool = False,
) -> int | float | None:
    """Parse and validate a finite numeric attribute for graph construction.

    Invalid or missing values are recorded in ``validation`` and return
    ``None``. Integral values are returned as integers when requested.
    """
    raw = _text(value)
    if not raw:
        validation["invalid_attributes"].append({
            "source_id": source_id,
            "attribute": attribute,
            "value": raw,
            "reason": "missing_numeric_value",
        })
        return None
    try:
        number = float(raw)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        validation["invalid_attributes"].append({
            "source_id": source_id,
            "attribute": attribute,
            "value": raw,
            "reason": "not_a_finite_number",
        })
        return None
    return int(number) if integer and number.is_integer() else number


def _attributes_agree(rows: pd.DataFrame, columns: list[str]) -> bool:
    """Return whether each selected column has at most one normalized value."""
    return all(
        rows[column].map(_text).nunique(dropna=False) <= 1
        for column in columns
        if column in rows.columns
    )


def _add_node(
    graph: nx.DiGraph,
    node_id: str,
    attributes: dict[str, Any],
) -> None:
    """Add a node only if its identifier is not already present."""
    if node_id not in graph:
        graph.add_node(node_id, **attributes)


def create_subject_nodes(
    graph: nx.DiGraph,
    subjects: pd.DataFrame,
    validation: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Create one graph node per subject and return source-ID mappings."""
    subject_nodes: dict[str, str] = {}
    for subject_id, rows in subjects.groupby("subject_id", sort=True):
        subject_id = _text(subject_id)
        if not subject_id:
            validation["missing_subject_references"].append({
                "source": "subjects.csv",
                "row_count": len(rows),
                "reason": "empty_subject_id",
            })
            continue
        if len(rows) > 1:
            validation["duplicate_identifiers"].append({
                "node_type": "subject",
                "source_id": subject_id,
                "row_count": len(rows),
            })
        row = rows.iloc[0]
        node_id = f"SUBJECT:{subject_id}"
        _add_node(graph, node_id, {
            "node_type": "subject",
            "source_id": subject_id,
            "label": _text(row["subject_name"]) or subject_id,
            "subject_id": subject_id,
            "subject_name": _text(row["subject_name"]),
            "educational_tier": _text(row["educational_tier"]),
        })
        subject_nodes[subject_id] = node_id
    return subject_nodes


def create_topic_nodes(
    graph: nx.DiGraph,
    topics: pd.DataFrame,
    subject_nodes: dict[str, str],
    validation: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Create topic nodes and connect resolvable subjects to their topics."""
    topic_nodes: dict[str, str] = {}
    for topic_id, rows in topics.groupby("topic_id", sort=True):
        topic_id = _text(topic_id)
        if not topic_id:
            validation["missing_topic_references"].append({
                "source": "topics.csv",
                "row_count": len(rows),
                "reason": "empty_topic_id",
            })
            continue
        if not _attributes_agree(rows, ["subject_id", "topic_name"]):
            validation["invalid_attributes"].append({
                "source_id": topic_id,
                "node_type": "topic",
                "reason": "conflicting_topic_attributes",
            })
        row = rows.iloc[0]
        subject_id = _text(row["subject_id"])
        node_id = f"TOPIC:{topic_id}"
        grade_levels = sorted({
            grade
            for value in rows["grade_level"]
            if (grade := _number(
                value,
                source_id=topic_id,
                attribute="grade_level",
                validation=validation,
                integer=True,
            )) is not None
        })
        topic_grade_level: int | float | list[int | float] | None
        if len(grade_levels) == 1:
            topic_grade_level = grade_levels[0]
        elif grade_levels:
            topic_grade_level = grade_levels
        else:
            topic_grade_level = None
        _add_node(graph, node_id, {
            "node_type": "topic",
            "source_id": topic_id,
            "label": _text(row["topic_name"]) or topic_id,
            "topic_id": topic_id,
            "topic_name": _text(row["topic_name"]),
            "subject_id": subject_id,
            "grade_level": topic_grade_level,
        })
        if len(grade_levels) > 1:
            graph.nodes[node_id]["grade_levels"] = grade_levels
        topic_nodes[topic_id] = node_id
        if not subject_id or subject_id not in subject_nodes:
            validation["missing_subject_references"].append({
                "topic_id": topic_id,
                "subject_id": subject_id or None,
            })
        else:
            graph.add_edge(
                subject_nodes[subject_id],
                node_id,
                relation="contains_topic",
            )
    return topic_nodes


def create_subtopic_nodes(
    graph: nx.DiGraph,
    topics: pd.DataFrame,
    topic_nodes: dict[str, str],
    validation: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Create subtopic nodes and connect resolvable parent topics."""
    subtopic_nodes: dict[str, str] = {}
    for subtopic_id, rows in topics.groupby("subtopic_id", sort=True):
        subtopic_id = _text(subtopic_id)
        if not subtopic_id:
            validation["missing_subtopic_references"].append({
                "source": "topics.csv",
                "row_count": len(rows),
                "reason": "empty_subtopic_id",
            })
            continue
        if len(rows) > 1:
            validation["duplicate_identifiers"].append({
                "node_type": "subtopic",
                "source_id": subtopic_id,
                "row_count": len(rows),
            })
        if not _attributes_agree(
            rows,
            ["topic_id", "subject_id", "subtopic_name", "grade_level"],
        ):
            validation["invalid_attributes"].append({
                "source_id": subtopic_id,
                "node_type": "subtopic",
                "reason": "conflicting_subtopic_attributes",
            })
        row = rows.iloc[0]
        topic_id = _text(row["topic_id"])
        node_id = f"SUBTOPIC:{subtopic_id}"
        _add_node(graph, node_id, {
            "node_type": "subtopic",
            "source_id": subtopic_id,
            "label": _text(row["subtopic_name"]) or subtopic_id,
            "subtopic_id": subtopic_id,
            "subtopic_name": _text(row["subtopic_name"]),
            "topic_id": topic_id,
            "subject_id": _text(row["subject_id"]),
            "grade_level": _number(
                row["grade_level"],
                source_id=subtopic_id,
                attribute="grade_level",
                validation=validation,
                integer=True,
            ),
        })
        subtopic_nodes[subtopic_id] = node_id
        if not topic_id or topic_id not in topic_nodes:
            validation["missing_topic_references"].append({
                "subtopic_id": subtopic_id,
                "topic_id": topic_id or None,
            })
        else:
            graph.add_edge(
                topic_nodes[topic_id],
                node_id,
                relation="contains_subtopic",
            )
    return subtopic_nodes


def create_question_nodes(
    graph: nx.DiGraph,
    questions: pd.DataFrame,
    subtopic_nodes: dict[str, str],
    validation: dict[str, list[dict[str, Any]]],
) -> None:
    """Create question nodes and connect them to valid subtopic nodes."""
    for question_id, rows in questions.groupby("question_id", sort=True):
        question_id = _text(question_id)
        if not question_id:
            validation["invalid_attributes"].append({
                "source": "questions.csv",
                "reason": "empty_question_id",
                "row_count": len(rows),
            })
            continue
        if len(rows) > 1:
            validation["duplicate_identifiers"].append({
                "node_type": "question",
                "source_id": question_id,
                "row_count": len(rows),
            })
        row = rows.iloc[0]
        subtopic_id = _text(row["subtopic_id"])
        node_id = f"QUESTION:{question_id}"
        _add_node(graph, node_id, {
            "node_type": "question",
            "source_id": question_id,
            "label": _text(row["question_text"]) or question_id,
            "question_id": question_id,
            "question_text": _text(row["question_text"]),
            "question_type": _text(row["question_type"]),
            "difficulty_level": _number(
                row["difficulty_level"],
                source_id=question_id,
                attribute="difficulty_level",
                validation=validation,
            ),
            "concept_tag": _text(row["concept_tag"]),
            "estimated_solving_time_sec": _number(
                row["estimated_solving_time_sec"],
                source_id=question_id,
                attribute="estimated_solving_time_sec",
                validation=validation,
            ),
            "marks_allocated": _number(
                row["marks_allocated"],
                source_id=question_id,
                attribute="marks_allocated",
                validation=validation,
            ),
            "subtopic_id": subtopic_id,
        })
        if not subtopic_id or subtopic_id not in subtopic_nodes:
            validation["questions_without_valid_subtopics"].append({
                "question_id": question_id,
                "subtopic_id": subtopic_id or None,
            })
            if subtopic_id:
                validation["missing_subtopic_references"].append({
                    "question_id": question_id,
                    "subtopic_id": subtopic_id,
                })
        else:
            graph.add_edge(
                subtopic_nodes[subtopic_id],
                node_id,
                relation="contains_question",
            )


def _parse_prerequisite_ids(raw_value: Any) -> tuple[list[str], str | None]:
    """Decode a prerequisite cell as a JSON list of string or integer IDs."""
    raw = _text(raw_value)
    if not raw:
        return [], None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"invalid_json: {exc.msg}"
    if not isinstance(values, list):
        return [], "prerequisite value is not a JSON list"
    ids = []
    for value in values:
        if not isinstance(value, (str, int)):
            return [], f"prerequisite ID must be a string or integer: {value!r}"
        prerequisite_id = _text(value)
        if prerequisite_id:
            ids.append(prerequisite_id)
    return ids, None


def create_prerequisite_relationships(
    graph: nx.DiGraph,
    topics: pd.DataFrame,
    topic_nodes: dict[str, str],
    subtopic_nodes: dict[str, str],
    validation: dict[str, list[dict[str, Any]]],
) -> None:
    """Add prerequisite edges and record malformed or unresolved references."""
    seen: set[tuple[str, str]] = set()
    for row in topics.to_dict("records"):
        topic_id = _text(row["topic_id"])
        subtopic_id = _text(row["subtopic_id"])
        prerequisites, parse_error = _parse_prerequisite_ids(
            row["prerequisite_topic_ids"]
        )
        if parse_error:
            validation["unresolved_prerequisites"].append({
                "dependent_topic_id": topic_id or None,
                "dependent_subtopic_id": subtopic_id or None,
                "raw_value": _text(row["prerequisite_topic_ids"]),
                "reason": parse_error,
            })
            continue
        for prerequisite_id in prerequisites:
            if prerequisite_id in topic_nodes:
                source = topic_nodes[prerequisite_id]
                target = topic_nodes.get(topic_id)
                target_type = "topic"
            elif prerequisite_id in subtopic_nodes:
                source = subtopic_nodes[prerequisite_id]
                target = subtopic_nodes.get(subtopic_id)
                target_type = "subtopic"
            else:
                validation["unresolved_prerequisites"].append({
                    "prerequisite_id": prerequisite_id,
                    "dependent_topic_id": topic_id or None,
                    "dependent_subtopic_id": subtopic_id or None,
                    "reason": "prerequisite ID does not match a known topic or subtopic",
                })
                continue
            if target is None:
                validation["unresolved_prerequisites"].append({
                    "prerequisite_id": prerequisite_id,
                    "dependent_topic_id": topic_id or None,
                    "dependent_subtopic_id": subtopic_id or None,
                    "reason": f"dependent {target_type} does not resolve",
                })
                continue
            if source == target:
                validation["self_prerequisites"].append({
                    "prerequisite_id": prerequisite_id,
                    "dependent_id": topic_id if target_type == "topic" else subtopic_id,
                    "node_type": target_type,
                })
                continue
            edge = (source, target)
            if edge not in seen:
                graph.add_edge(source, target, relation="prerequisite_of")
                seen.add(edge)


def check_missing_references(
    validation: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return all reference-validation findings collected during construction."""
    return validation


def detect_prerequisite_cycles(
    graph: nx.DiGraph,
    validation: dict[str, list[dict[str, Any]]] | None = None,
) -> list[list[str]]:
    """Find prerequisite cycles and optionally store normalized cycle reports."""
    prerequisite_graph = nx.DiGraph()
    for source, target, attributes in graph.edges(data=True):
        if attributes.get("relation") == "prerequisite_of":
            prerequisite_graph.add_edge(source, target)

    cycles = []
    for cycle in nx.simple_cycles(prerequisite_graph):
        if cycle:
            smallest_index = min(range(len(cycle)), key=cycle.__getitem__)
            normalized = cycle[smallest_index:] + cycle[:smallest_index]
            cycles.append(normalized)
    cycles.sort()
    if validation is not None:
        validation["prerequisite_cycles"] = [
            {
                "node_ids": cycle,
                "source_ids": [
                    graph.nodes[node_id]["source_id"] for node_id in cycle
                ],
            }
            for cycle in cycles
        ]
    return cycles


def _resolve_node_id(graph: nx.DiGraph, identifier: str) -> str | None:
    """Resolve a graph ID or a unique source ID to its graph node ID."""
    identifier = _text(identifier)
    if identifier in graph:
        return identifier
    matches = [
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if attributes.get("source_id") == identifier
    ]
    return matches[0] if len(matches) == 1 else None


def _related_nodes(
    graph: nx.DiGraph,
    identifier: str,
    *,
    direction: str,
    relation: str | None = None,
) -> list[dict[str, Any]]:
    """Return sorted adjacent node records filtered by direction and relation."""
    node_id = _resolve_node_id(graph, identifier)
    if node_id is None:
        return []
    adjacent = (
        graph.successors(node_id)
        if direction == "out"
        else graph.predecessors(node_id)
    )
    result = []
    for other_id in adjacent:
        attributes = graph.nodes[other_id]
        edge = (
            graph[node_id][other_id]
            if direction == "out"
            else graph[other_id][node_id]
        )
        if relation is None or edge.get("relation") == relation:
            result.append({"node_id": other_id, **attributes})
    return sorted(result, key=lambda item: (item.get("label", ""), item["node_id"]))


def get_subjects(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Return all subject nodes in stable label order."""
    return sorted(
        (
            {"node_id": node_id, **attributes}
            for node_id, attributes in graph.nodes(data=True)
            if attributes.get("node_type") == "subject"
        ),
        key=lambda item: (item.get("label", ""), item["node_id"]),
    )


def get_topics_for_subject(
    graph: nx.DiGraph,
    subject_id: str,
) -> list[dict[str, Any]]:
    """Return topics linked to the requested subject."""
    return _related_nodes(
        graph, subject_id, direction="out", relation="contains_topic"
    )


def get_subtopics_for_topic(
    graph: nx.DiGraph,
    topic_id: str,
) -> list[dict[str, Any]]:
    """Return subtopics linked to the requested topic."""
    return _related_nodes(
        graph, topic_id, direction="out", relation="contains_subtopic"
    )


def get_questions_for_subtopic(
    graph: nx.DiGraph,
    subtopic_id: str,
) -> list[dict[str, Any]]:
    """Return questions linked to the requested subtopic."""
    return _related_nodes(
        graph, subtopic_id, direction="out", relation="contains_question"
    )


def get_prerequisites(
    graph: nx.DiGraph,
    concept_id: str,
) -> list[dict[str, Any]]:
    """Return concepts that are prerequisites of the requested concept."""
    return _related_nodes(
        graph, concept_id, direction="in", relation="prerequisite_of"
    )


def get_dependent_concepts(
    graph: nx.DiGraph,
    concept_id: str,
) -> list[dict[str, Any]]:
    """Return concepts that depend on the requested concept."""
    return _related_nodes(
        graph, concept_id, direction="out", relation="prerequisite_of"
    )


def load_student_mastery(
    path: str | Path = DEFAULT_BKT_DIR / "student_subtopic_mastery.csv",
) -> pd.DataFrame | None:
    """Load optional BKT results; return None if that output is not available."""
    path = Path(path)
    if not path.is_file():
        print(f"BKT mastery file not found; mastery lookup is unavailable: {path}")
        return None
    mastery = _read_csv(path, MASTERY_COLUMNS)
    mastery["mastery_probability"] = pd.to_numeric(
        mastery["mastery_probability"], errors="coerce"
    )
    mastery["last_practice_timestamp"] = pd.to_datetime(
        mastery["last_practice_timestamp"], errors="coerce"
    )
    return mastery


def get_student_mastery(
    student_id: str,
    subtopic_id: str | None = None,
    mastery_data: pd.DataFrame | None = None,
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """Return BKT mastery records without mutating the curriculum graph."""
    if mastery_data is None:
        mastery_data = load_student_mastery()
    if mastery_data is None:
        return None
    records = mastery_data.loc[
        mastery_data["student_id"].map(_text) == _text(student_id)
    ]
    if subtopic_id is not None:
        records = records.loc[
            records["subtopic_id"].map(_text) == _text(subtopic_id)
        ]
        if records.empty:
            return None
        return (
            records.sort_values("last_practice_timestamp", kind="mergesort")
            .iloc[-1]
            .to_dict()
        )
    latest_records = (
        records.sort_values("last_practice_timestamp", kind="mergesort")
        .drop_duplicates("subtopic_id", keep="last")
    )
    return latest_records.to_dict("records")


def _json_safe(value: Any) -> Any:
    """Convert nested values and non-finite floats to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_graph_and_reports(
    graph: nx.DiGraph,
    validation: dict[str, list[dict[str, Any]]],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write graph nodes, edges, summary, and validation reports to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = [
        {"node_id": node_id, **attributes}
        for node_id, attributes in sorted(graph.nodes(data=True))
    ]
    edges = [
        {
            "source_node": source,
            "target_node": target,
            "relation": attributes["relation"],
        }
        for source, target, attributes in sorted(
            graph.edges(data=True),
            key=lambda edge: (edge[0], edge[1], edge[2].get("relation", "")),
        )
    ]
    graph_json = {
        "directed": True,
        "multigraph": False,
        "nodes": nodes,
        "edges": edges,
    }

    prerequisite_edges = sum(
        edge["relation"] == "prerequisite_of" for edge in edges
    )
    node_counts = {
        node_type: sum(
            attributes.get("node_type") == node_type
            for _, attributes in graph.nodes(data=True)
        )
        for node_type in ("subject", "topic", "subtopic", "question")
    }
    summary = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "subject_count": node_counts["subject"],
        "topic_count": node_counts["topic"],
        "subtopic_count": node_counts["subtopic"],
        "question_count": node_counts["question"],
        "prerequisite_relationship_count": prerequisite_edges,
        "unresolved_prerequisite_count": len(
            validation["unresolved_prerequisites"]
        ),
        "cycle_count": len(validation["prerequisite_cycles"]),
    }

    node_attribute_columns = sorted({
        key for node in nodes for key in node if key != "node_id"
    })
    node_fields = ["node_id", *node_attribute_columns]
    node_path = output_dir / "knowledge_graph_nodes.csv"
    with node_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=node_fields)
        writer.writeheader()
        for node in nodes:
            writer.writerow({
                key: (
                    json.dumps(_json_safe(value), ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else _json_safe(value)
                )
                for key, value in node.items()
            })

    edge_path = output_dir / "knowledge_graph_edges.csv"
    with edge_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["source_node", "target_node", "relation"]
        )
        writer.writeheader()
        writer.writerows(edges)

    paths = {
        "graph": output_dir / "knowledge_graph.json",
        "edges": edge_path,
        "nodes": node_path,
        "summary": output_dir / "knowledge_graph_summary.json",
        "validation": output_dir / "knowledge_graph_validation.json",
    }
    for key, payload in (
        ("graph", graph_json),
        ("summary", summary),
        ("validation", validation),
    ):
        with paths[key].open("w", encoding="utf-8") as file:
            json.dump(_json_safe(payload), file, indent=2, ensure_ascii=False)
            file.write("\n")
    return paths


def build_knowledge_graph(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[nx.DiGraph, dict[str, list[dict[str, Any]]]]:
    """Build the curriculum graph and collect structural validation findings."""
    subjects, topics, questions = load_and_validate_csvs(data_dir)
    graph = nx.DiGraph()
    validation = _blank_validation()

    print("Creating subject nodes...")
    subject_nodes = create_subject_nodes(graph, subjects, validation)
    print("Creating topic and subtopic nodes...")
    topic_nodes = create_topic_nodes(graph, topics, subject_nodes, validation)
    subtopic_nodes = create_subtopic_nodes(
        graph, topics, topic_nodes, validation
    )
    print("Creating question nodes...")
    create_question_nodes(graph, questions, subtopic_nodes, validation)
    print("Resolving prerequisite relationships...")
    create_prerequisite_relationships(
        graph, topics, topic_nodes, subtopic_nodes, validation
    )
    detect_prerequisite_cycles(graph, validation)
    return graph, check_missing_references(validation)


def main() -> None:
    """Parse paths, build and export the graph, and report optional BKT data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--bkt-dir", type=Path, default=DEFAULT_BKT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    graph, validation = build_knowledge_graph(args.data_dir)
    outputs = export_graph_and_reports(graph, validation, args.output_dir)
    mastery = load_student_mastery(
        args.bkt_dir / "student_subtopic_mastery.csv"
    )
    print(
        f"Graph built: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges."
    )
    print(
        f"Prerequisite cycles: {len(validation['prerequisite_cycles'])}; "
        f"unresolved prerequisites: {len(validation['unresolved_prerequisites'])}."
    )
    print(f"BKT mastery loaded: {'yes' if mastery is not None else 'no'}")
    for path in outputs.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
