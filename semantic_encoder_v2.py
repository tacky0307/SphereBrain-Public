from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import sqlite3

from brain import SphereBrain, SignalResult

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BRAIN_FILE = DATA / "brain_semantic_v2.json"
DB_FILE = DATA / "semantic_v2.db"
SOURCE_BRAIN_FILE = DATA / "brain.json"


@dataclass(frozen=True)
class StructuredInput:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


@dataclass
class StructuredExperience:
    input: StructuredInput
    subject_result: SignalResult
    relation_result: SignalResult
    content_result: SignalResult

    @property
    def all_nodes(self) -> list[int]:
        return sorted(set(self.subject_result.activated_nodes) | set(self.relation_result.activated_nodes) | set(self.content_result.activated_nodes))

    @property
    def all_edges(self) -> list[tuple[int, int]]:
        return sorted(set(self.subject_result.traversed_edges) | set(self.relation_result.traversed_edges) | set(self.content_result.traversed_edges))


def initialize_db() -> None:
    DATA.mkdir(exist_ok=True)
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantic_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                content TEXT NOT NULL,
                subject_nodes TEXT NOT NULL,
                relation_nodes TEXT NOT NULL,
                content_nodes TEXT NOT NULL,
                subject_edges TEXT NOT NULL,
                relation_edges TEXT NOT NULL,
                content_edges TEXT NOT NULL,
                all_nodes TEXT NOT NULL,
                all_edges TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_experiences(subject);
            CREATE INDEX IF NOT EXISTS idx_semantic_content ON semantic_experiences(content);
            """
        )


def create_clean_brain() -> SphereBrain:
    DATA.mkdir(exist_ok=True)
    if SOURCE_BRAIN_FILE.exists():
        source = SphereBrain.load(SOURCE_BRAIN_FILE)
        brain = SphereBrain(
            node_count=source.node_count,
            neighbors_per_node=source.neighbors_per_node,
            seed=source.seed,
            learning_rate=source.learning_rate,
            decay_rate=source.decay_rate,
            propagation_mode=source.propagation_mode,
            signal_decay=source.signal_decay,
            max_branches=source.max_branches,
            max_active_per_step=source.max_active_per_step,
            max_total_active_nodes=source.max_total_active_nodes,
        )
    else:
        brain = SphereBrain(node_count=600)
    brain.save(BRAIN_FILE)
    return brain


def load_brain() -> SphereBrain:
    if BRAIN_FILE.exists():
        return SphereBrain.load(BRAIN_FILE)
    return create_clean_brain()


def reset_experiment() -> None:
    if BRAIN_FILE.exists():
        BRAIN_FILE.unlink()
    if DB_FILE.exists():
        DB_FILE.unlink()
    initialize_db()
    create_clean_brain()


def component_nodes(brain: SphereBrain, namespace: str, value: str, count: int = 3) -> list[int]:
    material = f"semantic-v2|{namespace}|{value.strip()}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    nodes: list[int] = []
    offset = 0
    while len(nodes) < count:
        if offset + 4 > len(digest):
            digest = hashlib.sha256(digest).digest()
            offset = 0
        node = int.from_bytes(digest[offset:offset + 4], "big") % brain.node_count
        if node not in nodes:
            nodes.append(node)
        offset += 4
    return nodes


def _context_tail(result: SignalResult, limit: int = 18) -> list[int]:
    ordered: list[int] = []
    for step in reversed(result.activation_history):
        for node in step:
            if node not in ordered:
                ordered.append(int(node))
            if len(ordered) >= limit:
                return ordered
    return ordered


def encode_and_experience(brain: SphereBrain, item: StructuredInput, learn: bool = True) -> StructuredExperience:
    subject_sources = component_nodes(brain, "role:subject", "subject", 2) + component_nodes(brain, "entity", item.subject, 3)
    subject_result = brain.propagate(subject_sources, steps=8, threshold=0.18, noise=0.004 if learn else 0.0, learn=learn)

    relation_sources = component_nodes(brain, "role:relation", "relation", 2) + component_nodes(brain, "relation", item.relation, 3)
    relation_result = brain.propagate(
        relation_sources,
        steps=8,
        threshold=0.18,
        noise=0.004 if learn else 0.0,
        learn=learn,
        context_nodes=_context_tail(subject_result),
    )

    content_sources = component_nodes(brain, "role:content", "content", 2) + component_nodes(brain, "content", item.content, 3)
    content_result = brain.propagate(
        content_sources,
        steps=10,
        threshold=0.18,
        noise=0.004 if learn else 0.0,
        learn=learn,
        context_nodes=_context_tail(relation_result),
    )

    return StructuredExperience(item, subject_result, relation_result, content_result)


def save_experience(experience: StructuredExperience) -> None:
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            """
            INSERT INTO semantic_experiences(
                subject, relation, content,
                subject_nodes, relation_nodes, content_nodes,
                subject_edges, relation_edges, content_edges,
                all_nodes, all_edges
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                experience.input.subject,
                experience.input.relation,
                experience.input.content,
                json.dumps(experience.subject_result.activated_nodes),
                json.dumps(experience.relation_result.activated_nodes),
                json.dumps(experience.content_result.activated_nodes),
                json.dumps(experience.subject_result.traversed_edges),
                json.dumps(experience.relation_result.traversed_edges),
                json.dumps(experience.content_result.traversed_edges),
                json.dumps(experience.all_nodes),
                json.dumps(experience.all_edges),
            ),
        )


def train(subject: str, relation: str, content: str, repeats: int = 1) -> StructuredExperience:
    subject, relation, content = subject.strip(), relation.strip(), content.strip()
    if not subject or not relation or not content:
        raise ValueError("主体・関係・内容をすべて入力してください。")
    brain = load_brain()
    latest: StructuredExperience | None = None
    for _ in range(max(1, int(repeats))):
        latest = encode_and_experience(brain, StructuredInput(subject, relation, content), learn=True)
        save_experience(latest)
    brain.save(BRAIN_FILE)
    assert latest is not None
    return latest


def jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _score_probe_against_rows(probe_nodes: set[int], probe_edges: set[tuple[int, int]], rows: list[sqlite3.Row], *, group_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[sqlite3.Row]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    results: list[dict] = []
    for key, items in grouped.items():
        node_scores: list[float] = []
        edge_scores: list[float] = []
        for row in items:
            stored_nodes = json.loads(row["all_nodes"])
            stored_edges = [tuple(v) for v in json.loads(row["all_edges"])]
            node_scores.append(jaccard(probe_nodes, stored_nodes))
            edge_scores.append(jaccard(probe_edges, stored_edges))
        score = 0.42 * max(node_scores, default=0.0) + 0.58 * max(edge_scores, default=0.0)
        item = {field: value for field, value in zip(group_fields, key)}
        item.update({"score": score, "experiences": len(items)})
        results.append(item)
    results.sort(key=lambda x: (-x["score"], -x["experiences"], tuple(x[field] for field in group_fields)))
    return results


def recall_probe(subject: str, relation: str = "動作", limit: int = 8) -> dict:
    brain = load_brain()
    cue = StructuredInput(subject.strip(), relation.strip(), "__cue__")
    subject_sources = component_nodes(brain, "role:subject", "subject", 2) + component_nodes(brain, "entity", cue.subject, 3)
    subject_result = brain.propagate(subject_sources, steps=8, threshold=0.18, noise=0.0, learn=False)
    relation_sources = component_nodes(brain, "role:relation", "relation", 2) + component_nodes(brain, "relation", cue.relation, 3)
    relation_result = brain.propagate(relation_sources, steps=10, threshold=0.18, noise=0.0, learn=False, context_nodes=_context_tail(subject_result))
    probe_nodes = set(subject_result.activated_nodes) | set(relation_result.activated_nodes)
    probe_edges = set(subject_result.traversed_edges) | set(relation_result.traversed_edges)
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM semantic_experiences WHERE subject=? AND relation=? ORDER BY id DESC", (cue.subject, cue.relation)).fetchall()
    results = _score_probe_against_rows(probe_nodes, probe_edges, list(rows), group_fields=("content",))
    return {"subject": cue.subject, "relation": cue.relation, "source_nodes": subject_result.source_nodes, "activated_nodes": sorted(probe_nodes), "traversed_edges": sorted(probe_edges), "matches": results[: max(1, int(limit))]}


def cross_subject_probe(subject: str, relation: str, limit: int = 20) -> dict:
    brain = load_brain()
    subject = subject.strip()
    relation = relation.strip()
    subject_sources = component_nodes(brain, "role:subject", "subject", 2) + component_nodes(brain, "entity", subject, 3)
    subject_result = brain.propagate(subject_sources, steps=8, threshold=0.18, noise=0.0, learn=False)
    relation_sources = component_nodes(brain, "role:relation", "relation", 2) + component_nodes(brain, "relation", relation, 3)
    relation_result = brain.propagate(relation_sources, steps=10, threshold=0.18, noise=0.0, learn=False, context_nodes=_context_tail(subject_result))
    probe_nodes = set(subject_result.activated_nodes) | set(relation_result.activated_nodes)
    probe_edges = set(subject_result.traversed_edges) | set(relation_result.traversed_edges)
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM semantic_experiences WHERE relation=? ORDER BY id DESC", (relation,)).fetchall()
    results = _score_probe_against_rows(probe_nodes, probe_edges, list(rows), group_fields=("subject", "content"))
    return {"subject": subject, "relation": relation, "source_nodes": subject_result.source_nodes, "activated_nodes": sorted(probe_nodes), "traversed_edges": sorted(probe_edges), "matches": results[: max(1, int(limit))]}


def subject_only_probe(subject: str, limit: int = 24) -> dict:
    brain = load_brain()
    subject = subject.strip()
    sources = component_nodes(brain, "role:subject", "subject", 2) + component_nodes(brain, "entity", subject, 3)
    result = brain.propagate(sources, steps=12, threshold=0.18, noise=0.0, learn=False)
    probe_nodes = set(result.activated_nodes)
    probe_edges = set(result.traversed_edges)
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM semantic_experiences WHERE subject=? ORDER BY id DESC", (subject,)).fetchall()
    results = _score_probe_against_rows(probe_nodes, probe_edges, list(rows), group_fields=("relation", "content"))
    return {"subject": subject, "source_nodes": result.source_nodes, "activated_nodes": result.activated_nodes, "traversed_edges": result.traversed_edges, "matches": results[: max(1, int(limit))]}


def _row_content_signature(row: sqlite3.Row) -> tuple[set[int], set[tuple[int, int]]]:
    return (
        {int(v) for v in json.loads(row["content_nodes"])},
        {tuple(int(x) for x in edge) for edge in json.loads(row["content_edges"])},
    )


def _content_pair_score(left: list[sqlite3.Row], right: list[sqlite3.Row]) -> float:
    scores: list[float] = []
    for left_row in left:
        left_nodes, left_edges = _row_content_signature(left_row)
        for right_row in right:
            right_nodes, right_edges = _row_content_signature(right_row)
            scores.append(0.42 * jaccard(left_nodes, right_nodes) + 0.58 * jaccard(left_edges, right_edges))
    return sum(scores) / len(scores) if scores else 0.0


def content_phase_analysis(relation: str, limit: int = 40) -> dict:
    """Compare only the content-stage activity, excluding subject/relation paths.

    The result tests whether identical contents across different subjects are
    structurally closer than different contents under the same relation.
    """
    relation = relation.strip()
    if not relation:
        raise ValueError("関係を入力してください。")
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT * FROM semantic_experiences WHERE relation=? ORDER BY id", (relation,)).fetchall())

    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault((str(row["subject"]), str(row["content"])), []).append(row)

    pairs: list[dict] = []
    keys = sorted(groups)
    same_scores: list[float] = []
    different_scores: list[float] = []
    for index, left_key in enumerate(keys):
        for right_key in keys[index + 1:]:
            left_subject, left_content = left_key
            right_subject, right_content = right_key
            score = _content_pair_score(groups[left_key], groups[right_key])
            same_content = left_content == right_content and left_subject != right_subject
            if same_content:
                same_scores.append(score)
            else:
                different_scores.append(score)
            pairs.append({
                "left_subject": left_subject,
                "left_content": left_content,
                "right_subject": right_subject,
                "right_content": right_content,
                "score": score,
                "same_content": same_content,
                "left_experiences": len(groups[left_key]),
                "right_experiences": len(groups[right_key]),
            })

    pairs.sort(key=lambda item: (not item["same_content"], -item["score"], item["left_subject"], item["right_subject"]))
    same_average = sum(same_scores) / len(same_scores) if same_scores else 0.0
    different_average = sum(different_scores) / len(different_scores) if different_scores else 0.0
    return {
        "relation": relation,
        "group_count": len(groups),
        "same_pair_count": len(same_scores),
        "different_pair_count": len(different_scores),
        "same_average": same_average,
        "different_average": different_average,
        "separation": same_average - different_average,
        "pairs": pairs[: max(1, int(limit))],
    }


def stats() -> dict:
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        total = conn.execute("SELECT COUNT(*) FROM semantic_experiences").fetchone()[0]
        subjects = conn.execute("SELECT COUNT(DISTINCT subject) FROM semantic_experiences").fetchone()[0]
        relations = conn.execute("SELECT COUNT(DISTINCT relation) FROM semantic_experiences").fetchone()[0]
        contents = conn.execute("SELECT COUNT(DISTINCT content) FROM semantic_experiences").fetchone()[0]
    return {"total": total, "subjects": subjects, "relations": relations, "contents": contents}
