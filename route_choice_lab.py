from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import math
import random
import re
import sqlite3
import uuid
import webbrowser

import numpy as np
from flask import Flask, render_template_string, request
from waitress import serve

from brain import SphereBrain


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BRAIN_FILE = DATA / "brain.json"
MEMORY_DB = DATA / "memory.db"
PATTERN_DB = DATA / "pattern_candidates.db"
FEEDBACK_DB = DATA / "route_choice_feedback.db"

app = Flask(__name__)

GRADE_META = {
    "positive": {"symbol": "○", "label": "正しい", "class": "positive"},
    "partial": {"symbol": "△", "label": "近い・保留", "class": "partial"},
    "negative": {"symbol": "×", "label": "違う", "class": "negative"},
}


@dataclass
class Candidate:
    key: str
    nodes: list[int]
    edges: list[tuple[int, int]]
    evidence_texts: list[str] = field(default_factory=list)
    source_kinds: set[str] = field(default_factory=set)
    decoy: bool = False
    score: float = 0.0
    percent: float = 0.0
    rank: int = 0
    decoded_text: str = "未解釈の経路"
    decode_confidence: float = 0.0
    decode_evidence: list[str] = field(default_factory=list)
    grade: str = ""


def norm(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def jload(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def clean_nodes(nodes: list[int]) -> list[int]:
    clean: list[int] = []
    for raw in nodes:
        node = int(raw)
        if not clean or clean[-1] != node:
            clean.append(node)
    return clean


def key_for(edges: list[tuple[int, int]]) -> str:
    material = ";".join(f"{a}-{b}" for a, b in edges)
    return sha256(material.encode("utf-8")).hexdigest()[:20]


def strip_observer_annotation(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+\[[^\]]+\]\s*$", "", value)
    return value.strip()


def decode_suffix(prefix: str, source_text: str) -> tuple[str, float]:
    """Human-facing decoder. Core never receives the returned language."""
    prefix_clean = str(prefix or "").strip()
    source = strip_observer_annotation(source_text)
    if not source:
        return "未解釈の経路", 0.0
    if not prefix_clean:
        return source, 45.0

    if source.startswith(prefix_clean):
        suffix = source[len(prefix_clean) :].lstrip(" 　、。・:：-—→")
        if suffix:
            return suffix, 100.0
        return source, 58.0

    position = source.find(prefix_clean)
    if position >= 0:
        suffix = source[position + len(prefix_clean) :].lstrip(" 　、。・:：-—→")
        if suffix:
            return suffix, 86.0

    common = 0
    for left, right in zip(prefix_clean, source):
        if left != right:
            break
        common += 1
    if common >= 2 and common / max(1, len(prefix_clean)) >= 0.55:
        suffix = source[common:].lstrip(" 　、。・:：-—→")
        if suffix:
            return suffix, 68.0

    return source, 42.0


def add_route(
    route_map: dict[str, Candidate],
    text: str,
    nodes: list[int],
    source_kind: str,
) -> None:
    clean = clean_nodes(nodes)
    edges = [norm(a, b) for a, b in zip(clean, clean[1:])]
    if len(edges) < 2:
        return
    route_key = key_for(edges)
    candidate = route_map.get(route_key)
    if candidate is None:
        candidate = Candidate(route_key, clean, edges)
        route_map[route_key] = candidate
    label = strip_observer_annotation(text) or "(名称なし)"
    if label not in candidate.evidence_texts:
        candidate.evidence_texts.append(label)
    candidate.source_kinds.add(source_kind)


def load_routes(limit: int = 2500) -> tuple[list[Candidate], dict[str, int]]:
    route_map: dict[str, Candidate] = {}
    counts = {"memory": 0, "reflection": 0}

    if MEMORY_DB.exists():
        with sqlite3.connect(
            f"file:{MEMORY_DB.as_posix()}?mode=ro", uri=True, timeout=30
        ) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT input_text, activated_nodes, traversed_edges "
                "FROM memories WHERE kind='input' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        before = len(route_map)
        for row in rows:
            raw_edges = jload(row["traversed_edges"], [])
            nodes: list[int] = []
            if len(raw_edges) >= 2:
                for raw_a, raw_b in raw_edges:
                    a, b = int(raw_a), int(raw_b)
                    if not nodes:
                        nodes.extend([a, b])
                    elif nodes[-1] == a:
                        nodes.append(b)
                    elif nodes[-1] == b:
                        nodes.append(a)
                    else:
                        nodes.extend([a, b])
            else:
                nodes = [int(node) for node in jload(row["activated_nodes"], [])]
            add_route(route_map, row["input_text"] or "", nodes, "memory")
        counts["memory"] = len(route_map) - before

    if PATTERN_DB.exists():
        with sqlite3.connect(
            f"file:{PATTERN_DB.as_posix()}?mode=ro", uri=True, timeout=30
        ) as connection:
            connection.row_factory = sqlite3.Row
            latest = connection.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()[0]
            rows = (
                []
                if latest is None
                else connection.execute(
                    "SELECT pattern_json, target_texts, classification "
                    "FROM reflection_pattern_snapshots "
                    "WHERE run_id=? ORDER BY pattern_id",
                    (latest,),
                ).fetchall()
            )
        before = len(route_map)
        for row in rows:
            nodes = [int(node) for node in jload(row["pattern_json"], [])]
            texts = [str(value) for value in jload(row["target_texts"], [])]
            if not texts:
                texts = [f"Reflection経路 [{row['classification'] or 'pattern'}]"]
            for text in texts:
                add_route(route_map, text, nodes, "reflection")
        counts["reflection"] = len(route_map) - before

    return list(route_map.values()), counts


def latest_reflection_run() -> int | None:
    if not PATTERN_DB.exists():
        return None
    try:
        with sqlite3.connect(
            f"file:{PATTERN_DB.as_posix()}?mode=ro", uri=True, timeout=30
        ) as connection:
            row = connection.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()
        return int(row[0]) if row and row[0] is not None else None
    except sqlite3.Error:
        return None


def route_similarity(left: Candidate, right: Candidate) -> float:
    left_edges, right_edges = set(left.edges), set(right.edges)
    left_nodes, right_nodes = set(left.nodes), set(right.nodes)
    edge_union = left_edges | right_edges
    node_union = left_nodes | right_nodes
    edge_overlap = len(left_edges & right_edges) / len(edge_union) if edge_union else 0.0
    node_overlap = len(left_nodes & right_nodes) / len(node_union) if node_union else 0.0
    return 0.68 * edge_overlap + 0.32 * node_overlap


def decode_candidate(prefix: str, candidate: Candidate, references: list[Candidate]) -> None:
    if not candidate.decoy and not candidate.evidence_texts:
        exact = next((reference for reference in references if reference.key == candidate.key), None)
        if exact is not None:
            candidate.evidence_texts = list(exact.evidence_texts)
            candidate.source_kinds = set(exact.source_kinds)

    suggestions: dict[str, tuple[float, str]] = {}

    def offer(label: str, confidence: float, evidence: str) -> None:
        label = label.strip() or "未解釈の経路"
        score = max(0.0, min(1.0, confidence / 100.0))
        previous = suggestions.get(label)
        if previous is None or score > previous[0]:
            suggestions[label] = (score, evidence)

    if not candidate.decoy:
        for text in candidate.evidence_texts[:12]:
            label, lexical = decode_suffix(prefix, text)
            offer(label, min(100.0, 72.0 + 0.28 * lexical), text)

    ranked_references = sorted(
        (
            (route_similarity(candidate, reference), reference)
            for reference in references
            if reference.key != candidate.key or candidate.decoy
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:24]
    for similarity, reference in ranked_references:
        if similarity <= 0.0:
            continue
        for text in reference.evidence_texts[:5]:
            label, lexical = decode_suffix(prefix, text)
            exact_prefix = strip_observer_annotation(text).startswith(prefix.strip())
            confidence = 76.0 * similarity + 0.24 * lexical + (8.0 if exact_prefix else 0.0)
            offer(label, min(96.0 if candidate.decoy else 100.0, confidence), text)

    if not suggestions:
        candidate.decoded_text = "未解釈の経路"
        candidate.decode_confidence = 0.0
        candidate.decode_evidence = []
        return

    ordered = sorted(suggestions.items(), key=lambda item: (-item[1][0], len(item[0]), item[0]))
    candidate.decoded_text = ordered[0][0]
    candidate.decode_confidence = round(ordered[0][1][0] * 100.0, 1)
    evidence: list[str] = []
    for _, (_, source_text) in ordered:
        source_text = strip_observer_annotation(source_text)
        if source_text and source_text not in evidence:
            evidence.append(source_text)
        if len(evidence) >= 3:
            break
    candidate.decode_evidence = evidence


def init_feedback() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB, timeout=30) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS route_feedback("
            "prefix TEXT NOT NULL, route_key TEXT NOT NULL, "
            "positive INTEGER NOT NULL DEFAULT 0, negative INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY(prefix, route_key))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS route_feedback_v3("
            "prefix TEXT NOT NULL, route_key TEXT NOT NULL, "
            "positive REAL NOT NULL DEFAULT 0, partial REAL NOT NULL DEFAULT 0, "
            "negative REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL, "
            "PRIMARY KEY(prefix, route_key))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS feedback_sessions("
            "session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "prefix_text TEXT NOT NULL, prefix_signature TEXT NOT NULL, "
            "latest_reflection_run INTEGER, before_json TEXT NOT NULL, "
            "after_json TEXT NOT NULL, summary_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS feedback_items("
            "session_id TEXT NOT NULL, route_key TEXT NOT NULL, "
            "decoded_text TEXT NOT NULL, grade TEXT NOT NULL, "
            "before_percent REAL NOT NULL, after_percent REAL NOT NULL, "
            "score_delta REAL NOT NULL, route_weight_before REAL NOT NULL, "
            "route_weight_after REAL NOT NULL, bridge_weight_before REAL NOT NULL, "
            "bridge_weight_after REAL NOT NULL, route_json TEXT NOT NULL, "
            "PRIMARY KEY(session_id, route_key))"
        )


def feedback_map(prefix: str) -> dict[str, tuple[float, float, float]]:
    init_feedback()
    totals: dict[str, list[float]] = {}
    with sqlite3.connect(FEEDBACK_DB, timeout=30) as connection:
        for route_key, positive, negative in connection.execute(
            "SELECT route_key, positive, negative FROM route_feedback WHERE prefix=?",
            (prefix,),
        ):
            totals[str(route_key)] = [float(positive), 0.0, float(negative)]
        for route_key, positive, partial, negative in connection.execute(
            "SELECT route_key, positive, partial, negative FROM route_feedback_v3 WHERE prefix=?",
            (prefix,),
        ):
            values = totals.setdefault(str(route_key), [0.0, 0.0, 0.0])
            values[0] += float(positive)
            values[1] += float(partial)
            values[2] += float(negative)
    return {key: (value[0], value[1], value[2]) for key, value in totals.items()}


def feedback_totals(prefix: str, route_key: str) -> tuple[float, float, float]:
    return feedback_map(prefix).get(route_key, (0.0, 0.0, 0.0))


def bias_from_totals(totals: tuple[float, float, float]) -> float:
    positive, partial, negative = totals
    total = positive + partial + negative
    if total <= 0:
        return 0.0
    return (positive + 0.35 * partial - negative) / (total + 3.0)


def fbias(prefix: str, route_key: str) -> float:
    return bias_from_totals(feedback_totals(prefix, route_key))


def decoys(real: list[Candidate], count: int, seed: str) -> list[Candidate]:
    if len(real) < 2 or count <= 0:
        return []
    rng = random.Random(int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big"))
    output: list[Candidate] = []
    used = {candidate.key for candidate in real}
    for _ in range(count * 40):
        if len(output) >= count:
            break
        left, right = rng.sample(real, 2)
        left_size = max(2, len(left.nodes) // 2)
        right_size = max(2, len(right.nodes) // 2)
        nodes = clean_nodes(left.nodes[:left_size] + right.nodes[-right_size:])
        if len(nodes) < 4:
            continue
        edges = [norm(a, b) for a, b in zip(nodes, nodes[1:])]
        route_key = key_for(edges)
        if route_key in used:
            continue
        used.add(route_key)
        output.append(
            Candidate(
                route_key,
                nodes,
                edges,
                evidence_texts=["偽経路（実経験の断片を組み替え）"],
                source_kinds={"decoy"},
                decoy=True,
            )
        )
    return output


def build_source_tree(brain: SphereBrain, sources: list[int]) -> tuple[np.ndarray, list[int]]:
    distance = np.full(brain.node_count, -1, dtype=int)
    parent = [-1] * brain.node_count
    queue: deque[int] = deque()
    for source in sources:
        if 0 <= source < brain.node_count and distance[source] < 0:
            distance[source] = 0
            queue.append(source)
    while queue:
        current = queue.popleft()
        for raw_neighbor in np.flatnonzero(brain.adjacency[current]):
            neighbor = int(raw_neighbor)
            if distance[neighbor] >= 0:
                continue
            distance[neighbor] = distance[current] + 1
            parent[neighbor] = current
            queue.append(neighbor)
    return distance, parent


def bridge_edges_for(
    brain: SphereBrain,
    candidate: Candidate,
    distance: np.ndarray,
    parent: list[int],
) -> list[tuple[int, int]]:
    targets = [node for node in candidate.nodes if 0 <= node < brain.node_count and distance[node] >= 0]
    if not targets:
        return []
    target = min(targets, key=lambda node: (int(distance[node]), -int(brain.node_usage[node]), node))
    path: list[tuple[int, int]] = []
    current = target
    while parent[current] >= 0:
        previous = parent[current]
        path.append(norm(previous, current))
        current = previous
    path.reverse()
    return path


def edge_average(brain: SphereBrain, edges: list[tuple[int, int]]) -> float:
    values = [
        float(brain.weights[a, b])
        for a, b in edges
        if 0 <= a < brain.node_count
        and 0 <= b < brain.node_count
        and bool(brain.adjacency[a, b])
    ]
    return float(np.mean(values)) if values else 0.0


def candidate_metrics(
    brain: SphereBrain,
    sources: list[int],
    candidate: Candidate,
    source_tree: tuple[np.ndarray, list[int]] | None = None,
) -> dict[str, Any]:
    distance, parent = source_tree or build_source_tree(brain, sources)
    valid_edges = [
        (a, b)
        for a, b in candidate.edges
        if 0 <= a < brain.node_count
        and 0 <= b < brain.node_count
        and bool(brain.adjacency[a, b])
    ]
    weights = [float(brain.weights[a, b]) for a, b in valid_edges]
    usages = [int(brain.usage[a, b]) for a, b in valid_edges]
    strength = float(np.mean(weights)) if weights else 0.0
    familiarity = float(np.mean([usage / (usage + 5.0) for usage in usages])) if usages else 0.0

    valid_nodes = [node for node in candidate.nodes if 0 <= node < brain.node_count]
    geometry = 0.0
    if valid_nodes and sources:
        node_array = np.array(sorted(set(valid_nodes)), dtype=int)
        proximity: list[float] = []
        for source in sources:
            distances = np.linalg.norm(brain.positions[node_array] - brain.positions[source], axis=1)
            proximity.append(1.0 / (1.0 + float(np.min(distances))))
        geometry = float(np.mean(proximity))

    bridge_edges = bridge_edges_for(brain, candidate, distance, parent)
    bridge_weight = edge_average(brain, bridge_edges)
    if bridge_edges:
        bridge_affinity = bridge_weight / (1.0 + 0.10 * max(0, len(bridge_edges) - 1))
    elif any(source in valid_nodes for source in sources):
        bridge_affinity = 1.0
    else:
        bridge_affinity = 0.0

    return {
        "strength": strength,
        "familiarity": familiarity,
        "geometry": geometry,
        "bridge_affinity": bridge_affinity,
        "bridge_weight": bridge_weight,
        "bridge_edges": bridge_edges,
        "valid_edges": valid_edges,
    }


def score_candidate(
    brain: SphereBrain,
    sources: list[int],
    prefix: str,
    candidate: Candidate,
    source_tree: tuple[np.ndarray, list[int]] | None = None,
    feedback_lookup: dict[str, tuple[float, float, float]] | None = None,
) -> float:
    metrics = candidate_metrics(brain, sources, candidate, source_tree)
    feedback = (
        feedback_lookup.get(candidate.key, (0.0, 0.0, 0.0))
        if feedback_lookup is not None
        else feedback_totals(prefix, candidate.key)
    )
    return (
        0.34 * metrics["strength"]
        + 0.24 * metrics["familiarity"]
        + 0.18 * metrics["bridge_affinity"]
        + 0.10 * metrics["geometry"]
        + 0.22 * bias_from_totals(feedback)
        - (0.055 if candidate.decoy else 0.0)
    )


def normalize_candidates(candidates: list[Candidate]) -> None:
    if not candidates:
        return
    peak = max(candidate.score for candidate in candidates)
    exponentials = [math.exp((candidate.score - peak) * 7.0) for candidate in candidates]
    total = sum(exponentials) or 1.0
    for candidate, value in zip(candidates, exponentials):
        candidate.percent = 100.0 * value / total
    candidates.sort(key=lambda candidate: (-candidate.percent, candidate.key))
    for rank, candidate in enumerate(candidates, 1):
        candidate.rank = rank


def prefix_signature(sources: list[int]) -> str:
    return sha256(",".join(map(str, sorted(sources))).encode("utf-8")).hexdigest()[:20]


def evaluate(text: str, count: int, decoy_count: int) -> dict[str, Any]:
    if not BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")
    brain = SphereBrain.load(BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = prefix_signature(sources)
    routes, counts = load_routes()
    if not routes:
        raise RuntimeError(
            "memory.dbとpattern_candidates.dbの両方を調べましたが候補経路がありません。"
            "通常入力またはReflectionを一度実行してください。"
        )

    source_tree = build_source_tree(brain, sources)
    feedback_lookup = feedback_map(prefix)
    for candidate in routes:
        candidate.score = score_candidate(
            brain, sources, prefix, candidate, source_tree, feedback_lookup
        )
    routes.sort(key=lambda candidate: (-candidate.score, candidate.key))

    real_count = max(1, count - decoy_count)
    selected = routes[:real_count]
    selected += decoys(routes[: min(80, len(routes))], decoy_count, text)
    for candidate in selected:
        decode_candidate(text, candidate, routes)
        candidate.score = score_candidate(
            brain, sources, prefix, candidate, source_tree, feedback_lookup
        )
    normalize_candidates(selected)

    return {
        "prefix": prefix,
        "sources": sources,
        "candidates": selected,
        "source_counts": counts,
        "latest_reflection_run": latest_reflection_run(),
    }


def serialize_candidates(candidates: list[Candidate]) -> str:
    payload = [
        {
            "key": candidate.key,
            "nodes": candidate.nodes,
            "edges": candidate.edges,
            "decoy": candidate.decoy,
        }
        for candidate in candidates
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def candidates_from_payload(payload: Any, brain: SphereBrain) -> list[Candidate]:
    if not isinstance(payload, list) or not (1 <= len(payload) <= 8):
        raise ValueError("候補経路の情報が不正です。もう一度評価してください。")
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("候補経路の形式が不正です。")
        nodes = clean_nodes([int(value) for value in raw.get("nodes", [])])
        if not nodes or len(nodes) > 160 or any(node < 0 or node >= brain.node_count for node in nodes):
            raise ValueError("候補経路の地点情報が不正です。")
        edges = [norm(int(pair[0]), int(pair[1])) for pair in raw.get("edges", []) if len(pair) == 2]
        if not edges or len(edges) > 160:
            raise ValueError("候補経路のエッジ情報が不正です。")
        route_key = str(raw.get("key", ""))
        if route_key != key_for(edges) or route_key in seen:
            raise ValueError("候補経路の識別情報が一致しません。もう一度評価してください。")
        seen.add(route_key)
        candidates.append(
            Candidate(
                route_key,
                nodes,
                edges,
                decoy=bool(raw.get("decoy", False)),
            )
        )
    return candidates


def snapshot_candidates(
    brain: SphereBrain,
    text: str,
    candidates: list[Candidate],
    references: list[Candidate],
) -> dict[str, dict[str, Any]]:
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = prefix_signature(sources)
    source_tree = build_source_tree(brain, sources)
    feedback_lookup = feedback_map(prefix)
    for candidate in candidates:
        decode_candidate(text, candidate, references)
        candidate.score = score_candidate(
            brain, sources, prefix, candidate, source_tree, feedback_lookup
        )
    normalize_candidates(candidates)
    output: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        metrics = candidate_metrics(brain, sources, candidate, source_tree)
        positive, partial, negative = feedback_lookup.get(
            candidate.key, (0.0, 0.0, 0.0)
        )
        output[candidate.key] = {
            "rank": candidate.rank,
            "percent": candidate.percent,
            "score": candidate.score,
            "route_weight": metrics["strength"],
            "bridge_weight": metrics["bridge_weight"],
            "bridge_edges": metrics["bridge_edges"],
            "feedback": {"positive": positive, "partial": partial, "negative": negative},
        }
    return output


def apply_edge_signals(
    brain: SphereBrain,
    signals: dict[tuple[int, int], float],
    usage_edges: set[tuple[int, int]],
    used_nodes: set[int],
) -> None:
    for (a, b), signal in signals.items():
        if not (0 <= a < brain.node_count and 0 <= b < brain.node_count and brain.adjacency[a, b]):
            continue
        current = float(brain.weights[a, b])
        if signal > 0:
            rate = min(0.18, 0.028 * signal)
            updated = current + rate * (1.0 - current)
        elif signal < 0:
            rate = min(0.18, 0.04 * abs(signal))
            updated = current * (1.0 - rate)
        else:
            continue
        updated = float(np.clip(updated, 0.03, 1.0))
        brain.weights[a, b] = brain.weights[b, a] = updated
        if (a, b) in usage_edges:
            brain.usage[a, b] += 1
            brain.usage[b, a] += 1
    for node in used_nodes:
        if 0 <= node < brain.node_count:
            brain.node_usage[node] += 1


def save_brain_atomic(brain: SphereBrain) -> None:
    temporary = BRAIN_FILE.with_name(BRAIN_FILE.name + ".route-choice.tmp")
    brain.save(temporary)
    temporary.replace(BRAIN_FILE)


def comparison_status(delta: float) -> str:
    if delta >= 0.5:
        return "強化"
    if delta <= -0.5:
        return "弱化"
    return "ほぼ不変"


def teach(text: str, payload: Any, grades: dict[str, str]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")
    original_json = BRAIN_FILE.read_text(encoding="utf-8")
    brain = SphereBrain.load(BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = prefix_signature(sources)
    candidates = candidates_from_payload(payload, brain)
    references, counts = load_routes()
    if not references:
        raise RuntimeError("翻訳・比較に使える過去経験経路がありません。")

    normalized_grades = {
        key: value
        for key, value in grades.items()
        if value in GRADE_META and any(candidate.key == key for candidate in candidates)
    }
    if not normalized_grades:
        raise ValueError("少なくとも1件に○・△・×を付けてください。")

    before = snapshot_candidates(brain, text, candidates, references)
    source_tree = build_source_tree(brain, sources)
    signals: dict[tuple[int, int], float] = defaultdict(float)
    usage_edges: set[tuple[int, int]] = set()
    used_nodes: set[int] = set()

    grade_counts = {"positive": 0, "partial": 0, "negative": 0}
    for candidate in candidates:
        grade = normalized_grades.get(candidate.key)
        candidate.grade = grade or ""
        if not grade:
            continue
        grade_counts[grade] += 1
        metrics = candidate_metrics(brain, sources, candidate, source_tree)
        bridge_edges = list(metrics["bridge_edges"])
        route_edges = list(metrics["valid_edges"])

        if grade == "positive":
            for edge in route_edges:
                signals[edge] += 1.0
                usage_edges.add(edge)
            for edge in bridge_edges:
                signals[edge] += 1.55
                usage_edges.add(edge)
            used_nodes.update(candidate.nodes)
            for edge in bridge_edges:
                used_nodes.update(edge)
        elif grade == "partial":
            for edge in route_edges:
                signals[edge] += 0.24
            for edge in bridge_edges:
                signals[edge] += 0.38
        else:
            # ×は経路そのものを消さず、現在の入口から向かう橋を主に弱める。
            for edge in route_edges:
                signals[edge] -= 0.06
            for edge in bridge_edges:
                signals[edge] -= 1.0

    apply_edge_signals(brain, signals, usage_edges, used_nodes)

    init_feedback()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        save_brain_atomic(brain)
        with sqlite3.connect(FEEDBACK_DB, timeout=30) as connection:
            for route_key, grade in normalized_grades.items():
                positive = 1.0 if grade == "positive" else 0.0
                partial = 1.0 if grade == "partial" else 0.0
                negative = 1.0 if grade == "negative" else 0.0
                connection.execute(
                    "INSERT INTO route_feedback_v3("
                    "prefix, route_key, positive, partial, negative, updated_at) "
                    "VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(prefix, route_key) DO UPDATE SET "
                    "positive=positive+excluded.positive, "
                    "partial=partial+excluded.partial, "
                    "negative=negative+excluded.negative, "
                    "updated_at=excluded.updated_at",
                    (prefix, route_key, positive, partial, negative, timestamp),
                )
    except Exception:
        BRAIN_FILE.write_text(original_json, encoding="utf-8")
        raise

    after = snapshot_candidates(brain, text, candidates, references)
    session_id = uuid.uuid4().hex[:16]
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        grade = normalized_grades.get(candidate.key)
        if not grade:
            continue
        before_item = before[candidate.key]
        after_item = after[candidate.key]
        delta = after_item["percent"] - before_item["percent"]
        items.append(
            {
                "key": candidate.key,
                "decoded_text": candidate.decoded_text,
                "grade": grade,
                "grade_symbol": GRADE_META[grade]["symbol"],
                "before_percent": before_item["percent"],
                "after_percent": after_item["percent"],
                "delta": delta,
                "status": comparison_status(delta),
                "rank_before": before_item["rank"],
                "rank_after": after_item["rank"],
                "route_weight_before": before_item["route_weight"],
                "route_weight_after": after_item["route_weight"],
                "bridge_weight_before": before_item["bridge_weight"],
                "bridge_weight_after": after_item["bridge_weight"],
            }
        )

    summary = {
        **grade_counts,
        "strengthened": sum(1 for item in items if item["status"] == "強化"),
        "weakened": sum(1 for item in items if item["status"] == "弱化"),
        "stable": sum(1 for item in items if item["status"] == "ほぼ不変"),
    }
    comparison = {
        "session_id": session_id,
        "latest_reflection_run": latest_reflection_run(),
        "summary": summary,
        "items": items,
    }

    with sqlite3.connect(FEEDBACK_DB, timeout=30) as connection:
        connection.execute(
            "INSERT INTO feedback_sessions("
            "session_id, created_at, prefix_text, prefix_signature, latest_reflection_run, "
            "before_json, after_json, summary_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                session_id,
                timestamp,
                text,
                prefix,
                comparison["latest_reflection_run"],
                json.dumps(before, ensure_ascii=False),
                json.dumps(after, ensure_ascii=False),
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        for item in items:
            candidate = next(value for value in candidates if value.key == item["key"])
            connection.execute(
                "INSERT INTO feedback_items("
                "session_id, route_key, decoded_text, grade, before_percent, after_percent, "
                "score_delta, route_weight_before, route_weight_after, bridge_weight_before, "
                "bridge_weight_after, route_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    item["key"],
                    item["decoded_text"],
                    item["grade"],
                    item["before_percent"],
                    item["after_percent"],
                    item["delta"],
                    item["route_weight_before"],
                    item["route_weight_after"],
                    item["bridge_weight_before"],
                    item["bridge_weight_after"],
                    json.dumps(
                        {"nodes": candidate.nodes, "edges": candidate.edges},
                        ensure_ascii=False,
                    ),
                ),
            )

    candidates.sort(key=lambda candidate: candidate.rank)
    result = {
        "prefix": prefix,
        "sources": sources,
        "candidates": candidates,
        "source_counts": counts,
        "latest_reflection_run": comparison["latest_reflection_run"],
    }
    message = (
        f"○ {grade_counts['positive']}件 / △ {grade_counts['partial']}件 / "
        f"× {grade_counts['negative']}件を、選択経験としてCoreへ返しました。"
    )
    return message, result, comparison


PAGE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Route Choice Learning Lab v0.3</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--panel2:#0b192a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--green:#8ce3a9;--yellow:#ffd36f;--red:#ff8585;--orange:#ed8447}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}.wrap{max-width:1240px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 180px 180px;gap:14px}.summary-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.summary{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:15px}.summary strong{display:block;font-size:28px;margin-top:6px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.13em}.muted{color:var(--muted)}input[type=text],input[type=number]{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:13px;font-size:16px}button{background:var(--orange);border:0;color:white;border-radius:10px;padding:13px 19px;font-weight:800;cursor:pointer}.message{color:var(--green)}.error{color:var(--red)}.candidate{background:var(--panel2);border:1px solid var(--line);border-radius:16px;padding:18px;margin:14px 0}.candidate-head{display:grid;grid-template-columns:1fr 150px auto;gap:16px;align-items:center}.decoded{font-size:26px;font-weight:850}.percent{font-size:32px;font-weight:900;text-align:right}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 10px;font-size:13px}.badge.real{color:var(--green)}.badge.decoy{color:var(--red)}.confidence{margin-top:5px;color:var(--muted)}.route{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#bfeeff;overflow-wrap:anywhere}.evidence{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}.evidence span{border:1px solid #31577e;border-radius:999px;padding:5px 9px;color:#b9d1ec;font-size:13px}.grades{display:flex;gap:9px;flex-wrap:wrap;margin-top:15px}.grade input{position:absolute;opacity:0;pointer-events:none}.grade span{display:inline-flex;align-items:center;gap:7px;border:1px solid #42668a;border-radius:11px;padding:10px 14px;cursor:pointer;font-weight:800}.grade input:checked+span{outline:2px solid currentColor;background:#162d48}.grade.positive{color:var(--green)}.grade.partial{color:var(--yellow)}.grade.negative{color:var(--red)}.grade.none{color:var(--muted)}.teach-bar{position:sticky;bottom:12px;background:#0b192aee;border:1px solid var(--line);border-radius:14px;padding:14px;display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:18px;backdrop-filter:blur(8px)}details{margin-top:10px}summary{cursor:pointer;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--line);padding:12px 9px}.delta-up{color:var(--green)}.delta-down{color:var(--red)}.delta-flat{color:var(--muted)}.note{border-left:3px solid var(--cyan);padding-left:13px}.source-line{font-size:13px;color:var(--muted)}
@media(max-width:900px){.summary-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:760px){.grid,.candidate-head{grid-template-columns:1fr}.percent{text-align:left}.summary-grid{grid-template-columns:repeat(2,1fr)}.teach-bar{position:static;display:block}.teach-bar button{margin-top:10px;width:100%}table{display:block;overflow-x:auto}}
</style>
</head>
<body><main class="wrap">
<div class="card"><div class="eyebrow">ROUTE CHOICE LEARNING LAB v0.3</div><h1>経路を日本語へ翻訳し、○・△・×を経験として返す</h1><p class="muted">Coreへ提示する候補は言葉ではなく経路です。日本語は人間が確認するためにObserverが後からデコードした表示なので、誤訳もそのまま△・×で教えられます。</p></div>
<div class="card"><form method="post"><input type="hidden" name="action" value="evaluate"><div class="grid"><div><div class="eyebrow">PARTIAL INPUT</div><h2>途中入力</h2><input type="text" name="text" value="{{text}}" required></div><div><div class="eyebrow">CANDIDATES</div><h2>候補数</h2><input name="count" type="number" min="3" max="8" value="{{count}}"></div><div><div class="eyebrow">DECOYS</div><h2>偽経路数</h2><input name="decoys" type="number" min="0" max="4" value="{{decoys}}"></div></div><p><button>経路候補を評価する</button></p></form></div>
{% if message %}<div class="card message">{{message}}</div>{% endif %}
{% if error %}<div class="card error">{{error}}</div>{% endif %}
{% if result %}
<form method="post" class="card"><input type="hidden" name="action" value="teach"><input type="hidden" name="text" value="{{text}}"><input type="hidden" name="count" value="{{count}}"><input type="hidden" name="decoys" value="{{decoys}}"><textarea name="payload" hidden>{{payload}}</textarea>
<div class="eyebrow">DECODED ROUTE CANDIDATES</div><h2>「{{text}}」の先として提示された経路</h2><p class="source-line">候補元：通常記憶 {{result.source_counts.memory}}本 / Reflection {{result.source_counts.reflection}}本{% if result.latest_reflection_run %}（最新Reflection #{{result.latest_reflection_run}}）{% endif %}</p>
{% for c in result.candidates %}
<section class="candidate"><div class="candidate-head"><div><div class="decoded">{{c.decoded_text}}</div><div class="confidence">日本語デコード信頼度 {{'%.1f'|format(c.decode_confidence)}}% ／ 順位 {{c.rank}}</div></div><div class="percent">{{'%.1f'|format(c.percent)}}%</div><div class="badge {% if c.decoy %}decoy{% else %}real{% endif %}">{% if c.decoy %}偽経路候補{% else %}過去経験経路{% endif %}</div></div>
{% if c.decode_evidence %}<div class="evidence"><span>翻訳根拠</span>{% for value in c.decode_evidence %}<span>{{value}}</span>{% endfor %}</div>{% endif %}
<details><summary>内部経路を確認</summary><p class="route">{% for e in c.edges %}{{e[0]}}→{{e[1]}}{% if not loop.last %} / {% endif %}{% endfor %}</p><p class="muted">候補ID {{c.key}}</p></details>
<div class="grades" role="group" aria-label="{{c.decoded_text}}の判定"><label class="grade positive"><input type="radio" name="grade_{{c.key}}" value="positive" {% if c.grade=='positive' %}checked{% endif %}><span>○ 正しい</span></label><label class="grade partial"><input type="radio" name="grade_{{c.key}}" value="partial" {% if c.grade=='partial' %}checked{% endif %}><span>△ 近い</span></label><label class="grade negative"><input type="radio" name="grade_{{c.key}}" value="negative" {% if c.grade=='negative' %}checked{% endif %}><span>× 違う</span></label><label class="grade none"><input type="radio" name="grade_{{c.key}}" value="" {% if not c.grade %}checked{% endif %}><span>— 未判定</span></label></div></section>
{% endfor %}
<div class="teach-bar"><div class="muted">複数候補をまとめて判定できます。未判定の候補は変更しません。</div><button>選んだ○・△・×で教育を確定</button></div></form>
{% endif %}
{% if comparison %}
<div class="card"><div class="eyebrow">FEEDBACK REFLECTION</div><h2>教育前後のReflection比較</h2><p class="note muted">これは保存済みCoreを教育直前・直後に読み取った即時Reflectionです。既存のReflection DBへ答えを書き込まず、教育によって経路の共鳴と重みがどう変わったかだけを比較します。</p>
<div class="summary-grid"><div class="summary"><span>○</span><strong>{{comparison.summary.positive}}</strong></div><div class="summary"><span>△</span><strong>{{comparison.summary.partial}}</strong></div><div class="summary"><span>×</span><strong>{{comparison.summary.negative}}</strong></div><div class="summary"><span>強化</span><strong>{{comparison.summary.strengthened}}</strong></div><div class="summary"><span>弱化</span><strong>{{comparison.summary.weakened}}</strong></div><div class="summary"><span>ほぼ不変</span><strong>{{comparison.summary.stable}}</strong></div></div>
<p class="muted">観測セッション {{comparison.session_id}}{% if comparison.latest_reflection_run %} ／ 基準となる最新Reflection #{{comparison.latest_reflection_run}}{% endif %}</p>
<table><thead><tr><th>デコード</th><th>判定</th><th>教育前</th><th>教育後</th><th>差</th><th>順位</th><th>経路重み</th><th>入口橋重み</th></tr></thead><tbody>{% for item in comparison["items"] %}<tr><td>{{item.decoded_text}}</td><td>{{item.grade_symbol}}</td><td>{{'%.1f'|format(item.before_percent)}}%</td><td>{{'%.1f'|format(item.after_percent)}}%</td><td class="{% if item.delta>0.05 %}delta-up{% elif item.delta<-0.05 %}delta-down{% else %}delta-flat{% endif %}">{{'%+.1f'|format(item.delta)}} pt<br><small>{{item.status}}</small></td><td>{{item.rank_before}} → {{item.rank_after}}</td><td>{{'%.3f'|format(item.route_weight_before)}} → {{'%.3f'|format(item.route_weight_after)}}</td><td>{{'%.3f'|format(item.bridge_weight_before)}} → {{'%.3f'|format(item.bridge_weight_after)}}</td></tr>{% endfor %}</tbody></table></div>
{% endif %}
</main></body></html>'''


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    text = "犬は"
    count = 5
    decoy_count = 2
    result: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    error = ""
    message = ""
    payload = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        action = request.form.get("action", "evaluate")
        try:
            count = max(3, min(8, int(request.form.get("count", "5"))))
            decoy_count = max(0, min(4, int(request.form.get("decoys", "2"))))
            decoy_count = min(decoy_count, count - 1)
            if action == "teach":
                raw_payload = json.loads(request.form.get("payload", "[]"))
                grades = {
                    str(item.get("key", "")): request.form.get(
                        f"grade_{item.get('key', '')}", ""
                    )
                    for item in raw_payload
                    if isinstance(item, dict)
                }
                message, result, comparison = teach(text, raw_payload, grades)
            else:
                result = evaluate(text, count, decoy_count)
            if result:
                payload = serialize_candidates(result["candidates"])
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        PAGE,
        text=text,
        count=count,
        decoys=decoy_count,
        result=result,
        comparison=comparison,
        error=error,
        message=message,
        payload=payload,
    )


def main() -> None:
    webbrowser.open("http://127.0.0.1:5077")
    serve(app, host="127.0.0.1", port=5077, threads=4)


if __name__ == "__main__":
    main()
