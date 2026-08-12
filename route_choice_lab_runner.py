from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3
from typing import Any

import numpy as np

import route_choice_lab as lab


MODE_COMBINED = "combined"
MODE_CORE_ONLY = "core_only"
MODE_FEEDBACK_ONLY = "feedback_only"
MODE_ORDER = (MODE_COMBINED, MODE_CORE_ONLY, MODE_FEEDBACK_ONLY)

ORIGINAL_TEACH = lab.teach


def structural_route_signature(
    candidate: lab.Candidate,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return the canonical identity of one traversed route.

    A route and the same route traversed in reverse are identical. Traversal
    order and repeated nodes/edges are otherwise preserved, so routes that only
    share the same topology are not accidentally collapsed.
    """

    nodes = tuple(int(node) for node in lab.clean_nodes(candidate.nodes))
    edges = tuple(lab.norm(int(a), int(b)) for a, b in candidate.edges)
    forward = (nodes, edges)
    reverse = (tuple(reversed(nodes)), tuple(reversed(edges)))
    return min(forward, reverse)


def route_fingerprint_from_signature(
    signature: tuple[tuple[int, ...], tuple[tuple[int, int], ...]],
) -> str:
    material = json.dumps(signature, ensure_ascii=True, separators=(",", ":"))
    return "R-" + sha256(material.encode("utf-8")).hexdigest()[:10].upper()


def route_fingerprint(candidate: lab.Candidate) -> str:
    return route_fingerprint_from_signature(structural_route_signature(candidate))


def _candidate_alias_keys(candidate: lab.Candidate) -> list[str]:
    aliases = getattr(candidate, "route_alias_keys", None)
    if aliases:
        return [str(value) for value in aliases]
    return [str(candidate.key)]


def _decorate_identity(candidate: lab.Candidate) -> None:
    candidate.route_fingerprint = route_fingerprint(candidate)
    aliases = sorted(set(_candidate_alias_keys(candidate) + [str(candidate.key)]))
    candidate.route_alias_keys = aliases
    candidate.route_alias_count = len(aliases)


def _merge_duplicate_metadata(primary: lab.Candidate, duplicate: lab.Candidate) -> None:
    for text in duplicate.evidence_texts:
        if text not in primary.evidence_texts:
            primary.evidence_texts.append(text)
    primary.source_kinds.update(duplicate.source_kinds)
    # If a generated decoy collides with a real route, the real route wins.
    primary.decoy = bool(primary.decoy and duplicate.decoy)
    aliases = set(_candidate_alias_keys(primary))
    aliases.update(_candidate_alias_keys(duplicate))
    aliases.update((str(primary.key), str(duplicate.key)))
    primary.route_alias_keys = sorted(aliases)
    _decorate_identity(primary)


def deduplicate_routes(
    candidates: list[lab.Candidate],
) -> tuple[list[lab.Candidate], int]:
    """Remove only canonical, structurally identical routes.

    Different internal routes remain separate even when both decode to the same
    Japanese label.
    """

    grouped: dict[
        tuple[tuple[int, ...], tuple[tuple[int, int], ...]],
        lab.Candidate,
    ] = {}
    order: list[
        tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
    ] = []
    removed = 0

    for candidate in candidates:
        signature = structural_route_signature(candidate)
        existing = grouped.get(signature)
        if existing is None:
            grouped[signature] = candidate
            order.append(signature)
            _decorate_identity(candidate)
            continue
        _merge_duplicate_metadata(existing, candidate)
        removed += 1

    return [grouped[signature] for signature in order], removed


def feedback_cycle_count(text: str) -> int:
    """Return the persisted number of completed teaching cycles for this text."""

    try:
        lab.init_feedback()
        with sqlite3.connect(lab.FEEDBACK_DB, timeout=30) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM feedback_sessions WHERE prefix_text=?",
                (str(text or "").strip(),),
            ).fetchone()
        return int(row[0]) if row else 0
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return 0


def _feedback_for_candidate(
    candidate: lab.Candidate,
    feedback_lookup: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float]:
    positive = partial = negative = 0.0
    for route_key in _candidate_alias_keys(candidate):
        values = feedback_lookup.get(route_key, (0.0, 0.0, 0.0))
        positive += float(values[0])
        partial += float(values[1])
        negative += float(values[2])
    return positive, partial, negative


def mode_scores_from_metrics(
    metrics: dict[str, Any],
    feedback: tuple[float, float, float],
    *,
    decoy: bool,
) -> dict[str, float]:
    """Return separated Core, external-feedback, and combined scores."""

    core_score = (
        0.34 * float(metrics["strength"])
        + 0.24 * float(metrics["familiarity"])
        + 0.18 * float(metrics["bridge_affinity"])
        + 0.10 * float(metrics["geometry"])
    )
    feedback_score = float(lab.bias_from_totals(feedback))
    combined_score = (
        core_score
        + 0.22 * feedback_score
        - (0.055 if decoy else 0.0)
    )
    return {
        MODE_COMBINED: combined_score,
        MODE_CORE_ONLY: core_score,
        MODE_FEEDBACK_ONLY: feedback_score,
    }


def _normalized_mode_stats(
    candidates: list[lab.Candidate],
    scores: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    if not candidates:
        return {}

    peak = max(scores[candidate.key] for candidate in candidates)
    exponentials = {
        candidate.key: float(np.exp((scores[candidate.key] - peak) * 7.0))
        for candidate in candidates
    }
    total = sum(exponentials.values()) or 1.0
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -(100.0 * exponentials[candidate.key] / total),
            route_fingerprint(candidate),
            candidate.key,
        ),
    )

    output: dict[str, dict[str, float | int]] = {}
    for rank, candidate in enumerate(ordered, 1):
        output[candidate.key] = {
            "score": float(scores[candidate.key]),
            "percent": 100.0 * exponentials[candidate.key] / total,
            "rank": rank,
        }
    return output


def decorate_candidate_evaluations(
    brain: lab.SphereBrain,
    sources: list[int],
    prefix: str,
    candidates: list[lab.Candidate],
    *,
    source_tree: tuple[np.ndarray, list[int]] | None = None,
    feedback_lookup: dict[str, tuple[float, float, float]] | None = None,
    primary_mode: str = MODE_COMBINED,
) -> None:
    """Attach three independent evaluation layers to every candidate."""

    if primary_mode not in MODE_ORDER:
        raise ValueError(f"unknown evaluation mode: {primary_mode}")

    source_tree = source_tree or lab.build_source_tree(brain, sources)
    feedback_lookup = feedback_lookup if feedback_lookup is not None else lab.feedback_map(prefix)
    score_maps: dict[str, dict[str, float]] = {
        mode: {} for mode in MODE_ORDER
    }

    for candidate in candidates:
        _decorate_identity(candidate)
        metrics = lab.candidate_metrics(brain, sources, candidate, source_tree)
        feedback = _feedback_for_candidate(candidate, feedback_lookup)
        scores = mode_scores_from_metrics(
            metrics,
            feedback,
            decoy=bool(candidate.decoy),
        )
        candidate.research_metrics = metrics
        candidate.feedback_totals = feedback
        for mode in MODE_ORDER:
            score_maps[mode][candidate.key] = scores[mode]

    mode_stats = {
        mode: _normalized_mode_stats(candidates, score_maps[mode])
        for mode in MODE_ORDER
    }

    for candidate in candidates:
        candidate.mode_evaluations = {
            mode: mode_stats[mode][candidate.key] for mode in MODE_ORDER
        }
        candidate.combined_score = float(
            candidate.mode_evaluations[MODE_COMBINED]["score"]
        )
        candidate.combined_percent = float(
            candidate.mode_evaluations[MODE_COMBINED]["percent"]
        )
        candidate.combined_rank = int(
            candidate.mode_evaluations[MODE_COMBINED]["rank"]
        )
        candidate.core_only_score = float(
            candidate.mode_evaluations[MODE_CORE_ONLY]["score"]
        )
        candidate.core_only_percent = float(
            candidate.mode_evaluations[MODE_CORE_ONLY]["percent"]
        )
        candidate.core_only_rank = int(
            candidate.mode_evaluations[MODE_CORE_ONLY]["rank"]
        )
        candidate.feedback_only_score = float(
            candidate.mode_evaluations[MODE_FEEDBACK_ONLY]["score"]
        )
        candidate.feedback_only_percent = float(
            candidate.mode_evaluations[MODE_FEEDBACK_ONLY]["percent"]
        )
        candidate.feedback_only_rank = int(
            candidate.mode_evaluations[MODE_FEEDBACK_ONLY]["rank"]
        )

        primary = candidate.mode_evaluations[primary_mode]
        candidate.score = float(primary["score"])
        candidate.percent = float(primary["percent"])
        candidate.rank = int(primary["rank"])

    candidates.sort(key=lambda candidate: (candidate.rank, candidate.key))


def evaluate_with_observability(
    text: str,
    count: int,
    decoy_count: int,
) -> dict[str, Any]:
    """Evaluate unique routes and expose Core/Feedback score separation."""

    if not lab.BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")

    brain = lab.SphereBrain.load(lab.BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = lab.prefix_signature(sources)
    loaded_routes, counts = lab.load_routes()
    if not loaded_routes:
        raise RuntimeError(
            "memory.dbとpattern_candidates.dbの両方を調べましたが候補経路がありません。"
            "通常入力またはReflectionを一度実行してください。"
        )

    routes, removed = deduplicate_routes(loaded_routes)
    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix)
    decorate_candidate_evaluations(
        brain,
        sources,
        prefix,
        routes,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
    )

    requested_count = max(1, int(count))
    requested_decoys = max(0, min(int(decoy_count), requested_count - 1))
    requested_real = max(1, requested_count - requested_decoys)

    selected: list[lab.Candidate] = []
    seen_structures: set[
        tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
    ] = set()

    def append_if_unique(candidate: lab.Candidate) -> bool:
        signature = structural_route_signature(candidate)
        if signature in seen_structures:
            return False
        seen_structures.add(signature)
        _decorate_identity(candidate)
        selected.append(candidate)
        return True

    for candidate in routes:
        if append_if_unique(candidate) and sum(
            not item.decoy for item in selected
        ) >= requested_real:
            break

    if requested_decoys:
        real_pool = routes[: min(120, len(routes))]
        for attempt in range(6):
            remaining = requested_decoys - sum(item.decoy for item in selected)
            if remaining <= 0:
                break
            generated = lab.decoys(
                real_pool,
                max(8, remaining * 6),
                f"{text}|structural-dedup-v04|{attempt}",
            )
            for candidate in generated:
                if append_if_unique(candidate):
                    remaining -= 1
                    if remaining <= 0:
                        break

    # If unique decoys are exhausted, preserve the requested card count with
    # additional real routes rather than repeating an identical route.
    if len(selected) < requested_count:
        for candidate in routes:
            append_if_unique(candidate)
            if len(selected) >= requested_count:
                break

    selected = selected[:requested_count]
    for candidate in selected:
        lab.decode_candidate(text, candidate, routes)
    decorate_candidate_evaluations(
        brain,
        sources,
        prefix,
        selected,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
    )

    return {
        "prefix": prefix,
        "sources": sources,
        "candidates": selected,
        "source_counts": counts,
        "latest_reflection_run": lab.latest_reflection_run(),
        "dedup_removed": removed,
    }


def snapshot_candidates_with_modes(
    brain: lab.SphereBrain,
    text: str,
    candidates: list[lab.Candidate],
    references: list[lab.Candidate],
) -> dict[str, dict[str, Any]]:
    """Snapshot the same candidates through all three evaluation layers."""

    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = lab.prefix_signature(sources)
    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix)
    unique_references, _ = deduplicate_routes(references)
    reference_by_signature = {
        structural_route_signature(reference): reference
        for reference in unique_references
    }

    for candidate in candidates:
        matching_reference = reference_by_signature.get(
            structural_route_signature(candidate)
        )
        if matching_reference is not None:
            candidate.route_alias_keys = list(
                getattr(
                    matching_reference,
                    "route_alias_keys",
                    [matching_reference.key],
                )
            )
        _decorate_identity(candidate)
        lab.decode_candidate(text, candidate, unique_references)

    decorate_candidate_evaluations(
        brain,
        sources,
        prefix,
        candidates,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
    )

    output: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        metrics = candidate.research_metrics
        positive, partial, negative = candidate.feedback_totals
        output[candidate.key] = {
            "rank": candidate.combined_rank,
            "percent": candidate.combined_percent,
            "score": candidate.combined_score,
            "route_weight": float(metrics["strength"]),
            "bridge_weight": float(metrics["bridge_weight"]),
            "bridge_edges": list(metrics["bridge_edges"]),
            "feedback": {
                "positive": positive,
                "partial": partial,
                "negative": negative,
            },
            "route_fingerprint": candidate.route_fingerprint,
            "route_alias_keys": list(candidate.route_alias_keys),
            "modes": {
                mode: {
                    "rank": int(candidate.mode_evaluations[mode]["rank"]),
                    "percent": float(candidate.mode_evaluations[mode]["percent"]),
                    "score": float(candidate.mode_evaluations[mode]["score"]),
                }
                for mode in MODE_ORDER
            },
        }
    return output


def _file_sha256(path: Any) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _expected_teaching_signals(
    brain: lab.SphereBrain,
    text: str,
    payload: Any,
    grades: dict[str, str],
) -> tuple[
    dict[tuple[int, int], float],
    set[tuple[int, int]],
    set[int],
]:
    sources = [int(node) for node in brain.text_to_sources(text)]
    candidates = lab.candidates_from_payload(payload, brain)
    source_tree = lab.build_source_tree(brain, sources)
    normalized_grades = {
        key: value
        for key, value in grades.items()
        if value in lab.GRADE_META
        and any(candidate.key == key for candidate in candidates)
    }

    signals: dict[tuple[int, int], float] = defaultdict(float)
    usage_edges: set[tuple[int, int]] = set()
    used_nodes: set[int] = set()

    for candidate in candidates:
        grade = normalized_grades.get(candidate.key)
        if not grade:
            continue
        metrics = lab.candidate_metrics(brain, sources, candidate, source_tree)
        route_edges = list(metrics["valid_edges"])
        bridge_edges = list(metrics["bridge_edges"])

        if grade == "positive":
            for edge in route_edges:
                signals[lab.norm(*edge)] += 1.0
                usage_edges.add(lab.norm(*edge))
            for edge in bridge_edges:
                signals[lab.norm(*edge)] += 1.55
                usage_edges.add(lab.norm(*edge))
            used_nodes.update(int(node) for node in candidate.nodes)
            for edge in bridge_edges:
                used_nodes.update(int(node) for node in edge)
        elif grade == "partial":
            for edge in route_edges:
                signals[lab.norm(*edge)] += 0.24
            for edge in bridge_edges:
                signals[lab.norm(*edge)] += 0.38
        else:
            for edge in route_edges:
                signals[lab.norm(*edge)] -= 0.06
            for edge in bridge_edges:
                signals[lab.norm(*edge)] -= 1.0

    signals = {
        edge: float(signal)
        for edge, signal in signals.items()
        if abs(float(signal)) > 1e-15
    }
    return signals, usage_edges, used_nodes


def build_core_audit(
    before_brain: Any,
    after_brain: Any,
    *,
    hash_before: str,
    hash_after: str,
    expected_signals: dict[tuple[int, int], float],
    expected_usage_edges: set[tuple[int, int]],
    expected_used_nodes: set[int],
) -> dict[str, Any]:
    """Measure exactly what changed inside Core during one teaching cycle."""

    if int(before_brain.node_count) != int(after_brain.node_count):
        raise ValueError("Core node_count changed during teaching.")

    node_count = int(before_brain.node_count)
    upper = np.triu(np.ones((node_count, node_count), dtype=bool), k=1)
    adjacency_before = np.asarray(before_brain.adjacency, dtype=bool)
    adjacency_after = np.asarray(after_brain.adjacency, dtype=bool)
    edge_mask = upper & (adjacency_before | adjacency_after)

    weight_before = np.asarray(before_brain.weights, dtype=float)
    weight_after = np.asarray(after_brain.weights, dtype=float)
    weight_delta = weight_after - weight_before
    changed_mask = edge_mask & (np.abs(weight_delta) > 1e-12)
    strengthened_mask = changed_mask & (weight_delta > 0)
    weakened_mask = changed_mask & (weight_delta < 0)

    changed_pairs = {
        (int(a), int(b))
        for a, b in zip(*np.where(changed_mask))
    }
    expected_pairs = {
        lab.norm(int(a), int(b))
        for (a, b), signal in expected_signals.items()
        if abs(float(signal)) > 1e-15
    }
    unexpected_pairs = changed_pairs - expected_pairs
    expected_changed = changed_pairs & expected_pairs
    expected_unchanged = expected_pairs - changed_pairs

    usage_before = np.asarray(before_brain.usage, dtype=int)
    usage_after = np.asarray(after_brain.usage, dtype=int)
    usage_delta = usage_after - usage_before
    usage_changed_mask = upper & (usage_delta != 0)
    usage_changed_pairs = {
        (int(a), int(b))
        for a, b in zip(*np.where(usage_changed_mask))
    }
    normalized_expected_usage_edges = {
        lab.norm(int(a), int(b))
        for a, b in expected_usage_edges
    }
    unexpected_usage_pairs = (
        usage_changed_pairs - normalized_expected_usage_edges
    )
    expected_usage_changed = (
        usage_changed_pairs & normalized_expected_usage_edges
    )

    node_usage_before = np.asarray(before_brain.node_usage, dtype=int)
    node_usage_after = np.asarray(after_brain.node_usage, dtype=int)
    node_usage_delta = node_usage_after - node_usage_before
    changed_nodes = {
        int(node)
        for node in np.flatnonzero(node_usage_delta != 0)
    }
    normalized_expected_nodes = {
        int(node)
        for node in expected_used_nodes
        if 0 <= int(node) < node_count
    }
    unexpected_changed_nodes = changed_nodes - normalized_expected_nodes
    expected_nodes_changed = changed_nodes & normalized_expected_nodes

    adjacency_delta = upper & (adjacency_before != adjacency_after)
    changed_values = np.abs(weight_delta[changed_mask])
    total_abs_delta = float(changed_values.sum()) if changed_values.size else 0.0
    max_abs_delta = float(changed_values.max()) if changed_values.size else 0.0

    samples: list[dict[str, Any]] = []
    for a, b in sorted(changed_pairs)[:20]:
        samples.append(
            {
                "edge": [a, b],
                "before": float(weight_before[a, b]),
                "after": float(weight_after[a, b]),
                "delta": float(weight_delta[a, b]),
                "expected_signal": float(expected_signals.get((a, b), 0.0)),
            }
        )

    core_changed = bool(
        changed_pairs
        or usage_changed_pairs
        or changed_nodes
        or np.any(adjacency_delta)
    )
    warnings: list[str] = []
    if expected_pairs and not changed_pairs:
        warnings.append("教育信号はありましたが、Core重みの変化を検出できませんでした。")
    if unexpected_pairs:
        warnings.append("教育対象外の重みエッジ変化を検出しました。")
    if unexpected_usage_pairs:
        warnings.append("教育対象外のusageエッジ変化を検出しました。")
    if unexpected_changed_nodes:
        warnings.append("教育対象外のnode_usage変化を検出しました。")
    if np.any(adjacency_delta):
        warnings.append("教育中に隣接構造そのものが変化しました。")
    if hash_before == hash_after and core_changed:
        warnings.append("内部配列は変化しましたが、brain.jsonハッシュが変わっていません。")
    if hash_before and hash_after and hash_before != hash_after and not core_changed:
        warnings.append("brain.jsonは書き換わりましたが、Core内部配列の変化はありません。")

    audit_ok = bool(
        core_changed
        and not unexpected_pairs
        and not unexpected_usage_pairs
        and not unexpected_changed_nodes
        and not np.any(adjacency_delta)
    )
    return {
        "core_changed": core_changed,
        "audit_ok": audit_ok,
        "hash_before": hash_before,
        "hash_after": hash_after,
        "hash_before_short": hash_before[:12] if hash_before else "(取得失敗)",
        "hash_after_short": hash_after[:12] if hash_after else "(取得失敗)",
        "hash_changed": bool(hash_before and hash_after and hash_before != hash_after),
        "changed_edges": len(changed_pairs),
        "strengthened_edges": int(np.count_nonzero(strengthened_mask)),
        "weakened_edges": int(np.count_nonzero(weakened_mask)),
        "total_abs_weight_delta": total_abs_delta,
        "max_abs_weight_delta": max_abs_delta,
        "expected_signal_edges": len(expected_pairs),
        "expected_signal_edges_changed": len(expected_changed),
        "expected_signal_edges_unchanged": len(expected_unchanged),
        "unexpected_edges_changed": len(unexpected_pairs),
        "usage_edges_expected": len(normalized_expected_usage_edges),
        "usage_edges_expected_changed": len(expected_usage_changed),
        "usage_edges_changed": len(usage_changed_pairs),
        "unexpected_usage_edges_changed": len(unexpected_usage_pairs),
        "usage_total_delta": int(usage_delta[usage_changed_mask].sum())
        if np.any(usage_changed_mask)
        else 0,
        "expected_used_nodes": len(normalized_expected_nodes),
        "expected_used_nodes_changed": len(expected_nodes_changed),
        "node_usage_changed": len(changed_nodes),
        "unexpected_node_usage_changed": len(unexpected_changed_nodes),
        "node_usage_total_delta": int(node_usage_delta.sum()),
        "adjacency_changes": int(np.count_nonzero(adjacency_delta)),
        "changed_edge_samples": samples,
        "warnings": warnings,
        "status_label": "正常" if audit_ok else "要確認",
    }


def _failed_core_audit(
    hash_before: str,
    hash_after: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "core_changed": False,
        "audit_ok": False,
        "hash_before": hash_before,
        "hash_after": hash_after,
        "hash_before_short": hash_before[:12] if hash_before else "(取得失敗)",
        "hash_after_short": hash_after[:12] if hash_after else "(取得失敗)",
        "hash_changed": bool(hash_before and hash_after and hash_before != hash_after),
        "changed_edges": 0,
        "strengthened_edges": 0,
        "weakened_edges": 0,
        "total_abs_weight_delta": 0.0,
        "max_abs_weight_delta": 0.0,
        "expected_signal_edges": 0,
        "expected_signal_edges_changed": 0,
        "expected_signal_edges_unchanged": 0,
        "unexpected_edges_changed": 0,
        "usage_edges_expected": 0,
        "usage_edges_expected_changed": 0,
        "usage_edges_changed": 0,
        "unexpected_usage_edges_changed": 0,
        "usage_total_delta": 0,
        "expected_used_nodes": 0,
        "expected_used_nodes_changed": 0,
        "node_usage_changed": 0,
        "unexpected_node_usage_changed": 0,
        "node_usage_total_delta": 0,
        "adjacency_changes": 0,
        "changed_edge_samples": [],
        "warnings": [f"教育は完了しましたが、Core監査に失敗しました: {error}"],
        "status_label": "監査失敗",
    }


def _persist_core_audit(session_id: str, audit: dict[str, Any]) -> None:
    lab.init_feedback()
    with sqlite3.connect(lab.FEEDBACK_DB, timeout=30) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS feedback_core_audits("
            "session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "brain_hash_before TEXT NOT NULL, brain_hash_after TEXT NOT NULL, "
            "changed_edges INTEGER NOT NULL, strengthened_edges INTEGER NOT NULL, "
            "weakened_edges INTEGER NOT NULL, unexpected_edges_changed INTEGER NOT NULL, "
            "audit_json TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO feedback_core_audits("
            "session_id, created_at, brain_hash_before, brain_hash_after, "
            "changed_edges, strengthened_edges, weakened_edges, "
            "unexpected_edges_changed, audit_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                datetime.now(timezone.utc).isoformat(),
                audit["hash_before"],
                audit["hash_after"],
                audit["changed_edges"],
                audit["strengthened_edges"],
                audit["weakened_edges"],
                audit["unexpected_edges_changed"],
                json.dumps(audit, ensure_ascii=False),
            ),
        )


def _augment_comparison_from_saved_snapshots(
    comparison: dict[str, Any],
) -> None:
    for item in comparison.get("items", []):
        route_key = str(item.get("key", ""))
        item["route_fingerprint"] = route_key[:10].upper()
        for mode in MODE_ORDER:
            item[f"{mode}_before_percent"] = float(
                item.get("before_percent", 0.0)
            )
            item[f"{mode}_after_percent"] = float(
                item.get("after_percent", 0.0)
            )
            item[f"{mode}_delta"] = (
                item[f"{mode}_after_percent"]
                - item[f"{mode}_before_percent"]
            )
            item[f"{mode}_rank_before"] = int(
                item.get("rank_before", 0)
            )
            item[f"{mode}_rank_after"] = int(
                item.get("rank_after", 0)
            )

    session_id = str(comparison.get("session_id", ""))
    if not session_id:
        return

    try:
        with sqlite3.connect(lab.FEEDBACK_DB, timeout=30) as connection:
            row = connection.execute(
                "SELECT before_json, after_json FROM feedback_sessions "
                "WHERE session_id=?",
                (session_id,),
            ).fetchone()
    except sqlite3.Error:
        return
    if not row:
        return

    try:
        before = json.loads(row[0])
        after = json.loads(row[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return

    for item in comparison.get("items", []):
        route_key = str(item.get("key", ""))
        before_item = before.get(route_key, {})
        after_item = after.get(route_key, {})
        before_modes = before_item.get("modes", {})
        after_modes = after_item.get("modes", {})
        item["route_fingerprint"] = (
            before_item.get("route_fingerprint")
            or after_item.get("route_fingerprint")
            or route_key[:10].upper()
        )
        for mode in MODE_ORDER:
            before_mode = before_modes.get(mode, {})
            after_mode = after_modes.get(mode, {})
            item[f"{mode}_before_percent"] = float(
                before_mode.get("percent", item.get("before_percent", 0.0))
            )
            item[f"{mode}_after_percent"] = float(
                after_mode.get("percent", item.get("after_percent", 0.0))
            )
            item[f"{mode}_delta"] = (
                item[f"{mode}_after_percent"]
                - item[f"{mode}_before_percent"]
            )
            item[f"{mode}_rank_before"] = int(
                before_mode.get("rank", item.get("rank_before", 0))
            )
            item[f"{mode}_rank_after"] = int(
                after_mode.get("rank", item.get("rank_after", 0))
            )


def teach_with_core_audit(
    text: str,
    payload: Any,
    grades: dict[str, str],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run the original teaching logic and independently audit Core changes."""

    if not lab.BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")
    before_brain = lab.SphereBrain.load(lab.BRAIN_FILE)
    hash_before = _file_sha256(lab.BRAIN_FILE)
    expected_signals, expected_usage_edges, expected_used_nodes = (
        _expected_teaching_signals(before_brain, text, payload, grades)
    )

    message, result, comparison = ORIGINAL_TEACH(text, payload, grades)

    hash_after = _file_sha256(lab.BRAIN_FILE)
    try:
        after_brain = lab.SphereBrain.load(lab.BRAIN_FILE)
        audit = build_core_audit(
            before_brain,
            after_brain,
            hash_before=hash_before,
            hash_after=hash_after,
            expected_signals=expected_signals,
            expected_usage_edges=expected_usage_edges,
            expected_used_nodes=expected_used_nodes,
        )
    except Exception as exc:  # Education already succeeded; never invite a retry.
        audit = _failed_core_audit(hash_before, hash_after, exc)

    _augment_comparison_from_saved_snapshots(comparison)
    comparison["core_audit"] = audit
    try:
        _persist_core_audit(str(comparison["session_id"]), audit)
    except (OSError, sqlite3.Error) as exc:
        audit["audit_ok"] = False
        audit["status_label"] = "要確認"
        audit["warnings"].append(f"Core監査ログの保存に失敗しました: {exc}")

    for candidate in result.get("candidates", []):
        _decorate_identity(candidate)

    message += (
        f" Core監査: 重み変更 {audit['changed_edges']}エッジ、"
        f"想定外 {audit['unexpected_edges_changed']}エッジ、"
        f"判定 {audit['status_label']}。"
    )
    return message, result, comparison


def _replace_once(page: str, old: str, new: str, label: str) -> str:
    if old not in page:
        raise RuntimeError(f"Route Choice UI patch point not found: {label}")
    return page.replace(old, new, 1)


def install_research_ui() -> None:
    """Install v0.4 observability and Work-friendly controls."""

    page = lab.PAGE
    page = page.replace(
        "<title>Route Choice Learning Lab v0.3</title>",
        "<title>Route Choice Learning Lab v0.4</title>",
        1,
    )
    page = page.replace(
        "ROUTE CHOICE LEARNING LAB v0.3",
        "ROUTE CHOICE LEARNING LAB v0.4",
        1,
    )
    page = page.replace(
        "Coreへ提示する候補は言葉ではなく経路です。日本語は人間が確認するためにObserverが後からデコードした表示なので、誤訳もそのまま△・×で教えられます。",
        "Coreへ提示する候補は言葉ではなく経路です。完全同一経路だけを除外し、"
        "異なる経路が同じ日本語へデコードされた場合は別候補として残します。"
        "総合・Core-only・Feedback-onlyを同時に観測し、教育直後はCore本体の変更も監査します。",
        1,
    )
    page = page.replace(
        ".source-line{font-size:13px;color:var(--muted)}",
        ".source-line{font-size:13px;color:var(--muted)}"
        ".fingerprint{display:inline-block;margin-left:9px;border:1px solid #4b759d;"
        "border-radius:999px;padding:4px 9px;color:var(--cyan);font:700 12px "
        "ui-monospace,SFMono-Regular,Consolas,monospace;vertical-align:middle}"
        ".mode-grid{display:grid;grid-template-columns:repeat(3,minmax(105px,1fr));"
        "gap:7px;min-width:360px}.mode-cell{background:#10223a;border:1px solid "
        "#284a70;border-radius:10px;padding:8px 10px}.mode-cell span{display:block;"
        "font-size:11px;color:var(--muted)}.mode-cell strong{font-size:20px}"
        ".mode-cell small{display:block;color:var(--muted)}"
        ".audit-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;"
        "margin-top:14px}.audit-box{background:var(--panel2);border:1px solid "
        "var(--line);border-radius:12px;padding:13px}.audit-box strong{display:block;"
        "font-size:22px;margin-top:4px}.audit-ok{color:var(--green)}"
        ".audit-warn{color:var(--yellow)}.audit-errors{color:var(--red);margin-top:12px}"
        ".mode-delta{white-space:nowrap}",
        1,
    )
    page = page.replace(
        ".candidate-head{display:grid;grid-template-columns:1fr 150px auto;",
        ".candidate-head{display:grid;grid-template-columns:minmax(240px,1fr) "
        "minmax(330px,520px) auto;",
        1,
    )
    page = page.replace(
        "@media(max-width:900px){.summary-grid{grid-template-columns:repeat(3,1fr)}}",
        "@media(max-width:1100px){.candidate-head{grid-template-columns:1fr}"
        ".mode-grid{min-width:0}.audit-grid{grid-template-columns:repeat(2,1fr)}}"
        "@media(max-width:900px){.summary-grid{grid-template-columns:repeat(3,1fr)}}",
        1,
    )
    page = page.replace(
        "@media(max-width:760px){.grid,.candidate-head{grid-template-columns:1fr}",
        "@media(max-width:760px){.grid,.candidate-head{grid-template-columns:1fr}"
        ".mode-grid,.audit-grid{grid-template-columns:1fr;min-width:0}",
        1,
    )
    page = _replace_once(
        page,
        "<p><button>経路候補を評価する</button></p>",
        '<p><button id="evaluate-button">経路候補を評価する</button></p>',
        "evaluate button",
    )
    page = _replace_once(
        page,
        "</form></div>\n{% if message %}",
        "</form></div>\n"
        '<div class="card" id="education-progress">'
        '<div class="eyebrow">EDUCATION PROGRESS</div>'
        '<h2>「{{text}}」の累積教育回数：'
        '<strong>{{feedback_cycle_count(text)}}</strong>回</h2>'
        '<p class="muted">教育を確定し、feedback_sessionsへ正常保存された時だけ1回増えます。'
        "一時停止や再読込の後も、この数字を基準に再開できます。</p>"
        "</div>\n{% if message %}",
        "education progress",
    )
    page = page.replace(
        "{% if result %}\n<form method=\"post\" class=\"card\">",
        "{% if result %}\n"
        '<div class="card"><div class="eyebrow">EVALUATION LAYERS</div>'
        "<h2>同じ候補を3つの層で比較</h2>"
        '<p class="muted"><strong>総合</strong>＝Core＋外部Feedback＋現行の偽経路補正、'
        "<strong>Core-only</strong>＝経路重み・使用経験・入口橋・幾何だけ、"
        "<strong>Feedback-only</strong>＝○・△・×の外部教師履歴だけ。"
        "候補カードの3数値は同じ候補集合内で正規化しています。</p></div>\n"
        '<form method="post" class="card">',
        1,
    )
    page = page.replace(
        '<p class="source-line">候補元：通常記憶 {{result.source_counts.memory}}本 / Reflection {{result.source_counts.reflection}}本{% if result.latest_reflection_run %}（最新Reflection #{{result.latest_reflection_run}}）{% endif %}</p>',
        '<p class="source-line">候補元：通常記憶 {{result.source_counts.memory}}本 / '
        "Reflection {{result.source_counts.reflection}}本"
        "{% if result.latest_reflection_run %}（最新Reflection "
        "#{{result.latest_reflection_run}}）{% endif %}"
        "{% if result.dedup_removed %} ／ 完全同一経路 "
        "{{result.dedup_removed}}件を除外{% endif %}</p>",
        1,
    )
    page = _replace_once(
        page,
        '<div class="decoded">{{c.decoded_text}}</div>',
        '<div class="decoded">{{c.decoded_text}}'
        '<span class="fingerprint">{{c.route_fingerprint}}</span></div>',
        "candidate fingerprint",
    )
    page = _replace_once(
        page,
        '<div class="percent">{{\'%.1f\'|format(c.percent)}}%</div>',
        '<div class="mode-grid">'
        '<div class="mode-cell"><span>総合</span><strong>'
        "{{'%.1f'|format(c.combined_percent)}}%</strong><small>順位 "
        "{{c.combined_rank}}</small></div>"
        '<div class="mode-cell"><span>Core-only</span><strong>'
        "{{'%.1f'|format(c.core_only_percent)}}%</strong><small>順位 "
        "{{c.core_only_rank}}</small></div>"
        '<div class="mode-cell"><span>Feedback-only</span><strong>'
        "{{'%.1f'|format(c.feedback_only_percent)}}%</strong><small>順位 "
        "{{c.feedback_only_rank}}</small></div></div>",
        "mode percentages",
    )
    page = _replace_once(
        page,
        '<p class="muted">候補ID {{c.key}}</p>',
        '<p class="muted">経路指紋 {{c.route_fingerprint}} ／ 候補ID {{c.key}}'
        "{% if c.route_alias_count > 1 %} ／ 統合した同一構造ID "
        "{{c.route_alias_count}}件{% endif %}</p>",
        "route details fingerprint",
    )
    page = _replace_once(
        page,
        "<button>選んだ○・△・×で教育を確定</button>",
        '<button id="confirm-teaching-button">選んだ○・△・×で教育を確定</button>',
        "confirm teaching button",
    )

    old_table = (
        '<table><thead><tr><th>デコード</th><th>判定</th><th>教育前</th>'
        '<th>教育後</th><th>差</th><th>順位</th><th>経路重み</th>'
        '<th>入口橋重み</th></tr></thead><tbody>{% for item in '
        'comparison["items"] %}<tr><td>{{item.decoded_text}}</td>'
        '<td>{{item.grade_symbol}}</td><td>{{\'%.1f\'|format(item.before_percent)}}%'
        '</td><td>{{\'%.1f\'|format(item.after_percent)}}%</td><td class="'
        '{% if item.delta>0.05 %}delta-up{% elif item.delta<-0.05 %}'
        'delta-down{% else %}delta-flat{% endif %}">{{\'%+.1f\'|format(item.delta)}} '
        'pt<br><small>{{item.status}}</small></td><td>{{item.rank_before}} → '
        '{{item.rank_after}}</td><td>{{\'%.3f\'|format(item.route_weight_before)}} '
        '→ {{\'%.3f\'|format(item.route_weight_after)}}</td><td>'
        "{{'%.3f'|format(item.bridge_weight_before)}} → "
        "{{'%.3f'|format(item.bridge_weight_after)}}</td></tr>{% endfor %}"
        "</tbody></table></div>"
    )
    new_table = (
        '<table><thead><tr><th>デコード／経路指紋</th><th>判定</th>'
        '<th>総合</th><th>Core-only</th><th>Feedback-only</th>'
        '<th>経路重み</th><th>入口橋重み</th></tr></thead><tbody>'
        '{% for item in comparison["items"] %}<tr><td>{{item.decoded_text}}'
        '<br><span class="fingerprint">{{item.route_fingerprint}}</span></td>'
        '<td>{{item.grade_symbol}}</td>'
        '<td class="mode-delta">{{\'%.1f\'|format(item.combined_before_percent)}}'
        '% → {{\'%.1f\'|format(item.combined_after_percent)}}%<br><small>'
        "{{'%+.1f'|format(item.combined_delta)}} pt ／ "
        "{{item.combined_rank_before}}→{{item.combined_rank_after}}</small></td>"
        '<td class="mode-delta">{{\'%.1f\'|format(item.core_only_before_percent)}}'
        '% → {{\'%.1f\'|format(item.core_only_after_percent)}}%<br><small>'
        "{{'%+.1f'|format(item.core_only_delta)}} pt ／ "
        "{{item.core_only_rank_before}}→{{item.core_only_rank_after}}</small></td>"
        '<td class="mode-delta">{{\'%.1f\'|format(item.feedback_only_before_percent)}}'
        '% → {{\'%.1f\'|format(item.feedback_only_after_percent)}}%<br><small>'
        "{{'%+.1f'|format(item.feedback_only_delta)}} pt ／ "
        "{{item.feedback_only_rank_before}}→{{item.feedback_only_rank_after}}</small></td>"
        "<td>{{'%.6f'|format(item.route_weight_before)}} → "
        "{{'%.6f'|format(item.route_weight_after)}}</td>"
        "<td>{{'%.6f'|format(item.bridge_weight_before)}} → "
        "{{'%.6f'|format(item.bridge_weight_after)}}</td></tr>{% endfor %}"
        "</tbody></table>"
        "{% if comparison.core_audit %}"
        '<div class="eyebrow" style="margin-top:22px">CORE CHANGE AUDIT</div>'
        "<h2>教育前後のCore変更監査 "
        '<span class="{% if comparison.core_audit.audit_ok %}audit-ok'
        '{% else %}audit-warn{% endif %}">{{comparison.core_audit.status_label}}'
        "</span></h2>"
        '<div class="audit-grid">'
        '<div class="audit-box"><span>brain.json hash</span><strong>'
        "{{comparison.core_audit.hash_before_short}} → "
        "{{comparison.core_audit.hash_after_short}}</strong></div>"
        '<div class="audit-box"><span>変更エッジ</span><strong>'
        "{{comparison.core_audit.changed_edges}}</strong><small>強化 "
        "{{comparison.core_audit.strengthened_edges}} ／ 弱化 "
        "{{comparison.core_audit.weakened_edges}}</small></div>"
        '<div class="audit-box"><span>教育信号の反映</span><strong>'
        "{{comparison.core_audit.expected_signal_edges_changed}} / "
        "{{comparison.core_audit.expected_signal_edges}}</strong><small>未変化 "
        "{{comparison.core_audit.expected_signal_edges_unchanged}}</small></div>"
        '<div class="audit-box"><span>想定外変更</span><strong>'
        "{{comparison.core_audit.unexpected_edges_changed}}</strong><small>"
        "重み / usage {{comparison.core_audit.unexpected_usage_edges_changed}} / "
        "node {{comparison.core_audit.unexpected_node_usage_changed}} / "
        "隣接 {{comparison.core_audit.adjacency_changes}}</small></div>"
        '<div class="audit-box"><span>重み変化量 L1</span><strong>'
        "{{'%.8f'|format(comparison.core_audit.total_abs_weight_delta)}}"
        "</strong><small>最大 "
        "{{'%.8f'|format(comparison.core_audit.max_abs_weight_delta)}}</small></div>"
        '<div class="audit-box"><span>usage変更エッジ</span><strong>'
        "{{comparison.core_audit.usage_edges_changed}}</strong><small>増分 "
        "{{comparison.core_audit.usage_total_delta}}</small></div>"
        '<div class="audit-box"><span>node_usage変更</span><strong>'
        "{{comparison.core_audit.node_usage_changed}}</strong><small>増分 "
        "{{comparison.core_audit.node_usage_total_delta}}</small></div>"
        '<div class="audit-box"><span>Core変更</span><strong>'
        "{% if comparison.core_audit.core_changed %}あり{% else %}なし{% endif %}"
        "</strong><small>hash変更 "
        "{% if comparison.core_audit.hash_changed %}あり{% else %}なし{% endif %}"
        "</small></div></div>"
        "{% if comparison.core_audit.warnings %}"
        '<div class="audit-errors"><strong>監査上の注意</strong><ul>'
        "{% for warning in comparison.core_audit.warnings %}<li>{{warning}}</li>"
        "{% endfor %}</ul></div>{% endif %}"
        "{% endif %}</div>"
    )
    page = _replace_once(page, old_table, new_table, "comparison table and audit")

    page = _replace_once(
        page,
        "</main></body></html>",
        "{% if comparison %}"
        '<div class="card" id="next-cycle-card">'
        '<div class="eyebrow">NEXT CYCLE</div>'
        "<h2>同じ設定で次の教育サイクルへ進む</h2>"
        '<form method="post">'
        '<input type="hidden" name="action" value="evaluate">'
        '<input type="hidden" name="text" value="{{text}}">'
        '<input type="hidden" name="count" value="{{count}}">'
        '<input type="hidden" name="decoys" value="{{decoys}}">'
        '<button id="next-cycle-button">同じ設定で次の候補を評価</button>'
        "</form>"
        '<p class="muted">途中入力・候補数・偽経路数を変えずに次へ進みます。</p>'
        "</div>"
        "{% endif %}"
        "</main></body></html>",
        "next cycle card",
    )
    lab.PAGE = page
    lab.app.jinja_env.globals["feedback_cycle_count"] = feedback_cycle_count


def main() -> None:
    # The Flask route resolves these module globals at request time. Patching
    # here preserves the established v0.3 teaching flow while adding v0.4
    # observability without rewriting the user's existing data.
    lab.evaluate = evaluate_with_observability
    lab.snapshot_candidates = snapshot_candidates_with_modes
    lab.teach = teach_with_core_audit
    install_research_ui()
    lab.main()


if __name__ == "__main__":
    main()
