from __future__ import annotations

from collections import Counter
import math
import sqlite3
from typing import Any

import route_choice_lab as lab
import route_choice_lab_runner as research
import route_choice_real_only_runner as real_only


CONTEXT_WEIGHT = 0.55
ROUTE_STRENGTH_WEIGHT = 0.16
FAMILIARITY_WEIGHT = 0.11
BRIDGE_WEIGHT = 0.12
GEOMETRY_WEIGHT = 0.06
FEEDBACK_WEIGHT = 0.15


def normalize_text(text: str) -> str:
    return "".join(str(text or "").strip().split())


def load_input_counts(prefix_text: str) -> Counter[str]:
    """Read repeated normal inputs directly from memory.db.

    No extra registration step is needed. Every saved input whose text starts
    with the current partial input contributes to contextual experience.
    """

    prefix = normalize_text(prefix_text)
    counts: Counter[str] = Counter()
    if not prefix or not lab.MEMORY_DB.exists():
        return counts

    try:
        with sqlite3.connect(
            f"file:{lab.MEMORY_DB.as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        ) as connection:
            rows = connection.execute(
                "SELECT input_text, COUNT(*) FROM memories "
                "WHERE kind='input' GROUP BY input_text"
            ).fetchall()
    except sqlite3.Error:
        return counts

    for raw_text, raw_count in rows:
        text = normalize_text(str(raw_text or ""))
        if text.startswith(prefix) and len(text) > len(prefix):
            counts[text] += int(raw_count or 0)
    return counts


def candidate_context_count(
    prefix_text: str,
    candidate: lab.Candidate,
    input_counts: Counter[str],
) -> int:
    """Count exact normal-input experiences represented by this route."""

    prefix = normalize_text(prefix_text)
    matched: set[str] = set()
    for evidence in getattr(candidate, "evidence_texts", []):
        text = normalize_text(lab.strip_observer_annotation(str(evidence)))
        if text.startswith(prefix) and len(text) > len(prefix):
            matched.add(text)
    return sum(int(input_counts.get(text, 0)) for text in matched)


def context_affinity(experience_count: int) -> float:
    """Saturating contextual strength: 1, 2, 5, 20 inputs remain distinct."""

    count = max(0, int(experience_count))
    return count / (count + 4.0) if count else 0.0


def contextual_mode_scores(
    metrics: dict[str, Any],
    feedback: tuple[float, float, float],
) -> dict[str, float]:
    core_score = (
        CONTEXT_WEIGHT * float(metrics.get("context_affinity", 0.0))
        + ROUTE_STRENGTH_WEIGHT * float(metrics["strength"])
        + FAMILIARITY_WEIGHT * float(metrics["familiarity"])
        + BRIDGE_WEIGHT * float(metrics["bridge_affinity"])
        + GEOMETRY_WEIGHT * float(metrics["geometry"])
    )
    feedback_score = float(lab.bias_from_totals(feedback))
    return {
        research.MODE_CORE_ONLY: core_score,
        research.MODE_FEEDBACK_ONLY: feedback_score,
        research.MODE_COMBINED: core_score + FEEDBACK_WEIGHT * feedback_score,
    }


def decorate_contextual_evaluations(
    brain: lab.SphereBrain,
    sources: list[int],
    prefix_signature: str,
    candidates: list[lab.Candidate],
    *,
    prefix_text: str,
    source_tree: tuple[Any, list[int]] | None = None,
    feedback_lookup: dict[str, tuple[float, float, float]] | None = None,
    primary_mode: str = research.MODE_COMBINED,
) -> None:
    if primary_mode not in research.MODE_ORDER:
        raise ValueError(f"unknown evaluation mode: {primary_mode}")

    source_tree = source_tree or lab.build_source_tree(brain, sources)
    feedback_lookup = (
        feedback_lookup
        if feedback_lookup is not None
        else lab.feedback_map(prefix_signature)
    )
    input_counts = load_input_counts(prefix_text)
    score_maps: dict[str, dict[str, float]] = {
        mode: {} for mode in research.MODE_ORDER
    }

    for candidate in candidates:
        research._decorate_identity(candidate)
        metrics = lab.candidate_metrics(brain, sources, candidate, source_tree)
        count = candidate_context_count(prefix_text, candidate, input_counts)
        affinity = context_affinity(count)
        metrics["context_experience_count"] = count
        metrics["context_affinity"] = affinity
        candidate.context_experience_count = count
        candidate.context_affinity = affinity
        feedback = research._feedback_for_candidate(candidate, feedback_lookup)
        scores = contextual_mode_scores(metrics, feedback)
        candidate.research_metrics = metrics
        candidate.feedback_totals = feedback
        for mode in research.MODE_ORDER:
            score_maps[mode][candidate.key] = scores[mode]

    mode_stats = {
        mode: research._normalized_mode_stats(candidates, score_maps[mode])
        for mode in research.MODE_ORDER
    }
    for candidate in candidates:
        candidate.mode_evaluations = {
            mode: mode_stats[mode][candidate.key]
            for mode in research.MODE_ORDER
        }
        for mode, prefix in (
            (research.MODE_COMBINED, "combined"),
            (research.MODE_CORE_ONLY, "core_only"),
            (research.MODE_FEEDBACK_ONLY, "feedback_only"),
        ):
            setattr(candidate, f"{prefix}_score", float(candidate.mode_evaluations[mode]["score"]))
            setattr(candidate, f"{prefix}_percent", float(candidate.mode_evaluations[mode]["percent"]))
            setattr(candidate, f"{prefix}_rank", int(candidate.mode_evaluations[mode]["rank"]))
        primary = candidate.mode_evaluations[primary_mode]
        candidate.score = float(primary["score"])
        candidate.percent = float(primary["percent"])
        candidate.rank = int(primary["rank"])

    candidates.sort(key=lambda candidate: (candidate.rank, candidate.key))


def evaluate_contextual(text: str, count: int, decoy_count: int = 0) -> dict[str, Any]:
    if not lab.BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")

    brain = lab.SphereBrain.load(lab.BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix_signature = lab.prefix_signature(sources)
    loaded_routes, source_counts = lab.load_routes()
    if not loaded_routes:
        raise RuntimeError("通常入力またはReflection由来の候補経路がありません。")

    routes, removed = research.deduplicate_routes(loaded_routes)
    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix_signature)
    decorate_contextual_evaluations(
        brain,
        sources,
        prefix_signature,
        routes,
        prefix_text=text,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
    )

    requested_count = max(1, int(count))
    selected = routes[:requested_count]
    for candidate in selected:
        lab.decode_candidate(text, candidate, routes)
    decorate_contextual_evaluations(
        brain,
        sources,
        prefix_signature,
        selected,
        prefix_text=text,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
    )
    if any(bool(candidate.decoy) for candidate in selected):
        raise RuntimeError("文脈経験モードで偽経路が検出されました。")

    return {
        "prefix": prefix_signature,
        "sources": sources,
        "candidates": selected,
        "source_counts": source_counts,
        "latest_reflection_run": lab.latest_reflection_run(),
        "dedup_removed": removed,
        "real_routes_only": True,
        "context_experience_enabled": True,
    }


def snapshot_contextual(
    brain: lab.SphereBrain,
    text: str,
    candidates: list[lab.Candidate],
    references: list[lab.Candidate],
) -> dict[str, dict[str, Any]]:
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix_signature = lab.prefix_signature(sources)
    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix_signature)
    unique_references, _ = research.deduplicate_routes(references)
    reference_by_signature = {
        research.structural_route_signature(reference): reference
        for reference in unique_references
    }

    for candidate in candidates:
        matching = reference_by_signature.get(
            research.structural_route_signature(candidate)
        )
        if matching is not None:
            candidate.route_alias_keys = list(
                getattr(matching, "route_alias_keys", [matching.key])
            )
            candidate.evidence_texts = list(matching.evidence_texts)
        research._decorate_identity(candidate)
        lab.decode_candidate(text, candidate, unique_references)

    decorate_contextual_evaluations(
        brain,
        sources,
        prefix_signature,
        candidates,
        prefix_text=text,
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
            "context_experience_count": int(metrics["context_experience_count"]),
            "context_affinity": float(metrics["context_affinity"]),
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
                for mode in research.MODE_ORDER
            },
        }
    return output


def install_context_ui() -> None:
    real_only.install_real_only_ui()
    page = lab.PAGE
    page = page.replace(
        "<strong>Core-only</strong>＝経路重み・使用経験・入口橋・幾何だけ、",
        "<strong>Core-only</strong>＝通常入力の文脈経験を中心に、経路重み・使用経験・入口橋・幾何を加えた値、",
        1,
    )
    page = page.replace(
        '<p class="muted">経路指紋 {{c.route_fingerprint}} ／ 候補ID {{c.key}}',
        '<p class="muted">文脈経験 {{c.context_experience_count}}回 ／ '
        '経路指紋 {{c.route_fingerprint}} ／ 候補ID {{c.key}}',
        1,
    )
    marker = '<div class="card"><div class="eyebrow">EVALUATION LAYERS</div>'
    notice = (
        '<div class="card"><div class="eyebrow">CONTEXT EXPERIENCE</div>'
        '<h2>通常入力が候補順位を直接育てます</h2>'
        '<p class="muted">例：「犬は走る」を通常入力すると、「犬は」で評価した際に'
        'その実経路の文脈経験が1回増えます。反復回数はmemory.dbから毎回自動集計し、'
        '別の登録操作や正解ラベルは不要です。</p></div>'
    )
    if marker in page:
        page = page.replace(marker, notice + marker, 1)
    lab.PAGE = page


def main() -> None:
    lab.evaluate = evaluate_contextual
    lab.snapshot_candidates = snapshot_contextual
    lab.teach = research.teach_with_core_audit
    install_context_ui()
    lab.main()


if __name__ == "__main__":
    main()
