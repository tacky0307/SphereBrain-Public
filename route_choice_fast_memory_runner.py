from __future__ import annotations

from collections import Counter
import json
import sqlite3
from typing import Any

import route_choice_lab as lab
import route_choice_lab_runner as research
import route_choice_context_runner as contextual
import route_choice_context_direct_runner as direct
import route_choice_context_diverse_runner as diverse


def _normalize(value: str) -> str:
    return contextual.normalize_text(value)


def _nodes_from_row(activated_json: str | None, edges_json: str | None) -> tuple[list[int], list[tuple[int, int]]]:
    raw_edges = lab.jload(edges_json, [])
    edges: list[tuple[int, int]] = []
    nodes: list[int] = []

    for raw in raw_edges:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        a, b = int(raw[0]), int(raw[1])
        edge = lab.norm(a, b)
        edges.append(edge)
        if not nodes:
            nodes.extend([a, b])
        elif nodes[-1] == a:
            nodes.append(b)
        elif nodes[-1] == b:
            nodes.append(a)

    if not nodes:
        nodes = lab.clean_nodes([int(value) for value in lab.jload(activated_json, [])])
    if not edges and len(nodes) >= 2:
        edges = [lab.norm(a, b) for a, b in zip(nodes, nodes[1:])]
    return lab.clean_nodes(nodes), edges


def load_matching_memory_candidates(prefix_text: str, limit: int = 500) -> list[lab.Candidate]:
    """Load only normal inputs beginning with the current partial input.

    Unlike the legacy route loader, a route with one traversed edge is accepted.
    This is important for short early-stage experiences that otherwise disappear
    and leave Reflection as the only candidate source.
    """

    prefix = _normalize(prefix_text)
    if not prefix or not lab.MEMORY_DB.exists():
        return []

    try:
        with sqlite3.connect(
            f"file:{lab.MEMORY_DB.as_posix()}?mode=ro", uri=True, timeout=30
        ) as connection:
            rows = connection.execute(
                "SELECT input_text, activated_nodes, traversed_edges, COUNT(*) AS repetitions, "
                "MAX(id) AS latest_id FROM memories WHERE kind='input' AND input_text LIKE ? "
                "GROUP BY input_text, activated_nodes, traversed_edges "
                "ORDER BY repetitions DESC, latest_id DESC LIMIT ?",
                (f"{prefix_text.strip()}%", int(limit)),
            ).fetchall()
    except sqlite3.Error:
        return []

    by_key: dict[str, lab.Candidate] = {}
    repetitions_by_key: Counter[str] = Counter()
    for raw_text, activated_json, edges_json, raw_repetitions, _ in rows:
        source_text = str(raw_text or "").strip()
        normalized = _normalize(source_text)
        if not normalized.startswith(prefix) or len(normalized) <= len(prefix):
            continue
        nodes, edges = _nodes_from_row(activated_json, edges_json)
        if len(nodes) < 2 or not edges:
            continue
        route_key = lab.key_for(edges)
        candidate = by_key.get(route_key)
        if candidate is None:
            candidate = lab.Candidate(
                route_key,
                nodes,
                edges,
                evidence_texts=[source_text],
                source_kinds={"memory"},
            )
            by_key[route_key] = candidate
        elif source_text not in candidate.evidence_texts:
            candidate.evidence_texts.append(source_text)
        repetitions_by_key[route_key] += int(raw_repetitions or 0)

    output = list(by_key.values())
    for candidate in output:
        candidate.direct_memory_repetitions = int(repetitions_by_key[candidate.key])
    return output


def _label_key(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


def evaluate_fast_memory_first(text: str, count: int, decoy_count: int = 0) -> dict[str, Any]:
    if not lab.BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")

    brain = lab.SphereBrain.load(lab.BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix_signature = lab.prefix_signature(sources)
    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix_signature)
    requested_count = max(1, int(count))

    direct_routes = load_matching_memory_candidates(text)
    if direct_routes:
        contextual.decorate_contextual_evaluations(
            brain,
            sources,
            prefix_signature,
            direct_routes,
            prefix_text=text,
            source_tree=source_tree,
            feedback_lookup=feedback_lookup,
            primary_mode=research.MODE_CORE_ONLY,
        )
        for candidate in direct_routes:
            direct.direct_experience_decode(text, candidate, direct_routes)
        direct_routes.sort(
            key=lambda candidate: (
                -int(getattr(candidate, "direct_memory_repetitions", 0)),
                int(candidate.core_only_rank),
                candidate.key,
            )
        )

    selected: list[lab.Candidate] = []
    seen_labels: set[str] = set()
    seen_structures: set[Any] = set()

    def append(candidate: lab.Candidate, allow_duplicate_label: bool = False) -> bool:
        signature = research.structural_route_signature(candidate)
        label = _label_key(candidate.decoded_text)
        if signature in seen_structures:
            return False
        if label and label in seen_labels and not allow_duplicate_label:
            return False
        seen_structures.add(signature)
        if label:
            seen_labels.add(label)
        selected.append(candidate)
        return True

    for candidate in direct_routes:
        append(candidate)
        if len(selected) >= requested_count:
            break

    fallback_count = 0
    reflection_count = 0
    if len(selected) < requested_count:
        loaded_routes, _ = lab.load_routes(limit=160)
        fallback_routes, _ = research.deduplicate_routes(loaded_routes[:160])
        fallback_routes = [
            candidate
            for candidate in fallback_routes
            if research.structural_route_signature(candidate) not in seen_structures
        ][:80]
        if fallback_routes:
            contextual.decorate_contextual_evaluations(
                brain,
                sources,
                prefix_signature,
                fallback_routes,
                prefix_text=text,
                source_tree=source_tree,
                feedback_lookup=feedback_lookup,
                primary_mode=research.MODE_CORE_ONLY,
            )
            for candidate in fallback_routes:
                direct.direct_experience_decode(text, candidate, direct_routes + fallback_routes)
            fallback_routes.sort(
                key=lambda candidate: (
                    -int("memory" in getattr(candidate, "source_kinds", set())),
                    int("reflection" in getattr(candidate, "source_kinds", set())),
                    int(candidate.core_only_rank),
                    candidate.key,
                )
            )
            for candidate in fallback_routes:
                if append(candidate):
                    fallback_count += 1
                    reflection_count += int(
                        "reflection" in getattr(candidate, "source_kinds", set())
                    )
                if len(selected) >= requested_count:
                    break

    if len(selected) < requested_count:
        for candidate in direct_routes:
            append(candidate, allow_duplicate_label=True)
            if len(selected) >= requested_count:
                break

    if not selected:
        raise RuntimeError(
            "現在の途中入力に対応する通常入力経路が見つかりません。"
            "先に完全な文章を通常入力してください。"
        )

    selected = selected[:requested_count]
    contextual.decorate_contextual_evaluations(
        brain,
        sources,
        prefix_signature,
        selected,
        prefix_text=text,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
        primary_mode=research.MODE_COMBINED,
    )

    return {
        "prefix": prefix_signature,
        "sources": sources,
        "candidates": selected,
        "source_counts": {
            "memory": len(direct_routes),
            "reflection": reflection_count,
        },
        "latest_reflection_run": lab.latest_reflection_run(),
        "dedup_removed": 0,
        "real_routes_only": True,
        "context_experience_enabled": True,
        "context_first_discovery": True,
        "fast_memory_first": True,
        "fallback_count": fallback_count,
    }


def snapshot_fast_memory(
    brain: lab.SphereBrain,
    text: str,
    candidates: list[lab.Candidate],
    references: list[lab.Candidate],
) -> dict[str, dict[str, Any]]:
    direct_routes = load_matching_memory_candidates(text)
    combined_references = direct_routes + list(references)
    by_signature = {
        research.structural_route_signature(reference): reference
        for reference in combined_references
    }
    for candidate in candidates:
        matching = by_signature.get(research.structural_route_signature(candidate))
        if matching is not None:
            candidate.evidence_texts = list(matching.evidence_texts)
            candidate.source_kinds = set(matching.source_kinds)
    return contextual.snapshot_contextual(brain, text, candidates, combined_references)


def install_compact_ui() -> None:
    diverse.install_diverse_notice()
    page = lab.PAGE

    removable = [
        (
            '<div class="card"><div class="eyebrow">CONTEXT-FIRST DISCOVERY</div>'
            '<h2>通常入力に一致する異なる続きから候補を選択</h2>'
            '<p class="muted">候補発見には外部Feedbackを使いません。現在の途中入力で始まる通常経験を最優先にし、'
            '同じ続きは原則1件だけ表示します。枠が余った場合のみ、他の実経路やReflection経路で補います。</p></div>'
        ),
        (
            '<div class="card"><div class="eyebrow">DIRECT EXPERIENCE DECODER</div>'
            '<h2>経路自身の通常入力を最優先で翻訳</h2>'
            '<p class="muted">現在の途中入力で始まる通常経験をその経路自身が持つ場合、'
            '類似経路からの推測より先に、その文章の続きで表示します。直接経験がない経路だけ、'
            '従来のDecoderへフォールバックします。</p></div>'
        ),
        (
            '<div class="card"><div class="eyebrow">CONTEXT EXPERIENCE</div>'
            '<h2>通常入力が候補順位を直接育てます</h2>'
            '<p class="muted">例：「犬は走る」を通常入力すると、「犬は」で評価した際に'
            'その実経路の文脈経験が1回増えます。反復回数はmemory.dbから毎回自動集計し、'
            '別の登録操作や正解ラベルは不要です。</p></div>'
        ),
        (
            '<div class="card"><div class="eyebrow">EVALUATION LAYERS</div>'
            '<h2>同じ候補を3つの層で比較</h2>'
            '<p class="muted"><strong>総合</strong>＝Core＋外部Feedback、'
            '<strong>Core-only</strong>＝通常入力の文脈経験を中心に、経路重み・使用経験・入口橋・幾何を加えた値、'
            '<strong>Feedback-only</strong>＝○・△・×の外部教師履歴だけ。候補カードの3数値は同じ候補集合内で正規化しています。</p></div>'
        ),
    ]
    for block in removable:
        page = page.replace(block, "", 1)
    lab.PAGE = page


def main() -> None:
    lab.decode_candidate = direct.direct_experience_decode
    lab.evaluate = evaluate_fast_memory_first
    lab.snapshot_candidates = snapshot_fast_memory
    lab.teach = research.teach_with_core_audit
    install_compact_ui()
    lab.main()


if __name__ == "__main__":
    main()
