from __future__ import annotations

from typing import Any

import route_choice_lab as lab
import route_choice_lab_runner as research
import route_choice_context_runner as contextual
import route_choice_context_direct_runner as direct


def _label_key(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


def evaluate_context_first_diverse(
    text: str,
    count: int,
    decoy_count: int = 0,
) -> dict[str, Any]:
    """Prefer matching normal experiences and keep one card per decoded continuation.

    Candidate discovery is driven by contextual Core-only score, not external
    feedback. Routes backed by normal inputs beginning with the current partial
    input are considered first. Duplicate decoded continuations are suppressed
    on the teaching screen while the underlying routes remain untouched.
    """

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

    # Discovery must not be monopolized by previous teacher feedback.
    contextual.decorate_contextual_evaluations(
        brain,
        sources,
        prefix_signature,
        routes,
        prefix_text=text,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
        primary_mode=research.MODE_CORE_ONLY,
    )

    # Decode every route before selecting cards. Direct route-local experience
    # is already installed by the runner and therefore wins over similarity.
    for candidate in routes:
        direct.direct_experience_decode(text, candidate, routes)

    # Matching normal experiences first, then other memory routes, Reflection last.
    routes.sort(
        key=lambda candidate: (
            -int(getattr(candidate, "context_experience_count", 0) > 0),
            -int(getattr(candidate, "context_experience_count", 0)),
            -int("memory" in getattr(candidate, "source_kinds", set())),
            int("reflection" in getattr(candidate, "source_kinds", set())),
            int(candidate.core_only_rank),
            candidate.key,
        )
    )

    requested_count = max(1, int(count))
    selected: list[lab.Candidate] = []
    seen_labels: set[str] = set()
    seen_structures: set[Any] = set()

    def append(candidate: lab.Candidate, *, allow_duplicate_label: bool = False) -> bool:
        signature = research.structural_route_signature(candidate)
        if signature in seen_structures:
            return False
        label = _label_key(candidate.decoded_text)
        if label and label in seen_labels and not allow_duplicate_label:
            return False
        seen_structures.add(signature)
        if label:
            seen_labels.add(label)
        selected.append(candidate)
        return True

    # First pass: one representative route for each continuation.
    for candidate in routes:
        append(candidate)
        if len(selected) >= requested_count:
            break

    # Only if there are fewer than requested distinct continuations, fill the
    # remaining cards with structurally different routes.
    if len(selected) < requested_count:
        for candidate in routes:
            append(candidate, allow_duplicate_label=True)
            if len(selected) >= requested_count:
                break

    selected = selected[:requested_count]

    # Display all three layers after discovery. Combined/Feedback can change the
    # shown rank, but cannot decide which five routes enter the candidate set.
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
    if any(bool(candidate.decoy) for candidate in selected):
        raise RuntimeError("文脈候補モードで偽経路が検出されました。")

    return {
        "prefix": prefix_signature,
        "sources": sources,
        "candidates": selected,
        "source_counts": source_counts,
        "latest_reflection_run": lab.latest_reflection_run(),
        "dedup_removed": removed,
        "real_routes_only": True,
        "context_experience_enabled": True,
        "context_first_discovery": True,
        "distinct_continuations": len(seen_labels),
    }


def install_diverse_notice() -> None:
    direct.install_direct_decoder_notice()
    page = lab.PAGE
    marker = '<div class="card"><div class="eyebrow">DIRECT EXPERIENCE DECODER</div>'
    notice = (
        '<div class="card"><div class="eyebrow">CONTEXT-FIRST DISCOVERY</div>'
        '<h2>通常入力に一致する異なる続きから候補を選択</h2>'
        '<p class="muted">候補発見には外部Feedbackを使いません。現在の途中入力で始まる通常経験を最優先にし、'
        '同じ続きは原則1件だけ表示します。枠が余った場合のみ、他の実経路やReflection経路で補います。</p></div>'
    )
    if marker in page:
        page = page.replace(marker, notice + marker, 1)
    lab.PAGE = page


def main() -> None:
    lab.decode_candidate = direct.direct_experience_decode
    lab.evaluate = evaluate_context_first_diverse
    lab.snapshot_candidates = contextual.snapshot_contextual
    lab.teach = research.teach_with_core_audit
    install_diverse_notice()
    lab.main()


if __name__ == "__main__":
    main()
