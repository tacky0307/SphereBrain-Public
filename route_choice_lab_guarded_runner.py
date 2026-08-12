from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import route_choice_lab as lab
import route_choice_lab_runner as research


MAX_SAME_DECODED_LABEL = 2
MAX_UNKNOWN_LABEL = 1
DIRECT_CONSENSUS_SHARE = 0.67
INDIRECT_CONSENSUS_SHARE = 0.72
INDIRECT_MIN_SIMILARITY = 0.82


@dataclass
class DecodeVote:
    weight: float = 0.0
    exact_prefix_weight: float = 0.0
    support_count: int = 0
    evidence: list[str] | None = None

    def add(self, weight: float, evidence: str, exact_prefix: bool) -> None:
        self.weight += float(weight)
        self.support_count += 1
        if exact_prefix:
            self.exact_prefix_weight += float(weight)
        if self.evidence is None:
            self.evidence = []
        clean = lab.strip_observer_annotation(evidence)
        if clean and clean not in self.evidence and len(self.evidence) < 4:
            self.evidence.append(clean)


def _is_usable_label(label: str) -> bool:
    value = str(label or "").strip()
    if not value:
        return False
    blocked = (
        "未解釈",
        "Reflection経路",
        "偽経路",
        "名称なし",
        "pattern",
    )
    return not any(token in value for token in blocked)


def _label_key(label: str) -> str:
    return "".join(str(label or "").split()).casefold()


def _unknown(candidate: lab.Candidate, evidence: list[str] | None = None) -> None:
    candidate.decoded_text = "未解釈の経路"
    candidate.decode_confidence = 0.0
    candidate.decode_evidence = list(evidence or [])[:4]
    candidate.decode_method = "unresolved"


def _choose_consensus(
    votes: dict[str, DecodeVote],
    labels: dict[str, str],
    *,
    minimum_share: float,
    minimum_margin: float,
    minimum_support: int,
) -> tuple[str, float, list[str]] | None:
    if not votes:
        return None
    ordered = sorted(votes.items(), key=lambda item: (-item[1].weight, item[0]))
    total = sum(vote.weight for _, vote in ordered) or 1.0
    top_key, top = ordered[0]
    second_weight = ordered[1][1].weight if len(ordered) > 1 else 0.0
    share = top.weight / total
    margin = (top.weight - second_weight) / total
    if share < minimum_share or margin < minimum_margin:
        return None
    if top.support_count < minimum_support:
        return None
    confidence = min(100.0, 42.0 + 48.0 * share + 10.0 * (top.exact_prefix_weight / max(top.weight, 1e-9)))
    return labels[top_key], confidence, list(top.evidence or [])


def conservative_decode_candidate(
    prefix: str,
    candidate: lab.Candidate,
    references: list[lab.Candidate],
) -> None:
    """Decode only when route evidence agrees; otherwise preserve uncertainty.

    Direct evidence always has priority. Similar routes are used only when no
    direct semantic evidence exists, and only with high structural agreement.
    """

    prefix_clean = str(prefix or "").strip()
    direct_votes: dict[str, DecodeVote] = defaultdict(DecodeVote)
    direct_labels: dict[str, str] = {}
    raw_evidence: list[str] = []

    for text in list(candidate.evidence_texts)[:24]:
        source = lab.strip_observer_annotation(text)
        if not source or any(token in source for token in ("偽経路", "Reflection経路")):
            continue
        label, lexical = lab.decode_suffix(prefix_clean, source)
        if not _is_usable_label(label):
            continue
        key = _label_key(label)
        exact = bool(prefix_clean and source.startswith(prefix_clean))
        contains = bool(prefix_clean and prefix_clean in source)
        weight = 2.0 if exact else (1.25 if contains else 0.65)
        weight *= 0.75 + 0.25 * max(0.0, min(100.0, float(lexical))) / 100.0
        direct_labels[key] = label
        direct_votes[key].add(weight, source, exact)
        if source not in raw_evidence:
            raw_evidence.append(source)

    if direct_votes:
        chosen = _choose_consensus(
            direct_votes,
            direct_labels,
            minimum_share=DIRECT_CONSENSUS_SHARE,
            minimum_margin=0.18,
            minimum_support=1,
        )
        if chosen is None:
            _unknown(candidate, raw_evidence)
            candidate.decode_method = "direct-conflict"
            return
        label, confidence, evidence = chosen
        candidate.decoded_text = label
        candidate.decode_confidence = round(confidence, 1)
        candidate.decode_evidence = evidence
        candidate.decode_method = "direct-consensus"
        return

    indirect_votes: dict[str, DecodeVote] = defaultdict(DecodeVote)
    indirect_labels: dict[str, str] = {}
    ranked = sorted(
        ((research.lab.route_similarity(candidate, reference), reference) for reference in references if reference.key != candidate.key),
        key=lambda item: item[0],
        reverse=True,
    )[:40]

    for similarity, reference in ranked:
        if similarity < INDIRECT_MIN_SIMILARITY:
            break
        for text in list(reference.evidence_texts)[:8]:
            source = lab.strip_observer_annotation(text)
            if not source or any(token in source for token in ("偽経路", "Reflection経路")):
                continue
            label, lexical = lab.decode_suffix(prefix_clean, source)
            if not _is_usable_label(label):
                continue
            key = _label_key(label)
            exact = bool(prefix_clean and source.startswith(prefix_clean))
            lexical_factor = 0.70 + 0.30 * max(0.0, min(100.0, float(lexical))) / 100.0
            weight = (float(similarity) ** 3) * lexical_factor * (1.15 if exact else 1.0)
            indirect_labels[key] = label
            indirect_votes[key].add(weight, source, exact)

    chosen = _choose_consensus(
        indirect_votes,
        indirect_labels,
        minimum_share=INDIRECT_CONSENSUS_SHARE,
        minimum_margin=0.20,
        minimum_support=2,
    )
    if chosen is None:
        _unknown(candidate)
        return
    label, confidence, evidence = chosen
    candidate.decoded_text = label
    candidate.decode_confidence = round(min(88.0, confidence), 1)
    candidate.decode_evidence = evidence
    candidate.decode_method = "indirect-high-consensus"


def _can_add_label(candidate: lab.Candidate, counts: dict[str, int], *, relaxed: bool = False) -> bool:
    key = _label_key(candidate.decoded_text)
    unknown = candidate.decoded_text == "未解釈の経路"
    limit = (2 if unknown else 3) if relaxed else (MAX_UNKNOWN_LABEL if unknown else MAX_SAME_DECODED_LABEL)
    return counts.get(key, 0) < limit


def _append_unique(
    candidate: lab.Candidate,
    selected: list[lab.Candidate],
    signatures: set[Any],
    label_counts: dict[str, int],
    *,
    relaxed: bool = False,
) -> bool:
    signature = research.structural_route_signature(candidate)
    if signature in signatures or not _can_add_label(candidate, label_counts, relaxed=relaxed):
        return False
    signatures.add(signature)
    research._decorate_identity(candidate)
    selected.append(candidate)
    key = _label_key(candidate.decoded_text)
    label_counts[key] = label_counts.get(key, 0) + 1
    return True


def evaluate_core_first_guarded(text: str, count: int, decoy_count: int) -> dict[str, Any]:
    """Select candidates from Core-only scores, then display all three layers."""

    if not lab.BRAIN_FILE.exists():
        raise RuntimeError("data/brain.json がありません。")
    brain = lab.SphereBrain.load(lab.BRAIN_FILE)
    sources = [int(node) for node in brain.text_to_sources(text)]
    prefix = lab.prefix_signature(sources)
    loaded_routes, source_counts = lab.load_routes()
    if not loaded_routes:
        raise RuntimeError("候補経路がありません。通常入力またはReflectionを実行してください。")

    routes, removed = research.deduplicate_routes(loaded_routes)
    source_tree = lab.build_source_tree(brain, sources)
    feedback_lookup = lab.feedback_map(prefix)

    # Critical guard: candidate discovery is driven only by Core. Feedback is
    # still displayed and used in the combined score, but cannot monopolize the
    # candidate pool.
    research.decorate_candidate_evaluations(
        brain,
        sources,
        prefix,
        routes,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
        primary_mode=research.MODE_CORE_ONLY,
    )

    requested_count = max(1, int(count))
    requested_decoys = max(0, min(int(decoy_count), requested_count - 1))
    requested_real = max(1, requested_count - requested_decoys)
    pool = routes[: min(240, len(routes))]
    for candidate in pool:
        conservative_decode_candidate(text, candidate, routes)

    selected: list[lab.Candidate] = []
    signatures: set[Any] = set()
    label_counts: dict[str, int] = {}

    for candidate in pool:
        if _append_unique(candidate, selected, signatures, label_counts) and sum(not item.decoy for item in selected) >= requested_real:
            break

    if sum(not item.decoy for item in selected) < requested_real:
        for candidate in pool:
            if _append_unique(candidate, selected, signatures, label_counts, relaxed=True) and sum(not item.decoy for item in selected) >= requested_real:
                break

    real_pool = routes[: min(160, len(routes))]
    if requested_decoys:
        for attempt in range(8):
            remaining = requested_decoys - sum(item.decoy for item in selected)
            if remaining <= 0:
                break
            generated = lab.decoys(real_pool, max(10, remaining * 8), f"{text}|guarded|{attempt}")
            for candidate in generated:
                conservative_decode_candidate(text, candidate, routes)
                if _append_unique(candidate, selected, signatures, label_counts):
                    remaining -= 1
                    if remaining <= 0:
                        break

    if len(selected) < requested_count:
        for candidate in pool:
            if _append_unique(candidate, selected, signatures, label_counts, relaxed=True):
                if len(selected) >= requested_count:
                    break

    selected = selected[:requested_count]
    # After Core-only discovery, compute the three display layers. Combined
    # ordering is acceptable here because it no longer controls discovery.
    research.decorate_candidate_evaluations(
        brain,
        sources,
        prefix,
        selected,
        source_tree=source_tree,
        feedback_lookup=feedback_lookup,
        primary_mode=research.MODE_COMBINED,
    )

    return {
        "prefix": prefix,
        "sources": sources,
        "candidates": selected,
        "source_counts": source_counts,
        "latest_reflection_run": lab.latest_reflection_run(),
        "dedup_removed": removed,
        "selection_mode": "Core-only",
        "decoder_mode": "direct-consensus guarded",
    }


def install_guard_notice() -> None:
    page = lab.PAGE
    marker = '<div class="card"><div class="eyebrow">EVALUATION LAYERS</div>'
    notice = (
        '<div class="card"><div class="eyebrow">BIAS GUARD</div>'
        '<h2>候補抽出はCore-only／翻訳は合意優先</h2>'
        '<p class="muted">外部Feedbackは候補抽出に使いません。候補自身の根拠が複数の意味に割れる場合は、'
        '一語へ強制せず「未解釈の経路」とします。同じ翻訳語は1画面最大2経路までです。</p></div>'
    )
    if marker in page:
        page = page.replace(marker, notice + marker, 1)
    lab.PAGE = page


def main() -> None:
    lab.decode_candidate = conservative_decode_candidate
    lab.evaluate = evaluate_core_first_guarded
    lab.snapshot_candidates = research.snapshot_candidates_with_modes
    lab.teach = research.teach_with_core_audit
    research.install_research_ui()
    install_guard_notice()
    lab.main()


if __name__ == "__main__":
    main()
