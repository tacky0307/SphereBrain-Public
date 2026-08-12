from __future__ import annotations

from collections import Counter
from typing import Any

import route_choice_lab as lab
import route_choice_lab_runner as research
import route_choice_context_runner as contextual


ORIGINAL_DECODE = lab.decode_candidate


def normalize_text(text: str) -> str:
    return contextual.normalize_text(text)


def direct_experience_decode(
    prefix_text: str,
    candidate: lab.Candidate,
    references: list[lab.Candidate],
) -> None:
    """Decode from the candidate's own matching normal experiences first.

    Similar-route inference is used only when the route itself has no evidence
    beginning with the current partial input. This prevents unrelated strong
    routes from being repeatedly rendered as the same word such as 泳ぐ or 青.
    """

    prefix = normalize_text(prefix_text)
    input_counts = contextual.load_input_counts(prefix_text)
    direct_votes: Counter[str] = Counter()
    direct_evidence: dict[str, list[str]] = {}

    for raw_evidence in getattr(candidate, "evidence_texts", []):
        source = lab.strip_observer_annotation(str(raw_evidence or "")).strip()
        normalized = normalize_text(source)
        if not prefix or not normalized.startswith(prefix) or len(normalized) <= len(prefix):
            continue

        suffix = normalized[len(prefix):].lstrip("、。・:：-—→ 　")
        if not suffix:
            continue

        # Exact repetition count from normal input is the primary vote. A direct
        # route-local label still receives one vote even when the original row is
        # old or outside the current aggregation window.
        repetitions = max(1, int(input_counts.get(normalized, 0)))
        direct_votes[suffix] += repetitions
        direct_evidence.setdefault(suffix, [])
        if source not in direct_evidence[suffix]:
            direct_evidence[suffix].append(source)

    if direct_votes:
        ordered = sorted(
            direct_votes.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        label, vote_count = ordered[0]
        candidate.decoded_text = label
        candidate.decode_confidence = 100.0
        candidate.decode_evidence = direct_evidence.get(label, [])[:3]
        candidate.decode_method = "route-local-direct-experience"
        candidate.direct_experience_votes = int(vote_count)
        return

    ORIGINAL_DECODE(prefix_text, candidate, references)
    candidate.decode_method = getattr(candidate, "decode_method", "similar-route-fallback")
    candidate.direct_experience_votes = 0


def install_direct_decoder_notice() -> None:
    contextual.install_context_ui()
    page = lab.PAGE
    marker = '<div class="card"><div class="eyebrow">CONTEXT EXPERIENCE</div>'
    notice = (
        '<div class="card"><div class="eyebrow">DIRECT EXPERIENCE DECODER</div>'
        '<h2>経路自身の通常入力を最優先で翻訳</h2>'
        '<p class="muted">現在の途中入力で始まる通常経験をその経路自身が持つ場合、'
        '類似経路からの推測より先に、その文章の続きで表示します。直接経験がない経路だけ、'
        '従来のDecoderへフォールバックします。</p></div>'
    )
    if marker in page:
        page = page.replace(marker, notice + marker, 1)

    page = page.replace(
        '<p class="muted">文脈経験 {{c.context_experience_count}}回 ／ ',
        '<p class="muted">文脈経験 {{c.context_experience_count}}回 ／ '
        '直接翻訳票 {{c.direct_experience_votes|default(0)}}票 ／ ',
        1,
    )
    lab.PAGE = page


def main() -> None:
    lab.decode_candidate = direct_experience_decode
    lab.evaluate = contextual.evaluate_contextual
    lab.snapshot_candidates = contextual.snapshot_contextual
    lab.teach = research.teach_with_core_audit
    install_direct_decoder_notice()
    lab.main()


if __name__ == "__main__":
    main()
