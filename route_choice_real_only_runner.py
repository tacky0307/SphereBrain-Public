from __future__ import annotations

from typing import Any

import route_choice_lab as lab
import route_choice_lab_runner as research


def evaluate_real_routes_only(
    text: str,
    count: int,
    decoy_count: int = 0,
) -> dict[str, Any]:
    """Return only saved experience routes; never generate synthetic decoys."""

    result = research.evaluate_with_observability(text, count, 0)
    candidates = list(result.get("candidates", []))
    if any(bool(candidate.decoy) for candidate in candidates):
        raise RuntimeError("実経路限定モードで偽経路候補が検出されました。")
    result["real_routes_only"] = True
    return result


def install_real_only_ui() -> None:
    """Keep the v0.4 research UI while removing the decoy control entirely."""

    research.install_research_ui()
    page = lab.PAGE

    page = page.replace(
        "Coreへ提示する候補は言葉ではなく経路です。完全同一経路だけを除外し、"
        "異なる経路が同じ日本語へデコードされた場合は別候補として残します。"
        "総合・Core-only・Feedback-onlyを同時に観測し、教育直後はCore本体の変更も監査します。",
        "Coreへ提示する候補は言葉ではなく、保存済み経験から得た実経路だけです。"
        "偽経路は生成しません。完全同一経路だけを除外し、異なる経路が同じ日本語へ"
        "デコードされた場合は別候補として残します。総合・Core-only・Feedback-onlyを"
        "同時に観測し、教育直後はCore本体の変更も監査します。",
        1,
    )

    old_control = (
        '<div><div class="eyebrow">DECOYS</div><h2>偽経路数</h2>'
        '<input name="decoys" type="number" min="0" max="4" value="{{decoys}}"></div>'
    )
    new_control = (
        '<div><div class="eyebrow">CANDIDATE SOURCE</div><h2>実経路のみ</h2>'
        '<input type="hidden" name="decoys" value="0">'
        '<p class="muted">偽経路は生成しません</p></div>'
    )
    if old_control not in page:
        raise RuntimeError("偽経路入力欄のUI置換位置が見つかりません。")
    page = page.replace(old_control, new_control, 1)

    page = page.replace(
        "<strong>総合</strong>＝Core＋外部Feedback＋現行の偽経路補正、",
        "<strong>総合</strong>＝Core＋外部Feedback、",
        1,
    )
    page = page.replace(
        "途中入力・候補数・偽経路数を変えずに次へ進みます。",
        "途中入力と候補数を変えず、実経路だけで次へ進みます。",
        1,
    )

    lab.PAGE = page


def main() -> None:
    # Preserve the established decoder, teaching logic, fingerprints, separated
    # evaluation layers, progress tracking, and Core audit. Only decoy generation
    # is disabled.
    lab.evaluate = evaluate_real_routes_only
    lab.snapshot_candidates = research.snapshot_candidates_with_modes
    lab.teach = research.teach_with_core_audit
    install_real_only_ui()
    lab.main()


if __name__ == "__main__":
    main()
