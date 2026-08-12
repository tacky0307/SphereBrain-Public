from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_llm_core_in_out_sentence_demo as sentence_demo

base = sentence_demo.base
expressive_decoder = sentence_demo.decoder_output


def literal_output(metrics: dict) -> str:
    """Deterministic, low-expression translation of Core metrics only."""
    sunny = float(metrics["sunny_affinity"])
    rainy = float(metrics["rainy_affinity"])
    ambiguous = float(metrics["ambiguous_affinity"])
    delta = sunny - rainy

    ranked = sorted(
        (("晴れ", sunny), ("雨", rainy), ("曖昧", ambiguous)),
        key=lambda item: item[1],
        reverse=True,
    )
    first_name, first_value = ranked[0]
    second_name, second_value = ranked[1]
    lead = first_value - second_value

    if lead < 0.015:
        center = f"{first_name}と{second_name}がほぼ同じ強さで現れている"
    elif lead < 0.05:
        center = f"{first_name}がわずかに強く、{second_name}も近い強さで残っている"
    else:
        center = f"{first_name}が最も強く、{second_name}がそれに続いている"

    if abs(delta) < 0.015:
        weather_side = "晴れと雨の偏りはほとんどない"
    elif delta > 0:
        strength = "わずかに" if delta < 0.05 else "明確に"
        weather_side = f"晴れ側が雨側より{strength}強い"
    else:
        strength = "わずかに" if -delta < 0.05 else "明確に"
        weather_side = f"雨側が晴れ側より{strength}強い"

    return f"{center}状態で、{weather_side}。"


def analyze_input(text: str) -> list[dict]:
    state = base.load_state()
    adapter = base.CachedAdapter()
    results = []

    for order_name, info in state["orders"].items():
        base.configure_core(ROOT / info["path"])
        route = base.observe(text, adapter)
        refs = info["references"]
        sunny = base.mean_overlap(route, refs["晴れ"])
        rainy = base.mean_overlap(route, refs["雨"])
        ambiguous = base.mean_overlap(route, refs["曖昧"])
        affinities = {"晴れ": sunny, "雨": rainy, "曖昧": ambiguous}
        ranked = sorted(affinities.items(), key=lambda item: item[1], reverse=True)
        metrics = {
            "sunny_affinity": sunny,
            "rainy_affinity": rainy,
            "ambiguous_affinity": ambiguous,
            "sunny_minus_rainy": sunny - rainy,
            "bridge_strength": ambiguous - abs(sunny - rainy) * 0.5,
            "node_count": route["node_count"],
            "edge_count": route["edge_count"],
            "top_group": ranked[0][0],
        }
        results.append(
            {
                "order": order_name,
                "literal_output": literal_output(metrics),
                "expressive_output": expressive_decoder(text, order_name, metrics, adapter),
                "metrics": {
                    "晴れ親和性": round(sunny * 100, 1),
                    "雨親和性": round(rainy * 100, 1),
                    "曖昧親和性": round(ambiguous * 100, 1),
                    "晴れ−雨": round((sunny - rainy) * 100, 1),
                    "活動Node": route["node_count"],
                    "通過Edge": route["edge_count"],
                },
            }
        )
    return results


base.analyze_input = analyze_input

# Reuse the existing page and only replace the result-card rendering and labels.
base.PAGE = (
    base.PAGE
    .replace("3つのCoreから文章を形成する", "Core直訳OUTとLLM表現OUTを比較する")
    .replace(
        "Decoderは解説や返答をせず、Core指標から独立した一文を形成します。会話として成立しなくても構いません。",
        "Core直訳OUTは数値状態を決め打ちで訳し、LLM表現OUTは同じ状態を自然な一文へ整えます。",
    )
    .replace(
        "Coreを観測し、数値状態から文章を形成しています…",
        "Core直訳OUTとLLM表現OUTを生成しています…",
    )
    .replace(
        "同じ入力を3つのCoreへ通し、それぞれの数値状態から一文を形成します。",
        "同じCore状態を、最小限の直訳とLLMによる表現の2段階で表示します。",
    )
    .replace(
        "<div class=\"out\">${item.output}</div>",
        "<div class=\"note\"><b>Core直訳OUT</b></div><div class=\"out\">${item.literal_output}</div><div class=\"note\"><b>LLM表現OUT</b></div><div class=\"out\">${item.expressive_output}</div>",
    )
)


if __name__ == "__main__":
    base.main()
