from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

TRIALS_PER_GROUP = 5
REPEATS = 20
BASE_SEED = pipeline.PROJECTION_SEED
NAME_SETS = [
    ("ラモ", "キト", "セナ"),
    ("ネフ", "ポラ", "ジム"),
    ("トア", "ミグ", "レナ"),
    ("バル", "ソク", "ニア"),
    ("ケム", "ルタ", "ホノ"),
]
CONDITIONS = ("推移連鎖", "同語彙非連鎖")


def configure_data(path: Path, projection_seed: int) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def route_overlap(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def merge_routes(routes: list[dict]) -> dict:
    return {
        "nodes": set().union(*(r["nodes"] for r in routes)) if routes else set(),
        "edges": set().union(*(r["edges"] for r in routes)) if routes else set(),
    }


def component_texts(subject: str, obj: str) -> list[str]:
    return [
        f"主体スロット::{subject}",
        "方向関係スロット::より強い::主体から対象へ",
        f"対象スロット::{obj}",
        f"関係結合::{subject}::より強い::{obj}",
    ]


def propagate_text(
    text: str,
    adapter: pipeline.OpenAIAdapter,
    *,
    learn: bool,
    context_nodes: Iterable[int] | None = None,
) -> dict:
    _embedding, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(
        sources,
        steps=14,
        threshold=0.18,
        noise=0.004 if learn else 0.0,
        learn=learn,
        context_nodes=context_nodes,
    )
    if learn:
        brain.save(pipeline.BRAIN_FILE)
    terminal = np.flatnonzero(result.final_activation > 0).tolist()
    if not terminal and result.activation_history:
        terminal = list(result.activation_history[-1])
    return {
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
        "terminal": terminal,
    }


def experience_relation(subject: str, obj: str, adapter: pipeline.OpenAIAdapter) -> None:
    for _ in range(REPEATS):
        for text in component_texts(subject, obj):
            propagate_text(text, adapter, learn=True)


def observe_relation(subject: str, obj: str, adapter: pipeline.OpenAIAdapter) -> dict:
    routes = [propagate_text(text, adapter, learn=False) for text in component_texts(subject, obj)]
    merged = merge_routes(routes)
    merged["terminal"] = routes[-1]["terminal"] if routes else []
    return merged


def train_condition(condition: str, names: tuple[str, str, str], adapter: pipeline.OpenAIAdapter) -> None:
    a, b, c = names
    experience_relation(a, b, adapter)
    if condition == "推移連鎖":
        experience_relation(b, c, adapter)
    elif condition == "同語彙非連鎖":
        experience_relation(c, b, adapter)
    else:
        raise ValueError(condition)


def contextual_stage(
    subject: str,
    obj: str,
    adapter: pipeline.OpenAIAdapter,
    context_nodes: Iterable[int],
) -> dict:
    routes: list[dict] = []
    current_context = list(context_nodes)
    for text in component_texts(subject, obj):
        route = propagate_text(text, adapter, learn=False, context_nodes=current_context)
        routes.append(route)
        current_context = route["terminal"] or current_context
    merged = merge_routes(routes)
    merged["terminal"] = current_context
    return merged


def evaluate(names: tuple[str, str, str], condition: str, adapter: pipeline.OpenAIAdapter) -> dict:
    a, b, c = names
    premise1 = observe_relation(a, b, adapter)
    if condition == "推移連鎖":
        premise2 = observe_relation(b, c, adapter)
        stage2_subject, stage2_object = b, c
    else:
        premise2 = observe_relation(c, b, adapter)
        stage2_subject, stage2_object = c, b

    candidate = observe_relation(a, c, adapter)
    reverse = observe_relation(c, a, adapter)

    one_pass = merge_routes([premise1, premise2])
    one_candidate = route_overlap(one_pass, candidate)
    one_reverse = route_overlap(one_pass, reverse)

    stage2 = contextual_stage(
        stage2_subject,
        stage2_object,
        adapter,
        context_nodes=premise1["terminal"],
    )
    two_stage = merge_routes([premise1, stage2])
    two_candidate = route_overlap(two_stage, candidate)
    two_reverse = route_overlap(two_stage, reverse)

    stage_bridge = len(stage2["edges"] & premise1["edges"]) / len(premise1["edges"]) if premise1["edges"] else 0.0
    candidate_new = len(candidate["edges"] & (two_stage["edges"] - one_pass["edges"]))
    candidate_new_ratio = candidate_new / len(candidate["edges"]) if candidate["edges"] else 0.0

    return {
        "one_pass_candidate": one_candidate,
        "one_pass_reverse": one_reverse,
        "one_pass_margin": one_candidate - one_reverse,
        "two_stage_candidate": two_candidate,
        "two_stage_reverse": two_reverse,
        "two_stage_margin": two_candidate - two_reverse,
        "margin_gain": (two_candidate - two_reverse) - (one_candidate - one_reverse),
        "stage_bridge_use": stage_bridge,
        "candidate_new_edge_ratio": candidate_new_ratio,
        "premise1_terminal_count": len(premise1["terminal"]),
        "two_stage_node_count": len(two_stage["nodes"]),
        "two_stage_edge_count": len(two_stage["edges"]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    fields = [
        "one_pass_margin", "two_stage_margin", "margin_gain",
        "stage_bridge_use", "candidate_new_edge_ratio",
        "two_stage_node_count", "two_stage_edge_count",
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"])].append(row)
    output: list[dict] = []
    for (group, condition), items in sorted(grouped.items()):
        record: dict = {"group": group, "condition": condition, "trials": len(items)}
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=float)
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=0))
        output.append(record)
    return output


def write_html(path: Path, summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    figure = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "一段結合と二段連鎖の候補−逆向き支持差",
            "二段化による支持差の改善量",
            "第1段終点を文脈にした第2段の接続指標",
        ),
        vertical_spacing=0.13,
    )
    x = [f"{r['condition']} / {r['group']}" for r in summary]
    figure.add_trace(go.Bar(x=x, y=[r["one_pass_margin_mean"]*100 for r in summary], name="一段結合"), row=1, col=1)
    figure.add_trace(go.Bar(x=x, y=[r["two_stage_margin_mean"]*100 for r in summary], name="二段連鎖"), row=1, col=1)
    figure.add_trace(go.Bar(x=x, y=[r["margin_gain_mean"]*100 for r in summary], error_y={"type":"data","array":[r["margin_gain_std"]*100 for r in summary]}, name="改善量"), row=2, col=1)
    figure.add_trace(go.Bar(x=x, y=[r["stage_bridge_use_mean"]*100 for r in summary], name="第1段経路再利用"), row=3, col=1)
    figure.add_trace(go.Bar(x=x, y=[r["candidate_new_edge_ratio_mean"]*100 for r in summary], name="候補の新規Edge利用"), row=3, col=1)
    figure.add_hline(y=0, line_dash="dot", row=1, col=1)
    figure.add_hline(y=0, line_dash="dot", row=2, col=1)
    figure.update_yaxes(title_text="支持差 (%)", row=1, col=1)
    figure.update_yaxes(title_text="改善量 (%)", row=2, col=1)
    figure.update_yaxes(title_text="割合 (%)", row=3, col=1)
    figure.update_layout(height=1200, title="SphereBrain 最小二段連鎖伝播実験", barmode="group")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_two_stage_chain_propagation" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    rows: list[dict] = []
    partial_csv = run_dir / "two_stage_chain_trials.partial.csv"

    try:
        for group in ("固定射影", "異なる射影"):
            for trial in range(1, TRIALS_PER_GROUP + 1):
                names = NAME_SETS[(trial - 1) % len(NAME_SETS)]
                for condition in CONDITIONS:
                    seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
                    data_path = run_dir / "data" / group / f"trial_{trial}" / condition
                    configure_data(data_path, seed)
                    pipeline.reset_experiment()
                    train_condition(condition, names, adapter)
                    metrics = evaluate(names, condition, adapter)
                    row = {
                        "group": group,
                        "trial": trial,
                        "condition": condition,
                        "subject": names[0],
                        "middle": names[1],
                        "object": names[2],
                        **metrics,
                    }
                    rows.append(row)
                    write_csv(partial_csv, rows)
                    print(
                        f"[{group} {trial}] {condition}: "
                        f"one={metrics['one_pass_margin']*100:.1f}% "
                        f"two={metrics['two_stage_margin']*100:.1f}% "
                        f"gain={metrics['margin_gain']*100:.1f}%"
                    )
    finally:
        write_csv(partial_csv, rows)

    summary = summarize(rows)
    write_csv(run_dir / "two_stage_chain_trials.csv", rows)
    write_csv(run_dir / "two_stage_chain_summary.csv", summary)
    payload = {
        "experiment": "two_stage_chain_propagation",
        "description": "Use terminal activation from premise 1 as context for premise 2, then test whether the composite route favors A>C over C>A.",
        "repeats": REPEATS,
        "trials_per_group": TRIALS_PER_GROUP,
        "rows": rows,
        "summary": summary,
    }
    (run_dir / "two_stage_chain_propagation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "two_stage_chain_propagation.html", summary)

    print("\n完了しました。")
    print(f"HTML: {run_dir / 'two_stage_chain_propagation.html'}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
