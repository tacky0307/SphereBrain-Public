from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

TRAIN_TEXT = "今日は晴れて気持ちいい"
TEXTS = {
    "学習文": TRAIN_TEXT,
    "類似文": "今日の天気は最高だ",
    "無関係文": "犬は公園を走っている",
}
TRIALS_PER_GROUP = 5
BASE_SEED = pipeline.PROJECTION_SEED
ABLATION_FRACTION = 0.25


def configure_data(path: Path, projection_seed: int) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def observe(text: str, adapter: pipeline.OpenAIAdapter) -> dict:
    embedding, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(
        sources,
        steps=14,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )
    return {
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
        "node_count": len(result.activated_nodes),
        "edge_count": len(result.traversed_edges),
    }


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def route_score(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def retention(reference: dict, current: dict) -> float:
    node = len(reference["nodes"] & current["nodes"]) / len(reference["nodes"]) if reference["nodes"] else 1.0
    edge = len(reference["edges"] & current["edges"]) / len(reference["edges"]) if reference["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def shared_learning_similar_edges(routes: dict[str, dict]) -> set[tuple[int, int]]:
    return (routes["学習文"]["edges"] & routes["類似文"]["edges"]) - routes["無関係文"]["edges"]


def edge_usage(brain, edge: tuple[int, int]) -> int:
    a, b = edge
    return int(max(brain.usage[a, b], brain.usage[b, a]))


def choose_conditions(
    brain,
    shared_edges: set[tuple[int, int]],
    rng: random.Random,
) -> dict[str, list[tuple[int, int]]]:
    ranked = sorted(shared_edges, key=lambda edge: (edge_usage(brain, edge), edge))
    if not ranked:
        return {"無処置": [], "高頻度": [], "中頻度": [], "低頻度": [], "ランダム": []}

    k = max(1, math.ceil(len(ranked) * ABLATION_FRACTION))
    low = ranked[:k]
    high = ranked[-k:]

    midpoint = len(ranked) // 2
    start = max(0, midpoint - k // 2)
    middle = ranked[start : start + k]
    if len(middle) < k:
        middle = ranked[-k:]

    random_edges = rng.sample(ranked, k=min(k, len(ranked)))
    return {
        "無処置": [],
        "高頻度": high,
        "中頻度": middle,
        "低頻度": low,
        "ランダム": random_edges,
    }


def measure_with_ablation(
    brain,
    edges: list[tuple[int, int]],
    adapter: pipeline.OpenAIAdapter,
) -> dict[str, dict]:
    originals: list[tuple[int, int, bool, bool, float, float]] = []
    try:
        for a, b in edges:
            originals.append((a, b, bool(brain.adjacency[a, b]), bool(brain.adjacency[b, a]), float(brain.weights[a, b]), float(brain.weights[b, a])))
            brain.adjacency[a, b] = False
            brain.adjacency[b, a] = False
            brain.weights[a, b] = 0.0
            brain.weights[b, a] = 0.0
        return {label: observe(text, adapter) for label, text in TEXTS.items()}
    finally:
        for a, b, adj_ab, adj_ba, weight_ab, weight_ba in originals:
            brain.adjacency[a, b] = adj_ab
            brain.adjacency[b, a] = adj_ba
            brain.weights[a, b] = weight_ab
            brain.weights[b, a] = weight_ba


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"], row["label"])].append(row)

    fields = [
        "route_retention",
        "node_count",
        "edge_count",
        "learning_similar_overlap",
        "learning_unrelated_overlap",
        "removed_edge_count",
        "removed_usage_mean",
        "removed_usage_min",
        "removed_usage_max",
    ]
    summary: list[dict] = []
    for key, items in sorted(grouped.items()):
        record = {"group": key[0], "condition": key[1], "label": key[2], "trials": len(items)}
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=float)
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=0))
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    order = ["無処置", "高頻度", "中頻度", "低頻度", "ランダム"]
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "役割別Edge切断後の各入力経路保持率",
            "学習文と類似文の経路重なり",
            "切断後の通過Edge数",
            "実際に切断したEdgeの平均使用回数",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        dash = "solid" if group == "固定射影" else "dash"
        for label in TEXTS:
            items = [r for condition in order for r in summary if r["group"] == group and r["condition"] == condition and r["label"] == label]
            if not items:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[r["condition"] for r in items],
                    y=[r["route_retention_mean"] * 100 for r in items],
                    error_y={"type": "data", "array": [r["route_retention_std"] * 100 for r in items]},
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"{label} / {group}",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=[r["condition"] for r in items],
                    y=[r["edge_count_mean"] for r in items],
                    error_y={"type": "data", "array": [r["edge_count_std"] for r in items]},
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"通過Edge: {label} / {group}",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

        base_items = [r for condition in order for r in summary if r["group"] == group and r["condition"] == condition and r["label"] == "学習文"]
        figure.add_trace(
            go.Scatter(
                x=[r["condition"] for r in base_items],
                y=[r["learning_similar_overlap_mean"] * 100 for r in base_items],
                error_y={"type": "data", "array": [r["learning_similar_overlap_std"] * 100 for r in base_items]},
                mode="lines+markers",
                line={"dash": dash},
                name=f"学習↔類似 / {group}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[r["condition"] for r in base_items],
                y=[r["removed_usage_mean_mean"] for r in base_items],
                error_y={"type": "data", "array": [r["removed_usage_mean_std"] for r in base_items]},
                mode="lines+markers",
                line={"dash": dash},
                name=f"切断Edge使用回数 / {group}",
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    figure.update_yaxes(title_text="保持率 (%)", row=1, col=1)
    figure.update_yaxes(title_text="重なり (%)", row=2, col=1)
    figure.update_yaxes(title_text="edges", row=3, col=1)
    figure.update_yaxes(title_text="usage", row=4, col=1)
    figure.update_xaxes(title_text="切断条件", row=4, col=1)
    figure.update_layout(
        height=1400,
        title=(
            "SphereBrain 共有Edgeの重要度階層"
            f"<br><sup>共有Edgeの各{ABLATION_FRACTION:.0%}を同数切断 / 各群{TRIALS_PER_GROUP}試行</sup>"
        ),
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_edge_importance" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    all_rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            data_dir = run_dir / "data" / group / f"trial_{trial}"
            configure_data(data_dir, seed)
            pipeline.reset_experiment()
            pipeline.experience(TRAIN_TEXT, repeats=50, adapter=adapter)

            brain = pipeline.load_brain()
            baseline = {label: observe(text, adapter) for label, text in TEXTS.items()}
            shared = shared_learning_similar_edges(baseline)
            rng = random.Random(seed + trial * 7919)
            conditions = choose_conditions(brain, shared, rng)
            print(f"\n[{group} 試行{trial}] 共有Edge={len(shared)}")

            for condition, removed_edges in conditions.items():
                routes = baseline if condition == "無処置" else measure_with_ablation(brain, removed_edges, adapter)
                learning_similar = route_score(routes["学習文"], routes["類似文"])
                learning_unrelated = route_score(routes["学習文"], routes["無関係文"])
                usages = [edge_usage(brain, edge) for edge in removed_edges]
                for label in TEXTS:
                    all_rows.append({
                        "group": group,
                        "trial": trial,
                        "projection_seed": seed,
                        "condition": condition,
                        "label": label,
                        "text": TEXTS[label],
                        "shared_edge_total": len(shared),
                        "removed_edge_count": len(removed_edges),
                        "removed_usage_mean": float(np.mean(usages)) if usages else 0.0,
                        "removed_usage_min": min(usages) if usages else 0,
                        "removed_usage_max": max(usages) if usages else 0,
                        "route_retention": round(retention(baseline[label], routes[label]), 8),
                        "node_count": routes[label]["node_count"],
                        "edge_count": routes[label]["edge_count"],
                        "learning_similar_overlap": round(learning_similar, 8),
                        "learning_unrelated_overlap": round(learning_unrelated, 8),
                    })
                print(
                    f"  {condition}: cut={len(removed_edges)} usage={float(np.mean(usages)) if usages else 0:.1f} "
                    f"学習↔類似={learning_similar*100:.1f}%"
                )

    summary = summarize(all_rows)
    write_csv(run_dir / "edge_importance_all_trials.csv", all_rows)
    write_csv(run_dir / "edge_importance_summary.csv", summary)
    (run_dir / "edge_importance.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "train_text": TRAIN_TEXT,
                "texts": TEXTS,
                "ablation_fraction": ABLATION_FRACTION,
                "rows": all_rows,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(run_dir / "edge_importance.html", summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
