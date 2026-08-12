from __future__ import annotations

import csv
import json
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
SIMILAR_TEXT = "今日の天気は最高だ"
UNRELATED_TEXT = "犬は公園を走っている"
TEXTS = {
    "学習文": TRAIN_TEXT,
    "類似文": SIMILAR_TEXT,
    "無関係文": UNRELATED_TEXT,
}
TRIALS_PER_GROUP = 5
BASE_SEED = pipeline.PROJECTION_SEED
TOP_FRACTION = 0.25


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
    result = brain.propagate(sources, steps=14, threshold=0.18, noise=0.0, learn=False)
    return {
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
    }


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def route_overlap(a: dict, b: dict) -> float:
    return 0.35 * jaccard(a["nodes"], b["nodes"]) + 0.65 * jaccard(a["edges"], b["edges"])


def retention(reference: dict, current: dict) -> float:
    node = len(reference["nodes"] & current["nodes"]) / len(reference["nodes"]) if reference["nodes"] else 1.0
    edge = len(reference["edges"] & current["edges"]) / len(reference["edges"]) if reference["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def ablate_and_observe(edge: tuple[int, int], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
    brain = pipeline.load_brain()
    a, b = edge
    original_weight_ab = float(brain.weights[a, b])
    original_weight_ba = float(brain.weights[b, a])
    original_adj_ab = bool(brain.adjacency[a, b])
    original_adj_ba = bool(brain.adjacency[b, a])
    brain.weights[a, b] = brain.weights[b, a] = 0.0
    brain.adjacency[a, b] = brain.adjacency[b, a] = False
    brain.save(pipeline.BRAIN_FILE)
    try:
        return {label: observe(text, adapter) for label, text in TEXTS.items()}
    finally:
        restored = pipeline.load_brain()
        restored.weights[a, b] = original_weight_ab
        restored.weights[b, a] = original_weight_ba
        restored.adjacency[a, b] = original_adj_ab
        restored.adjacency[b, a] = original_adj_ba
        restored.save(pipeline.BRAIN_FILE)


def cut_set_and_observe(edges: list[tuple[int, int]], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
    brain = pipeline.load_brain()
    backup = []
    for a, b in edges:
        backup.append((a, b, float(brain.weights[a, b]), float(brain.weights[b, a]), bool(brain.adjacency[a, b]), bool(brain.adjacency[b, a])))
        brain.weights[a, b] = brain.weights[b, a] = 0.0
        brain.adjacency[a, b] = brain.adjacency[b, a] = False
    brain.save(pipeline.BRAIN_FILE)
    try:
        return {label: observe(text, adapter) for label, text in TEXTS.items()}
    finally:
        restored = pipeline.load_brain()
        for a, b, wab, wba, aab, aba in backup:
            restored.weights[a, b] = wab
            restored.weights[b, a] = wba
            restored.adjacency[a, b] = aab
            restored.adjacency[b, a] = aba
        restored.save(pipeline.BRAIN_FILE)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"])].append(row)
    fields = [
        "train_retention", "similar_retention", "unrelated_retention",
        "train_similar_overlap", "train_unrelated_overlap",
        "train_edges", "similar_edges", "unrelated_edges",
    ]
    output = []
    for (group, condition), items in sorted(grouped.items()):
        rec = {"group": group, "condition": condition, "trials": len(items)}
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=float)
            rec[f"{field}_mean"] = float(values.mean())
            rec[f"{field}_std"] = float(values.std(ddof=0))
        output.append(rec)
    return output


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
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=(
        "切断群別の各入力経路保持率",
        "学習文と類似文の経路重なり",
        "切断後の通過Edge数",
    ))
    conditions = ["無処置", "因果上位25%", "使用頻度上位25%", "ランダム25%"]
    for group, dash in (("固定射影", "solid"), ("異なる射影", "dash")):
        items = {r["condition"]: r for r in summary if r["group"] == group}
        x = conditions
        for field, label in (("train_retention", "学習文"), ("similar_retention", "類似文"), ("unrelated_retention", "無関係文")):
            fig.add_trace(go.Scatter(x=x, y=[items[c][field+"_mean"]*100 for c in x], error_y={"type":"data","array":[items[c][field+"_std"]*100 for c in x]}, mode="lines+markers", line={"dash":dash}, name=f"{label}/{group}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=[items[c]["train_similar_overlap_mean"]*100 for c in x], error_y={"type":"data","array":[items[c]["train_similar_overlap_std"]*100 for c in x]}, mode="lines+markers", line={"dash":dash}, name=f"学習↔類似/{group}"), row=2, col=1)
        for field, label in (("train_edges", "学習文"), ("similar_edges", "類似文"), ("unrelated_edges", "無関係文")):
            fig.add_trace(go.Scatter(x=x, y=[items[c][field+"_mean"] for c in x], error_y={"type":"data","array":[items[c][field+"_std"] for c in x]}, mode="lines+markers", line={"dash":dash}, name=f"{label} edges/{group}", showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text="保持率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="重なり (%)", row=2, col=1)
    fig.update_yaxes(title_text="edges", row=3, col=1)
    fig.update_xaxes(title_text="切断条件", row=3, col=1)
    fig.update_layout(height=1100, title="SphereBrain 因果的重要度による共有Edge比較", hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_causal_edge_importance" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    all_rows: list[dict] = []
    edge_scores_all: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            pipeline.experience(TRAIN_TEXT, repeats=50, adapter=adapter)
            baseline = {label: observe(text, adapter) for label, text in TEXTS.items()}
            shared_edges = sorted((baseline["学習文"]["edges"] & baseline["類似文"]["edges"]) - baseline["無関係文"]["edges"])
            brain = pipeline.load_brain()
            print(f"\n[{group} 試行{trial}] 共有Edge={len(shared_edges)}")

            scored = []
            for index, edge in enumerate(shared_edges, start=1):
                ablated = ablate_and_observe(edge, adapter)
                score = (
                    (1.0 - retention(baseline["学習文"], ablated["学習文"])) * 0.35
                    + (1.0 - retention(baseline["類似文"], ablated["類似文"])) * 0.45
                    + max(0.0, route_overlap(baseline["学習文"], baseline["類似文"]) - route_overlap(ablated["学習文"], ablated["類似文"])) * 0.20
                )
                a, b = edge
                usage = int(brain.usage[a, b])
                scored.append((edge, score, usage))
                edge_scores_all.append({"group":group,"trial":trial,"edge_a":a,"edge_b":b,"causal_score":round(score,8),"usage":usage})
                if index % 10 == 0 or index == len(shared_edges):
                    print(f"  単独アブレーション {index}/{len(shared_edges)}")

            count = max(1, int(round(len(shared_edges) * TOP_FRACTION))) if shared_edges else 0
            causal_edges = [edge for edge, _, _ in sorted(scored, key=lambda x: (-x[1], -x[2], x[0]))[:count]]
            usage_edges = [edge for edge, _, _ in sorted(scored, key=lambda x: (-x[2], -x[1], x[0]))[:count]]
            rng = random.Random(seed + 424242)
            random_edges = rng.sample(shared_edges, count) if count and len(shared_edges) >= count else list(shared_edges)
            conditions = {
                "無処置": [],
                "因果上位25%": causal_edges,
                "使用頻度上位25%": usage_edges,
                "ランダム25%": random_edges,
            }
            for condition, edges in conditions.items():
                current = baseline if not edges else cut_set_and_observe(edges, adapter)
                row = {
                    "group": group,
                    "trial": trial,
                    "projection_seed": seed,
                    "condition": condition,
                    "cut_edges": len(edges),
                    "shared_edges": len(shared_edges),
                    "train_retention": round(retention(baseline["学習文"], current["学習文"]), 8),
                    "similar_retention": round(retention(baseline["類似文"], current["類似文"]), 8),
                    "unrelated_retention": round(retention(baseline["無関係文"], current["無関係文"]), 8),
                    "train_similar_overlap": round(route_overlap(current["学習文"], current["類似文"]), 8),
                    "train_unrelated_overlap": round(route_overlap(current["学習文"], current["無関係文"]), 8),
                    "train_edges": len(current["学習文"]["edges"]),
                    "similar_edges": len(current["類似文"]["edges"]),
                    "unrelated_edges": len(current["無関係文"]["edges"]),
                }
                all_rows.append(row)

    summary = summarize(all_rows)
    write_csv(run_dir / "causal_edge_importance_all_trials.csv", all_rows)
    write_csv(run_dir / "causal_edge_scores.csv", edge_scores_all)
    write_csv(run_dir / "causal_edge_importance_summary.csv", summary)
    (run_dir / "causal_edge_importance.json").write_text(json.dumps({"created_at":datetime.now().isoformat(timespec="seconds"),"train_text":TRAIN_TEXT,"similar_text":SIMILAR_TEXT,"unrelated_text":UNRELATED_TEXT,"rows":all_rows,"edge_scores":edge_scores_all,"summary":summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "causal_edge_importance.html", summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
