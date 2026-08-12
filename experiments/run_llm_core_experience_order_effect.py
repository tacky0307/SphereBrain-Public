from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

SUNNY = [
    "今日は晴れて気持ちいい",
    "青い空が広がって爽やかだ",
    "暖かな日差しが心地よい",
]
RAINY = [
    "今日は雨で肌寒い",
    "暗い空から雨が降っている",
    "冷たい雨で気分が沈む",
]
AMBIGUOUS = [
    "今日は天気が変わりやすい",
    "晴れたり雨が降ったりしている",
    "空模様が落ち着かない",
]
GROUPS = {"晴れ": SUNNY, "雨": RAINY, "曖昧": AMBIGUOUS}
ORDERS = {
    "晴れ→雨→曖昧": ("晴れ", "雨", "曖昧"),
    "雨→晴れ→曖昧": ("雨", "晴れ", "曖昧"),
    "曖昧→晴れ→雨": ("曖昧", "晴れ", "雨"),
}
PROBES = {
    "晴れ未経験": "よく晴れた穏やかな日だ",
    "雨未経験": "雨雲が広がって寒く感じる",
    "曖昧天気": "今日は天気が読みにくい",
    "遠い対照": "会計ソフトが請求書を処理する",
}
TRAIN_REPEATS = 20
TRIALS_PER_GROUP = 3
BASE_SEED = pipeline.PROJECTION_SEED
K_VALUES = range(2, 9)


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


def route_retention(previous: dict, current: dict) -> float:
    node = len(previous["nodes"] & current["nodes"]) / len(previous["nodes"]) if previous["nodes"] else 1.0
    edge = len(previous["edges"] & current["edges"]) / len(previous["edges"]) if previous["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def kmeans(matrix: np.ndarray, k: int, seed: int, restarts: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_labels = np.zeros(len(matrix), dtype=int)
    best_centers = np.zeros((k, matrix.shape[1]))
    best_loss = float("inf")
    for _ in range(restarts):
        centers = matrix[rng.choice(len(matrix), k, replace=False)].astype(float).copy()
        labels = np.zeros(len(matrix), dtype=int)
        for iteration in range(80):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if iteration and np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = matrix[labels == cluster]
                centers[cluster] = members.mean(axis=0) if len(members) else matrix[rng.integers(len(matrix))]
        loss = float(((matrix - centers[labels]) ** 2).sum())
        if loss < best_loss:
            best_loss = loss
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def coassignment(labels: np.ndarray) -> set[tuple[int, int]]:
    return {(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if labels[i] == labels[j]}


def summarize(rows: list[dict], keys: tuple[str, ...], fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        record = {name: value for name, value in zip(keys, key)}
        record["trials"] = len(items)
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=float)
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=0))
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, stage_summary: list[dict], final_summary: list[dict], order_summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "経験順序ごとの段階間経路保持率",
            "最終段階の未経験Probe親和差",
            "同じ最終経験でも順序が違うCore間の最終経路類似度",
            "順序間のクラスタ関係類似度（k=2〜8）",
        ),
        vertical_spacing=0.10,
    )

    for order in ORDERS:
        items = [r for r in stage_summary if r["order"] == order]
        figure.add_trace(
            go.Scatter(
                x=[r["transition"] for r in items],
                y=[r["route_retention_mean"] * 100 for r in items],
                error_y={"type": "data", "array": [r["route_retention_std"] * 100 for r in items]},
                mode="lines+markers",
                name=order,
            ),
            row=1,
            col=1,
        )

    for probe in PROBES:
        items = [r for r in final_summary if r["probe"] == probe]
        figure.add_trace(
            go.Bar(
                x=[r["order"] for r in items],
                y=[r["margin_mean"] * 100 for r in items],
                error_y={"type": "data", "array": [r["margin_std"] * 100 for r in items]},
                name=probe,
            ),
            row=2,
            col=1,
        )

    route_items = [r for r in order_summary if r["metric"] == "final_route_similarity"]
    figure.add_trace(
        go.Bar(
            x=[r["order_pair"] for r in route_items],
            y=[r["value_mean"] * 100 for r in route_items],
            error_y={"type": "data", "array": [r["value_std"] * 100 for r in route_items]},
            name="最終経路類似度",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    for pair in sorted(set(r["order_pair"] for r in order_summary if r["metric"] == "cluster_similarity")):
        items = [r for r in order_summary if r["metric"] == "cluster_similarity" and r["order_pair"] == pair]
        figure.add_trace(
            go.Scatter(
                x=[r["k"] for r in items],
                y=[r["value_mean"] * 100 for r in items],
                error_y={"type": "data", "array": [r["value_std"] * 100 for r in items]},
                mode="lines+markers",
                name=pair,
            ),
            row=4,
            col=1,
        )

    figure.update_yaxes(title_text="保持率 (%)", row=1, col=1)
    figure.update_yaxes(title_text="親和差 (%)", row=2, col=1)
    figure.update_yaxes(title_text="類似度 (%)", row=3, col=1)
    figure.update_yaxes(title_text="類似度 (%)", row=4, col=1)
    figure.update_xaxes(title_text="段階", row=1, col=1)
    figure.update_xaxes(title_text="経験順序", row=2, col=1)
    figure.update_xaxes(title_text="順序ペア", row=3, col=1)
    figure.update_xaxes(title_text="クラスタ数 k", row=4, col=1)
    figure.update_layout(height=1500, title="SphereBrain 経験順序が最終認識構造へ与える影響", barmode="group", hovermode="x unified")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_experience_order_effect" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()

    stage_rows: list[dict] = []
    final_rows: list[dict] = []
    comparison_rows: list[dict] = []
    final_by_key: dict[tuple[str, int, str], dict] = {}

    for projection_group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            projection_seed = BASE_SEED if projection_group == "固定射影" else BASE_SEED + trial * 1009
            for order_name, order in ORDERS.items():
                data_dir = run_dir / "data" / projection_group / f"trial_{trial}" / order_name
                configure_data(data_dir, projection_seed)
                pipeline.reset_experiment()

                probe_snapshots = {"初期": {label: observe(text, adapter) for label, text in PROBES.items()}}
                previous_stage = "初期"
                previous_routes = probe_snapshots[previous_stage]

                for stage_index, group_name in enumerate(order, start=1):
                    for text in GROUPS[group_name]:
                        pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)
                    stage_name = f"{stage_index}:{group_name}追加後"
                    current_routes = {label: observe(text, adapter) for label, text in PROBES.items()}
                    probe_snapshots[stage_name] = current_routes
                    for label in PROBES:
                        stage_rows.append({
                            "projection_group": projection_group,
                            "trial": trial,
                            "order": order_name,
                            "transition": f"{previous_stage}→{stage_name}",
                            "probe": label,
                            "route_retention": round(route_retention(previous_routes[label], current_routes[label]), 8),
                            "new_edges": len(current_routes[label]["edges"] - previous_routes[label]["edges"]),
                            "removed_edges": len(previous_routes[label]["edges"] - current_routes[label]["edges"]),
                        })
                    previous_stage = stage_name
                    previous_routes = current_routes

                sunny_refs = [observe(text, adapter) for text in SUNNY]
                rainy_refs = [observe(text, adapter) for text in RAINY]
                final_routes = probe_snapshots[previous_stage]
                edge_universe = sorted(set().union(*(r["edges"] for r in sunny_refs + rainy_refs + list(final_routes.values()))))
                matrix_routes = sunny_refs + rainy_refs + list(final_routes.values())
                matrix = np.asarray([[1.0 if edge in route["edges"] else 0.0 for route in matrix_routes] for edge in edge_universe], dtype=float)

                cluster_sets = {}
                for k in K_VALUES:
                    labels, _ = kmeans(matrix, k, projection_seed + trial * 307 + k)
                    cluster_sets[k] = coassignment(labels)

                final_by_key[(projection_group, trial, order_name)] = {
                    "routes": final_routes,
                    "edge_universe": edge_universe,
                    "matrix": matrix,
                    "cluster_sets": cluster_sets,
                }

                for label, route in final_routes.items():
                    sunny_affinity = float(np.mean([route_overlap(route, ref) for ref in sunny_refs]))
                    rainy_affinity = float(np.mean([route_overlap(route, ref) for ref in rainy_refs]))
                    final_rows.append({
                        "projection_group": projection_group,
                        "trial": trial,
                        "order": order_name,
                        "probe": label,
                        "sunny_affinity": round(sunny_affinity, 8),
                        "rainy_affinity": round(rainy_affinity, 8),
                        "margin": round(sunny_affinity - rainy_affinity, 8),
                        "edge_count": len(route["edges"]),
                    })
                print(f"[{projection_group} 試行{trial}] {order_name} 完了")

    for projection_group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            for order_a, order_b in combinations(ORDERS, 2):
                left = final_by_key[(projection_group, trial, order_a)]
                right = final_by_key[(projection_group, trial, order_b)]
                for probe in PROBES:
                    comparison_rows.append({
                        "projection_group": projection_group,
                        "trial": trial,
                        "order_pair": f"{order_a} vs {order_b}",
                        "metric": "final_route_similarity",
                        "probe": probe,
                        "k": 0,
                        "value": round(route_overlap(left["routes"][probe], right["routes"][probe]), 8),
                    })
                if left["edge_universe"] == right["edge_universe"]:
                    for k in K_VALUES:
                        comparison_rows.append({
                            "projection_group": projection_group,
                            "trial": trial,
                            "order_pair": f"{order_a} vs {order_b}",
                            "metric": "cluster_similarity",
                            "probe": "all",
                            "k": k,
                            "value": round(jaccard(left["cluster_sets"][k], right["cluster_sets"][k]), 8),
                        })

    stage_summary = summarize(stage_rows, ("order", "transition"), ["route_retention", "new_edges", "removed_edges"])
    final_summary = summarize(final_rows, ("order", "probe"), ["sunny_affinity", "rainy_affinity", "margin", "edge_count"])
    order_summary = summarize(comparison_rows, ("metric", "order_pair", "k"), ["value"])

    write_csv(run_dir / "experience_order_stage_trials.csv", stage_rows)
    write_csv(run_dir / "experience_order_stage_summary.csv", stage_summary)
    write_csv(run_dir / "experience_order_final_trials.csv", final_rows)
    write_csv(run_dir / "experience_order_final_summary.csv", final_summary)
    write_csv(run_dir / "experience_order_comparisons.csv", comparison_rows)
    write_csv(run_dir / "experience_order_comparison_summary.csv", order_summary)
    (run_dir / "experience_order_effect.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "orders": ORDERS,
            "stage_rows": stage_rows,
            "final_rows": final_rows,
            "comparison_rows": comparison_rows,
            "stage_summary": stage_summary,
            "final_summary": final_summary,
            "order_summary": order_summary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(run_dir / "experience_order_effect.html", stage_summary, final_summary, order_summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
