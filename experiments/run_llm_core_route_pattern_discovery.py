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
PROBES = {
    "晴れ未経験": "よく晴れた穏やかな日だ",
    "雨未経験": "雨雲が広がって寒く感じる",
    "曖昧天気": "今日は天気が変わりやすい",
    "遠い対照": "会計ソフトが請求書を処理する",
}
TRAIN_REPEATS = 20
TRIALS_PER_GROUP = 5
BASE_SEED = pipeline.PROJECTION_SEED
CLUSTERS = 5
FEATURE_LABELS = [
    "晴れ学習1", "晴れ学習2", "晴れ学習3",
    "雨学習1", "雨学習2", "雨学習3",
    "晴れ未経験", "雨未経験", "曖昧天気", "遠い対照",
]


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


def kmeans_binary(matrix: np.ndarray, k: int, seed: int, restarts: int = 12) -> tuple[np.ndarray, np.ndarray, float]:
    if matrix.shape[0] < k:
        k = matrix.shape[0]
    best_labels = np.zeros(matrix.shape[0], dtype=int)
    best_centers = np.zeros((k, matrix.shape[1]), dtype=float)
    best_loss = float("inf")
    rng = np.random.default_rng(seed)

    for _ in range(restarts):
        indices = rng.choice(matrix.shape[0], size=k, replace=False)
        centers = matrix[indices].astype(float).copy()
        labels = np.zeros(matrix.shape[0], dtype=int)
        for _iteration in range(100):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _iteration > 0:
                break
            labels = new_labels
            for cluster in range(k):
                members = matrix[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
                else:
                    centers[cluster] = matrix[rng.integers(0, matrix.shape[0])]
        loss = float(((matrix - centers[labels]) ** 2).sum())
        if loss < best_loss:
            best_loss = loss
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_loss


def semantic_name(center: np.ndarray) -> str:
    sunny_train = float(center[0:3].mean())
    rainy_train = float(center[3:6].mean())
    sunny_probe = float(center[6])
    rainy_probe = float(center[7])
    ambiguous = float(center[8])
    control = float(center[9])
    weather = np.mean([sunny_train, rainy_train, sunny_probe, rainy_probe, ambiguous])

    if control >= 0.45 and weather >= 0.45:
        return "汎用クラスタ"
    if weather >= 0.55 and control < 0.25 and abs(sunny_train - rainy_train) < 0.25:
        return "天気共通クラスタ"
    if sunny_train + sunny_probe >= rainy_train + rainy_probe + 0.55:
        return "晴れ中心クラスタ"
    if rainy_train + rainy_probe >= sunny_train + sunny_probe + 0.55:
        return "雨中心クラスタ"
    if ambiguous >= 0.45 and control < 0.25:
        return "曖昧・橋渡しクラスタ"
    return "混合・補助クラスタ"


def cut_edges_and_observe(edges: list[tuple[int, int]], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
    brain = pipeline.load_brain()
    backup = []
    for a, b in edges:
        backup.append((a, b, float(brain.weights[a, b]), float(brain.weights[b, a]), bool(brain.adjacency[a, b]), bool(brain.adjacency[b, a])))
        brain.weights[a, b] = brain.weights[b, a] = 0.0
        brain.adjacency[a, b] = brain.adjacency[b, a] = False
    brain.save(pipeline.BRAIN_FILE)
    try:
        return {label: observe(text, adapter) for label, text in PROBES.items()}
    finally:
        restored = pipeline.load_brain()
        for a, b, wab, wba, aab, aba in backup:
            restored.weights[a, b] = wab
            restored.weights[b, a] = wba
            restored.adjacency[a, b] = aab
            restored.adjacency[b, a] = aba
        restored.save(pipeline.BRAIN_FILE)


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


def write_html(path: Path, cluster_summary: list[dict], ablation_summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "発見されたEdgeクラスタの平均利用パターン",
            "クラスタ別Edge本数",
            "クラスタ切断後の未経験Probe経路保持率",
            "クラスタ切断後の晴れ親和差・雨親和差",
        ),
        vertical_spacing=0.10,
    )

    names = sorted(set(r["semantic_name"] for r in cluster_summary))
    for name in names:
        items = [r for r in cluster_summary if r["semantic_name"] == name]
        if not items:
            continue
        values = []
        errors = []
        for index in range(len(FEATURE_LABELS)):
            field = f"feature_{index}"
            values.append(float(np.mean([r[f"{field}_mean"] for r in items])) * 100)
            errors.append(float(np.mean([r[f"{field}_std"] for r in items])) * 100)
        figure.add_trace(go.Scatter(x=FEATURE_LABELS, y=values, error_y={"type":"data","array":errors}, mode="lines+markers", name=name), row=1, col=1)

    bar_x = [f"{r['group']} / {r['semantic_name']}" for r in cluster_summary]
    figure.add_trace(go.Bar(x=bar_x, y=[r["edge_count_mean"] for r in cluster_summary], error_y={"type":"data","array":[r["edge_count_std"] for r in cluster_summary]}, name="Edge本数", showlegend=False), row=2, col=1)

    conditions = ["無処置"] + names
    for label in PROBES:
        items = {r["condition"]: r for r in ablation_summary if r["label"] == label}
        x = [c for c in conditions if c in items]
        figure.add_trace(go.Scatter(x=x, y=[items[c]["retention_mean"]*100 for c in x], error_y={"type":"data","array":[items[c]["retention_std"]*100 for c in x]}, mode="lines+markers", name=f"保持率: {label}"), row=3, col=1)

    base_items = {r["condition"]: r for r in ablation_summary if r["label"] == "晴れ未経験"}
    x = [c for c in conditions if c in base_items]
    figure.add_trace(go.Scatter(x=x, y=[base_items[c]["sunny_margin_mean"]*100 for c in x], error_y={"type":"data","array":[base_items[c]["sunny_margin_std"]*100 for c in x]}, mode="lines+markers", name="晴れ親和差"), row=4, col=1)
    figure.add_trace(go.Scatter(x=x, y=[base_items[c]["rainy_margin_mean"]*100 for c in x], error_y={"type":"data","array":[base_items[c]["rainy_margin_std"]*100 for c in x]}, mode="lines+markers", name="雨親和差"), row=4, col=1)

    figure.update_yaxes(title_text="利用率 (%)", row=1, col=1)
    figure.update_yaxes(title_text="edges", row=2, col=1)
    figure.update_yaxes(title_text="保持率 (%)", row=3, col=1)
    figure.update_yaxes(title_text="親和差 (%)", row=4, col=1)
    figure.update_xaxes(title_text="入力", row=1, col=1)
    figure.update_xaxes(title_text="クラスタ", row=2, col=1)
    figure.update_xaxes(title_text="切断条件", row=4, col=1)
    figure.update_layout(height=1550, title="SphereBrain Core自身の経路利用パターンから概念構造を発見", hovermode="x unified")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_route_pattern_discovery" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    cluster_rows: list[dict] = []
    edge_rows: list[dict] = []
    ablation_rows: list[dict] = []

    all_texts = SUNNY + RAINY + list(PROBES.values())

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            for text in SUNNY + RAINY:
                pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)

            routes = [observe(text, adapter) for text in all_texts]
            edge_universe = sorted(set().union(*(route["edges"] for route in routes)))
            matrix = np.asarray([[1.0 if edge in route["edges"] else 0.0 for route in routes] for edge in edge_universe], dtype=float)
            labels, centers, loss = kmeans_binary(matrix, CLUSTERS, seed + trial * 313)
            baseline_probes = {label: routes[6 + index] for index, label in enumerate(PROBES)}
            sunny_centroid_routes = routes[0:3]
            rainy_centroid_routes = routes[3:6]

            cluster_edges_by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for cluster_id in range(centers.shape[0]):
                name = semantic_name(centers[cluster_id])
                members = [edge_universe[i] for i in range(len(edge_universe)) if labels[i] == cluster_id]
                cluster_edges_by_name[name].extend(members)
                record = {
                    "group": group,
                    "trial": trial,
                    "cluster_id": cluster_id,
                    "semantic_name": name,
                    "edge_count": len(members),
                    "loss": loss,
                }
                for index, value in enumerate(centers[cluster_id]):
                    record[f"feature_{index}"] = float(value)
                cluster_rows.append(record)
                for edge in members:
                    edge_rows.append({
                        "group": group,
                        "trial": trial,
                        "cluster_id": cluster_id,
                        "semantic_name": name,
                        "edge_a": edge[0],
                        "edge_b": edge[1],
                        "usage_pattern": "".join(str(int(v)) for v in matrix[edge_universe.index(edge)]),
                    })

            conditions = {"無処置": []}
            conditions.update({name: sorted(set(edges)) for name, edges in cluster_edges_by_name.items()})
            for condition, edges in conditions.items():
                current = baseline_probes if condition == "無処置" else cut_edges_and_observe(edges, adapter)
                sunny_affinity = {label: float(np.mean([route_overlap(current[label], route) for route in sunny_centroid_routes])) for label in PROBES}
                rainy_affinity = {label: float(np.mean([route_overlap(current[label], route) for route in rainy_centroid_routes])) for label in PROBES}
                for label in PROBES:
                    ablation_rows.append({
                        "group": group,
                        "trial": trial,
                        "condition": condition,
                        "label": label,
                        "cut_edges": len(edges),
                        "retention": round(retention(baseline_probes[label], current[label]), 8),
                        "sunny_affinity": round(sunny_affinity[label], 8),
                        "rainy_affinity": round(rainy_affinity[label], 8),
                        "sunny_margin": round(sunny_affinity["晴れ未経験"] - rainy_affinity["晴れ未経験"], 8),
                        "rainy_margin": round(rainy_affinity["雨未経験"] - sunny_affinity["雨未経験"], 8),
                    })
            print(f"[{group} 試行{trial}] Edge={len(edge_universe)} clusters={len(cluster_edges_by_name)}")

    cluster_summary = summarize(
        cluster_rows,
        ("group", "semantic_name"),
        ["edge_count"] + [f"feature_{i}" for i in range(len(FEATURE_LABELS))],
    )
    ablation_summary = summarize(
        ablation_rows,
        ("condition", "label"),
        ["retention", "sunny_affinity", "rainy_affinity", "sunny_margin", "rainy_margin", "cut_edges"],
    )

    write_csv(run_dir / "route_pattern_clusters.csv", cluster_rows)
    write_csv(run_dir / "route_pattern_edges.csv", edge_rows)
    write_csv(run_dir / "route_pattern_ablation_trials.csv", ablation_rows)
    write_csv(run_dir / "route_pattern_cluster_summary.csv", cluster_summary)
    write_csv(run_dir / "route_pattern_ablation_summary.csv", ablation_summary)
    (run_dir / "route_pattern_discovery.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sunny": SUNNY,
        "rainy": RAINY,
        "probes": PROBES,
        "feature_labels": FEATURE_LABELS,
        "clusters": CLUSTERS,
        "cluster_rows": cluster_rows,
        "ablation_rows": ablation_rows,
        "cluster_summary": cluster_summary,
        "ablation_summary": ablation_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "route_pattern_discovery.html", cluster_summary, ablation_summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
