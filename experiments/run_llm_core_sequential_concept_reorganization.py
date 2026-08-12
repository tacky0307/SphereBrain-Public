from __future__ import annotations

import csv
import json
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
AMBIGUOUS = [
    "今日は天気が変わりやすい",
    "晴れたり雨が降ったりしている",
    "空模様が落ち着かない",
]
PROBES = {
    "晴れ未経験": "よく晴れた穏やかな日だ",
    "雨未経験": "雨雲が広がって寒く感じる",
    "曖昧天気": "晴れるか雨になるか分からない",
    "遠い対照": "会計ソフトが請求書を処理する",
}
STAGES = (
    ("初期", []),
    ("晴れ学習後", SUNNY),
    ("雨追加後", RAINY),
    ("曖昧追加後", AMBIGUOUS),
)
REPEATS = 20
TRIALS_PER_GROUP = 4
K_VALUES = range(2, 9)
BASE_SEED = pipeline.PROJECTION_SEED


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


def kmeans(matrix: np.ndarray, k: int, seed: int, restarts: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_labels = np.zeros(matrix.shape[0], dtype=int)
    best_centers = np.zeros((k, matrix.shape[1]), dtype=float)
    best_loss = float("inf")
    for _ in range(restarts):
        centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].astype(float).copy()
        labels = np.zeros(matrix.shape[0], dtype=int)
        for iteration in range(100):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if iteration and np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = matrix[labels == cluster]
                centers[cluster] = members.mean(axis=0) if len(members) else matrix[rng.integers(0, len(matrix))]
        loss = float(((matrix - centers[labels]) ** 2).sum())
        if loss < best_loss:
            best_loss = loss
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers


def silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.0
    distances = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(matrix)):
        same = labels == labels[i]
        same[i] = False
        a = float(distances[i, same].mean()) if same.any() else 0.0
        b_values = []
        for cluster in set(labels.tolist()):
            if cluster == labels[i]:
                continue
            mask = labels == cluster
            if mask.any():
                b_values.append(float(distances[i, mask].mean()))
        b = min(b_values) if b_values else 0.0
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(scores))


def partition_similarity(old_edges: list[tuple[int, int]], old_labels: np.ndarray, new_edges: list[tuple[int, int]], new_labels: np.ndarray) -> float:
    common = sorted(set(old_edges) & set(new_edges))
    if len(common) < 2:
        return 0.0
    old_index = {edge: i for i, edge in enumerate(old_edges)}
    new_index = {edge: i for i, edge in enumerate(new_edges)}
    agreements = total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            old_same = old_labels[old_index[a]] == old_labels[old_index[b]]
            new_same = new_labels[new_index[a]] == new_labels[new_index[b]]
            agreements += int(old_same == new_same)
            total += 1
    return agreements / total if total else 0.0


def summarize(rows: list[dict], keys: tuple[str, ...], fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
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


def write_html(path: Path, structure_summary: list[dict], affinity_summary: list[dict], transition_summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "経験追加に伴う各解像度のシルエット係数",
            "段階間で既存Edgeのクラスタ関係がどれだけ維持されたか",
            "未経験Probeの晴れ・雨親和差",
            "Edge数と新規Edge数",
        ),
        vertical_spacing=0.09,
    )

    stages = [stage for stage, _ in STAGES]
    for k in K_VALUES:
        items = {r["stage"]: r for r in structure_summary if r["k"] == k}
        x = [s for s in stages if s in items]
        fig.add_trace(go.Scatter(x=x, y=[items[s]["silhouette_mean"] for s in x], error_y={"type":"data","array":[items[s]["silhouette_std"] for s in x]}, mode="lines+markers", name=f"k={k}"), row=1, col=1)

    for k in K_VALUES:
        items = [r for r in transition_summary if r["k"] == k]
        fig.add_trace(go.Scatter(x=[r["transition"] for r in items], y=[r["partition_retention_mean"]*100 for r in items], error_y={"type":"data","array":[r["partition_retention_std"]*100 for r in items]}, mode="lines+markers", name=f"保持 k={k}", showlegend=False), row=2, col=1)

    for label in ("晴れ未経験", "雨未経験", "曖昧天気"):
        items = [r for r in affinity_summary if r["label"] == label]
        fig.add_trace(go.Scatter(x=[r["stage"] for r in items], y=[r["sunny_margin_mean"]*100 for r in items], error_y={"type":"data","array":[r["sunny_margin_std"]*100 for r in items]}, mode="lines+markers", name=f"晴れ差:{label}"), row=3, col=1)

    base = {r["stage"]: r for r in structure_summary if r["k"] == 5}
    x = [s for s in stages if s in base]
    fig.add_trace(go.Bar(x=x, y=[base[s]["edge_count_mean"] for s in x], name="総Edge", showlegend=False), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=[base[s]["new_edge_count_mean"] for s in x], name="新規Edge", showlegend=False), row=4, col=1)

    fig.update_yaxes(title_text="silhouette", row=1, col=1)
    fig.update_yaxes(title_text="保持率 (%)", row=2, col=1)
    fig.update_yaxes(title_text="親和差 (%)", row=3, col=1)
    fig.update_yaxes(title_text="edges", row=4, col=1)
    fig.update_layout(height=1450, title="SphereBrain 経験追加による概念構造の再編成", hovermode="x unified", barmode="group")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_sequential_concept_reorganization" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    structure_rows: list[dict] = []
    affinity_rows: list[dict] = []
    transition_rows: list[dict] = []

    analysis_texts = SUNNY + RAINY + AMBIGUOUS + list(PROBES.values())

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            previous_edges: list[tuple[int, int]] | None = None
            previous_labels_by_k: dict[int, np.ndarray] = {}

            for stage_index, (stage_name, new_experiences) in enumerate(STAGES):
                for text in new_experiences:
                    pipeline.experience(text, repeats=REPEATS, adapter=adapter)

                routes = [observe(text, adapter) for text in analysis_texts]
                edge_universe = sorted(set().union(*(route["edges"] for route in routes)))
                matrix = np.asarray([[1.0 if edge in route["edges"] else 0.0 for route in routes] for edge in edge_universe], dtype=float)
                new_edge_count = len(set(edge_universe) - set(previous_edges or []))

                labels_by_k: dict[int, np.ndarray] = {}
                for k in K_VALUES:
                    labels, _ = kmeans(matrix, k, seed + stage_index * 997 + k * 37)
                    labels_by_k[k] = labels
                    structure_rows.append({
                        "group": group,
                        "trial": trial,
                        "stage": stage_name,
                        "stage_index": stage_index,
                        "k": k,
                        "silhouette": round(silhouette(matrix, labels), 8),
                        "edge_count": len(edge_universe),
                        "new_edge_count": new_edge_count,
                    })
                    if previous_edges is not None and k in previous_labels_by_k:
                        transition_rows.append({
                            "group": group,
                            "trial": trial,
                            "transition": f"{STAGES[stage_index-1][0]}→{stage_name}",
                            "k": k,
                            "partition_retention": round(partition_similarity(previous_edges, previous_labels_by_k[k], edge_universe, labels), 8),
                        })

                sunny_refs = routes[0:3]
                rainy_refs = routes[3:6]
                for probe_index, label in enumerate(PROBES):
                    route = routes[9 + probe_index]
                    sunny_affinity = float(np.mean([route_overlap(route, ref) for ref in sunny_refs]))
                    rainy_affinity = float(np.mean([route_overlap(route, ref) for ref in rainy_refs]))
                    affinity_rows.append({
                        "group": group,
                        "trial": trial,
                        "stage": stage_name,
                        "stage_index": stage_index,
                        "label": label,
                        "sunny_affinity": round(sunny_affinity, 8),
                        "rainy_affinity": round(rainy_affinity, 8),
                        "sunny_margin": round(sunny_affinity - rainy_affinity, 8),
                    })

                previous_edges = edge_universe
                previous_labels_by_k = labels_by_k
                print(f"[{group} 試行{trial}] {stage_name}: edges={len(edge_universe)}")

    structure_summary = summarize(structure_rows, ("stage", "stage_index", "k"), ["silhouette", "edge_count", "new_edge_count"])
    affinity_summary = summarize(affinity_rows, ("stage", "stage_index", "label"), ["sunny_affinity", "rainy_affinity", "sunny_margin"])
    transition_summary = summarize(transition_rows, ("transition", "k"), ["partition_retention"])

    write_csv(run_dir / "sequential_structure_trials.csv", structure_rows)
    write_csv(run_dir / "sequential_structure_summary.csv", structure_summary)
    write_csv(run_dir / "sequential_affinity_trials.csv", affinity_rows)
    write_csv(run_dir / "sequential_affinity_summary.csv", affinity_summary)
    write_csv(run_dir / "sequential_transition_trials.csv", transition_rows)
    write_csv(run_dir / "sequential_transition_summary.csv", transition_summary)
    (run_dir / "sequential_concept_reorganization.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stages": [name for name, _ in STAGES],
        "sunny": SUNNY,
        "rainy": RAINY,
        "ambiguous": AMBIGUOUS,
        "probes": PROBES,
        "structure_rows": structure_rows,
        "affinity_rows": affinity_rows,
        "transition_rows": transition_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "sequential_concept_reorganization.html", structure_summary, affinity_summary, transition_summary)

    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
