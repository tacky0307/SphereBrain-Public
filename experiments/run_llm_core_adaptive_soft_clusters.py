from __future__ import annotations

import csv
import json
import math
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
FEATURE_LABELS = [
    "晴れ学習1", "晴れ学習2", "晴れ学習3",
    "雨学習1", "雨学習2", "雨学習3",
    "晴れ未経験", "雨未経験", "曖昧天気", "遠い対照",
]
TRAIN_REPEATS = 20
TRIALS_PER_GROUP = 6
MIN_K = 2
MAX_K = 8
BASE_SEED = pipeline.PROJECTION_SEED
SOFT_TEMPERATURE = 0.35
SECONDARY_THRESHOLD = 0.22


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


def kmeans(matrix: np.ndarray, k: int, seed: int, restarts: int = 16) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_labels = np.zeros(matrix.shape[0], dtype=int)
    best_centers = np.zeros((k, matrix.shape[1]), dtype=float)

    for _ in range(restarts):
        centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].astype(float).copy()
        labels = np.full(matrix.shape[0], -1, dtype=int)
        for _iteration in range(100):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = matrix[labels == cluster]
                centers[cluster] = members.mean(axis=0) if len(members) else matrix[rng.integers(0, matrix.shape[0])]
        loss = float(((matrix - centers[labels]) ** 2).sum())
        if loss < best_loss:
            best_loss = loss
            best_labels = labels.copy()
            best_centers = centers.copy()
    return best_labels, best_centers, best_loss


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2 or matrix.shape[0] < 3:
        return -1.0
    distances = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for index in range(matrix.shape[0]):
        same = np.flatnonzero(labels == labels[index])
        same = same[same != index]
        a = float(distances[index, same].mean()) if same.size else 0.0
        b_values = []
        for other in sorted(set(labels.tolist())):
            if other == labels[index]:
                continue
            members = np.flatnonzero(labels == other)
            if members.size:
                b_values.append(float(distances[index, members].mean()))
        b = min(b_values) if b_values else 0.0
        denominator = max(a, b)
        scores.append((b - a) / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores))


def choose_k(matrix: np.ndarray, seed: int) -> tuple[int, list[dict], np.ndarray, np.ndarray]:
    candidates = []
    best = None
    upper = min(MAX_K, max(MIN_K, matrix.shape[0] - 1))
    for k in range(MIN_K, upper + 1):
        labels, centers, loss = kmeans(matrix, k, seed + k * 977)
        silhouette = silhouette_score(matrix, labels)
        cluster_sizes = np.bincount(labels, minlength=k)
        tiny_penalty = float(np.mean(cluster_sizes <= 2)) * 0.12
        complexity_penalty = (k - MIN_K) * 0.012
        selection_score = silhouette - tiny_penalty - complexity_penalty
        record = {
            "k": k,
            "silhouette": silhouette,
            "loss": loss,
            "tiny_cluster_fraction": float(np.mean(cluster_sizes <= 2)),
            "selection_score": selection_score,
        }
        candidates.append(record)
        if best is None or selection_score > best[0]:
            best = (selection_score, k, labels, centers)
    assert best is not None
    return best[1], candidates, best[2], best[3]


def soft_membership(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sqrt(((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    logits = -distances / max(SOFT_TEMPERATURE, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def semantic_profile(center: np.ndarray) -> dict:
    sunny = float(np.mean(np.r_[center[0:3], center[6]]))
    rainy = float(np.mean(np.r_[center[3:6], center[7]]))
    ambiguous = float(center[8])
    control = float(center[9])
    weather = float(np.mean(center[:9]))
    return {
        "sunny": sunny,
        "rainy": rainy,
        "ambiguous": ambiguous,
        "control": control,
        "weather": weather,
    }


def semantic_name(center: np.ndarray) -> str:
    p = semantic_profile(center)
    if p["control"] >= 0.40 and p["weather"] >= 0.35:
        return "汎用型"
    if p["sunny"] >= p["rainy"] + 0.18:
        return "晴れ中心型"
    if p["rainy"] >= p["sunny"] + 0.18:
        return "雨中心型"
    if p["ambiguous"] >= 0.48 and p["control"] < 0.25:
        return "橋渡し型"
    if p["weather"] >= 0.45 and p["control"] < 0.25:
        return "天気共通型"
    return "混合補助型"


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


def retention(reference: dict, current: dict) -> float:
    node = len(reference["nodes"] & current["nodes"]) / len(reference["nodes"]) if reference["nodes"] else 1.0
    edge = len(reference["edges"] & current["edges"]) / len(reference["edges"]) if reference["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def summarize(rows: list[dict], keys: tuple[str, ...], fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        record = {name: value for name, value in zip(keys, key)}
        record["n"] = len(items)
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


def write_html(path: Path, trial_rows: list[dict], cluster_summary: list[dict], ablation_summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=(
            "Coreが選んだクラスタ数",
            "候補クラスタ数ごとの選択スコア",
            "機能タイプ別の平均利用パターン",
            "Edgeの重複所属率と所属エントロピー",
            "機能タイプ切断後の未経験Probe保持率",
        ),
        vertical_spacing=0.08,
    )

    groups = ("固定射影", "異なる射影")
    for group in groups:
        items = [r for r in trial_rows if r["group"] == group and r["record_type"] == "trial"]
        figure.add_trace(go.Histogram(x=[r["selected_k"] for r in items], name=f"選択k/{group}", opacity=0.65), row=1, col=1)
        score_items = [r for r in trial_rows if r["group"] == group and r["record_type"] == "candidate"]
        for k in range(MIN_K, MAX_K + 1):
            values = [r["selection_score"] for r in score_items if r["candidate_k"] == k]
            if values:
                figure.add_trace(go.Box(y=values, x=[str(k)] * len(values), name=f"k={k}/{group}", boxmean=True, showlegend=False), row=2, col=1)

    names = sorted(set(r["semantic_name"] for r in cluster_summary))
    for name in names:
        items = [r for r in cluster_summary if r["semantic_name"] == name]
        y = [float(np.mean([r[f"feature_{i}_mean"] for r in items])) * 100 for i in range(len(FEATURE_LABELS))]
        figure.add_trace(go.Scatter(x=FEATURE_LABELS, y=y, mode="lines+markers", name=name), row=3, col=1)

    overlap_items = [r for r in trial_rows if r["record_type"] == "trial"]
    figure.add_trace(go.Bar(x=[f"{r['group']}-{r['trial']}" for r in overlap_items], y=[r["multi_membership_rate"] * 100 for r in overlap_items], name="重複所属率"), row=4, col=1)
    figure.add_trace(go.Scatter(x=[f"{r['group']}-{r['trial']}" for r in overlap_items], y=[r["membership_entropy"] * 100 for r in overlap_items], mode="lines+markers", name="所属エントロピー"), row=4, col=1)

    for label in PROBES:
        items = [r for r in ablation_summary if r["label"] == label]
        figure.add_trace(go.Scatter(x=[r["condition"] for r in items], y=[r["retention_mean"] * 100 for r in items], error_y={"type":"data","array":[r["retention_std"]*100 for r in items]}, mode="lines+markers", name=f"保持率:{label}"), row=5, col=1)

    figure.update_yaxes(title_text="試行数", row=1, col=1)
    figure.update_yaxes(title_text="選択スコア", row=2, col=1)
    figure.update_yaxes(title_text="利用率 (%)", row=3, col=1)
    figure.update_yaxes(title_text="率 / entropy (%)", row=4, col=1)
    figure.update_yaxes(title_text="保持率 (%)", row=5, col=1)
    figure.update_xaxes(title_text="選択クラスタ数", row=1, col=1)
    figure.update_xaxes(title_text="候補k", row=2, col=1)
    figure.update_xaxes(title_text="入力", row=3, col=1)
    figure.update_xaxes(title_text="試行", row=4, col=1)
    figure.update_xaxes(title_text="切断条件", row=5, col=1)
    figure.update_layout(height=1850, title="SphereBrain 適応的クラスタ数＋重複所属による柔らかい機能発見", hovermode="x unified", barmode="overlay")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_adaptive_soft_clusters" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()

    trial_rows: list[dict] = []
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
            selected_k, candidates, labels, centers = choose_k(matrix, seed + trial * 313)
            memberships = soft_membership(matrix, centers)
            entropy = -np.sum(memberships * np.log(np.clip(memberships, 1e-12, 1.0)), axis=1) / math.log(max(selected_k, 2))
            multi_rate = float(np.mean(np.sum(memberships >= SECONDARY_THRESHOLD, axis=1) >= 2))

            trial_rows.append({
                "record_type": "trial", "group": group, "trial": trial,
                "selected_k": selected_k, "candidate_k": 0, "selection_score": 0.0,
                "silhouette": 0.0, "loss": 0.0,
                "multi_membership_rate": multi_rate,
                "membership_entropy": float(entropy.mean()),
                "edge_total": len(edge_universe),
            })
            for candidate in candidates:
                trial_rows.append({
                    "record_type": "candidate", "group": group, "trial": trial,
                    "selected_k": selected_k, "candidate_k": candidate["k"],
                    "selection_score": candidate["selection_score"],
                    "silhouette": candidate["silhouette"], "loss": candidate["loss"],
                    "multi_membership_rate": multi_rate,
                    "membership_entropy": float(entropy.mean()),
                    "edge_total": len(edge_universe),
                })

            baseline_probes = {label: routes[6 + index] for index, label in enumerate(PROBES)}
            type_edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for cluster_id in range(selected_k):
                name = semantic_name(centers[cluster_id])
                hard_members = np.flatnonzero(labels == cluster_id)
                record = {
                    "group": group, "trial": trial, "cluster_id": cluster_id,
                    "semantic_name": name, "edge_count": int(hard_members.size),
                    "mean_primary_membership": float(np.mean(memberships[hard_members, cluster_id])) if hard_members.size else 0.0,
                }
                for i, value in enumerate(centers[cluster_id]):
                    record[f"feature_{i}"] = float(value)
                cluster_rows.append(record)

            for edge_index, edge in enumerate(edge_universe):
                primary = int(np.argmax(memberships[edge_index]))
                primary_name = semantic_name(centers[primary])
                secondary_ids = [i for i, value in enumerate(memberships[edge_index]) if value >= SECONDARY_THRESHOLD and i != primary]
                secondary_names = sorted({semantic_name(centers[i]) for i in secondary_ids})
                type_edges[primary_name].append(edge)
                edge_rows.append({
                    "group": group, "trial": trial, "edge_a": edge[0], "edge_b": edge[1],
                    "primary_cluster": primary, "primary_type": primary_name,
                    "primary_membership": float(memberships[edge_index, primary]),
                    "secondary_types": "|".join(secondary_names),
                    "secondary_count": len(secondary_names),
                    "membership_entropy": float(entropy[edge_index]),
                    "usage_pattern": "".join(str(int(v)) for v in matrix[edge_index]),
                })

            conditions = {"無処置": []}
            conditions.update({name: sorted(set(edges)) for name, edges in type_edges.items()})
            for condition, edges in conditions.items():
                current = baseline_probes if condition == "無処置" else cut_edges_and_observe(edges, adapter)
                for label in PROBES:
                    ablation_rows.append({
                        "group": group, "trial": trial, "condition": condition,
                        "label": label, "cut_edges": len(edges),
                        "retention": retention(baseline_probes[label], current[label]),
                    })
            print(f"[{group} 試行{trial}] selected_k={selected_k} edges={len(edge_universe)} overlap={multi_rate*100:.1f}%")

    cluster_summary = summarize(cluster_rows, ("group", "semantic_name"), ["edge_count", "mean_primary_membership"] + [f"feature_{i}" for i in range(len(FEATURE_LABELS))])
    ablation_summary = summarize(ablation_rows, ("condition", "label"), ["retention", "cut_edges"])

    write_csv(run_dir / "adaptive_cluster_trials.csv", trial_rows)
    write_csv(run_dir / "adaptive_cluster_summary.csv", cluster_summary)
    write_csv(run_dir / "adaptive_soft_edge_memberships.csv", edge_rows)
    write_csv(run_dir / "adaptive_cluster_ablation_trials.csv", ablation_rows)
    write_csv(run_dir / "adaptive_cluster_ablation_summary.csv", ablation_summary)
    (run_dir / "adaptive_soft_clusters.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "k_range": [MIN_K, MAX_K],
        "soft_temperature": SOFT_TEMPERATURE,
        "secondary_threshold": SECONDARY_THRESHOLD,
        "trial_rows": trial_rows,
        "cluster_rows": cluster_rows,
        "edge_rows": edge_rows,
        "ablation_rows": ablation_rows,
        "cluster_summary": cluster_summary,
        "ablation_summary": ablation_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "adaptive_soft_clusters.html", trial_rows, cluster_summary, ablation_summary)

    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
