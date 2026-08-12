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
PROBES = [
    "よく晴れた穏やかな日だ",
    "雨雲が広がって寒く感じる",
    "今日は天気が変わりやすい",
    "会計ソフトが請求書を処理する",
]
FEATURE_LABELS = [
    "晴れ学習1", "晴れ学習2", "晴れ学習3",
    "雨学習1", "雨学習2", "雨学習3",
    "晴れ未経験", "雨未経験", "曖昧天気", "遠い対照",
]
TRAIN_REPEATS = 20
TRIALS_PER_GROUP = 6
K_VALUES = tuple(range(2, 9))
RESTARTS = 16
STABILITY_RUNS = 8
BASE_SEED = pipeline.PROJECTION_SEED


def configure_data(path: Path, projection_seed: int) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def observe(text: str, adapter: pipeline.OpenAIAdapter) -> set[tuple[int, int]]:
    embedding, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(sources, steps=14, threshold=0.18, noise=0.0, learn=False)
    return {tuple(edge) for edge in result.traversed_edges}


def kmeans(matrix: np.ndarray, k: int, seed: int, restarts: int = RESTARTS) -> tuple[np.ndarray, np.ndarray, float]:
    k = min(k, matrix.shape[0])
    rng = np.random.default_rng(seed)
    best_labels = np.zeros(matrix.shape[0], dtype=int)
    best_centers = np.zeros((k, matrix.shape[1]), dtype=float)
    best_loss = float("inf")
    for _ in range(restarts):
        indices = rng.choice(matrix.shape[0], size=k, replace=False)
        centers = matrix[indices].astype(float).copy()
        labels = np.zeros(matrix.shape[0], dtype=int)
        for iteration in range(100):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if iteration and np.array_equal(new_labels, labels):
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


def silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    n = len(matrix)
    unique = np.unique(labels)
    if n < 3 or len(unique) < 2:
        return 0.0
    distances = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))
    values = []
    for i in range(n):
        own = labels == labels[i]
        own[i] = False
        a = float(distances[i, own].mean()) if own.any() else 0.0
        b_values = [float(distances[i, labels == other].mean()) for other in unique if other != labels[i] and (labels == other).any()]
        b = min(b_values) if b_values else 0.0
        denominator = max(a, b)
        values.append((b - a) / denominator if denominator else 0.0)
    return float(np.mean(values))


def pairwise_agreement(left: np.ndarray, right: np.ndarray) -> float:
    n = len(left)
    if n < 2:
        return 1.0
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            agree += int((left[i] == left[j]) == (right[i] == right[j]))
            total += 1
    return agree / total if total else 1.0


def split_purity(parent: np.ndarray, child: np.ndarray) -> float:
    """How cleanly child clusters refine parent clusters; 1 means no child crosses parents."""
    total = 0
    weighted = 0.0
    for cluster in np.unique(child):
        members = np.flatnonzero(child == cluster)
        if not len(members):
            continue
        counts = np.bincount(parent[members])
        weighted += len(members) * (counts.max() / len(members))
        total += len(members)
    return weighted / total if total else 1.0


def semantic_profile(center: np.ndarray) -> dict:
    sunny = float(np.mean(np.r_[center[0:3], center[6]]))
    rainy = float(np.mean(np.r_[center[3:6], center[7]]))
    ambiguous = float(center[8])
    control = float(center[9])
    return {"sunny": sunny, "rainy": rainy, "ambiguous": ambiguous, "control": control}


def semantic_name(center: np.ndarray) -> str:
    p = semantic_profile(center)
    weather = np.mean([p["sunny"], p["rainy"], p["ambiguous"]])
    if p["control"] >= 0.45 and weather >= 0.40:
        return "汎用型"
    if p["sunny"] >= p["rainy"] + 0.20:
        return "晴れ中心型"
    if p["rainy"] >= p["sunny"] + 0.20:
        return "雨中心型"
    if p["ambiguous"] >= 0.45 and p["control"] < 0.25:
        return "橋渡し型"
    if weather >= 0.50 and p["control"] < 0.25:
        return "天気共通型"
    return "混合補助型"


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


def write_html(path: Path, metric_summary: list[dict], hierarchy_summary: list[dict], type_summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "解像度kごとのシルエット係数",
            "初期値を変えた分割安定度",
            "k→k+1 の分裂純度（高いほど階層的な細分化）",
            "各解像度で現れた機能タイプ構成",
        ),
        vertical_spacing=0.10,
    )

    for group, dash in (("固定射影", "solid"), ("異なる射影", "dash")):
        items = [r for r in metric_summary if r["group"] == group]
        fig.add_trace(go.Scatter(x=[r["k"] for r in items], y=[r["silhouette_mean"] for r in items], error_y={"type":"data","array":[r["silhouette_std"] for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"シルエット/{group}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[r["k"] for r in items], y=[r["stability_mean"]*100 for r in items], error_y={"type":"data","array":[r["stability_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"安定度/{group}"), row=2, col=1)
        hitems = [r for r in hierarchy_summary if r["group"] == group]
        fig.add_trace(go.Scatter(x=[f"{r['parent_k']}→{r['child_k']}" for r in hitems], y=[r["split_purity_mean"]*100 for r in hitems], error_y={"type":"data","array":[r["split_purity_std"]*100 for r in hitems]}, mode="lines+markers", line={"dash":dash}, name=f"分裂純度/{group}"), row=3, col=1)

    names = sorted(set(r["semantic_name"] for r in type_summary))
    for name in names:
        items = [r for r in type_summary if r["semantic_name"] == name]
        fig.add_trace(go.Bar(x=[f"{r['group']} k={r['k']}" for r in items], y=[r["cluster_count_mean"] for r in items], name=name), row=4, col=1)

    fig.update_yaxes(title_text="silhouette", row=1, col=1)
    fig.update_yaxes(title_text="安定度 (%)", row=2, col=1)
    fig.update_yaxes(title_text="純度 (%)", row=3, col=1)
    fig.update_yaxes(title_text="クラスタ数", row=4, col=1)
    fig.update_xaxes(title_text="クラスタ数 k", row=2, col=1)
    fig.update_xaxes(title_text="分裂段階", row=3, col=1)
    fig.update_xaxes(title_text="群・解像度", row=4, col=1)
    fig.update_layout(height=1500, barmode="stack", title="SphereBrain クラスタ分裂の階層と安定性", hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_cluster_hierarchy_stability" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()

    metric_rows: list[dict] = []
    hierarchy_rows: list[dict] = []
    type_rows: list[dict] = []
    all_texts = SUNNY + RAINY + PROBES

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            for text in SUNNY + RAINY:
                pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)

            routes = [observe(text, adapter) for text in all_texts]
            edge_universe = sorted(set().union(*routes))
            matrix = np.asarray([[1.0 if edge in route else 0.0 for route in routes] for edge in edge_universe], dtype=float)
            solutions: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            for k in K_VALUES:
                labels, centers, loss = kmeans(matrix, k, seed + trial * 101 + k)
                solutions[k] = (labels, centers)
                stability_scores = []
                for run in range(STABILITY_RUNS):
                    other_labels, _, _ = kmeans(matrix, k, seed + trial * 10007 + k * 100 + run, restarts=6)
                    stability_scores.append(pairwise_agreement(labels, other_labels))
                metric_rows.append({
                    "group": group,
                    "trial": trial,
                    "projection_seed": seed,
                    "k": k,
                    "edge_count": len(edge_universe),
                    "loss": loss,
                    "silhouette": silhouette(matrix, labels),
                    "stability": float(np.mean(stability_scores)),
                    "stability_std_within": float(np.std(stability_scores, ddof=0)),
                })
                counts = defaultdict(int)
                for center in centers:
                    counts[semantic_name(center)] += 1
                for name, count in counts.items():
                    type_rows.append({"group":group,"trial":trial,"k":k,"semantic_name":name,"cluster_count":count})

            for parent_k, child_k in zip(K_VALUES[:-1], K_VALUES[1:]):
                parent_labels = solutions[parent_k][0]
                child_labels = solutions[child_k][0]
                hierarchy_rows.append({
                    "group": group,
                    "trial": trial,
                    "parent_k": parent_k,
                    "child_k": child_k,
                    "split_purity": split_purity(parent_labels, child_labels),
                    "agreement": pairwise_agreement(parent_labels, child_labels),
                })
            print(f"[{group} 試行{trial}] edges={len(edge_universe)} hierarchy complete")

    metric_summary = summarize(metric_rows, ("group", "k"), ["silhouette", "stability", "loss"])
    hierarchy_summary = summarize(hierarchy_rows, ("group", "parent_k", "child_k"), ["split_purity", "agreement"])
    type_summary = summarize(type_rows, ("group", "k", "semantic_name"), ["cluster_count"])

    write_csv(run_dir / "cluster_hierarchy_metrics.csv", metric_rows)
    write_csv(run_dir / "cluster_hierarchy_summary.csv", metric_summary)
    write_csv(run_dir / "cluster_split_trials.csv", hierarchy_rows)
    write_csv(run_dir / "cluster_split_summary.csv", hierarchy_summary)
    write_csv(run_dir / "cluster_type_trials.csv", type_rows)
    write_csv(run_dir / "cluster_type_summary.csv", type_summary)
    (run_dir / "cluster_hierarchy_stability.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "k_values": K_VALUES,
        "feature_labels": FEATURE_LABELS,
        "metric_rows": metric_rows,
        "hierarchy_rows": hierarchy_rows,
        "type_rows": type_rows,
        "metric_summary": metric_summary,
        "hierarchy_summary": hierarchy_summary,
        "type_summary": type_summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "cluster_hierarchy_stability.html", metric_summary, hierarchy_summary, type_summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
