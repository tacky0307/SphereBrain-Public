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
FUNCTION_TYPES = [
    "天気共通クラスタ",
    "晴れ中心クラスタ",
    "雨中心クラスタ",
    "曖昧・橋渡しクラスタ",
    "混合・補助クラスタ",
]
TRAIN_REPEATS = 20
TRIALS_PER_GROUP = 8
CLUSTERS = 5
BASE_SEED = pipeline.PROJECTION_SEED


def configure_data(path: Path, seed: int) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = seed


def observe(text: str, adapter: pipeline.OpenAIAdapter) -> dict:
    _, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(sources, steps=14, threshold=0.18, noise=0.0, learn=False)
    return {
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
    }


def retention(reference: dict, current: dict) -> float:
    node = len(reference["nodes"] & current["nodes"]) / len(reference["nodes"]) if reference["nodes"] else 1.0
    edge = len(reference["edges"] & current["edges"]) / len(reference["edges"]) if reference["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def kmeans_binary(matrix: np.ndarray, k: int, seed: int, restarts: int = 16) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, matrix.shape[0])
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_labels = np.zeros(matrix.shape[0], dtype=int)
    best_centers = np.zeros((k, matrix.shape[1]), dtype=float)
    for _ in range(restarts):
        centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].astype(float).copy()
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
    return best_labels, best_centers


def semantic_name(center: np.ndarray) -> str:
    sunny_train = float(center[0:3].mean())
    rainy_train = float(center[3:6].mean())
    sunny_probe = float(center[6])
    rainy_probe = float(center[7])
    ambiguous = float(center[8])
    control = float(center[9])
    weather = float(np.mean([sunny_train, rainy_train, sunny_probe, rainy_probe, ambiguous]))
    if weather >= 0.55 and control < 0.25 and abs(sunny_train - rainy_train) < 0.25:
        return "天気共通クラスタ"
    if sunny_train + sunny_probe >= rainy_train + rainy_probe + 0.55:
        return "晴れ中心クラスタ"
    if rainy_train + rainy_probe >= sunny_train + sunny_probe + 0.55:
        return "雨中心クラスタ"
    if ambiguous >= 0.45 and control < 0.25:
        return "曖昧・橋渡しクラスタ"
    return "混合・補助クラスタ"


def cut_and_observe(edges: list[tuple[int, int]], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["function_type"])].append(row)
    output = []
    for group in ("固定射影", "異なる射影"):
        for function_type in FUNCTION_TYPES:
            items = grouped.get((group, function_type), [])
            present = [r for r in items if r["present"]]
            rec = {
                "group": group,
                "function_type": function_type,
                "trials": TRIALS_PER_GROUP,
                "present_trials": len(present),
                "occurrence_rate": len(present) / TRIALS_PER_GROUP,
            }
            for field in ["edge_count", "centroid_similarity", "signature_similarity", "role_match"]:
                values = np.asarray([r[field] for r in present], dtype=float) if present else np.asarray([], dtype=float)
                rec[f"{field}_mean"] = float(values.mean()) if len(values) else 0.0
                rec[f"{field}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
            for index in range(len(FEATURE_LABELS)):
                values = np.asarray([r[f"feature_{index}"] for r in present], dtype=float) if present else np.asarray([], dtype=float)
                rec[f"feature_{index}_mean"] = float(values.mean()) if len(values) else 0.0
            output.append(rec)
    return output


def write_html(path: Path, summary: list[dict], rows: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "機能タイプの再現率",
            "同じ機能タイプ内の利用パターン類似度",
            "同じ機能タイプ内のアブレーション機能類似度",
            "平均利用パターン（固定射影と異なる射影）",
        ),
        vertical_spacing=0.10,
    )
    for group in ("固定射影", "異なる射影"):
        items = [r for r in summary if r["group"] == group]
        fig.add_trace(go.Bar(x=[r["function_type"] for r in items], y=[r["occurrence_rate"]*100 for r in items], name=f"再現率/{group}"), row=1, col=1)
        fig.add_trace(go.Bar(x=[r["function_type"] for r in items], y=[r["centroid_similarity_mean"]*100 for r in items], error_y={"type":"data","array":[r["centroid_similarity_std"]*100 for r in items]}, name=f"利用類似/{group}"), row=2, col=1)
        fig.add_trace(go.Bar(x=[r["function_type"] for r in items], y=[r["signature_similarity_mean"]*100 for r in items], error_y={"type":"data","array":[r["signature_similarity_std"]*100 for r in items]}, name=f"機能類似/{group}"), row=3, col=1)
        dash = "solid" if group == "固定射影" else "dash"
        for function_type in FUNCTION_TYPES:
            item = next(r for r in items if r["function_type"] == function_type)
            fig.add_trace(go.Scatter(x=FEATURE_LABELS, y=[item[f"feature_{i}_mean"]*100 for i in range(len(FEATURE_LABELS))], mode="lines+markers", line={"dash":dash}, name=f"{function_type}/{group}"), row=4, col=1)
    fig.update_yaxes(title_text="出現率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="類似度 (%)", row=2, col=1)
    fig.update_yaxes(title_text="類似度 (%)", row=3, col=1)
    fig.update_yaxes(title_text="利用率 (%)", row=4, col=1)
    fig.update_xaxes(title_text="入力", row=4, col=1)
    fig.update_layout(height=1500, title=f"SphereBrain 機能パターンの再現性（各群{TRIALS_PER_GROUP}試行）", barmode="group")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_function_pattern_reproducibility" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    all_texts = SUNNY + RAINY + list(PROBES.values())
    raw_rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        trial_records: list[dict] = []
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            for text in SUNNY + RAINY:
                pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)
            routes = [observe(text, adapter) for text in all_texts]
            universe = sorted(set().union(*(r["edges"] for r in routes)))
            matrix = np.asarray([[1.0 if edge in route["edges"] else 0.0 for route in routes] for edge in universe], dtype=float)
            labels, centers = kmeans_binary(matrix, CLUSTERS, seed + trial * 313)
            baseline = {label: routes[6+i] for i, label in enumerate(PROBES)}

            by_type: dict[str, list[tuple[int, int]]] = defaultdict(list)
            center_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
            for cluster_id, center in enumerate(centers):
                name = semantic_name(center)
                members = [universe[i] for i in range(len(universe)) if labels[i] == cluster_id]
                by_type[name].extend(members)
                center_by_type[name].append(center)

            for function_type in FUNCTION_TYPES:
                edges = sorted(set(by_type.get(function_type, [])))
                if edges:
                    centroid = np.mean(center_by_type[function_type], axis=0)
                    current = cut_and_observe(edges, adapter)
                    signature = np.asarray([1.0 - retention(baseline[label], current[label]) for label in PROBES], dtype=float)
                    expected_index = {"晴れ中心クラスタ":0, "雨中心クラスタ":1, "曖昧・橋渡しクラスタ":2}.get(function_type)
                    role_match = 1.0 if expected_index is None or int(np.argmax(signature)) == expected_index else 0.0
                    trial_records.append({"trial":trial,"function_type":function_type,"present":1,"edge_count":len(edges),"centroid":centroid,"signature":signature,"role_match":role_match})
                else:
                    trial_records.append({"trial":trial,"function_type":function_type,"present":0,"edge_count":0,"centroid":np.zeros(len(FEATURE_LABELS)),"signature":np.zeros(len(PROBES)),"role_match":0.0})
            print(f"[{group} 試行{trial}] 完了")

        for function_type in FUNCTION_TYPES:
            present = [r for r in trial_records if r["function_type"] == function_type and r["present"]]
            mean_centroid = np.mean([r["centroid"] for r in present], axis=0) if present else np.zeros(len(FEATURE_LABELS))
            mean_signature = np.mean([r["signature"] for r in present], axis=0) if present else np.zeros(len(PROBES))
            for record in [r for r in trial_records if r["function_type"] == function_type]:
                row = {
                    "group":group,"trial":record["trial"],"function_type":function_type,
                    "present":record["present"],"edge_count":record["edge_count"],
                    "centroid_similarity":cosine(record["centroid"], mean_centroid) if record["present"] else 0.0,
                    "signature_similarity":cosine(record["signature"], mean_signature) if record["present"] else 0.0,
                    "role_match":record["role_match"],
                }
                for i, value in enumerate(record["centroid"]): row[f"feature_{i}"] = float(value)
                for i, value in enumerate(record["signature"]): row[f"drop_{i}"] = float(value)
                raw_rows.append(row)

    summary = summarize(raw_rows)
    write_csv(run_dir / "functional_pattern_trials.csv", raw_rows)
    write_csv(run_dir / "functional_pattern_summary.csv", summary)
    (run_dir / "functional_pattern_reproducibility.json").write_text(json.dumps({"created_at":datetime.now().isoformat(timespec="seconds"),"trials_per_group":TRIALS_PER_GROUP,"feature_labels":FEATURE_LABELS,"probe_labels":list(PROBES),"rows":raw_rows,"summary":summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "functional_pattern_reproducibility.html", summary, raw_rows)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
