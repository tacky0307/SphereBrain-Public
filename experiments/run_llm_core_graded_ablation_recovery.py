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

TRAIN_TEXT = "今日は晴れて気持ちいい"
TEXTS = {
    "学習文": TRAIN_TEXT,
    "類似文": "今日の天気は最高だ",
    "無関係文": "犬は公園を走っている",
}
FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
TRIALS_PER_GROUP = 5
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
        "node_count": len(result.activated_nodes),
        "edge_count": len(result.traversed_edges),
    }


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def route_overlap(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def retention(reference: dict, current: dict) -> float:
    node = len(reference["nodes"] & current["nodes"]) / len(reference["nodes"]) if reference["nodes"] else 1.0
    edge = len(reference["edges"] & current["edges"]) / len(reference["edges"]) if reference["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def shared_learning_similar_edges(routes: dict[str, dict]) -> set[tuple[int, int]]:
    return (routes["学習文"]["edges"] & routes["類似文"]["edges"]) - routes["無関係文"]["edges"]


def ranked_edges(brain, edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(
        edges,
        key=lambda e: (int(brain.usage[e[0], e[1]]), float(brain.weights[e[0], e[1]]), e),
        reverse=True,
    )


def set_edges_enabled(brain, edges: list[tuple[int, int]], enabled: bool, backups: dict) -> None:
    for a, b in edges:
        if (a, b) not in backups:
            backups[(a, b)] = (float(brain.weights[a, b]), float(brain.weights[b, a]))
        if enabled:
            brain.weights[a, b], brain.weights[b, a] = backups[(a, b)]
        else:
            brain.weights[a, b] = 0.0
            brain.weights[b, a] = 0.0


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["group"], row["fraction"], row["phase"], row["label"])
        grouped[key].append(row)
    fields = ["route_retention", "node_count", "edge_count", "train_similar_overlap", "train_unrelated_overlap"]
    output = []
    for key, items in sorted(grouped.items()):
        record = {"group": key[0], "fraction": key[1], "phase": key[2], "label": key[3], "trials": len(items)}
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


def write_html(path: Path, summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "段階的切断による各入力経路の保持率",
            "学習文と類似文の経路重なり",
            "復元後の回復率",
            "切断後の通過Edge数",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        dash = "solid" if group == "固定射影" else "dash"
        for label in TEXTS:
            cut = [r for r in summary if r["group"] == group and r["phase"] == "切断" and r["label"] == label]
            x = [r["fraction"] * 100 for r in cut]
            fig.add_trace(go.Scatter(x=x, y=[r["route_retention_mean"]*100 for r in cut], error_y={"type":"data","array":[r["route_retention_std"]*100 for r in cut]}, mode="lines+markers", line={"dash":dash}, name=f"{label}/{group}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=[r["edge_count_mean"] for r in cut], error_y={"type":"data","array":[r["edge_count_std"] for r in cut]}, mode="lines+markers", line={"dash":dash}, name=f"Edge:{label}/{group}", showlegend=False), row=4, col=1)

        cut_anchor = [r for r in summary if r["group"] == group and r["phase"] == "切断" and r["label"] == "学習文"]
        fig.add_trace(go.Scatter(x=[r["fraction"]*100 for r in cut_anchor], y=[r["train_similar_overlap_mean"]*100 for r in cut_anchor], error_y={"type":"data","array":[r["train_similar_overlap_std"]*100 for r in cut_anchor]}, mode="lines+markers", line={"dash":dash}, name=f"学習↔類似/{group}", showlegend=False), row=2, col=1)

        restored = [r for r in summary if r["group"] == group and r["phase"] == "復元" and r["label"] == "類似文"]
        fig.add_trace(go.Scatter(x=[r["fraction"]*100 for r in restored], y=[r["route_retention_mean"]*100 for r in restored], error_y={"type":"data","array":[r["route_retention_std"]*100 for r in restored]}, mode="lines+markers", line={"dash":dash}, name=f"類似文復元/{group}", showlegend=False), row=3, col=1)

    fig.update_yaxes(title_text="保持率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="重なり (%)", row=2, col=1)
    fig.update_yaxes(title_text="回復率 (%)", row=3, col=1)
    fig.update_yaxes(title_text="edges", row=4, col=1)
    fig.update_xaxes(title_text="共有Edge切断率 (%)", row=4, col=1)
    fig.update_layout(height=1400, title=f"SphereBrain 段階的アブレーション＋復元<br><sup>学習文: {TRAIN_TEXT} / 各群{TRIALS_PER_GROUP}試行</sup>", hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_graded_ablation_recovery" / timestamp
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
            baseline = {label: observe(text, adapter) for label, text in TEXTS.items()}
            brain = pipeline.load_brain()
            shared = ranked_edges(brain, shared_learning_similar_edges(baseline))
            print(f"[{group} 試行{trial}] 共有Edge={len(shared)}")

            for fraction in FRACTIONS:
                count = int(round(len(shared) * fraction))
                selected = shared[:count]
                backups: dict = {}
                set_edges_enabled(brain, selected, enabled=False, backups=backups)
                brain.save(pipeline.BRAIN_FILE)
                cut_routes = {label: observe(text, adapter) for label, text in TEXTS.items()}
                cut_overlap_similar = route_overlap(cut_routes["学習文"], cut_routes["類似文"])
                cut_overlap_unrelated = route_overlap(cut_routes["学習文"], cut_routes["無関係文"])
                for label in TEXTS:
                    all_rows.append({
                        "group": group, "trial": trial, "projection_seed": seed,
                        "fraction": fraction, "phase": "切断", "label": label,
                        "available_shared_edges": len(shared), "disabled_edges": count,
                        "route_retention": round(retention(baseline[label], cut_routes[label]), 8),
                        "node_count": cut_routes[label]["node_count"], "edge_count": cut_routes[label]["edge_count"],
                        "train_similar_overlap": round(cut_overlap_similar, 8),
                        "train_unrelated_overlap": round(cut_overlap_unrelated, 8),
                    })

                set_edges_enabled(brain, selected, enabled=True, backups=backups)
                brain.save(pipeline.BRAIN_FILE)
                restored_routes = {label: observe(text, adapter) for label, text in TEXTS.items()}
                restored_overlap_similar = route_overlap(restored_routes["学習文"], restored_routes["類似文"])
                restored_overlap_unrelated = route_overlap(restored_routes["学習文"], restored_routes["無関係文"])
                for label in TEXTS:
                    all_rows.append({
                        "group": group, "trial": trial, "projection_seed": seed,
                        "fraction": fraction, "phase": "復元", "label": label,
                        "available_shared_edges": len(shared), "disabled_edges": 0,
                        "route_retention": round(retention(baseline[label], restored_routes[label]), 8),
                        "node_count": restored_routes[label]["node_count"], "edge_count": restored_routes[label]["edge_count"],
                        "train_similar_overlap": round(restored_overlap_similar, 8),
                        "train_unrelated_overlap": round(restored_overlap_unrelated, 8),
                    })

    summary = summarize(all_rows)
    write_csv(run_dir / "graded_ablation_all_trials.csv", all_rows)
    write_csv(run_dir / "graded_ablation_summary.csv", summary)
    (run_dir / "graded_ablation_recovery.json").write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "train_text": TRAIN_TEXT, "texts": TEXTS, "fractions": FRACTIONS, "rows": all_rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "graded_ablation_recovery.html", summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
