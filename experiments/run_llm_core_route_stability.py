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

CHECKPOINTS = (0, 1, 5, 10, 20, 50)
TRIALS_PER_GROUP = 5
TRAIN_TEXT = "今日は晴れて気持ちいい"
TEXTS = {
    "学習文": TRAIN_TEXT,
    "類似文": "今日の天気は最高だ",
    "無関係文": "犬は公園を走っている",
}
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
    }


def ratio(part: set, whole: set) -> float:
    return len(part) / len(whole) if whole else 0.0


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def route_metrics(previous: dict, current: dict) -> dict:
    prev_nodes, curr_nodes = previous["nodes"], current["nodes"]
    prev_edges, curr_edges = previous["edges"], current["edges"]
    added_nodes = curr_nodes - prev_nodes
    removed_nodes = prev_nodes - curr_nodes
    added_edges = curr_edges - prev_edges
    removed_edges = prev_edges - curr_edges
    return {
        "node_jaccard": jaccard(prev_nodes, curr_nodes),
        "edge_jaccard": jaccard(prev_edges, curr_edges),
        "route_stability": 0.35 * jaccard(prev_nodes, curr_nodes) + 0.65 * jaccard(prev_edges, curr_edges),
        "node_retention": ratio(prev_nodes & curr_nodes, prev_nodes),
        "edge_retention": ratio(prev_edges & curr_edges, prev_edges),
        "added_nodes": len(added_nodes),
        "removed_nodes": len(removed_nodes),
        "added_edges": len(added_edges),
        "removed_edges": len(removed_edges),
        "current_nodes": len(curr_nodes),
        "current_edges": len(curr_edges),
        "added_node_set": added_nodes,
        "added_edge_set": added_edges,
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["group"], row["label"], row["from_checkpoint"], row["to_checkpoint"])
        grouped[key].append(row)

    fields = [
        "route_stability", "node_jaccard", "edge_jaccard", "node_retention", "edge_retention",
        "added_nodes", "removed_nodes", "added_edges", "removed_edges",
        "added_node_persistence", "added_edge_persistence", "current_nodes", "current_edges",
    ]
    summary = []
    for key, items in sorted(grouped.items()):
        record = {
            "group": key[0], "label": key[1],
            "from_checkpoint": key[2], "to_checkpoint": key[3],
            "trials": len(items),
        }
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=float)
            record[f"{field}_mean"] = float(values.mean())
            record[f"{field}_std"] = float(values.std(ddof=0))
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "前段階からの経路安定度（高いほど同じ経路を維持）",
            "前段階経路の保持率",
            "新規経路の次段階残存率",
            "追加・消失したNode / Edge数",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        for label in TEXTS:
            items = [r for r in summary if r["group"] == group and r["label"] == label]
            x = [f"{r['from_checkpoint']}→{r['to_checkpoint']}" for r in items]
            dash = "solid" if group == "固定射影" else "dash"
            name_base = f"{label} / {group}"
            figure.add_trace(go.Scatter(x=x, y=[r["route_stability_mean"]*100 for r in items], error_y={"type":"data","array":[r["route_stability_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=name_base), row=1, col=1)
            figure.add_trace(go.Scatter(x=x, y=[r["edge_retention_mean"]*100 for r in items], error_y={"type":"data","array":[r["edge_retention_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"Edge保持: {name_base}", showlegend=False), row=2, col=1)
            figure.add_trace(go.Scatter(x=x, y=[r["added_edge_persistence_mean"]*100 for r in items], error_y={"type":"data","array":[r["added_edge_persistence_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"新規Edge残存: {name_base}", showlegend=False), row=3, col=1)
            figure.add_trace(go.Scatter(x=x, y=[r["added_edges_mean"] for r in items], mode="lines+markers", line={"dash":dash}, name=f"追加Edge: {name_base}", showlegend=False), row=4, col=1)
            figure.add_trace(go.Scatter(x=x, y=[-r["removed_edges_mean"] for r in items], mode="lines+markers", line={"dash":dash}, name=f"消失Edge: {name_base}", showlegend=False), row=4, col=1)

    figure.update_yaxes(title_text="安定度 (%)", row=1, col=1)
    figure.update_yaxes(title_text="保持率 (%)", row=2, col=1)
    figure.update_yaxes(title_text="残存率 (%)", row=3, col=1)
    figure.update_yaxes(title_text="追加 (+) / 消失 (-)", row=4, col=1)
    figure.update_xaxes(title_text="学習段階", row=4, col=1)
    figure.update_layout(height=1350, title=f"SphereBrain 経路変化の方向と安定性<br><sup>学習文: {TRAIN_TEXT} / 各群{TRIALS_PER_GROUP}試行</sup>", hovermode="x unified")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_route_stability" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    all_rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            data_dir = run_dir / "data" / group / f"trial_{trial}"
            configure_data(data_dir, seed)
            pipeline.reset_experiment()
            print(f"\n[{group} 試行{trial}] seed={seed}")

            snapshots: dict[int, dict[str, dict]] = {}
            previous_checkpoint = 0
            for checkpoint in CHECKPOINTS:
                additional = checkpoint - previous_checkpoint
                if additional > 0:
                    pipeline.experience(TRAIN_TEXT, repeats=additional, adapter=adapter)
                snapshots[checkpoint] = {label: observe(text, adapter) for label, text in TEXTS.items()}
                print(f"  checkpoint {checkpoint} 完了")
                previous_checkpoint = checkpoint

            for index in range(1, len(CHECKPOINTS)):
                start, end = CHECKPOINTS[index - 1], CHECKPOINTS[index]
                next_checkpoint = CHECKPOINTS[index + 1] if index + 1 < len(CHECKPOINTS) else None
                for label in TEXTS:
                    metrics = route_metrics(snapshots[start][label], snapshots[end][label])
                    if next_checkpoint is not None:
                        next_nodes = snapshots[next_checkpoint][label]["nodes"]
                        next_edges = snapshots[next_checkpoint][label]["edges"]
                        node_persistence = ratio(metrics["added_node_set"] & next_nodes, metrics["added_node_set"])
                        edge_persistence = ratio(metrics["added_edge_set"] & next_edges, metrics["added_edge_set"])
                    else:
                        node_persistence = 0.0
                        edge_persistence = 0.0

                    row = {
                        "group": group,
                        "trial": trial,
                        "projection_seed": seed,
                        "label": label,
                        "text": TEXTS[label],
                        "from_checkpoint": start,
                        "to_checkpoint": end,
                        "route_stability": round(metrics["route_stability"], 8),
                        "node_jaccard": round(metrics["node_jaccard"], 8),
                        "edge_jaccard": round(metrics["edge_jaccard"], 8),
                        "node_retention": round(metrics["node_retention"], 8),
                        "edge_retention": round(metrics["edge_retention"], 8),
                        "added_nodes": metrics["added_nodes"],
                        "removed_nodes": metrics["removed_nodes"],
                        "added_edges": metrics["added_edges"],
                        "removed_edges": metrics["removed_edges"],
                        "added_node_persistence": round(node_persistence, 8),
                        "added_edge_persistence": round(edge_persistence, 8),
                        "current_nodes": metrics["current_nodes"],
                        "current_edges": metrics["current_edges"],
                    }
                    all_rows.append(row)

    summary = summarize(all_rows)
    write_csv(run_dir / "route_stability_all_trials.csv", all_rows)
    write_csv(run_dir / "route_stability_summary.csv", summary)
    (run_dir / "route_stability.json").write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "checkpoints": CHECKPOINTS, "train_text": TRAIN_TEXT, "texts": TEXTS, "rows": all_rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "route_stability.html", summary)

    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
