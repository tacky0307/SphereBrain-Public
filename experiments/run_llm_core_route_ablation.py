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
TRIALS_PER_GROUP = 5
BASE_SEED = pipeline.PROJECTION_SEED
TRAIN_REPEATS = 50


def configure_data(path: Path, projection_seed: int) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def observe(brain, text: str, adapter: pipeline.OpenAIAdapter) -> dict:
    _, stimulus = pipeline.encode_text(text, adapter)
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
        "edges": {tuple(sorted(edge)) for edge in result.traversed_edges},
    }


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def overlap(left: dict, right: dict) -> dict:
    node = jaccard(left["nodes"], right["nodes"])
    edge = jaccard(left["edges"], right["edges"])
    return {
        "node_overlap": node,
        "edge_overlap": edge,
        "route_overlap": 0.35 * node + 0.65 * edge,
    }


def classify_edges(routes: dict[str, dict]) -> dict[str, set[tuple[int, int]]]:
    train = routes["学習文"]["edges"]
    similar = routes["類似文"]["edges"]
    unrelated = routes["無関係文"]["edges"]
    universe = train | similar | unrelated
    classes: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for edge in universe:
        membership = (edge in train, edge in similar, edge in unrelated)
        if membership == (True, True, True):
            role = "全共有幹線"
        elif membership == (True, True, False):
            role = "学習・類似共有"
        elif membership == (True, False, False):
            role = "学習文専用"
        elif membership == (False, True, False):
            role = "類似文専用"
        elif membership == (False, False, True):
            role = "無関係文専用"
        elif membership == (True, False, True):
            role = "学習・無関係共有"
        else:
            role = "類似・無関係共有"
        classes[role].add(edge)
    return classes


def ablated_routes(brain, edges: set[tuple[int, int]], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
    if not edges:
        return {label: observe(brain, text, adapter) for label, text in TEXTS.items()}

    saved = []
    for left, right in edges:
        saved.append((left, right, bool(brain.adjacency[left, right]), float(brain.weights[left, right])))
        brain.adjacency[left, right] = False
        brain.adjacency[right, left] = False
        brain.weights[left, right] = 0.0
        brain.weights[right, left] = 0.0
    try:
        return {label: observe(brain, text, adapter) for label, text in TEXTS.items()}
    finally:
        for left, right, adjacency, weight in saved:
            brain.adjacency[left, right] = adjacency
            brain.adjacency[right, left] = adjacency
            brain.weights[left, right] = weight
            brain.weights[right, left] = weight


def sample_random_edges(brain, excluded: set[tuple[int, int]], count: int, seed: int) -> set[tuple[int, int]]:
    candidates = {
        (left, right)
        for left in range(brain.node_count)
        for right in range(left + 1, brain.node_count)
        if brain.adjacency[left, right] and (left, right) not in excluded
    }
    if count <= 0 or not candidates:
        return set()
    ordered = sorted(candidates)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ordered), size=min(count, len(ordered)), replace=False)
    return {ordered[int(index)] for index in np.atleast_1d(indices)}


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"], row["label"])].append(row)
    fields = [
        "ablated_edge_count", "node_count", "edge_count", "node_retention",
        "edge_retention", "route_retention", "train_similar_overlap",
        "train_unrelated_overlap", "similar_unrelated_overlap",
    ]
    output = []
    for key, items in sorted(grouped.items()):
        record = {"group": key[0], "condition": key[1], "label": key[2], "trials": len(items)}
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

    conditions = ["無処置", "学習・類似共有Edge", "全共有幹線Edge", "無関係文専用Edge", "ランダム同数Edge"]
    figure = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "各入力経路の保持率（無処置経路に対して）",
            "学習文と類似文の経路重なり",
            "学習文と無関係文の経路重なり",
            "アブレーション後の通過Edge数",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        dash = "solid" if group == "固定射影" else "dash"
        for label in TEXTS:
            items = [next((r for r in summary if r["group"] == group and r["condition"] == condition and r["label"] == label), None) for condition in conditions]
            items = [item for item in items if item]
            x = [item["condition"] for item in items]
            figure.add_trace(go.Scatter(x=x, y=[item["route_retention_mean"]*100 for item in items], error_y={"type":"data","array":[item["route_retention_std"]*100 for item in items]}, mode="lines+markers", line={"dash":dash}, name=f"{label} / {group}"), row=1, col=1)
            figure.add_trace(go.Scatter(x=x, y=[item["edge_count_mean"] for item in items], error_y={"type":"data","array":[item["edge_count_std"] for item in items]}, mode="lines+markers", line={"dash":dash}, name=f"Edge数: {label} / {group}", showlegend=False), row=4, col=1)

        representative = [next((r for r in summary if r["group"] == group and r["condition"] == condition and r["label"] == "学習文"), None) for condition in conditions]
        representative = [item for item in representative if item]
        x = [item["condition"] for item in representative]
        figure.add_trace(go.Scatter(x=x, y=[item["train_similar_overlap_mean"]*100 for item in representative], error_y={"type":"data","array":[item["train_similar_overlap_std"]*100 for item in representative]}, mode="lines+markers", line={"dash":dash}, name=f"学習↔類似 / {group}"), row=2, col=1)
        figure.add_trace(go.Scatter(x=x, y=[item["train_unrelated_overlap_mean"]*100 for item in representative], error_y={"type":"data","array":[item["train_unrelated_overlap_std"]*100 for item in representative]}, mode="lines+markers", line={"dash":dash}, name=f"学習↔無関係 / {group}"), row=3, col=1)

    figure.update_yaxes(title_text="保持率 (%)", row=1, col=1)
    figure.update_yaxes(title_text="重なり (%)", row=2, col=1)
    figure.update_yaxes(title_text="重なり (%)", row=3, col=1)
    figure.update_yaxes(title_text="edges", row=4, col=1)
    figure.update_xaxes(title_text="アブレーション条件", row=4, col=1)
    figure.update_layout(height=1450, title=f"SphereBrain 経路アブレーション実験<br><sup>50回学習後 / 各群{TRIALS_PER_GROUP}試行</sup>", hovermode="x unified")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_route_ablation" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            data_dir = run_dir / "data" / group / f"trial_{trial}"
            configure_data(data_dir, seed)
            pipeline.reset_experiment()
            pipeline.experience(TRAIN_TEXT, repeats=TRAIN_REPEATS, adapter=adapter)
            brain = pipeline.load_brain()
            baseline = {label: observe(brain, text, adapter) for label, text in TEXTS.items()}
            roles = classify_edges(baseline)
            target = roles.get("学習・類似共有", set())
            all_shared = roles.get("全共有幹線", set())
            unrelated_only = roles.get("無関係文専用", set())
            excluded = set().union(*roles.values()) if roles else set()
            random_matched = sample_random_edges(brain, excluded=set(), count=len(target), seed=seed + trial * 7919)

            conditions = {
                "無処置": set(),
                "学習・類似共有Edge": target,
                "全共有幹線Edge": all_shared,
                "無関係文専用Edge": unrelated_only,
                "ランダム同数Edge": random_matched,
            }
            print(f"\n[{group} 試行{trial}] target={len(target)} shared={len(all_shared)} unrelated={len(unrelated_only)}")

            for condition, ablated in conditions.items():
                current = baseline if condition == "無処置" else ablated_routes(brain, ablated, adapter)
                pair_ts = overlap(current["学習文"], current["類似文"])["route_overlap"]
                pair_tu = overlap(current["学習文"], current["無関係文"])["route_overlap"]
                pair_su = overlap(current["類似文"], current["無関係文"])["route_overlap"]
                for label in TEXTS:
                    retention = overlap(baseline[label], current[label])
                    rows.append({
                        "group": group,
                        "trial": trial,
                        "projection_seed": seed,
                        "condition": condition,
                        "label": label,
                        "text": TEXTS[label],
                        "ablated_edge_count": len(ablated),
                        "node_count": len(current[label]["nodes"]),
                        "edge_count": len(current[label]["edges"]),
                        "node_retention": round(retention["node_overlap"], 8),
                        "edge_retention": round(retention["edge_overlap"], 8),
                        "route_retention": round(retention["route_overlap"], 8),
                        "train_similar_overlap": round(pair_ts, 8),
                        "train_unrelated_overlap": round(pair_tu, 8),
                        "similar_unrelated_overlap": round(pair_su, 8),
                    })
                print(f"  {condition}: 学習↔類似={pair_ts*100:.1f}% 学習↔無関係={pair_tu*100:.1f}%")

    summary = summarize(rows)
    write_csv(run_dir / "route_ablation_all_trials.csv", rows)
    write_csv(run_dir / "route_ablation_summary.csv", summary)
    (run_dir / "route_ablation.json").write_text(json.dumps({"created_at": datetime.now().isoformat(timespec="seconds"), "train_repeats": TRAIN_REPEATS, "texts": TEXTS, "rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "route_ablation.html", summary)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
