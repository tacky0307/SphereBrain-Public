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
TEXTS = {
    "学習文": TRAIN_TEXT,
    "類似文": "今日の天気は最高だ",
    "無関係文": "犬は公園を走っている",
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


def edge_usage(brain, edge: tuple[int, int]) -> int:
    a, b = edge
    return int(max(brain.usage[a, b], brain.usage[b, a]))


def cut_and_observe(edges: list[tuple[int, int]], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
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


def score_single_edge(edge: tuple[int, int], baseline: dict[str, dict], adapter: pipeline.OpenAIAdapter) -> float:
    current = cut_and_observe([edge], adapter)
    baseline_overlap = route_overlap(baseline["学習文"], baseline["類似文"])
    current_overlap = route_overlap(current["学習文"], current["類似文"])
    return (
        (1.0 - retention(baseline["学習文"], current["学習文"])) * 0.35
        + (1.0 - retention(baseline["類似文"], current["類似文"])) * 0.45
        + max(0.0, baseline_overlap - current_overlap) * 0.20
    )


def set_overlap(a: set, b: set) -> dict:
    intersection = a & b
    union = a | b
    return {
        "intersection": len(intersection),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "a_coverage": len(intersection) / len(a) if a else 1.0,
        "b_coverage": len(intersection) / len(b) if b else 1.0,
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"])].append(row)
    fields = [
        "train_retention", "similar_retention", "unrelated_retention",
        "train_similar_overlap", "train_edges", "similar_edges", "unrelated_edges",
    ]
    result = []
    for (group, condition), items in sorted(grouped.items()):
        rec = {"group": group, "condition": condition, "trials": len(items)}
        for field in fields:
            values = np.asarray([item[field] for item in items], dtype=float)
            rec[f"{field}_mean"] = float(values.mean())
            rec[f"{field}_std"] = float(values.std(ddof=0))
        result.append(rec)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict], audits: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    conditions = ["無処置", "因果上位25%", "頻度上位25%", "共通部分", "因果のみ", "頻度のみ", "ランダム25%"]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=False,
        subplot_titles=(
            "切断集合の重複率",
            "各入力経路の保持率",
            "学習文と類似文の経路重なり",
            "切断後の通過Edge数",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        group_audits = [r for r in audits if r["group"] == group]
        fig.add_trace(go.Bar(
            x=[group],
            y=[np.mean([r["causal_frequency_jaccard"] for r in group_audits]) * 100],
            error_y={"type":"data","array":[np.std([r["causal_frequency_jaccard"] for r in group_audits]) * 100]},
            name=f"因果↔頻度 Jaccard / {group}",
        ), row=1, col=1)

        items = {r["condition"]: r for r in summary if r["group"] == group}
        available = [c for c in conditions if c in items]
        dash = "solid" if group == "固定射影" else "dash"
        for field, label in (("train_retention", "学習文"), ("similar_retention", "類似文"), ("unrelated_retention", "無関係文")):
            fig.add_trace(go.Scatter(
                x=available,
                y=[items[c][field+"_mean"]*100 for c in available],
                error_y={"type":"data","array":[items[c][field+"_std"]*100 for c in available]},
                mode="lines+markers", line={"dash":dash}, name=f"{label}/{group}",
            ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=available,
            y=[items[c]["train_similar_overlap_mean"]*100 for c in available],
            error_y={"type":"data","array":[items[c]["train_similar_overlap_std"]*100 for c in available]},
            mode="lines+markers", line={"dash":dash}, name=f"学習↔類似/{group}", showlegend=False,
        ), row=3, col=1)
        for field, label in (("train_edges", "学習文"), ("similar_edges", "類似文"), ("unrelated_edges", "無関係文")):
            fig.add_trace(go.Scatter(
                x=available,
                y=[items[c][field+"_mean"] for c in available],
                error_y={"type":"data","array":[items[c][field+"_std"] for c in available]},
                mode="lines+markers", line={"dash":dash}, name=f"{label} edges/{group}", showlegend=False,
            ), row=4, col=1)

    fig.update_yaxes(title_text="集合Jaccard (%)", row=1, col=1)
    fig.update_yaxes(title_text="保持率 (%)", row=2, col=1)
    fig.update_yaxes(title_text="重なり (%)", row=3, col=1)
    fig.update_yaxes(title_text="edges", row=4, col=1)
    fig.update_layout(height=1450, title="SphereBrain Edge選択・切断監査", hovermode="x unified", barmode="group")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_edge_selection_audit" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    rows: list[dict] = []
    audits: list[dict] = []
    edge_rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            pipeline.experience(TRAIN_TEXT, repeats=50, adapter=adapter)
            baseline = {label: observe(text, adapter) for label, text in TEXTS.items()}
            shared = sorted((baseline["学習文"]["edges"] & baseline["類似文"]["edges"]) - baseline["無関係文"]["edges"])
            brain = pipeline.load_brain()
            scored = []
            for index, edge in enumerate(shared, start=1):
                causal = score_single_edge(edge, baseline, adapter)
                usage = edge_usage(brain, edge)
                scored.append((edge, causal, usage))
                edge_rows.append({"group":group,"trial":trial,"edge_a":edge[0],"edge_b":edge[1],"causal_score":round(causal,8),"usage":usage})
                if index % 10 == 0 or index == len(shared):
                    print(f"[{group} 試行{trial}] 単独監査 {index}/{len(shared)}")

            count = max(1, round(len(shared) * TOP_FRACTION)) if shared else 0
            causal_set = {edge for edge, _, _ in sorted(scored, key=lambda x: (-x[1], -x[2], x[0]))[:count]}
            frequency_set = {edge for edge, _, _ in sorted(scored, key=lambda x: (-x[2], -x[1], x[0]))[:count]}
            common = causal_set & frequency_set
            causal_only = causal_set - frequency_set
            frequency_only = frequency_set - causal_set
            rng = random.Random(seed + trial * 424243)
            random_set = set(rng.sample(shared, count)) if count and len(shared) >= count else set(shared)
            overlap = set_overlap(causal_set, frequency_set)
            causal_scores = np.asarray([x[1] for x in scored], dtype=float)
            usages = np.asarray([x[2] for x in scored], dtype=float)
            correlation = float(np.corrcoef(causal_scores, usages)[0,1]) if len(scored) > 1 and np.std(causal_scores) > 0 and np.std(usages) > 0 else 0.0
            audits.append({
                "group":group,"trial":trial,"projection_seed":seed,"shared_edges":len(shared),"selected_each":count,
                "causal_frequency_intersection":overlap["intersection"],
                "causal_frequency_jaccard":round(overlap["jaccard"],8),
                "causal_covered_by_frequency":round(overlap["a_coverage"],8),
                "frequency_covered_by_causal":round(overlap["b_coverage"],8),
                "causal_usage_correlation":round(correlation,8),
                "causal_only_count":len(causal_only),"frequency_only_count":len(frequency_only),
            })

            conditions = {
                "無処置": set(),
                "因果上位25%": causal_set,
                "頻度上位25%": frequency_set,
                "共通部分": common,
                "因果のみ": causal_only,
                "頻度のみ": frequency_only,
                "ランダム25%": random_set,
            }
            for condition, cut_set in conditions.items():
                current = baseline if not cut_set else cut_and_observe(sorted(cut_set), adapter)
                rows.append({
                    "group":group,"trial":trial,"projection_seed":seed,"condition":condition,
                    "cut_edges":len(cut_set),"shared_edges":len(shared),
                    "train_retention":round(retention(baseline["学習文"], current["学習文"]),8),
                    "similar_retention":round(retention(baseline["類似文"], current["類似文"]),8),
                    "unrelated_retention":round(retention(baseline["無関係文"], current["無関係文"]),8),
                    "train_similar_overlap":round(route_overlap(current["学習文"], current["類似文"]),8),
                    "train_edges":len(current["学習文"]["edges"]),
                    "similar_edges":len(current["類似文"]["edges"]),
                    "unrelated_edges":len(current["無関係文"]["edges"]),
                })

    summary = summarize(rows)
    write_csv(run_dir / "edge_selection_audit_trials.csv", rows)
    write_csv(run_dir / "edge_selection_audit_sets.csv", audits)
    write_csv(run_dir / "edge_selection_audit_edges.csv", edge_rows)
    write_csv(run_dir / "edge_selection_audit_summary.csv", summary)
    (run_dir / "edge_selection_audit.json").write_text(json.dumps({"created_at":datetime.now().isoformat(timespec="seconds"),"rows":rows,"audits":audits,"edges":edge_rows,"summary":summary,"note":"Earlier run_llm_core_edge_importance.py modified an in-memory brain while observe() reloaded the saved brain, so its ablation did not reach measurement. This audit always saves the ablated brain before observation."}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "edge_selection_audit.html", summary, audits)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("監査注記: 旧頻度階層実験では切断が観測用Coreへ反映されない実装差を確認しました。")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
