from __future__ import annotations

import csv
import json
import math
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
    "無関係": "会計ソフトが請求書を処理する",
}
TRIALS_PER_GROUP = 5
BASE_SEED = pipeline.PROJECTION_SEED
TRAIN_REPEATS = 20
ABLATION_FRACTION = 1.0


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


def route_score(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def retention(reference: dict, current: dict) -> float:
    node = len(reference["nodes"] & current["nodes"]) / len(reference["nodes"]) if reference["nodes"] else 1.0
    edge = len(reference["edges"] & current["edges"]) / len(reference["edges"]) if reference["edges"] else 1.0
    return 0.35 * node + 0.65 * edge


def mean_affinity(route: dict, group_routes: list[dict]) -> float:
    return float(np.mean([route_score(route, item) for item in group_routes]))


def classify_edges(sunny_routes: list[dict], rainy_routes: list[dict]) -> dict[str, set[tuple[int, int]]]:
    all_edges = set().union(*(r["edges"] for r in sunny_routes + rainy_routes))
    categories = {
        "天気幹線": set(),
        "晴れ多数共有": set(),
        "雨多数共有": set(),
        "晴雨部分共有": set(),
        "その他": set(),
    }
    for edge in all_edges:
        sunny_count = sum(edge in route["edges"] for route in sunny_routes)
        rainy_count = sum(edge in route["edges"] for route in rainy_routes)
        if sunny_count == 3 and rainy_count == 3:
            categories["天気幹線"].add(edge)
        elif sunny_count >= 2 and rainy_count == 0:
            categories["晴れ多数共有"].add(edge)
        elif rainy_count >= 2 and sunny_count == 0:
            categories["雨多数共有"].add(edge)
        elif sunny_count >= 1 and rainy_count >= 1:
            categories["晴雨部分共有"].add(edge)
        else:
            categories["その他"].add(edge)
    return categories


def choose_edges(edges: set[tuple[int, int]], fraction: float) -> list[tuple[int, int]]:
    ranked = sorted(edges)
    if not ranked:
        return []
    count = max(1, math.ceil(len(ranked) * fraction))
    return ranked[:count]


def cut_set_and_observe(edges: list[tuple[int, int]], adapter: pipeline.OpenAIAdapter) -> dict[str, dict]:
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


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["condition"], row["probe_label"])].append(row)
    fields = [
        "route_retention", "sunny_affinity", "rainy_affinity", "affinity_margin",
        "node_count", "edge_count", "cut_edge_count",
    ]
    output = []
    for key, items in sorted(grouped.items()):
        record = {"group": key[0], "condition": key[1], "probe_label": key[2], "trials": len(items)}
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


def write_html(path: Path, summary: list[dict], category_rows: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    conditions = ["無処置", "天気幹線切断", "晴れ枝切断", "雨枝切断", "部分共有切断", "ランダム同数"]
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "未経験Probeの経路保持率",
            "晴れ親和性 − 雨親和性（正なら晴れ側）",
            "切断後の晴れ・雨親和性",
            "分類されたEdge本数",
        ),
    )

    for group, dash in (("固定射影", "solid"), ("異なる射影", "dash")):
        for probe_label in PROBES:
            items = [next(r for r in summary if r["group"] == group and r["condition"] == condition and r["probe_label"] == probe_label) for condition in conditions]
            figure.add_trace(go.Scatter(
                x=conditions,
                y=[r["route_retention_mean"] * 100 for r in items],
                error_y={"type":"data", "array":[r["route_retention_std"] * 100 for r in items]},
                mode="lines+markers", line={"dash":dash}, name=f"{probe_label}/{group}"
            ), row=1, col=1)
            if probe_label in ("晴れ未経験", "雨未経験", "曖昧天気"):
                figure.add_trace(go.Scatter(
                    x=conditions,
                    y=[r["affinity_margin_mean"] * 100 for r in items],
                    error_y={"type":"data", "array":[r["affinity_margin_std"] * 100 for r in items]},
                    mode="lines+markers", line={"dash":dash}, name=f"親和差:{probe_label}/{group}", showlegend=False
                ), row=2, col=1)
            if probe_label in ("晴れ未経験", "雨未経験"):
                figure.add_trace(go.Scatter(
                    x=conditions,
                    y=[r["sunny_affinity_mean"] * 100 for r in items],
                    mode="lines+markers", line={"dash":dash}, name=f"晴れ親和:{probe_label}/{group}", showlegend=False
                ), row=3, col=1)
                figure.add_trace(go.Scatter(
                    x=conditions,
                    y=[r["rainy_affinity_mean"] * 100 for r in items],
                    mode="lines+markers", line={"dash":dash}, name=f"雨親和:{probe_label}/{group}", showlegend=False
                ), row=3, col=1)

    categories = ["天気幹線", "晴れ多数共有", "雨多数共有", "晴雨部分共有", "その他"]
    for group in ("固定射影", "異なる射影"):
        vals = []
        errs = []
        for category in categories:
            items = [r["edge_count"] for r in category_rows if r["group"] == group and r["category"] == category]
            vals.append(float(np.mean(items)) if items else 0.0)
            errs.append(float(np.std(items)) if items else 0.0)
        figure.add_trace(go.Bar(x=categories, y=vals, error_y={"type":"data","array":errs}, name=f"{group}"), row=4, col=1)

    figure.update_yaxes(title_text="保持率 (%)", row=1, col=1)
    figure.update_yaxes(title_text="親和差 (%)", row=2, col=1)
    figure.update_yaxes(title_text="親和性 (%)", row=3, col=1)
    figure.update_yaxes(title_text="Edge数", row=4, col=1)
    figure.update_xaxes(title_text="切断条件", row=3, col=1)
    figure.update_xaxes(title_text="経路カテゴリ", row=4, col=1)
    figure.update_layout(height=1450, title="SphereBrain 概念階層の機能的アブレーション", hovermode="x unified", barmode="group")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_hierarchical_concept_ablation" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    all_rows: list[dict] = []
    category_rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            for text in SUNNY + RAINY:
                pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)

            sunny_routes = [observe(text, adapter) for text in SUNNY]
            rainy_routes = [observe(text, adapter) for text in RAINY]
            baseline = {label: observe(text, adapter) for label, text in PROBES.items()}
            categories = classify_edges(sunny_routes, rainy_routes)
            for category, edges in categories.items():
                category_rows.append({"group":group,"trial":trial,"projection_seed":seed,"category":category,"edge_count":len(edges)})

            trunk = choose_edges(categories["天気幹線"], ABLATION_FRACTION)
            sunny_branch = choose_edges(categories["晴れ多数共有"], ABLATION_FRACTION)
            rainy_branch = choose_edges(categories["雨多数共有"], ABLATION_FRACTION)
            partial = choose_edges(categories["晴雨部分共有"], ABLATION_FRACTION)
            rng = random.Random(seed + trial * 7919)
            random_count = max(len(trunk), len(sunny_branch), len(rainy_branch), 1)
            all_candidate_edges = sorted(set().union(*categories.values()))
            random_edges = rng.sample(all_candidate_edges, min(random_count, len(all_candidate_edges))) if all_candidate_edges else []
            conditions = {
                "無処置": [],
                "天気幹線切断": trunk,
                "晴れ枝切断": sunny_branch,
                "雨枝切断": rainy_branch,
                "部分共有切断": partial,
                "ランダム同数": random_edges,
            }

            print(f"\n[{group} 試行{trial}] 幹線={len(trunk)} 晴れ枝={len(sunny_branch)} 雨枝={len(rainy_branch)} 部分共有={len(partial)}")
            for condition, removed in conditions.items():
                current = baseline if not removed else cut_set_and_observe(removed, adapter)
                for probe_label in PROBES:
                    sunny_aff = mean_affinity(current[probe_label], sunny_routes)
                    rainy_aff = mean_affinity(current[probe_label], rainy_routes)
                    all_rows.append({
                        "group": group,
                        "trial": trial,
                        "projection_seed": seed,
                        "condition": condition,
                        "probe_label": probe_label,
                        "probe_text": PROBES[probe_label],
                        "cut_edge_count": len(removed),
                        "route_retention": round(retention(baseline[probe_label], current[probe_label]), 8),
                        "sunny_affinity": round(sunny_aff, 8),
                        "rainy_affinity": round(rainy_aff, 8),
                        "affinity_margin": round(sunny_aff - rainy_aff, 8),
                        "node_count": current[probe_label]["node_count"],
                        "edge_count": current[probe_label]["edge_count"],
                    })

    summary = summarize(all_rows)
    write_csv(run_dir / "hierarchical_ablation_trials.csv", all_rows)
    write_csv(run_dir / "hierarchical_ablation_summary.csv", summary)
    write_csv(run_dir / "hierarchical_edge_categories.csv", category_rows)
    (run_dir / "hierarchical_concept_ablation.json").write_text(json.dumps({
        "created_at":datetime.now().isoformat(timespec="seconds"),
        "sunny_experiences":SUNNY,
        "rainy_experiences":RAINY,
        "probes":PROBES,
        "rows":all_rows,
        "summary":summary,
        "categories":category_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "hierarchical_concept_ablation.html", summary, category_rows)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
