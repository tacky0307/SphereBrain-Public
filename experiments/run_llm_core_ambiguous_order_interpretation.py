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

ORDERS = {
    "晴れ→雨→曖昧": [SUNNY, RAINY, AMBIGUOUS],
    "雨→晴れ→曖昧": [RAINY, SUNNY, AMBIGUOUS],
    "曖昧→晴れ→雨": [AMBIGUOUS, SUNNY, RAINY],
}

PROBES = {
    "雨上がりの日差し": "雨上がりの空に日が差している",
    "曇りだが暖かい": "曇っているが少し暖かい",
    "晴れだが冷たい": "晴れているのに冷たい風が吹く",
    "雨でも明るい": "雨の日でも気分は明るい",
    "暗いが降らない": "空は暗いが雨は降っていない",
    "遠い対照": "会計ソフトが請求書を処理する",
}

TRAIN_REPEATS = 20
TRIALS_PER_GROUP = 4
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


def route_overlap(a: dict, b: dict) -> float:
    return 0.35 * jaccard(a["nodes"], b["nodes"]) + 0.65 * jaccard(a["edges"], b["edges"])


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["order"], row["probe_label"])].append(row)
    fields = [
        "sunny_affinity", "rainy_affinity", "ambiguous_affinity",
        "sunny_margin", "rainy_margin", "confidence_gap",
        "node_count", "edge_count",
    ]
    output = []
    for key, items in sorted(grouped.items()):
        record = {"group": key[0], "order": key[1], "probe_label": key[2], "trials": len(items)}
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


def write_html(path: Path, summary: list[dict], pair_rows: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "経験順序ごとの曖昧Probe親和差（晴れ−雨）",
            "経験順序ごとの曖昧群親和性",
            "経験順序ごとの通過Edge数",
            "同じProbeに対する異順序Core間の経路類似度",
        ),
        vertical_spacing=0.10,
    )

    orders = list(ORDERS)
    probe_labels = list(PROBES)
    for probe in probe_labels:
        items = {(r["order"], r["group"]): r for r in summary if r["probe_label"] == probe}
        for group, dash in (("固定射影", "solid"), ("異なる射影", "dash")):
            x = [o for o in orders if (o, group) in items]
            if not x:
                continue
            fig.add_trace(go.Scatter(
                x=x,
                y=[items[(o, group)]["sunny_margin_mean"] * 100 for o in x],
                error_y={"type": "data", "array": [items[(o, group)]["sunny_margin_std"] * 100 for o in x]},
                mode="lines+markers",
                line={"dash": dash},
                name=f"{probe}/{group}",
            ), row=1, col=1)

    for order in orders:
        items = {(r["probe_label"], r["group"]): r for r in summary if r["order"] == order}
        for group, dash in (("固定射影", "solid"), ("異なる射影", "dash")):
            x = [p for p in probe_labels if (p, group) in items]
            if not x:
                continue
            fig.add_trace(go.Scatter(
                x=x,
                y=[items[(p, group)]["ambiguous_affinity_mean"] * 100 for p in x],
                error_y={"type": "data", "array": [items[(p, group)]["ambiguous_affinity_std"] * 100 for p in x]},
                mode="lines+markers",
                line={"dash": dash},
                name=f"曖昧親和/{order}/{group}",
                showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=x,
                y=[items[(p, group)]["edge_count_mean"] for p in x],
                error_y={"type": "data", "array": [items[(p, group)]["edge_count_std"] for p in x]},
                mode="lines+markers",
                line={"dash": dash},
                name=f"edges/{order}/{group}",
                showlegend=False,
            ), row=3, col=1)

    if pair_rows:
        pair_labels = [r["pair"] for r in pair_rows]
        fig.add_trace(go.Bar(
            x=pair_labels,
            y=[r["similarity_mean"] * 100 for r in pair_rows],
            error_y={"type": "data", "array": [r["similarity_std"] * 100 for r in pair_rows]},
            name="経路類似度",
            showlegend=False,
        ), row=4, col=1)

    fig.update_yaxes(title_text="晴れ−雨 (%)", row=1, col=1)
    fig.update_yaxes(title_text="曖昧親和性 (%)", row=2, col=1)
    fig.update_yaxes(title_text="edges", row=3, col=1)
    fig.update_yaxes(title_text="類似度 (%)", row=4, col=1)
    fig.update_xaxes(title_text="経験順序", row=1, col=1)
    fig.update_xaxes(title_text="Probe", row=2, col=1)
    fig.update_xaxes(title_text="Probe", row=3, col=1)
    fig.update_xaxes(title_text="順序ペア", row=4, col=1)
    fig.update_layout(height=1500, title="SphereBrain 経験順序による曖昧入力の解釈差", hovermode="x unified")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_ambiguous_order_interpretation" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    rows: list[dict] = []
    final_routes: dict[tuple, dict[str, dict]] = {}

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            for order_name, stages in ORDERS.items():
                data_dir = run_dir / "data" / group / f"trial_{trial}" / order_name.replace("→", "_")
                configure_data(data_dir, seed)
                pipeline.reset_experiment()
                for stage in stages:
                    for text in stage:
                        pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)

                sunny_routes = [observe(text, adapter) for text in SUNNY]
                rainy_routes = [observe(text, adapter) for text in RAINY]
                ambiguous_routes = [observe(text, adapter) for text in AMBIGUOUS]
                probe_routes = {label: observe(text, adapter) for label, text in PROBES.items()}
                final_routes[(group, trial, order_name)] = probe_routes

                for label, route in probe_routes.items():
                    sunny_aff = float(np.mean([route_overlap(route, ref) for ref in sunny_routes]))
                    rainy_aff = float(np.mean([route_overlap(route, ref) for ref in rainy_routes]))
                    ambiguous_aff = float(np.mean([route_overlap(route, ref) for ref in ambiguous_routes]))
                    rows.append({
                        "group": group,
                        "trial": trial,
                        "projection_seed": seed,
                        "order": order_name,
                        "probe_label": label,
                        "probe_text": PROBES[label],
                        "sunny_affinity": round(sunny_aff, 8),
                        "rainy_affinity": round(rainy_aff, 8),
                        "ambiguous_affinity": round(ambiguous_aff, 8),
                        "sunny_margin": round(sunny_aff - rainy_aff, 8),
                        "rainy_margin": round(rainy_aff - sunny_aff, 8),
                        "confidence_gap": round(abs(sunny_aff - rainy_aff), 8),
                        "node_count": route["node_count"],
                        "edge_count": route["edge_count"],
                    })
                print(f"[{group} 試行{trial}] {order_name} 完了")

    pair_values: dict[str, list[float]] = defaultdict(list)
    order_names = list(ORDERS)
    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            for i in range(len(order_names)):
                for j in range(i + 1, len(order_names)):
                    a_name, b_name = order_names[i], order_names[j]
                    a_routes = final_routes[(group, trial, a_name)]
                    b_routes = final_routes[(group, trial, b_name)]
                    similarity = float(np.mean([route_overlap(a_routes[label], b_routes[label]) for label in PROBES]))
                    pair_values[f"{group}: {a_name} vs {b_name}"].append(similarity)

    pair_rows = []
    for pair, values in sorted(pair_values.items()):
        array = np.asarray(values, dtype=float)
        pair_rows.append({
            "pair": pair,
            "trials": len(values),
            "similarity_mean": float(array.mean()),
            "similarity_std": float(array.std(ddof=0)),
        })

    summary = summarize(rows)
    write_csv(run_dir / "ambiguous_order_trials.csv", rows)
    write_csv(run_dir / "ambiguous_order_summary.csv", summary)
    write_csv(run_dir / "ambiguous_order_pair_similarity.csv", pair_rows)
    (run_dir / "ambiguous_order_interpretation.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "orders": list(ORDERS),
            "probes": PROBES,
            "rows": rows,
            "summary": summary,
            "pair_similarity": pair_rows,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(run_dir / "ambiguous_order_interpretation.html", summary, pair_rows)
    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
