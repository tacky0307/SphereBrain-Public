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

TRIALS_PER_GROUP = 6
REPEATS = 30
BASE_SEED = pipeline.PROJECTION_SEED
CONDITIONS = ("未学習", "前提1のみ", "推移連鎖", "同語彙対照")

SYLLABLES = [
    "ラモ", "キト", "セナ", "ネフ", "ポラ", "ジム", "ヴァル", "トメ", "リク", "ハノ",
    "メザ", "クル", "ソナ", "デフ", "バリ", "ヨナ", "ケム", "フラ", "ニト", "ザム",
]


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
        "embedding": np.asarray(embedding, dtype=float),
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
    }


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def route_overlap(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def bridge_usage(probe: dict, premise1: dict, premise2: dict) -> float:
    bridge = premise1["edges"] & premise2["edges"]
    if not probe["edges"]:
        return 0.0
    return len(probe["edges"] & bridge) / len(probe["edges"])


def support(probe: dict, premise1: dict, premise2: dict) -> float:
    return 0.5 * (route_overlap(probe, premise1) + route_overlap(probe, premise2))


def embedding_support(probe: dict, premise1: dict, premise2: dict) -> float:
    return 0.5 * (cosine(probe["embedding"], premise1["embedding"]) + cosine(probe["embedding"], premise2["embedding"]))


def summarize(rows: list[dict], keys: tuple[str, ...], fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
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


def write_html(path: Path, summary: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "推移候補と逆向き候補のCore支持差",
            "前提間の共有Edgeを推移候補が利用した割合",
            "推移候補と前提群のCore支持",
            "Embedding支持差（学習で変わらない固定基準）",
        ),
        vertical_spacing=0.10,
    )

    for group in ("固定射影", "異なる射影"):
        items = [r for r in summary if r["group"] == group]
        x = [r["condition"] for r in items]
        dash = "solid" if group == "固定射影" else "dash"
        figure.add_trace(go.Scatter(x=x, y=[r["core_margin_mean"]*100 for r in items], error_y={"type":"data","array":[r["core_margin_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"Core差/{group}"), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=[r["bridge_usage_mean"]*100 for r in items], error_y={"type":"data","array":[r["bridge_usage_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"橋Edge/{group}"), row=2, col=1)
        figure.add_trace(go.Scatter(x=x, y=[r["transitive_support_mean"]*100 for r in items], error_y={"type":"data","array":[r["transitive_support_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"推移支持/{group}"), row=3, col=1)
        figure.add_trace(go.Scatter(x=x, y=[r["embedding_margin_mean"]*100 for r in items], error_y={"type":"data","array":[r["embedding_margin_std"]*100 for r in items]}, mode="lines+markers", line={"dash":dash}, name=f"Embedding差/{group}"), row=4, col=1)

    figure.update_yaxes(title_text="支持差 (%)", row=1, col=1)
    figure.update_yaxes(title_text="利用率 (%)", row=2, col=1)
    figure.update_yaxes(title_text="支持 (%)", row=3, col=1)
    figure.update_yaxes(title_text="固定差 (%)", row=4, col=1)
    figure.update_xaxes(title_text="学習条件", row=4, col=1)
    figure.update_layout(height=1450, title="SphereBrain 最小推移関係実験", hovermode="x unified")
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_transitive_relation" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            name_rng = random.Random(20260805 + trial * 97)
            a, b, c, d, e = name_rng.sample(SYLLABLES, 5)
            premise1_text = f"{a}は{b}より強い"
            premise2_text = f"{b}は{c}より強い"
            control2_text = f"{c}は{b}より強い"
            transitive_text = f"{a}は{c}より強い"
            reverse_text = f"{c}は{a}より強い"
            unrelated_text = f"{d}は{e}より強い"

            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            for condition in CONDITIONS:
                path = run_dir / "data" / group / f"trial_{trial}" / condition
                configure_data(path, seed)
                pipeline.reset_experiment()

                if condition in ("前提1のみ", "推移連鎖", "同語彙対照"):
                    pipeline.experience(premise1_text, repeats=REPEATS, adapter=adapter)
                if condition == "推移連鎖":
                    pipeline.experience(premise2_text, repeats=REPEATS, adapter=adapter)
                elif condition == "同語彙対照":
                    pipeline.experience(control2_text, repeats=REPEATS, adapter=adapter)

                premise1 = observe(premise1_text, adapter)
                premise2 = observe(premise2_text if condition != "同語彙対照" else control2_text, adapter)
                transitive = observe(transitive_text, adapter)
                reverse = observe(reverse_text, adapter)
                unrelated = observe(unrelated_text, adapter)

                trans_support = support(transitive, premise1, premise2)
                reverse_support = support(reverse, premise1, premise2)
                unrelated_support = support(unrelated, premise1, premise2)
                emb_trans = embedding_support(transitive, premise1, premise2)
                emb_reverse = embedding_support(reverse, premise1, premise2)

                rows.append({
                    "group": group,
                    "trial": trial,
                    "condition": condition,
                    "a": a,
                    "b": b,
                    "c": c,
                    "premise1": premise1_text,
                    "premise2": premise2_text if condition != "同語彙対照" else control2_text,
                    "transitive_probe": transitive_text,
                    "reverse_probe": reverse_text,
                    "transitive_support": trans_support,
                    "reverse_support": reverse_support,
                    "unrelated_support": unrelated_support,
                    "core_margin": trans_support - reverse_support,
                    "bridge_usage": bridge_usage(transitive, premise1, premise2),
                    "embedding_transitive": emb_trans,
                    "embedding_reverse": emb_reverse,
                    "embedding_margin": emb_trans - emb_reverse,
                    "transitive_nodes": len(transitive["nodes"]),
                    "transitive_edges": len(transitive["edges"]),
                })
                print(f"[{group} {trial}/{TRIALS_PER_GROUP}] {condition}: Core差={(trans_support-reverse_support)*100:.1f}% 橋={bridge_usage(transitive,premise1,premise2)*100:.1f}%")

    fields = [
        "transitive_support", "reverse_support", "unrelated_support", "core_margin",
        "bridge_usage", "embedding_transitive", "embedding_reverse", "embedding_margin",
        "transitive_nodes", "transitive_edges",
    ]
    summary = summarize(rows, ("group", "condition"), fields)
    write_csv(run_dir / "transitive_relation_trials.csv", rows)
    write_csv(run_dir / "transitive_relation_summary.csv", summary)
    payload = {"trials": rows, "summary": summary, "repeats": REPEATS}
    (run_dir / "transitive_relation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(run_dir / "transitive_relation.html", summary)

    print("\n完了しました。")
    print(f"HTML: {run_dir / 'transitive_relation.html'}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
