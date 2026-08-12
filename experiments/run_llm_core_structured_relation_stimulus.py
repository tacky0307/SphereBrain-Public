from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

TRIALS_PER_GROUP = 5
REPEATS = 20
BASE_SEED = pipeline.PROJECTION_SEED

NAME_SETS = [
    ("ラモ", "キト", "セナ"),
    ("ネフ", "ポラ", "ジム"),
    ("トア", "ミグ", "レナ"),
    ("バル", "ソク", "ニア"),
    ("ケム", "ルタ", "ホノ"),
]

MODES = ("文章全体", "役割分解")
CONDITIONS = ("未学習", "前提1のみ", "同語彙対照", "推移連鎖")


def configure_data(path: Path, projection_seed: int) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def route_overlap(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def merge_routes(routes: list[dict]) -> dict:
    return {
        "nodes": set().union(*(route["nodes"] for route in routes)) if routes else set(),
        "edges": set().union(*(route["edges"] for route in routes)) if routes else set(),
    }


def component_texts(subject: str, relation: str, obj: str) -> list[str]:
    # Role prefixes preserve who occupies which slot. Relation direction is explicit.
    return [
        f"主体スロット::{subject}",
        f"方向関係スロット::{relation}::主体から対象へ",
        f"対象スロット::{obj}",
        f"関係結合::{subject}::{relation}::{obj}",
    ]


def relation_sentence(subject: str, obj: str) -> str:
    return f"{subject}は{obj}より強い"


def propagate_text(text: str, adapter: pipeline.OpenAIAdapter, learn: bool) -> dict:
    _embedding, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(
        sources,
        steps=14,
        threshold=0.18,
        noise=0.004 if learn else 0.0,
        learn=learn,
    )
    if learn:
        brain.save(pipeline.BRAIN_FILE)
    return {
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
    }


def experience_relation(
    subject: str,
    obj: str,
    mode: str,
    adapter: pipeline.OpenAIAdapter,
    repeats: int = REPEATS,
) -> None:
    if mode == "文章全体":
        pipeline.experience(relation_sentence(subject, obj), repeats=repeats, adapter=adapter)
        return

    parts = component_texts(subject, "より強い", obj)
    for _ in range(repeats):
        for part in parts:
            propagate_text(part, adapter, learn=True)


def observe_relation(subject: str, obj: str, mode: str, adapter: pipeline.OpenAIAdapter) -> dict:
    if mode == "文章全体":
        return propagate_text(relation_sentence(subject, obj), adapter, learn=False)
    return merge_routes([propagate_text(part, adapter, learn=False) for part in component_texts(subject, "より強い", obj)])


def embedding_margin(subject: str, middle: str, obj: str, adapter: pipeline.OpenAIAdapter) -> float:
    candidate = np.asarray(adapter.embed(relation_sentence(subject, obj)), dtype=float)
    reverse = np.asarray(adapter.embed(relation_sentence(obj, subject)), dtype=float)
    p1 = np.asarray(adapter.embed(relation_sentence(subject, middle)), dtype=float)
    p2 = np.asarray(adapter.embed(relation_sentence(middle, obj)), dtype=float)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    candidate_support = (cosine(candidate, p1) + cosine(candidate, p2)) / 2
    reverse_support = (cosine(reverse, p1) + cosine(reverse, p2)) / 2
    return candidate_support - reverse_support


def train_condition(
    condition: str,
    names: tuple[str, str, str],
    mode: str,
    adapter: pipeline.OpenAIAdapter,
) -> None:
    a, b, c = names
    if condition == "未学習":
        return
    if condition == "前提1のみ":
        experience_relation(a, b, mode, adapter)
        return
    if condition == "推移連鎖":
        experience_relation(a, b, mode, adapter)
        experience_relation(b, c, mode, adapter)
        return
    if condition == "同語彙対照":
        # Same names and relation vocabulary, but B is not a directional bridge A→B→C.
        experience_relation(a, b, mode, adapter)
        experience_relation(c, b, mode, adapter)
        return
    raise ValueError(condition)


def evaluate(
    names: tuple[str, str, str],
    mode: str,
    adapter: pipeline.OpenAIAdapter,
) -> dict:
    a, b, c = names
    premise1 = observe_relation(a, b, mode, adapter)
    premise2 = observe_relation(b, c, mode, adapter)
    candidate = observe_relation(a, c, mode, adapter)
    reverse = observe_relation(c, a, mode, adapter)
    unrelated = observe_relation("フザ", "メコ", mode, adapter)

    premise_union = merge_routes([premise1, premise2])
    candidate_support = route_overlap(candidate, premise_union)
    reverse_support = route_overlap(reverse, premise_union)
    unrelated_support = route_overlap(unrelated, premise_union)

    bridge_edges = premise1["edges"] & premise2["edges"]
    bridge_use = len(candidate["edges"] & bridge_edges) / len(bridge_edges) if bridge_edges else 0.0
    reverse_bridge_use = len(reverse["edges"] & bridge_edges) / len(bridge_edges) if bridge_edges else 0.0

    # Does the candidate touch both premise routes, rather than only one lexical neighbor?
    candidate_p1 = route_overlap(candidate, premise1)
    candidate_p2 = route_overlap(candidate, premise2)
    reverse_p1 = route_overlap(reverse, premise1)
    reverse_p2 = route_overlap(reverse, premise2)
    two_premise_balance = min(candidate_p1, candidate_p2)
    reverse_two_premise_balance = min(reverse_p1, reverse_p2)

    return {
        "candidate_support": candidate_support,
        "reverse_support": reverse_support,
        "support_margin": candidate_support - reverse_support,
        "unrelated_support": unrelated_support,
        "candidate_vs_unrelated": candidate_support - unrelated_support,
        "bridge_use": bridge_use,
        "reverse_bridge_use": reverse_bridge_use,
        "bridge_margin": bridge_use - reverse_bridge_use,
        "two_premise_balance": two_premise_balance,
        "reverse_two_premise_balance": reverse_two_premise_balance,
        "composition_margin": two_premise_balance - reverse_two_premise_balance,
        "candidate_nodes": len(candidate["nodes"]),
        "candidate_edges": len(candidate["edges"]),
    }


def summarize(rows: list[dict], keys: tuple[str, ...], fields: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output: list[dict] = []
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
            "推移候補−逆向き候補のCore支持差",
            "二つの前提へ同時に接続した度合いの差",
            "前提共有Edgeの利用差",
            "候補と無関係候補の支持差",
        ),
        vertical_spacing=0.10,
    )

    x = [f"{r['mode']} / {r['condition']} / {r['group']}" for r in summary]
    specs = [
        ("support_margin", 1, "支持差 (%)"),
        ("composition_margin", 2, "合成差 (%)"),
        ("bridge_margin", 3, "共有Edge利用差 (%)"),
        ("candidate_vs_unrelated", 4, "支持差 (%)"),
    ]
    for field, row, _title in specs:
        figure.add_trace(
            go.Bar(
                x=x,
                y=[r[f"{field}_mean"] * 100 for r in summary],
                error_y={"type": "data", "array": [r[f"{field}_std"] * 100 for r in summary]},
                name=field,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        figure.add_hline(y=0, line_dash="dot", row=row, col=1)

    for _field, row, title in specs:
        figure.update_yaxes(title_text=title, row=row, col=1)
    figure.update_xaxes(title_text="入力方式 / 学習条件 / 射影群", row=4, col=1)
    figure.update_layout(
        height=1550,
        title="SphereBrain 関係を保った複数刺激による推移合成実験",
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_structured_relation_stimulus" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()
    rows: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            names = NAME_SETS[(trial - 1) % len(NAME_SETS)]
            for mode in MODES:
                for condition in CONDITIONS:
                    seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
                    data_path = run_dir / "data" / group / f"trial_{trial}" / mode / condition
                    configure_data(data_path, seed)
                    pipeline.reset_experiment()
                    train_condition(condition, names, mode, adapter)
                    metrics = evaluate(names, mode, adapter)
                    row = {
                        "group": group,
                        "trial": trial,
                        "mode": mode,
                        "condition": condition,
                        "subject": names[0],
                        "middle": names[1],
                        "object": names[2],
                        "embedding_margin": embedding_margin(*names, adapter),
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"[{group} {trial}] {mode} / {condition}: "
                        f"margin={metrics['support_margin']*100:.1f}% "
                        f"compose={metrics['composition_margin']*100:.1f}%"
                    )

    fields = [
        "candidate_support",
        "reverse_support",
        "support_margin",
        "unrelated_support",
        "candidate_vs_unrelated",
        "bridge_use",
        "reverse_bridge_use",
        "bridge_margin",
        "two_premise_balance",
        "reverse_two_premise_balance",
        "composition_margin",
        "candidate_nodes",
        "candidate_edges",
        "embedding_margin",
    ]
    summary = summarize(rows, ("mode", "condition", "group"), fields)

    write_csv(run_dir / "structured_relation_trials.csv", rows)
    write_csv(run_dir / "structured_relation_summary.csv", summary)
    payload = {
        "experiment": "structured_relation_stimulus",
        "description": "Compare whole-sentence embedding with role-preserving sequential stimuli for transitive relation composition.",
        "repeats": REPEATS,
        "trials_per_group": TRIALS_PER_GROUP,
        "rows": rows,
        "summary": summary,
    }
    (run_dir / "structured_relation_stimulus.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html(run_dir / "structured_relation_stimulus.html", summary)

    print("\n完了しました。")
    print(f"HTML: {run_dir / 'structured_relation_stimulus.html'}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
