from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

CHECKPOINTS = (0, 1, 5, 10, 20, 50)
DEFAULT_TRAIN_TEXT = "今日は晴れて気持ちいい"
DEFAULT_PROBES = (
    "今日の天気は最高だ",
    "犬は公園を走っている",
)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def jaccard(left: Iterable, right: Iterable) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def observe_route(text: str, adapter: pipeline.OpenAIAdapter) -> dict:
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
        "text": text,
        "embedding": embedding,
        "source_nodes": sources,
        "activated_nodes": result.activated_nodes,
        "traversed_edges": [tuple(edge) for edge in result.traversed_edges],
    }


def direct_core_overlap(reference: dict, probe: dict) -> dict:
    node_overlap = jaccard(reference["activated_nodes"], probe["activated_nodes"])
    edge_overlap = jaccard(reference["traversed_edges"], probe["traversed_edges"])
    score = 0.35 * node_overlap + 0.65 * edge_overlap
    return {
        "core_overlap": score,
        "node_overlap": node_overlap,
        "edge_overlap": edge_overlap,
    }


def configure_isolated_data(data_dir: Path) -> None:
    """Redirect llm_core_pipeline globals to a dedicated experimental area."""
    pipeline.DATA = data_dir
    pipeline.BRAIN_FILE = data_dir / "brain.json"
    pipeline.DB_FILE = data_dir / "experiences.db"
    pipeline.PROJECTION_FILE = data_dir / "projection.npy"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "checkpoint",
        "train_text",
        "probe_text",
        "embedding_similarity",
        "core_overlap",
        "node_overlap",
        "edge_overlap",
        "reference_nodes",
        "reference_edges",
        "probe_nodes",
        "probe_edges",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html_report(path: Path, rows: list[dict], train_text: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    probes = list(dict.fromkeys(row["probe_text"] for row in rows))
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Embedding類似度とCore経路重なり",
            "活動Node数",
            "通過Edge数",
        ),
    )

    for probe in probes:
        selected = [row for row in rows if row["probe_text"] == probe]
        x = [row["checkpoint"] for row in selected]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["embedding_similarity"] * 100 for row in selected],
                mode="lines+markers",
                name=f"Embedding: {probe}",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["core_overlap"] * 100 for row in selected],
                mode="lines+markers",
                name=f"Core: {probe}",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["probe_nodes"] for row in selected],
                mode="lines+markers",
                name=f"Nodes: {probe}",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["probe_edges"] for row in selected],
                mode="lines+markers",
                name=f"Edges: {probe}",
            ),
            row=3,
            col=1,
        )

    figure.update_yaxes(title_text="類似・重なり (%)", row=1, col=1)
    figure.update_yaxes(title_text="nodes", row=2, col=1)
    figure.update_yaxes(title_text="edges", row=3, col=1)
    figure.update_xaxes(title_text="累計学習回数", row=3, col=1)
    figure.update_layout(
        height=1050,
        title=f"SphereBrain LLM→Core 段階比較<br><sup>学習文: {train_text}</sup>",
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def run(train_text: str, probes: tuple[str, ...], output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    isolated_data = run_dir / "data"
    run_dir.mkdir(parents=True, exist_ok=False)

    configure_isolated_data(isolated_data)
    pipeline.reset_experiment()
    adapter = pipeline.OpenAIAdapter()

    train_embedding, _ = pipeline.encode_text(train_text, adapter)
    probe_embeddings = {
        text: pipeline.encode_text(text, adapter)[0]
        for text in probes
    }
    embedding_scores = {
        text: cosine_similarity(train_embedding, embedding)
        for text, embedding in probe_embeddings.items()
    }

    rows: list[dict] = []
    previous_checkpoint = 0

    for checkpoint in CHECKPOINTS:
        additional = checkpoint - previous_checkpoint
        if additional > 0:
            print(f"[{checkpoint:>2}回] {additional}回を追加学習しています...")
            pipeline.experience(train_text, repeats=additional, adapter=adapter)

        reference = observe_route(train_text, adapter)
        for probe_text in probes:
            probe = observe_route(probe_text, adapter)
            overlap = direct_core_overlap(reference, probe)
            row = {
                "checkpoint": checkpoint,
                "train_text": train_text,
                "probe_text": probe_text,
                "embedding_similarity": round(embedding_scores[probe_text], 8),
                "core_overlap": round(overlap["core_overlap"], 8),
                "node_overlap": round(overlap["node_overlap"], 8),
                "edge_overlap": round(overlap["edge_overlap"], 8),
                "reference_nodes": len(reference["activated_nodes"]),
                "reference_edges": len(reference["traversed_edges"]),
                "probe_nodes": len(probe["activated_nodes"]),
                "probe_edges": len(probe["traversed_edges"]),
            }
            rows.append(row)
            print(
                f"  {probe_text}: Embedding={row['embedding_similarity']*100:.1f}% "
                f"Core={row['core_overlap']*100:.1f}% "
                f"Nodes={row['probe_nodes']} Edges={row['probe_edges']}"
            )

        previous_checkpoint = checkpoint

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "llm_core_stage_comparison_v1",
        "checkpoints": list(CHECKPOINTS),
        "train_text": train_text,
        "probes": list(probes),
        "embedding_model": pipeline.EMBEDDING_MODEL,
        "projection_seed": pipeline.PROJECTION_SEED,
        "stimulus_dimension": pipeline.STIMULUS_DIM,
        "source_count": pipeline.SOURCE_COUNT,
        "data_isolation": str(isolated_data.relative_to(ROOT)),
        "measurement": (
            "At every checkpoint, the current Core route for the training sentence "
            "is compared directly with each probe route. Probe operations use learn=False."
        ),
        "rows": rows,
    }

    json_path = run_dir / "stage_comparison.json"
    csv_path = run_dir / "stage_comparison.csv"
    html_path = run_dir / "stage_comparison.html"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_html_report(html_path, rows, train_text)

    print("\n完了しました。")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    if html_path.exists():
        print(f"HTML: {html_path}")
    print("既存の data/llm_core_v1/ は変更していません。")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM EmbeddingとSphereBrain Core経路を学習段階ごとに分離比較します。"
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN_TEXT, help="学習文")
    parser.add_argument(
        "--probe",
        action="append",
        dest="probes",
        help="検証文。複数指定できます。省略時は標準の類似文・無関係文を使います。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "llm_core_stage_comparison",
        help="結果の保存先ルート",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probes = tuple(args.probes) if args.probes else DEFAULT_PROBES
    run(args.train.strip(), tuple(text.strip() for text in probes), args.output)


if __name__ == "__main__":
    main()
