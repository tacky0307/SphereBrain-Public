from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline
from experiments.run_llm_core_stage_comparison import (
    CHECKPOINTS,
    DEFAULT_PROBES,
    DEFAULT_TRAIN_TEXT,
    cosine_similarity,
    direct_core_overlap,
    observe_route,
)

DEFAULT_TRIALS = 5
BASE_PROJECTION_SEED = pipeline.PROJECTION_SEED


def configure_trial_data(data_dir: Path, projection_seed: int) -> None:
    pipeline.DATA = data_dir
    pipeline.BRAIN_FILE = data_dir / "brain.json"
    pipeline.DB_FILE = data_dir / "experiences.db"
    pipeline.PROJECTION_FILE = data_dir / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def train_encoded(
    text: str,
    embedding: list[float],
    stimulus: np.ndarray,
    repeats: int,
) -> None:
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    for _ in range(repeats):
        result = brain.propagate(
            sources,
            steps=14,
            threshold=0.18,
            noise=0.004,
            learn=True,
        )
        pipeline.save_experience(text, embedding, stimulus, sources, result)
    brain.save(pipeline.BRAIN_FILE)


def run_trial(
    trial: int,
    trial_dir: Path,
    train_text: str,
    probes: tuple[str, ...],
    adapter: pipeline.OpenAIAdapter,
    train_embedding: list[float],
    probe_embeddings: dict[str, list[float]],
) -> list[dict]:
    projection_seed = BASE_PROJECTION_SEED + trial - 1
    configure_trial_data(trial_dir / "data", projection_seed)
    pipeline.reset_experiment()

    train_stimulus = pipeline.project_embedding(train_embedding)
    embedding_scores = {
        probe: cosine_similarity(train_embedding, probe_embeddings[probe])
        for probe in probes
    }

    rows: list[dict] = []
    previous = 0
    print(f"\n===== 試行 {trial} / projection seed {projection_seed} =====")

    for checkpoint in CHECKPOINTS:
        additional = checkpoint - previous
        if additional > 0:
            print(f"[{checkpoint:>2}回] {additional}回を追加学習...")
            train_encoded(
                train_text,
                train_embedding,
                train_stimulus,
                additional,
            )

        reference = observe_route(train_text, adapter)
        for probe_text in probes:
            probe = observe_route(probe_text, adapter)
            overlap = direct_core_overlap(reference, probe)
            row = {
                "trial": trial,
                "projection_seed": projection_seed,
                "checkpoint": checkpoint,
                "train_text": train_text,
                "probe_text": probe_text,
                "embedding_similarity": embedding_scores[probe_text],
                "core_overlap": overlap["core_overlap"],
                "node_overlap": overlap["node_overlap"],
                "edge_overlap": overlap["edge_overlap"],
                "reference_nodes": len(reference["activated_nodes"]),
                "reference_edges": len(reference["traversed_edges"]),
                "probe_nodes": len(probe["activated_nodes"]),
                "probe_edges": len(probe["traversed_edges"]),
            }
            rows.append(row)
            print(
                f"  {probe_text}: Core={row['core_overlap']*100:.1f}% "
                f"Nodes={row['probe_nodes']} Edges={row['probe_edges']}"
            )
        previous = checkpoint

    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "embedding_similarity",
        "core_overlap",
        "node_overlap",
        "edge_overlap",
        "reference_nodes",
        "reference_edges",
        "probe_nodes",
        "probe_edges",
    )
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["probe_text"], row["checkpoint"]), []).append(row)

    summary: list[dict] = []
    for (probe_text, checkpoint), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        output = {
            "probe_text": probe_text,
            "checkpoint": checkpoint,
            "trial_count": len(group),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            output[f"{metric}_mean"] = statistics.fmean(values)
            output[f"{metric}_std"] = statistics.pstdev(values)
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
        summary.append(output)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict], train_text: str, trials: int) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    probes = list(dict.fromkeys(row["probe_text"] for row in summary))
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Core経路重なり：平均 ± 標準偏差",
            "活動Node数：平均 ± 標準偏差",
            "通過Edge数：平均 ± 標準偏差",
        ),
    )

    for probe in probes:
        selected = [row for row in summary if row["probe_text"] == probe]
        x = [row["checkpoint"] for row in selected]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["core_overlap_mean"] * 100 for row in selected],
                error_y={
                    "type": "data",
                    "array": [row["core_overlap_std"] * 100 for row in selected],
                    "visible": True,
                },
                mode="lines+markers",
                name=f"Core: {probe}",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["probe_nodes_mean"] for row in selected],
                error_y={
                    "type": "data",
                    "array": [row["probe_nodes_std"] for row in selected],
                    "visible": True,
                },
                mode="lines+markers",
                name=f"Nodes: {probe}",
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[row["probe_edges_mean"] for row in selected],
                error_y={
                    "type": "data",
                    "array": [row["probe_edges_std"] for row in selected],
                    "visible": True,
                },
                mode="lines+markers",
                name=f"Edges: {probe}",
            ),
            row=3,
            col=1,
        )

    figure.update_yaxes(title_text="Core overlap (%)", row=1, col=1)
    figure.update_yaxes(title_text="nodes", row=2, col=1)
    figure.update_yaxes(title_text="edges", row=3, col=1)
    figure.update_xaxes(title_text="累計学習回数", row=3, col=1)
    figure.update_layout(
        height=1050,
        hovermode="x unified",
        title=(
            f"SphereBrain LLM→Core 再現性実験（{trials}試行）"
            f"<br><sup>学習文: {train_text}</sup>"
        ),
    )
    figure.write_html(path, include_plotlyjs="cdn")


def run(
    train_text: str,
    probes: tuple[str, ...],
    trials: int,
    output_root: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    adapter = pipeline.OpenAIAdapter()
    print("Embeddingを取得しています...")
    train_embedding = adapter.embed(train_text)
    probe_embeddings = {probe: adapter.embed(probe) for probe in probes}

    all_rows: list[dict] = []
    for trial in range(1, trials + 1):
        trial_dir = run_dir / f"trial_{trial:02d}"
        all_rows.extend(
            run_trial(
                trial,
                trial_dir,
                train_text,
                probes,
                adapter,
                train_embedding,
                probe_embeddings,
            )
        )

    summary = aggregate(all_rows)
    raw_csv = run_dir / "all_trials.csv"
    summary_csv = run_dir / "summary_mean_std.csv"
    json_path = run_dir / "repeated_trials.json"
    html_path = run_dir / "repeated_trials.html"

    write_csv(raw_csv, all_rows)
    write_csv(summary_csv, summary)
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "experiment": "llm_core_repeated_trials_v1",
                "trial_count": trials,
                "checkpoints": list(CHECKPOINTS),
                "train_text": train_text,
                "probes": list(probes),
                "base_projection_seed": BASE_PROJECTION_SEED,
                "trial_seed_rule": "base_projection_seed + trial - 1",
                "embedding_model": pipeline.EMBEDDING_MODEL,
                "all_trials": all_rows,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(html_path, summary, train_text, trials)

    print("\n===== 全試行が完了しました =====")
    print(f"生データ : {raw_csv}")
    print(f"平均・標準偏差: {summary_csv}")
    print(f"JSON     : {json_path}")
    if html_path.exists():
        print(f"グラフ   : {html_path}")
    print("既存の data/llm_core_v1/ は変更していません。")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SphereBrain LLM-Core段階比較を複数の独立試行で再現します。"
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--probe", action="append", dest="probes")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "llm_core_repeated_trials",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 2:
        raise ValueError("再現性実験では2試行以上を指定してください。")
    probes = tuple(args.probes) if args.probes else DEFAULT_PROBES
    run(
        args.train.strip(),
        tuple(probe.strip() for probe in probes),
        args.trials,
        args.output,
    )


if __name__ == "__main__":
    main()
