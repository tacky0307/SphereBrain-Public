from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

CHECKPOINTS = (0, 1, 5, 10, 20, 50)
DEFAULT_TRAIN_TEXT = "今日は晴れて気持ちいい"
DEFAULT_PROBES = (
    ("train", "今日は晴れて気持ちいい"),
    ("similar", "今日の天気は最高だ"),
    ("unrelated", "犬は公園を走っている"),
)


class CachedAdapter:
    """Cache embeddings so repeated trials do not repeat API calls."""

    def __init__(self) -> None:
        self.base = pipeline.OpenAIAdapter()
        self.cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        clean = text.strip()
        if clean not in self.cache:
            self.cache[clean] = self.base.embed(clean)
        return list(self.cache[clean])

    def decode(self, observation: dict) -> str:
        return self.base.decode(observation)


def jaccard(left: Iterable, right: Iterable) -> float:
    a = set(left)
    b = set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def route_delta(baseline: dict, current: dict) -> dict:
    base_nodes = set(baseline["activated_nodes"])
    now_nodes = set(current["activated_nodes"])
    base_edges = set(tuple(edge) for edge in baseline["traversed_edges"])
    now_edges = set(tuple(edge) for edge in current["traversed_edges"])

    node_retention = jaccard(base_nodes, now_nodes)
    edge_retention = jaccard(base_edges, now_edges)
    combined_retention = 0.35 * node_retention + 0.65 * edge_retention

    return {
        "baseline_retention": combined_retention,
        "baseline_change": 1.0 - combined_retention,
        "node_retention": node_retention,
        "edge_retention": edge_retention,
        "added_nodes": len(now_nodes - base_nodes),
        "removed_nodes": len(base_nodes - now_nodes),
        "added_edges": len(now_edges - base_edges),
        "removed_edges": len(base_edges - now_edges),
    }


def configure_data(data_dir: Path, projection_seed: int) -> None:
    pipeline.DATA = data_dir
    pipeline.BRAIN_FILE = data_dir / "brain.json"
    pipeline.DB_FILE = data_dir / "experiences.db"
    pipeline.PROJECTION_FILE = data_dir / "projection.npy"
    pipeline.PROJECTION_SEED = projection_seed


def observe(text: str, adapter: CachedAdapter) -> dict:
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
        "source_nodes": list(sources),
        "activated_nodes": list(result.activated_nodes),
        "traversed_edges": [tuple(edge) for edge in result.traversed_edges],
    }


def run_trial(
    *,
    group: str,
    trial: int,
    projection_seed: int,
    noise_seed: int,
    train_text: str,
    probes: tuple[tuple[str, str], ...],
    trial_dir: Path,
    adapter: CachedAdapter,
) -> list[dict]:
    random.seed(noise_seed)
    np.random.seed(noise_seed)
    configure_data(trial_dir / "data", projection_seed)
    pipeline.reset_experiment()

    baselines = {role: observe(text, adapter) for role, text in probes}
    rows: list[dict] = []
    previous = 0

    print(f"\n[{group} / trial {trial}] projection={projection_seed} noise={noise_seed}")
    for checkpoint in CHECKPOINTS:
        additional = checkpoint - previous
        if additional > 0:
            pipeline.experience(train_text, repeats=additional, adapter=adapter)

        for role, text in probes:
            current = observe(text, adapter)
            delta = route_delta(baselines[role], current)
            row = {
                "group": group,
                "trial": trial,
                "projection_seed": projection_seed,
                "noise_seed": noise_seed,
                "checkpoint": checkpoint,
                "role": role,
                "text": text,
                **{key: round(value, 8) if isinstance(value, float) else value for key, value in delta.items()},
                "current_nodes": len(current["activated_nodes"]),
                "current_edges": len(current["traversed_edges"]),
                "baseline_nodes": len(baselines[role]["activated_nodes"]),
                "baseline_edges": len(baselines[role]["traversed_edges"]),
            }
            rows.append(row)
            print(
                f"  {checkpoint:>2}回 {role:<9} "
                f"change={row['baseline_change']*100:5.1f}% "
                f"node={100*(1-row['node_retention']):5.1f}% "
                f"edge={100*(1-row['edge_retention']):5.1f}% "
                f"+N={row['added_nodes']:>2} +E={row['added_edges']:>2}"
            )
        previous = checkpoint

    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["checkpoint"], row["role"])].append(row)

    metrics = (
        "baseline_change",
        "node_retention",
        "edge_retention",
        "added_nodes",
        "removed_nodes",
        "added_edges",
        "removed_edges",
        "current_nodes",
        "current_edges",
    )
    output: list[dict] = []
    for (group, checkpoint, role), items in sorted(grouped.items()):
        summary = {
            "group": group,
            "checkpoint": checkpoint,
            "role": role,
            "trials": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = round(mean(values), 8)
            summary[f"{metric}_std"] = round(pstdev(values), 8)
        output.append(summary)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict], train_text: str) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "0回時点の自分自身の経路からの変化量",
            "Node経路の変化量",
            "Edge経路の変化量",
            "追加されたNode / Edge数",
        ),
        vertical_spacing=0.08,
    )

    labels = {"train": "学習文", "similar": "類似文", "unrelated": "無関係文"}
    dashes = {"fixed_projection": "solid", "varied_projection": "dash"}

    for group in ("fixed_projection", "varied_projection"):
        for role in ("train", "similar", "unrelated"):
            selected = [row for row in summary if row["group"] == group and row["role"] == role]
            if not selected:
                continue
            x = [row["checkpoint"] for row in selected]
            name_base = f"{labels[role]} / {'固定射影' if group == 'fixed_projection' else '異なる射影'}"

            def add_error(metric_mean: str, metric_std: str, title: str, row_number: int) -> None:
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[100 * row[metric_mean] for row in selected],
                        error_y={
                            "type": "data",
                            "array": [100 * row[metric_std] for row in selected],
                            "visible": True,
                        },
                        mode="lines+markers",
                        line={"dash": dashes[group]},
                        name=f"{title}: {name_base}",
                    ),
                    row=row_number,
                    col=1,
                )

            add_error("baseline_change_mean", "baseline_change_std", "全経路", 1)
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[100 * (1 - row["node_retention_mean"]) for row in selected],
                    error_y={"type": "data", "array": [100 * row["node_retention_std"] for row in selected], "visible": True},
                    mode="lines+markers",
                    line={"dash": dashes[group]},
                    name=f"Node変化: {name_base}",
                ),
                row=2,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[100 * (1 - row["edge_retention_mean"]) for row in selected],
                    error_y={"type": "data", "array": [100 * row["edge_retention_std"] for row in selected], "visible": True},
                    mode="lines+markers",
                    line={"dash": dashes[group]},
                    name=f"Edge変化: {name_base}",
                ),
                row=3,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[row["added_nodes_mean"] for row in selected],
                    mode="lines+markers",
                    line={"dash": dashes[group]},
                    name=f"追加Node: {name_base}",
                ),
                row=4,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[row["added_edges_mean"] for row in selected],
                    mode="lines+markers",
                    line={"dash": "dot"},
                    name=f"追加Edge: {name_base}",
                ),
                row=4,
                col=1,
            )

    figure.update_yaxes(title_text="変化量 (%)", row=1, col=1)
    figure.update_yaxes(title_text="Node変化 (%)", row=2, col=1)
    figure.update_yaxes(title_text="Edge変化 (%)", row=3, col=1)
    figure.update_yaxes(title_text="追加数", row=4, col=1)
    figure.update_xaxes(title_text="累計学習回数", row=4, col=1)
    figure.update_layout(
        height=1500,
        title=f"SphereBrain 基準経路変化実験<br><sup>学習文: {train_text} / 平均 ± 標準偏差</sup>",
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def run(train_text: str, trials: int, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    probes = tuple(
        (role, train_text if role == "train" else text)
        for role, text in DEFAULT_PROBES
    )
    adapter = CachedAdapter()
    all_rows: list[dict] = []
    base_projection = int(pipeline.PROJECTION_SEED)

    for group in ("fixed_projection", "varied_projection"):
        for trial in range(1, trials + 1):
            projection_seed = base_projection if group == "fixed_projection" else base_projection + trial * 1009
            noise_seed = 20260805 + trial * 97 + (0 if group == "fixed_projection" else 10000)
            trial_dir = run_dir / group / f"trial_{trial:02d}"
            all_rows.extend(
                run_trial(
                    group=group,
                    trial=trial,
                    projection_seed=projection_seed,
                    noise_seed=noise_seed,
                    train_text=train_text,
                    probes=probes,
                    trial_dir=trial_dir,
                    adapter=adapter,
                )
            )

    summary = summarize(all_rows)
    all_csv = run_dir / "baseline_route_change_all_trials.csv"
    summary_csv = run_dir / "baseline_route_change_summary.csv"
    json_path = run_dir / "baseline_route_change.json"
    html_path = run_dir / "baseline_route_change.html"

    write_csv(all_csv, all_rows)
    write_csv(summary_csv, summary)
    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "experiment": "llm_core_baseline_route_change_v1",
                "train_text": train_text,
                "checkpoints": list(CHECKPOINTS),
                "trials_per_group": trials,
                "groups": {
                    "fixed_projection": "射影条件を固定し、試行ごとの揺らぎを測る",
                    "varied_projection": "射影条件を変え、入力配置を超えて傾向が再現するか測る",
                },
                "measurement": "各文の0回時点経路を基準とし、各学習段階の同じ文の経路変化をJaccard距離で測定する。",
                "rows": all_rows,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(html_path, summary, train_text)

    print("\n実験が完了しました。")
    print(f"ALL CSV : {all_csv}")
    print(f"SUMMARY : {summary_csv}")
    print(f"JSON    : {json_path}")
    if html_path.exists():
        print(f"HTML    : {html_path}")
    print("既存の data/llm_core_v1/ は変更していません。")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0回時点の基準経路からの変化量を測定します。")
    parser.add_argument("--train", default=DEFAULT_TRAIN_TEXT, help="学習文")
    parser.add_argument("--trials", type=int, default=5, help="各群の試行回数")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "llm_core_baseline_route_change",
        help="結果保存先",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.train.strip(), max(1, args.trials), args.output)


if __name__ == "__main__":
    main()
