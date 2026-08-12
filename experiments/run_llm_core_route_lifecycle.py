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


def longest_consecutive(indices: list[int]) -> int:
    if not indices:
        return 0
    longest = current = 1
    for previous, current_index in zip(indices, indices[1:]):
        if current_index == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def lifecycle_rows(
    snapshots: dict[int, dict[str, dict]],
    group: str,
    trial: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    final_index = len(CHECKPOINTS) - 1

    for label, text in TEXTS.items():
        for entity_type, key in (("node", "nodes"), ("edge", "edges")):
            universe = set().union(*(snapshots[c][label][key] for c in CHECKPOINTS))
            for entity in universe:
                present_indices = [
                    index
                    for index, checkpoint in enumerate(CHECKPOINTS)
                    if entity in snapshots[checkpoint][label][key]
                ]
                first_index = present_indices[0]
                last_index = present_indices[-1]
                appearances = len(present_indices)
                possible_from_birth = len(CHECKPOINTS) - first_index
                survival_fraction = appearances / possible_from_birth
                survived_to_final = final_index in present_indices
                continuous_to_final = present_indices == list(range(first_index, final_index + 1))
                longest_run = longest_consecutive(present_indices)

                if first_index == 0 and continuous_to_final:
                    category = "中核経路"
                elif first_index > 0 and continuous_to_final:
                    category = "定着した成長経路"
                elif survived_to_final:
                    category = "再出現・可変経路"
                else:
                    category = "一時経路"

                rows.append(
                    {
                        "group": group,
                        "trial": trial,
                        "projection_seed": seed,
                        "label": label,
                        "text": text,
                        "entity_type": entity_type,
                        "entity": json.dumps(entity, ensure_ascii=False),
                        "first_checkpoint": CHECKPOINTS[first_index],
                        "last_checkpoint": CHECKPOINTS[last_index],
                        "appearances": appearances,
                        "possible_stages_from_birth": possible_from_birth,
                        "survival_fraction": round(survival_fraction, 8),
                        "longest_consecutive_stages": longest_run,
                        "survived_to_final": int(survived_to_final),
                        "continuous_to_final": int(continuous_to_final),
                        "category": category,
                    }
                )
    return rows


def summarize_entities(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["group"],
            row["label"],
            row["entity_type"],
            row["first_checkpoint"],
        )
        grouped[key].append(row)

    summary: list[dict] = []
    for key, items in sorted(grouped.items()):
        survival = np.asarray([item["survival_fraction"] for item in items], dtype=float)
        longest = np.asarray([item["longest_consecutive_stages"] for item in items], dtype=float)
        final = np.asarray([item["survived_to_final"] for item in items], dtype=float)
        continuous = np.asarray([item["continuous_to_final"] for item in items], dtype=float)
        summary.append(
            {
                "group": key[0],
                "label": key[1],
                "entity_type": key[2],
                "first_checkpoint": key[3],
                "entities": len(items),
                "survival_fraction_mean": float(survival.mean()),
                "survival_fraction_std": float(survival.std(ddof=0)),
                "longest_consecutive_mean": float(longest.mean()),
                "longest_consecutive_std": float(longest.std(ddof=0)),
                "final_survival_rate": float(final.mean()),
                "continuous_to_final_rate": float(continuous.mean()),
            }
        )
    return summary


def summarize_categories(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["group"], row["label"], row["entity_type"], row["category"])
        grouped[key].append(row)

    totals: dict[tuple, int] = defaultdict(int)
    for row in rows:
        totals[(row["group"], row["label"], row["entity_type"])] += 1

    result = []
    for key, items in sorted(grouped.items()):
        total = totals[(key[0], key[1], key[2])]
        result.append(
            {
                "group": key[0],
                "label": key[1],
                "entity_type": key[2],
                "category": key[3],
                "count": len(items),
                "share": len(items) / total if total else 0.0,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: list[dict], categories: list[dict]) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "初登場段階別：最終段階まで残った割合",
            "初登場段階別：誕生後の平均生存割合",
            "初登場段階別：平均連続生存段階数",
            "経路の分類構成（中核・定着・可変・一時）",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        dash = "solid" if group == "固定射影" else "dash"
        for label in TEXTS:
            for entity_type in ("node", "edge"):
                items = [
                    row for row in summary
                    if row["group"] == group
                    and row["label"] == label
                    and row["entity_type"] == entity_type
                ]
                items.sort(key=lambda row: row["first_checkpoint"])
                x = [row["first_checkpoint"] for row in items]
                entity_name = "Node" if entity_type == "node" else "Edge"
                name = f"{entity_name}: {label} / {group}"
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[row["final_survival_rate"] * 100 for row in items],
                        mode="lines+markers",
                        line={"dash": dash},
                        name=name,
                    ),
                    row=1,
                    col=1,
                )
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[row["survival_fraction_mean"] * 100 for row in items],
                        error_y={
                            "type": "data",
                            "array": [row["survival_fraction_std"] * 100 for row in items],
                        },
                        mode="lines+markers",
                        line={"dash": dash},
                        name=name,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=[row["longest_consecutive_mean"] for row in items],
                        error_y={
                            "type": "data",
                            "array": [row["longest_consecutive_std"] for row in items],
                        },
                        mode="lines+markers",
                        line={"dash": dash},
                        name=name,
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

    category_order = ("中核経路", "定着した成長経路", "再出現・可変経路", "一時経路")
    for group in ("固定射影", "異なる射影"):
        for label in TEXTS:
            for entity_type in ("node", "edge"):
                selected = [
                    row for row in categories
                    if row["group"] == group
                    and row["label"] == label
                    and row["entity_type"] == entity_type
                ]
                shares = {row["category"]: row["share"] * 100 for row in selected}
                x_name = f"{label}<br>{'Node' if entity_type == 'node' else 'Edge'} / {group}"
                for category in category_order:
                    figure.add_trace(
                        go.Bar(
                            x=[x_name],
                            y=[shares.get(category, 0.0)],
                            name=category,
                            legendgroup=category,
                            showlegend=(group == "固定射影" and label == "学習文" and entity_type == "node"),
                        ),
                        row=4,
                        col=1,
                    )

    figure.update_yaxes(title_text="最終残存率 (%)", row=1, col=1)
    figure.update_yaxes(title_text="平均生存割合 (%)", row=2, col=1)
    figure.update_yaxes(title_text="連続段階数", row=3, col=1)
    figure.update_yaxes(title_text="構成比 (%)", row=4, col=1)
    figure.update_xaxes(title_text="初登場した累計学習回数", row=1, col=1)
    figure.update_xaxes(title_text="初登場した累計学習回数", row=2, col=1)
    figure.update_xaxes(title_text="初登場した累計学習回数", row=3, col=1)
    figure.update_layout(
        height=1500,
        barmode="stack",
        title=(
            "SphereBrain 経路の寿命と定着度"
            f"<br><sup>学習文: {TRAIN_TEXT} / 固定射影・異なる射影 各{TRIALS_PER_GROUP}試行</sup>"
        ),
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_route_lifecycle" / timestamp
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
                snapshots[checkpoint] = {
                    label: observe(text, adapter)
                    for label, text in TEXTS.items()
                }
                print(f"  checkpoint {checkpoint} 完了")
                previous_checkpoint = checkpoint

            all_rows.extend(lifecycle_rows(snapshots, group, trial, seed))

    summary = summarize_entities(all_rows)
    categories = summarize_categories(all_rows)
    write_csv(run_dir / "route_lifecycle_entities.csv", all_rows)
    write_csv(run_dir / "route_lifecycle_summary.csv", summary)
    write_csv(run_dir / "route_lifecycle_categories.csv", categories)
    (run_dir / "route_lifecycle.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "checkpoints": CHECKPOINTS,
                "train_text": TRAIN_TEXT,
                "texts": TEXTS,
                "rows": all_rows,
                "summary": summary,
                "categories": categories,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(run_dir / "route_lifecycle.html", summary, categories)

    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
