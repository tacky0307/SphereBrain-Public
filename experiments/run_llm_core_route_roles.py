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

ROLE_ORDER = (
    "全共有幹線",
    "学習・類似共有",
    "学習文専用",
    "類似文専用",
    "無関係文専用",
    "学習・無関係共有",
    "類似・無関係共有",
)


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


def role_for_presence(presence: frozenset[str]) -> str:
    mapping = {
        frozenset(("学習文", "類似文", "無関係文")): "全共有幹線",
        frozenset(("学習文", "類似文")): "学習・類似共有",
        frozenset(("学習文",)): "学習文専用",
        frozenset(("類似文",)): "類似文専用",
        frozenset(("無関係文",)): "無関係文専用",
        frozenset(("学習文", "無関係文")): "学習・無関係共有",
        frozenset(("類似文", "無関係文")): "類似・無関係共有",
    }
    return mapping[presence]


def classify(snapshot: dict[str, dict], entity_type: str) -> dict[str, set]:
    key = "nodes" if entity_type == "Node" else "edges"
    all_entities = set().union(*(snapshot[label][key] for label in TEXTS))
    roles: dict[str, set] = {role: set() for role in ROLE_ORDER}
    for entity in all_entities:
        presence = frozenset(label for label in TEXTS if entity in snapshot[label][key])
        roles[role_for_presence(presence)].add(entity)
    return roles


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["checkpoint"], row["entity_type"], row["role"])].append(row)

    summary = []
    for key, items in sorted(grouped.items()):
        counts = np.asarray([item["count"] for item in items], dtype=float)
        shares = np.asarray([item["share"] for item in items], dtype=float)
        summary.append({
            "group": key[0],
            "checkpoint": key[1],
            "entity_type": key[2],
            "role": key[3],
            "trials": len(items),
            "count_mean": float(counts.mean()),
            "count_std": float(counts.std(ddof=0)),
            "share_mean": float(shares.mean()),
            "share_std": float(shares.std(ddof=0)),
        })
    return summary


def transition_rows(group: str, trial: int, seed: int, role_snapshots: dict) -> list[dict]:
    rows = []
    for entity_type in ("Node", "Edge"):
        for index in range(1, len(CHECKPOINTS)):
            start, end = CHECKPOINTS[index - 1], CHECKPOINTS[index]
            previous = role_snapshots[start][entity_type]
            current = role_snapshots[end][entity_type]
            entities = set().union(*previous.values(), *current.values())
            previous_role = {entity: role for role, values in previous.items() for entity in values}
            current_role = {entity: role for role, values in current.items() for entity in values}
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for entity in entities:
                counts[(previous_role.get(entity, "未登場"), current_role.get(entity, "消失"))] += 1
            for (from_role, to_role), count in sorted(counts.items()):
                rows.append({
                    "group": group,
                    "trial": trial,
                    "projection_seed": seed,
                    "entity_type": entity_type,
                    "from_checkpoint": start,
                    "to_checkpoint": end,
                    "from_role": from_role,
                    "to_role": to_role,
                    "count": count,
                })
    return rows


def write_html(path: Path, summary: list[dict], transitions: list[dict]) -> None:
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
            "Nodeの役割構成比",
            "Edgeの役割構成比",
            "学習・類似共有経路の平均本数",
            "50回地点の役割構成比",
        ),
    )

    for group in ("固定射影", "異なる射影"):
        dash = "solid" if group == "固定射影" else "dash"
        for entity_type, row_index in (("Node", 1), ("Edge", 2)):
            for role in ROLE_ORDER:
                items = [r for r in summary if r["group"] == group and r["entity_type"] == entity_type and r["role"] == role]
                figure.add_trace(
                    go.Scatter(
                        x=[r["checkpoint"] for r in items],
                        y=[r["share_mean"] * 100 for r in items],
                        error_y={"type": "data", "array": [r["share_std"] * 100 for r in items]},
                        mode="lines+markers",
                        line={"dash": dash},
                        name=f"{role} / {group} / {entity_type}",
                        showlegend=(row_index == 1),
                    ),
                    row=row_index,
                    col=1,
                )

        shared_items = [r for r in summary if r["group"] == group and r["role"] == "学習・類似共有"]
        for entity_type in ("Node", "Edge"):
            items = [r for r in shared_items if r["entity_type"] == entity_type]
            figure.add_trace(
                go.Scatter(
                    x=[r["checkpoint"] for r in items],
                    y=[r["count_mean"] for r in items],
                    error_y={"type": "data", "array": [r["count_std"] for r in items]},
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"学習・類似共有 {entity_type} / {group}",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

    final_items = [r for r in summary if r["checkpoint"] == 50]
    labels = []
    values = []
    for group in ("固定射影", "異なる射影"):
        for entity_type in ("Node", "Edge"):
            for role in ROLE_ORDER:
                item = next(r for r in final_items if r["group"] == group and r["entity_type"] == entity_type and r["role"] == role)
                labels.append(f"{role}<br>{entity_type}/{group}")
                values.append(item["share_mean"] * 100)
    figure.add_trace(go.Bar(x=labels, y=values, showlegend=False), row=4, col=1)

    figure.update_yaxes(title_text="構成比 (%)", row=1, col=1)
    figure.update_yaxes(title_text="構成比 (%)", row=2, col=1)
    figure.update_yaxes(title_text="平均本数", row=3, col=1)
    figure.update_yaxes(title_text="構成比 (%)", row=4, col=1)
    figure.update_xaxes(title_text="累計学習回数", row=1, col=1)
    figure.update_xaxes(title_text="累計学習回数", row=2, col=1)
    figure.update_xaxes(title_text="累計学習回数", row=3, col=1)
    figure.update_layout(
        height=1600,
        title=f"SphereBrain 経路の役割分化<br><sup>学習文: {TRAIN_TEXT} / 各群{TRIALS_PER_GROUP}試行</sup>",
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_route_roles" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()

    all_rows: list[dict] = []
    all_transitions: list[dict] = []

    for group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if group == "固定射影" else BASE_SEED + trial * 1009
            data_dir = run_dir / "data" / group / f"trial_{trial}"
            configure_data(data_dir, seed)
            pipeline.reset_experiment()
            print(f"\n[{group} 試行{trial}] seed={seed}")

            role_snapshots: dict[int, dict[str, dict[str, set]]] = {}
            previous_checkpoint = 0
            for checkpoint in CHECKPOINTS:
                additional = checkpoint - previous_checkpoint
                if additional > 0:
                    pipeline.experience(TRAIN_TEXT, repeats=additional, adapter=adapter)
                snapshot = {label: observe(text, adapter) for label, text in TEXTS.items()}
                role_snapshots[checkpoint] = {
                    "Node": classify(snapshot, "Node"),
                    "Edge": classify(snapshot, "Edge"),
                }

                for entity_type in ("Node", "Edge"):
                    roles = role_snapshots[checkpoint][entity_type]
                    total = sum(len(values) for values in roles.values())
                    for role in ROLE_ORDER:
                        count = len(roles[role])
                        all_rows.append({
                            "group": group,
                            "trial": trial,
                            "projection_seed": seed,
                            "checkpoint": checkpoint,
                            "entity_type": entity_type,
                            "role": role,
                            "count": count,
                            "share": round(count / total if total else 0.0, 8),
                            "total_unique": total,
                        })
                print(f"  checkpoint {checkpoint} 完了")
                previous_checkpoint = checkpoint

            all_transitions.extend(transition_rows(group, trial, seed, role_snapshots))

    summary = summarize(all_rows)
    write_csv(run_dir / "route_roles_all_trials.csv", all_rows)
    write_csv(run_dir / "route_roles_summary.csv", summary)
    write_csv(run_dir / "route_role_transitions.csv", all_transitions)
    (run_dir / "route_roles.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoints": CHECKPOINTS,
            "train_text": TRAIN_TEXT,
            "texts": TEXTS,
            "role_definitions": {
                "全共有幹線": "3文すべてで使われる",
                "学習・類似共有": "学習文と類似文だけで共有される",
                "学習文専用": "学習文だけで使われる",
                "類似文専用": "類似文だけで使われる",
                "無関係文専用": "無関係文だけで使われる",
                "学習・無関係共有": "学習文と無関係文だけで共有される",
                "類似・無関係共有": "類似文と無関係文だけで共有される",
            },
            "rows": all_rows,
            "summary": summary,
            "transitions": all_transitions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(run_dir / "route_roles.html", summary, all_transitions)

    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
