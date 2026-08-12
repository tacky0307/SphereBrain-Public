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

TRIALS_PER_GROUP = 5
BASE_SEED = pipeline.PROJECTION_SEED
REPEATS_PER_EXPERIENCE = 20

EXPERIENCE_GROUPS = {
    "晴れ・快適": [
        "今日は晴れて気持ちいい",
        "青い空が広がって爽やかだ",
        "暖かな日差しが心地よい",
    ],
    "雨・不快": [
        "今日は雨で肌寒い",
        "暗い空から雨が降っている",
        "冷たい雨で気分が沈む",
    ],
}

PROBES = {
    "晴れ未経験": "よく晴れた穏やかな日だ",
    "雨未経験": "雨雲が広がって寒く感じる",
    "曖昧な天気": "今日は天気が変わりやすい",
    "無関係": "犬が庭で遊んでいる",
}


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
        "text": text,
        "embedding": np.asarray(embedding, dtype=float),
        "nodes": set(result.activated_nodes),
        "edges": {tuple(edge) for edge in result.traversed_edges},
        "node_count": len(result.activated_nodes),
        "edge_count": len(result.traversed_edges),
    }


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def route_overlap(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def mean_pairwise(routes: list[dict]) -> float:
    values = []
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            values.append(route_overlap(routes[i], routes[j]))
    return float(np.mean(values)) if values else 0.0


def group_shared_edges(routes: list[dict]) -> set[tuple[int, int]]:
    if not routes:
        return set()
    shared = set(routes[0]["edges"])
    for route in routes[1:]:
        shared &= route["edges"]
    return shared


def group_union_edges(routes: list[dict]) -> set[tuple[int, int]]:
    output: set[tuple[int, int]] = set()
    for route in routes:
        output |= route["edges"]
    return output


def summarize(rows: list[dict], keys: tuple[str, ...], fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, items in sorted(grouped.items()):
        record = {key: value for key, value in zip(keys, key_values)}
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


def write_html(path: Path, affinity_summary: list[dict], structure_summary: list[dict]) -> None:
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
            "未経験ProbeのCore経路親和性",
            "Embedding親和性（固定基準）",
            "経験群内の経路凝集度：学習前と学習後",
            "群固有共有Edgeと天気共通Edge",
        ),
    )

    probe_order = list(PROBES)
    group_order = list(EXPERIENCE_GROUPS)
    for projection_group, dash in (("固定射影", "solid"), ("異なる射影", "dash")):
        for target_group in group_order:
            items = {
                row["probe_label"]: row
                for row in affinity_summary
                if row["projection_group"] == projection_group and row["target_group"] == target_group
            }
            figure.add_trace(
                go.Scatter(
                    x=probe_order,
                    y=[items[p]["core_affinity_mean"] * 100 for p in probe_order],
                    error_y={"type": "data", "array": [items[p]["core_affinity_std"] * 100 for p in probe_order]},
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"Core→{target_group}/{projection_group}",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=probe_order,
                    y=[items[p]["embedding_affinity_mean"] * 100 for p in probe_order],
                    error_y={"type": "data", "array": [items[p]["embedding_affinity_std"] * 100 for p in probe_order]},
                    mode="lines+markers",
                    line={"dash": dash},
                    name=f"Embedding→{target_group}/{projection_group}",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        structures = {
            (row["metric"], row["concept_group"]): row
            for row in structure_summary
            if row["projection_group"] == projection_group
        }
        x = [f"{g} 学習前" for g in group_order] + [f"{g} 学習後" for g in group_order]
        y = [structures[("cohesion_before", g)]["value_mean"] * 100 for g in group_order] + [structures[("cohesion_after", g)]["value_mean"] * 100 for g in group_order]
        err = [structures[("cohesion_before", g)]["value_std"] * 100 for g in group_order] + [structures[("cohesion_after", g)]["value_std"] * 100 for g in group_order]
        figure.add_trace(
            go.Scatter(x=x, y=y, error_y={"type": "data", "array": err}, mode="lines+markers", line={"dash": dash}, name=f"凝集度/{projection_group}", showlegend=False),
            row=3,
            col=1,
        )

        edge_x = ["晴れ固有共有", "雨固有共有", "天気共通"]
        edge_y = [
            structures[("sunny_specific_shared_edges", "全体")]["value_mean"],
            structures[("rain_specific_shared_edges", "全体")]["value_mean"],
            structures[("weather_common_edges", "全体")]["value_mean"],
        ]
        edge_err = [
            structures[("sunny_specific_shared_edges", "全体")]["value_std"],
            structures[("rain_specific_shared_edges", "全体")]["value_std"],
            structures[("weather_common_edges", "全体")]["value_std"],
        ]
        figure.add_trace(
            go.Bar(x=edge_x, y=edge_y, error_y={"type": "data", "array": edge_err}, name=projection_group, opacity=0.75),
            row=4,
            col=1,
        )

    figure.update_yaxes(title_text="Core親和性 (%)", row=1, col=1)
    figure.update_yaxes(title_text="Embedding親和性 (%)", row=2, col=1)
    figure.update_yaxes(title_text="群内重なり (%)", row=3, col=1)
    figure.update_yaxes(title_text="Edge本数", row=4, col=1)
    figure.update_layout(
        height=1450,
        title=(
            "SphereBrain 複数経験からの概念群形成"
            f"<br><sup>各経験文{REPEATS_PER_EXPERIENCE}回 / 各射影群{TRIALS_PER_GROUP}試行</sup>"
        ),
        hovermode="x unified",
    )
    figure.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "results" / "llm_core_concept_groups" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter = pipeline.OpenAIAdapter()

    affinity_rows: list[dict] = []
    structure_rows: list[dict] = []

    for projection_group in ("固定射影", "異なる射影"):
        for trial in range(1, TRIALS_PER_GROUP + 1):
            seed = BASE_SEED if projection_group == "固定射影" else BASE_SEED + trial * 1009
            configure_data(run_dir / "data" / projection_group / f"trial_{trial}", seed)
            pipeline.reset_experiment()
            print(f"\n[{projection_group} 試行{trial}] seed={seed}")

            before_groups = {
                group: [observe(text, adapter) for text in texts]
                for group, texts in EXPERIENCE_GROUPS.items()
            }

            for group, texts in EXPERIENCE_GROUPS.items():
                for text in texts:
                    pipeline.experience(text, repeats=REPEATS_PER_EXPERIENCE, adapter=adapter)
                print(f"  {group} 学習完了")

            after_groups = {
                group: [observe(text, adapter) for text in texts]
                for group, texts in EXPERIENCE_GROUPS.items()
            }
            probes = {label: observe(text, adapter) for label, text in PROBES.items()}

            for group in EXPERIENCE_GROUPS:
                structure_rows.extend([
                    {"projection_group": projection_group, "trial": trial, "concept_group": group, "metric": "cohesion_before", "value": mean_pairwise(before_groups[group])},
                    {"projection_group": projection_group, "trial": trial, "concept_group": group, "metric": "cohesion_after", "value": mean_pairwise(after_groups[group])},
                ])

            sunny_shared = group_shared_edges(after_groups["晴れ・快適"])
            rainy_shared = group_shared_edges(after_groups["雨・不快"])
            sunny_union = group_union_edges(after_groups["晴れ・快適"])
            rainy_union = group_union_edges(after_groups["雨・不快"])
            weather_common = sunny_union & rainy_union
            structure_rows.extend([
                {"projection_group": projection_group, "trial": trial, "concept_group": "全体", "metric": "sunny_specific_shared_edges", "value": len(sunny_shared - rainy_union)},
                {"projection_group": projection_group, "trial": trial, "concept_group": "全体", "metric": "rain_specific_shared_edges", "value": len(rainy_shared - sunny_union)},
                {"projection_group": projection_group, "trial": trial, "concept_group": "全体", "metric": "weather_common_edges", "value": len(weather_common)},
            ])

            for probe_label, probe in probes.items():
                for target_group, routes in after_groups.items():
                    core_values = [route_overlap(probe, route) for route in routes]
                    embedding_values = [cosine(probe["embedding"], route["embedding"]) for route in routes]
                    affinity_rows.append({
                        "projection_group": projection_group,
                        "trial": trial,
                        "probe_label": probe_label,
                        "probe_text": PROBES[probe_label],
                        "target_group": target_group,
                        "core_affinity": float(np.mean(core_values)),
                        "core_affinity_max": float(np.max(core_values)),
                        "embedding_affinity": float(np.mean(embedding_values)),
                        "embedding_affinity_max": float(np.max(embedding_values)),
                        "probe_nodes": probe["node_count"],
                        "probe_edges": probe["edge_count"],
                    })

            print("  Probe測定完了")

    affinity_summary = summarize(
        affinity_rows,
        ("projection_group", "probe_label", "target_group"),
        ("core_affinity", "core_affinity_max", "embedding_affinity", "embedding_affinity_max", "probe_nodes", "probe_edges"),
    )
    structure_summary = summarize(
        structure_rows,
        ("projection_group", "concept_group", "metric"),
        ("value",),
    )

    write_csv(run_dir / "concept_group_affinity_trials.csv", affinity_rows)
    write_csv(run_dir / "concept_group_affinity_summary.csv", affinity_summary)
    write_csv(run_dir / "concept_group_structure_trials.csv", structure_rows)
    write_csv(run_dir / "concept_group_structure_summary.csv", structure_summary)
    (run_dir / "concept_groups.json").write_text(
        json.dumps({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experience_groups": EXPERIENCE_GROUPS,
            "probes": PROBES,
            "repeats_per_experience": REPEATS_PER_EXPERIENCE,
            "affinity_rows": affinity_rows,
            "affinity_summary": affinity_summary,
            "structure_rows": structure_rows,
            "structure_summary": structure_summary,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(run_dir / "concept_groups.html", affinity_summary, structure_summary)

    print("\n完了しました。")
    print(f"結果: {run_dir}")
    print("既存の data/llm_core_v1/ は変更していません。")


if __name__ == "__main__":
    main()
