from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain
from core_integration_shadow_v2 import CoreIntegrationShadowV2, ShadowV2Config

OUT = ROOT / "data" / "core_integration_shadow_v2" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_cases(node_count: int) -> dict[str, list[int]]:
    candidates = {
        "near_origin": [0, 1, 2],
        "quarter": [node_count // 4, node_count // 4 + 1, node_count // 4 + 2],
        "middle": [node_count // 2, node_count // 2 + 1, node_count // 2 + 2],
        "three_quarters": [3 * node_count // 4, 3 * node_count // 4 + 1, 3 * node_count // 4 + 2],
        "spread": [0, node_count // 3, 2 * node_count // 3],
        "end": [node_count - 3, node_count - 2, node_count - 1],
    }
    return {name: [min(node_count - 1, node) for node in nodes] for name, nodes in candidates.items()}


def main() -> None:
    if not BRAIN_PATH.exists():
        raise FileNotFoundError(f"brain file not found: {BRAIN_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)

    before_hash = file_hash(BRAIN_PATH)
    brain = SphereBrain.load(BRAIN_PATH)
    config = ShadowV2Config()
    engine = CoreIntegrationShadowV2(config)

    cases = []
    rows = []
    for name, sources in source_cases(brain.node_count).items():
        gain_runs = []
        for gain in config.gains:
            first = engine.run_gain(brain, sources, gain)
            second = engine.run_gain(brain, sources, gain)
            repeatable = first == second
            first["repeatable"] = repeatable
            gain_runs.append(first)
            rows.append({
                "case": name,
                "gain": gain,
                "route_edge_symmetric_difference": first["route_edge_symmetric_difference"],
                "route_edge_jaccard": first["route_edge_jaccard"],
                "activated_node_symmetric_difference": first["activated_node_symmetric_difference"],
                "activated_node_jaccard": first["activated_node_jaccard"],
                "step_count_difference": first["step_count_difference"],
                "shadow_top_change_ratio": first["shadow_top_change_ratio"],
                "tie_gate_activation_count": first["tie_gate_activation_count"],
                "max_modulation_to_margin_ratio": first["max_modulation_to_margin_ratio"],
                "core_unchanged": first["core_unchanged"],
                "repeatable": repeatable,
            })
        cases.append({"name": name, "sources": sources, "gains": gain_runs})

    after_hash = file_hash(BRAIN_PATH)
    all_runs = [run for case in cases for run in case["gains"]]

    # Candidate integration range: no Core mutation, deterministic, modulation
    # bounded, and route divergence remains limited across all source regions.
    safe_gains = []
    for gain in config.gains:
        selected = [run for run in all_runs if run["gain"] == gain]
        if all(
            run["core_unchanged"]
            and run["repeatable"]
            and run["max_modulation_to_margin_ratio"] <= config.modulation_cap_ratio + 1e-12
            and run["route_edge_jaccard"] >= 0.70
            and run["activated_node_jaccard"] >= 0.80
            for run in selected
        ):
            safe_gains.append(gain)

    payload = {
        "experiment": "Core Integration Shadow v2",
        "brain_path": str(BRAIN_PATH.relative_to(ROOT)),
        "brain_file_sha256_before": before_hash,
        "brain_file_sha256_after": after_hash,
        "read_only": True,
        "core_runtime_modified": False,
        "design": "Corrected structural context, tie-only bounded modulation, gain sweep, and fully diverging virtual routes beside an untouched Core.",
        "config": {
            "gains": list(config.gains),
            "tie_margin": config.tie_margin,
            "modulation_cap_ratio": config.modulation_cap_ratio,
        },
        "cases": cases,
        "safe_gain_candidates": safe_gains,
        "checks": {
            "brain_file_unchanged": before_hash == after_hash,
            "all_core_unchanged": all(run["core_unchanged"] for run in all_runs),
            "all_runs_repeatable": all(run["repeatable"] for run in all_runs),
            "cycle_false_positive_removed": all(
                run["baseline"]["records"][0]["closed_cycle"] == 0.0
                and run["shadow"]["records"][0]["closed_cycle"] == 0.0
                for run in all_runs
                if run["baseline"]["records"] and run["shadow"]["records"]
            ),
            "modulation_always_bounded": all(
                run["max_modulation_to_margin_ratio"] <= config.modulation_cap_ratio + 1e-12
                for run in all_runs
            ),
            "tie_gate_only": all(
                not record["tie_gate_active"] or (record["baseline_margin"] is not None and record["baseline_margin"] <= config.tie_margin)
                for run in all_runs for record in run["shadow"]["records"]
            ),
        },
    }

    json_path = OUT / "core_integration_shadow_v2.json"
    csv_path = OUT / "core_integration_shadow_v2.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Core Integration Shadow v2")
    print(f"brain unchanged: {before_hash == after_hash}")
    print(f"safe gain candidates: {safe_gains}")
    print(f"checks: {payload['checks']}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
