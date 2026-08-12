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
from core_integration_shadow_v3 import CoreIntegrationShadowV3, ShadowV3Config

OUT = ROOT / "data" / "core_integration_shadow_v3" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_cases(node_count: int) -> dict[str, list[int]]:
    raw = {
        "near_origin": [0, 1, 2],
        "quarter": [node_count // 4, node_count // 4 + 1, node_count // 4 + 2],
        "middle": [node_count // 2, node_count // 2 + 1, node_count // 2 + 2],
        "three_quarters": [3 * node_count // 4, 3 * node_count // 4 + 1, 3 * node_count // 4 + 2],
        "spread": [0, node_count // 3, 2 * node_count // 3],
        "end": [node_count - 3, node_count - 2, node_count - 1],
    }
    return {name: [min(node_count - 1, node) for node in nodes] for name, nodes in raw.items()}


def main() -> None:
    if not BRAIN_PATH.exists():
        raise FileNotFoundError(f"brain file not found: {BRAIN_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)

    before_hash = file_hash(BRAIN_PATH)
    brain = SphereBrain.load(BRAIN_PATH)
    config = ShadowV3Config()
    engine = CoreIntegrationShadowV3(config)

    cases = []
    rows = []
    for name, sources in source_cases(brain.node_count).items():
        first = engine.run(brain, sources)
        second = engine.run(brain, sources)
        first["repeatable"] = first == second
        cases.append({"name": name, "sources": sources, "result": first})
        rows.append({
            "case": name,
            "route_edge_symmetric_difference": first["route_edge_symmetric_difference"],
            "route_edge_jaccard": first["route_edge_jaccard"],
            "activated_node_symmetric_difference": first["activated_node_symmetric_difference"],
            "activated_node_jaccard": first["activated_node_jaccard"],
            "step_count_difference": first["step_count_difference"],
            "strong_decision_override_count": first["strong_decision_override_count"],
            "tie_resolved_by_structure_count": first["tie_resolved_by_structure_count"],
            "selected_set_difference_count": first["selected_set_difference_count"],
            "max_absolute_modulation": first["max_absolute_modulation"],
            "max_meaningful_relative_ratio": first["max_meaningful_relative_ratio"],
            "core_unchanged": first["core_unchanged"],
            "repeatable": first["repeatable"],
        })

    after_hash = file_hash(BRAIN_PATH)
    results = [case["result"] for case in cases]
    safe_for_core_flag = all(
        result["core_unchanged"]
        and result["repeatable"]
        and result["strong_decision_override_count"] == 0
        and result["selected_set_difference_count"] == 0
        and result["max_absolute_modulation"] <= config.absolute_cap + 1e-15
        and result["max_meaningful_relative_ratio"] <= config.relative_cap_ratio + 1e-12
        and result["route_edge_jaccard"] == 1.0
        and result["activated_node_jaccard"] == 1.0
        and result["step_count_difference"] == 0
        for result in results
    )

    payload = {
        "experiment": "Core Integration Shadow v3",
        "brain_path": str(BRAIN_PATH.relative_to(ROOT)),
        "brain_file_sha256_before": before_hash,
        "brain_file_sha256_after": after_hash,
        "read_only": True,
        "core_runtime_modified": False,
        "design": "Absolute and meaningful-relative caps, near-zero tie classification, strong-decision protection, and selected-set safety checks.",
        "config": {
            "gain": config.gain,
            "tie_margin": config.tie_margin,
            "near_zero_margin": config.near_zero_margin,
            "relative_cap_ratio": config.relative_cap_ratio,
            "absolute_cap": config.absolute_cap,
        },
        "cases": cases,
        "safe_for_core_feature_flag": safe_for_core_flag,
        "checks": {
            "brain_file_unchanged": before_hash == after_hash,
            "all_core_unchanged": all(result["core_unchanged"] for result in results),
            "all_runs_repeatable": all(result["repeatable"] for result in results),
            "absolute_cap_respected": all(result["max_absolute_modulation"] <= config.absolute_cap + 1e-15 for result in results),
            "meaningful_relative_cap_respected": all(result["max_meaningful_relative_ratio"] <= config.relative_cap_ratio + 1e-12 for result in results),
            "no_strong_decision_override": all(result["strong_decision_override_count"] == 0 for result in results),
            "no_selected_set_change": all(result["selected_set_difference_count"] == 0 for result in results),
            "routes_unchanged": all(result["route_edge_jaccard"] == 1.0 for result in results),
            "activated_nodes_unchanged": all(result["activated_node_jaccard"] == 1.0 for result in results),
        },
    }

    json_path = OUT / "core_integration_shadow_v3.json"
    csv_path = OUT / "core_integration_shadow_v3.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Core Integration Shadow v3")
    print(f"brain unchanged: {before_hash == after_hash}")
    print(f"safe for Core feature flag: {safe_for_core_flag}")
    print(f"checks: {payload['checks']}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
