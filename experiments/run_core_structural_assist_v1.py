from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

OUT = ROOT / "data" / "core_structural_assist_v1" / "results"
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


def result_snapshot(result) -> dict:
    return {
        "source_nodes": result.source_nodes,
        "activated_nodes": result.activated_nodes,
        "traversed_edges": [list(edge) for edge in result.traversed_edges],
        "activation_history": result.activation_history,
        "final_activation": result.final_activation.tolist(),
    }


def run_once(sources: list[int], enabled: bool) -> tuple[dict, list[dict]]:
    brain = SphereBrain.load(BRAIN_PATH)
    brain.set_structural_assist(enabled)
    result = brain.propagate(
        sources,
        steps=18,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )
    return result_snapshot(result), brain.last_structural_assist_trace


def main() -> None:
    if not BRAIN_PATH.exists():
        raise FileNotFoundError(f"brain file not found: {BRAIN_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)

    before_hash = file_hash(BRAIN_PATH)
    loaded = SphereBrain.load(BRAIN_PATH)
    cases = []
    rows = []

    for name, sources in source_cases(loaded.node_count).items():
        off_first, off_trace = run_once(sources, False)
        off_second, _ = run_once(sources, False)
        on_first, on_trace = run_once(sources, True)
        on_second, _ = run_once(sources, True)

        absolute_ok = all(
            item["absolute_modulation"] <= loaded.structural_absolute_cap + 1e-15
            for item in on_trace
        )
        relative_ok = all(
            item["near_zero_tie"]
            or item["meaningful_relative_ratio"] <= loaded.structural_relative_cap_ratio + 1e-12
            for item in on_trace
        )
        strong_override_count = sum(
            bool(item["top_candidate_changed"] and not item["near_zero_tie"])
            for item in on_trace
        )
        tie_resolved_count = sum(
            bool(item["top_candidate_changed"] and item["near_zero_tie"])
            for item in on_trace
        )
        assist_activation_count = sum(bool(item["tie_gate_active"]) for item in on_trace)

        same_routes = off_first["traversed_edges"] == on_first["traversed_edges"]
        same_nodes = off_first["activated_nodes"] == on_first["activated_nodes"]
        same_history = off_first["activation_history"] == on_first["activation_history"]
        same_final = np.array_equal(
            np.asarray(off_first["final_activation"]),
            np.asarray(on_first["final_activation"]),
        )

        item = {
            "name": name,
            "sources": sources,
            "off_repeatable": off_first == off_second,
            "on_repeatable": on_first == on_second,
            "off_trace_all_disabled": all(not record["enabled"] for record in off_trace),
            "assist_activation_count": assist_activation_count,
            "tie_resolved_by_structure_count": tie_resolved_count,
            "strong_decision_override_count": strong_override_count,
            "absolute_cap_respected": absolute_ok,
            "relative_cap_respected": relative_ok,
            "routes_unchanged": same_routes,
            "activated_nodes_unchanged": same_nodes,
            "activation_history_unchanged": same_history,
            "final_activation_unchanged": same_final,
            "off": off_first,
            "on": on_first,
            "on_trace": on_trace,
        }
        cases.append(item)
        rows.append({key: value for key, value in item.items() if key not in {"off", "on", "on_trace", "sources"}})

    after_hash = file_hash(BRAIN_PATH)

    # Existing brain.json has no structural fields. Loading must default to OFF,
    # and a round-trip save must preserve the explicit disabled setting.
    backward_compatible_default_off = loaded.structural_assist_enabled is False
    with tempfile.TemporaryDirectory() as directory:
        roundtrip_path = Path(directory) / "brain_roundtrip.json"
        loaded.save(roundtrip_path)
        roundtrip = SphereBrain.load(roundtrip_path)
        save_roundtrip_preserves_off = roundtrip.structural_assist_enabled is False

    checks = {
        "brain_file_unchanged": before_hash == after_hash,
        "backward_compatible_default_off": backward_compatible_default_off,
        "save_roundtrip_preserves_off": save_roundtrip_preserves_off,
        "all_off_repeatable": all(item["off_repeatable"] for item in cases),
        "all_on_repeatable": all(item["on_repeatable"] for item in cases),
        "off_path_is_inactive": all(item["off_trace_all_disabled"] for item in cases),
        "absolute_cap_respected": all(item["absolute_cap_respected"] for item in cases),
        "relative_cap_respected": all(item["relative_cap_respected"] for item in cases),
        "no_strong_decision_override": all(item["strong_decision_override_count"] == 0 for item in cases),
        "routes_unchanged": all(item["routes_unchanged"] for item in cases),
        "activated_nodes_unchanged": all(item["activated_nodes_unchanged"] for item in cases),
        "activation_history_unchanged": all(item["activation_history_unchanged"] for item in cases),
        "final_activation_unchanged": all(item["final_activation_unchanged"] for item in cases),
        "assist_activated_somewhere": any(item["assist_activation_count"] > 0 for item in cases),
        "tie_resolved_somewhere": any(item["tie_resolved_by_structure_count"] > 0 for item in cases),
    }
    payload = {
        "experiment": "Core Structural Assist v1",
        "brain_path": str(BRAIN_PATH.relative_to(ROOT)),
        "brain_file_sha256_before": before_hash,
        "brain_file_sha256_after": after_hash,
        "default_enabled": loaded.structural_assist_enabled,
        "config": {
            "gain": loaded.structural_gain,
            "tie_margin": loaded.structural_tie_margin,
            "near_zero_margin": loaded.structural_near_zero_margin,
            "relative_cap_ratio": loaded.structural_relative_cap_ratio,
            "absolute_cap": loaded.structural_absolute_cap,
        },
        "cases": cases,
        "checks": checks,
        "ready_for_manual_enable_experiment": all(checks.values()),
    }

    json_path = OUT / "core_structural_assist_v1.json"
    csv_path = OUT / "core_structural_assist_v1.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Core Structural Assist v1")
    print(f"default enabled: {loaded.structural_assist_enabled}")
    print(f"ready for manual enable experiment: {payload['ready_for_manual_enable_experiment']}")
    print(f"checks: {checks}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
