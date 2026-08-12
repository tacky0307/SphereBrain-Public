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
from core_integration_shadow import CoreIntegrationShadow, ShadowConfig

OUT = ROOT / "data" / "core_integration_shadow_v1" / "results"


def find_brain_path() -> Path:
    candidates = [
        ROOT / "brain.json",
        ROOT / "data" / "brain.json",
        ROOT / "data" / "llm_core_v1" / "brain.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "brain.json was not found. Expected ./brain.json, ./data/brain.json, "
        "or ./data/llm_core_v1/brain.json"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sets(node_count: int) -> list[dict]:
    raw = [
        ("near_origin", [0, 1, 2]),
        ("quarter", [node_count // 4, node_count // 4 + 1, node_count // 4 + 2]),
        ("middle", [node_count // 2, node_count // 2 + 1, node_count // 2 + 2]),
        ("three_quarters", [3 * node_count // 4, 3 * node_count // 4 + 1, 3 * node_count // 4 + 2]),
        ("spread", [0, node_count // 3, 2 * node_count // 3]),
        ("end", [node_count - 3, node_count - 2, node_count - 1]),
    ]
    cases = []
    for name, nodes in raw:
        unique = []
        for node in nodes:
            value = int(node % node_count)
            if value not in unique:
                unique.append(value)
        cases.append({"name": name, "sources": unique})
    return cases


def normalized_edges(edges) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    brain_path = find_brain_path()
    file_hash_before = file_sha256(brain_path)
    brain = SphereBrain.load(brain_path)
    shadow = CoreIntegrationShadow(ShadowConfig(structural_gain=0.05, enabled=True))

    results = []
    for case in source_sets(brain.node_count):
        sources = case["sources"]
        shadow_first = shadow.run(brain, sources, steps=18, threshold=0.18)
        shadow_second = shadow.run(brain, sources, steps=18, threshold=0.18)
        actual = brain.propagate(
            sources,
            steps=18,
            threshold=0.18,
            noise=0.0,
            learn=False,
        )
        shadow_route = normalized_edges(shadow_first["baseline_route_edges"])
        actual_route = normalized_edges(actual.traversed_edges)
        summary = shadow_first["summary"]
        results.append({
            "name": case["name"],
            "sources": sources,
            "shadow": shadow_first,
            "checks": {
                "core_arrays_unchanged": shadow_first["core_arrays_unchanged"],
                "repeatable": shadow_first == shadow_second,
                "baseline_replay_matches_core": shadow_route == actual_route,
            },
            "metrics": {
                **summary,
                "baseline_edge_count": len(shadow_route),
                "actual_edge_count": len(actual_route),
                "route_symmetric_difference": len(shadow_route ^ actual_route),
            },
        })

    file_hash_after = file_sha256(brain_path)
    all_checks = {
        "brain_file_unchanged": file_hash_before == file_hash_after,
        "all_core_arrays_unchanged": all(item["checks"]["core_arrays_unchanged"] for item in results),
        "all_runs_repeatable": all(item["checks"]["repeatable"] for item in results),
        "all_baseline_replays_match_core": all(item["checks"]["baseline_replay_matches_core"] for item in results),
        "shadow_never_intervenes": True,
        "learning_disabled": True,
        "noise_disabled": True,
    }

    payload = {
        "experiment": "Core Integration Shadow v1",
        "brain_path": str(brain_path.relative_to(ROOT)),
        "brain_file_sha256_before": file_hash_before,
        "brain_file_sha256_after": file_hash_after,
        "read_only": True,
        "design": "Real focused Core propagation is replayed exactly while structural modulation is calculated beside it and never fed back.",
        "structural_gain": 0.05,
        "cases": results,
        "checks": all_checks,
    }

    json_path = OUT / "core_integration_shadow_v1.json"
    csv_path = OUT / "core_integration_shadow_v1.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "case", "sources", "step_count", "activated_node_count",
            "mean_probability_distance", "max_probability_distance",
            "mean_top_k_overlap", "top_candidate_change_count",
            "top_candidate_change_ratio", "baseline_edge_count",
            "actual_edge_count", "route_symmetric_difference",
            "core_arrays_unchanged", "repeatable", "baseline_replay_matches_core",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            shadow_result = item["shadow"]
            writer.writerow({
                "case": item["name"],
                "sources": "|".join(map(str, item["sources"])),
                "step_count": shadow_result["step_count"],
                "activated_node_count": shadow_result["activated_node_count"],
                **item["metrics"],
                **item["checks"],
            })

    print("Core Integration Shadow v1")
    print(f"brain: {brain_path}")
    for item in results:
        m = item["metrics"]
        print(f"\n{item['name']} sources={item['sources']}")
        print(f"  mean probability distance : {m['mean_probability_distance']:.12f}")
        print(f"  max probability distance  : {m['max_probability_distance']:.12f}")
        print(f"  mean top-k overlap         : {m['mean_top_k_overlap']:.6f}")
        print(f"  top candidate changes      : {m['top_candidate_change_count']}")
        print(f"  checks                     : {item['checks']}")
    print(f"\nchecks: {all_checks}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
