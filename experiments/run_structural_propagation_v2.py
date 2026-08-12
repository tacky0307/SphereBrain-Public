from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_observer import StructuralEpisode
from structural_propagation_v2 import (
    CandidateEdgeState,
    PropagationV2Config,
    StructuralPropagationV2,
)

OUT = ROOT / "data" / "structural_propagation_v2" / "results"


def episode(steps, edges):
    return StructuralEpisode.from_lists(steps, edges)


def histories():
    direct = episode(
        [[1], [3], [4], [5], [6]],
        [[(1, 3)], [(3, 4)], [(4, 5)], [(5, 6)], []],
    )
    merge = episode(
        [[1, 2], [3], [4], [5], [6]],
        [[(1, 3), (2, 3)], [(3, 4)], [(4, 5)], [(5, 6)], []],
    )
    repeated = episode(
        [[10], [11], [12], [10], [11], [12], [13], [14]],
        [[(10, 11)], [(11, 12)], [(12, 10)], [(10, 11)], [(11, 12)], [(12, 13)], [(13, 14)], []],
    )
    nonrepeated = episode(
        [[20], [21], [22], [23], [24], [25], [13], [14]],
        [[(20, 21)], [(21, 22)], [(22, 23)], [(23, 24)], [(24, 25)], [(25, 13)], [(13, 14)], []],
    )
    direct_ids = episode(
        [[101], [103], [104], [105], [106]],
        [[(101, 103)], [(103, 104)], [(104, 105)], [(105, 106)], []],
    )
    return {
        "direct": (direct, 3),
        "merge": (merge, 3),
        "repeated": (repeated, 6),
        "nonrepeated": (nonrepeated, 6),
        "direct_ids": (direct_ids, 3),
    }


def candidates():
    # Equal base weight; local histories differ only in measurable Edge state.
    return [
        CandidateEdgeState(9001, weight=0.70, usage=0.25, recency=0.85, target_degree=0.30, direction=-0.60),
        CandidateEdgeState(9002, weight=0.70, usage=0.75, recency=0.35, target_degree=0.80, direction=0.60),
    ]


def swapped_candidate_ids(items):
    return [
        CandidateEdgeState(19002, items[1].weight, items[1].usage, items[1].recency, items[1].target_degree, items[1].direction),
        CandidateEdgeState(19001, items[0].weight, items[0].usage, items[0].recency, items[0].target_degree, items[0].direction),
    ]


def run(enabled: bool, candidate_items):
    engine = StructuralPropagationV2(PropagationV2Config(enabled=enabled))
    return {
        name: engine.propagate(ep, suffix, candidate_items)
        for name, (ep, suffix) in histories().items()
    }


def compare(left, right):
    engine = StructuralPropagationV2()
    return {
        "probability_distance": engine.distance(left["branch_probabilities"], right["branch_probabilities"]),
        "local_logit_distance": engine.distance(left["local_logits"], right["local_logits"]),
        "structural_modulation_distance": engine.distance(left["structural_modulation"], right["structural_modulation"]),
        "left_probabilities": left["branch_probabilities"],
        "right_probabilities": right["branch_probabilities"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_candidates = candidates()
    with_structure = run(True, base_candidates)
    without_structure = run(False, base_candidates)
    swapped = run(True, swapped_candidate_ids(base_candidates))

    results = {
        "direct_vs_merge": {
            "with_structure": compare(with_structure["direct"], with_structure["merge"]),
            "without_structure": compare(without_structure["direct"], without_structure["merge"]),
        },
        "repeated_vs_nonrepeated": {
            "with_structure": compare(with_structure["repeated"], with_structure["nonrepeated"]),
            "without_structure": compare(without_structure["repeated"], without_structure["nonrepeated"]),
        },
        "direct_id_invariance": {
            "with_structure": compare(with_structure["direct"], with_structure["direct_ids"]),
            "without_structure": compare(without_structure["direct"], without_structure["direct_ids"]),
        },
    }

    original = with_structure["direct"]["branch_probabilities"]
    swapped_probs = swapped["direct"]["branch_probabilities"]
    checks = {
        "same_edge_state_same_baseline": results["direct_vs_merge"]["without_structure"]["probability_distance"] < 1e-12,
        "merge_history_changes_edge_transmission": results["direct_vs_merge"]["with_structure"]["probability_distance"] > 1e-9,
        "repetition_history_changes_edge_transmission": results["repeated_vs_nonrepeated"]["with_structure"]["probability_distance"] > 1e-9,
        "node_id_invariant": results["direct_id_invariance"]["with_structure"]["probability_distance"] < 1e-12,
        "candidate_state_swap_is_symmetric": abs(original[0] - swapped_probs[1]) < 1e-12 and abs(original[1] - swapped_probs[0]) < 1e-12,
        "no_random_candidate_vectors": all(not item["random_candidate_vectors"] for item in with_structure.values()),
    }

    payload = {
        "experiment": "Structural Propagation v2",
        "language_free": True,
        "answer_labels": False,
        "long_term_learning": False,
        "design": "Ephemeral structural context modulates measurable candidate Edge state; no random candidate vectors or semantic labels.",
        "candidate_feature_names": ["weight", "usage", "recency", "target_degree", "direction"],
        "runs": {
            "with_structure": with_structure,
            "without_structure": without_structure,
            "swapped_candidate_states": swapped,
        },
        "results": results,
        "checks": checks,
    }

    json_path = OUT / "structural_propagation_v2.json"
    csv_path = OUT / "structural_propagation_v2.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "comparison", "mode", "probability_distance", "local_logit_distance",
            "structural_modulation_distance", "left_probabilities", "right_probabilities",
        ])
        writer.writeheader()
        for comparison, modes in results.items():
            for mode, item in modes.items():
                writer.writerow({
                    "comparison": comparison,
                    "mode": mode,
                    **item,
                })

    print("Structural Propagation v2")
    for name, value in checks.items():
        print(f"  {name}: {value}")
    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
