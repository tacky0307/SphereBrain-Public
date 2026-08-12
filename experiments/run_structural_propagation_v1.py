from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_observer import StructuralEpisode
from structural_propagation import PropagationConfig, StructuralPropagation

OUT = ROOT / "data" / "structural_propagation_v1" / "results"


def episode(steps, edges):
    return StructuralEpisode.from_lists(steps, edges)


def histories() -> dict:
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
        "direct": (direct, 6, 3),
        "merge": (merge, 6, 3),
        "repeated": (repeated, 14, 6),
        "nonrepeated": (nonrepeated, 14, 6),
        "direct_ids": (direct_ids, 106, 3),
    }


def one_run(enabled: bool, order=None) -> dict:
    engine = StructuralPropagation(PropagationConfig(enabled=enabled))
    output = {}
    for name, (ep, terminal, suffix) in histories().items():
        output[name] = engine.propagate(
            ep,
            terminal_node=terminal,
            common_suffix_start=suffix,
            candidate_order=order,
        )
    return output


def compare(left: dict, right: dict) -> dict:
    engine = StructuralPropagation()
    return {
        "probability_distance": engine.distance(left["branch_probabilities"], right["branch_probabilities"]),
        "modulation_distance": engine.distance(left["structural_modulation"], right["structural_modulation"]),
        "left_probabilities": left["branch_probabilities"],
        "right_probabilities": right["branch_probabilities"],
        "left_spread": left["probability_spread"],
        "right_spread": right["probability_spread"],
    }


def swapped_matches(original: dict, swapped: dict) -> bool:
    a = original["branch_probabilities"]
    b = swapped["branch_probabilities"]
    return abs(a[0] - b[1]) < 1e-12 and abs(a[1] - b[0]) < 1e-12


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    on = one_run(True)
    off = one_run(False)
    swapped = one_run(True, order=[1, 0])

    results = {
        "direct_vs_merge": {
            "with_structure": compare(on["direct"], on["merge"]),
            "without_structure": compare(off["direct"], off["merge"]),
        },
        "repeated_vs_nonrepeated": {
            "with_structure": compare(on["repeated"], on["nonrepeated"]),
            "without_structure": compare(off["repeated"], off["nonrepeated"]),
        },
        "direct_id_invariance": {
            "with_structure": compare(on["direct"], on["direct_ids"]),
            "without_structure": compare(off["direct"], off["direct_ids"]),
        },
    }

    checks = {
        "neutral_without_structure": all(
            abs(value - 0.5) < 1e-12
            for item in off.values()
            for value in item["branch_probabilities"]
        ),
        "merge_history_changes_propagation": results["direct_vs_merge"]["with_structure"]["probability_distance"] > 1e-9,
        "repetition_history_changes_propagation": results["repeated_vs_nonrepeated"]["with_structure"]["probability_distance"] > 1e-9,
        "node_id_invariant": results["direct_id_invariance"]["with_structure"]["probability_distance"] < 1e-12,
        "candidate_swap_is_symmetric": all(swapped_matches(on[name], swapped[name]) for name in on),
    }

    payload = {
        "experiment": "Structural Propagation v1",
        "language_free": True,
        "answer_labels": False,
        "long_term_learning": False,
        "design": "Ephemeral structural context weakly modulates a neutral two-way branch. Candidate directions have no semantic labels or correct answer.",
        "runs": {"with_structure": on, "without_structure": off, "swapped_candidates": swapped},
        "results": results,
        "checks": checks,
    }

    json_path = OUT / "structural_propagation_v1.json"
    csv_path = OUT / "structural_propagation_v1.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case", "with_probability_distance", "without_probability_distance",
            "with_modulation_distance", "without_modulation_distance",
            "left_probabilities", "right_probabilities",
        ])
        writer.writeheader()
        for name, item in results.items():
            writer.writerow({
                "case": name,
                "with_probability_distance": item["with_structure"]["probability_distance"],
                "without_probability_distance": item["without_structure"]["probability_distance"],
                "with_modulation_distance": item["with_structure"]["modulation_distance"],
                "without_modulation_distance": item["without_structure"]["modulation_distance"],
                "left_probabilities": json.dumps(item["with_structure"]["left_probabilities"]),
                "right_probabilities": json.dumps(item["with_structure"]["right_probabilities"]),
            })

    print("Structural Propagation v1")
    for name, item in results.items():
        print(f"\n{name}")
        print(f"  structure ON probability distance : {item['with_structure']['probability_distance']:.12f}")
        print(f"  structure OFF probability distance: {item['without_structure']['probability_distance']:.12f}")
        print(f"  left probabilities : {item['with_structure']['left_probabilities']}")
        print(f"  right probabilities: {item['with_structure']['right_probabilities']}")
    print(f"\nchecks: {checks}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
