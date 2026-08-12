from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_observer import StructuralEpisode
from structural_working_state_v2 import ControlledCase, StructuralWorkingStateV2, WorkingStateV2Config

OUT = ROOT / "data" / "structural_working_state_v2" / "results"


def episode(steps, edges):
    return StructuralEpisode.from_lists(steps, edges)


def cases() -> list[ControlledCase]:
    # Both paths end with exactly P -> B -> T. Only the earlier history differs.
    direct_history = episode(
        [[1], [3], [4], [5], [6]],
        [[(1, 3)], [(3, 4)], [(4, 5)], [(5, 6)], []],
    )
    merge_history = episode(
        [[1, 2], [3], [4], [5], [6]],
        [[(1, 3), (2, 3)], [(3, 4)], [(4, 5)], [(5, 6)], []],
    )

    # Equal-length histories: one reuses the same directed edges, the other does not.
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

    return [
        ControlledCase("direct_vs_merge_common_suffix", direct_history, merge_history, 6, 6, 3),
        ControlledCase("repeated_vs_nonrepeated_equal_length", repeated, nonrepeated, 14, 14, 6),
        ControlledCase("direct_id_invariance", direct_history, direct_ids, 6, 106, 3),
    ]


def run(enabled: bool) -> dict:
    worker = StructuralWorkingStateV2(WorkingStateV2Config(enabled=enabled))
    return {case.name: worker.run_case(case) for case in cases()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with_structure = run(True)
    without_structure = run(False)

    results = {}
    for name in with_structure:
        on = with_structure[name]
        off = without_structure[name]
        results[name] = {
            "with_structure": on,
            "without_structure": off,
            "checks": {
                "local_conditions_equal": on["comparison"]["local_distance"] < 1e-12,
                "without_structure_equal": off["comparison"]["distance"] < 1e-12,
                "structure_history_visible": on["comparison"]["structural_distance"] > 1e-9,
            },
        }

    payload = {
        "experiment": "Structural Working State v2",
        "language_free": True,
        "long_term_learning": False,
        "answer_labels": False,
        "design": "Terminal local state is computed only from an identical common suffix; earlier history can survive only through ephemeral structural state.",
        "results": results,
    }

    json_path = OUT / "structural_working_state_v2.json"
    csv_path = OUT / "structural_working_state_v2.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "case", "with_distance", "without_distance", "local_distance",
            "structural_distance", "with_similarity", "without_similarity",
            "local_conditions_equal", "without_structure_equal", "structure_history_visible",
        ])
        writer.writeheader()
        for name, item in results.items():
            on = item["with_structure"]["comparison"]
            off = item["without_structure"]["comparison"]
            writer.writerow({
                "case": name,
                "with_distance": on["distance"],
                "without_distance": off["distance"],
                "local_distance": on["local_distance"],
                "structural_distance": on["structural_distance"],
                "with_similarity": on["similarity"],
                "without_similarity": off["similarity"],
                **item["checks"],
            })

    print("Structural Working State v2")
    for name, item in results.items():
        checks = item["checks"]
        on = item["with_structure"]["comparison"]
        off = item["without_structure"]["comparison"]
        print(f"\n{name}")
        print(f"  local distance       : {on['local_distance']:.12f}")
        print(f"  structure OFF distance: {off['distance']:.12f}")
        print(f"  structure ON distance : {on['distance']:.12f}")
        print(f"  structural distance    : {on['structural_distance']:.12f}")
        print(f"  checks: {checks}")
    print(f"\nJSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
