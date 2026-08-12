from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_observer import StructuralEpisode
from structural_working_state import StructuralWorkingState, WorkingStateConfig

OUT = ROOT / "data" / "structural_working_state_v1" / "results"


def ep(steps, edges):
    return StructuralEpisode.from_lists(steps, edges)


def terminal_vector(result: dict, node: int) -> list[float]:
    return result["terminal_node_states"][str(node)]


def compare_case(name: str, left_ep, left_terminal: int, right_ep, right_terminal: int) -> dict:
    enabled = StructuralWorkingState(WorkingStateConfig(enabled=True))
    disabled = StructuralWorkingState(WorkingStateConfig(enabled=False))

    left_on = enabled.run(left_ep)
    right_on = enabled.run(right_ep)
    left_off = disabled.run(left_ep)
    right_off = disabled.run(right_ep)

    left_on_v = terminal_vector(left_on, left_terminal)
    right_on_v = terminal_vector(right_on, right_terminal)
    left_off_v = terminal_vector(left_off, left_terminal)
    right_off_v = terminal_vector(right_off, right_terminal)

    return {
        "case": name,
        "same_terminal_node": left_terminal == right_terminal,
        "with_structure": {
            "distance": enabled.distance(left_on_v, right_on_v),
            "similarity": enabled.cosine(left_on_v, right_on_v),
            "left_terminal_state": left_on_v,
            "right_terminal_state": right_on_v,
        },
        "without_structure": {
            "distance": disabled.distance(left_off_v, right_off_v),
            "similarity": disabled.cosine(left_off_v, right_off_v),
            "left_terminal_state": left_off_v,
            "right_terminal_state": right_off_v,
        },
        "left": left_on,
        "right": right_on,
    }


def main() -> None:
    direct_to_c = ep(
        steps=[[1], [3]],
        edges=[[(1, 3)], []],
    )
    merge_to_c = ep(
        steps=[[1, 2], [3]],
        edges=[[(1, 3), (2, 3)], []],
    )

    chain_once_to_z = ep(
        steps=[[10], [11], [12]],
        edges=[[(10, 11)], [(11, 12)], []],
    )
    repeated_chain_to_z = ep(
        steps=[[20], [21], [22], [20], [21], [22], [20], [21], [22]],
        edges=[
            [(20, 21)], [(21, 22)], [],
            [(20, 21)], [(21, 22)], [],
            [(20, 21)], [(21, 22)], [],
        ],
    )

    direct_other_ids = ep(
        steps=[[101], [103]],
        edges=[[(101, 103)], []],
    )
    merge_other_ids = ep(
        steps=[[101, 102], [103]],
        edges=[[(101, 103), (102, 103)], []],
    )

    results = {
        "experiment": "Structural Working State v1",
        "language_free": True,
        "long_term_learning": False,
        "answer_labels": False,
        "cases": {
            "direct_vs_merge_same_terminal": compare_case(
                "direct_vs_merge_same_terminal", direct_to_c, 3, merge_to_c, 3
            ),
            "single_vs_repeated_chain": compare_case(
                "single_vs_repeated_chain", chain_once_to_z, 12, repeated_chain_to_z, 22
            ),
            "direct_id_invariance": compare_case(
                "direct_id_invariance", direct_to_c, 3, direct_other_ids, 103
            ),
            "merge_id_invariance": compare_case(
                "merge_id_invariance", merge_to_c, 3, merge_other_ids, 103
            ),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "structural_working_state_v1.json"
    csv_path = OUT / "structural_working_state_v1.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for key, item in results["cases"].items():
        rows.append(
            {
                "case": key,
                "same_terminal_node": item["same_terminal_node"],
                "distance_with_structure": item["with_structure"]["distance"],
                "similarity_with_structure": item["with_structure"]["similarity"],
                "distance_without_structure": item["without_structure"]["distance"],
                "similarity_without_structure": item["without_structure"]["similarity"],
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Structural Working State v1 complete")
    for row in rows:
        print(
            f"{row['case']}: "
            f"distance ON={row['distance_with_structure']:.6f}, "
            f"OFF={row['distance_without_structure']:.6f}"
        )
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
