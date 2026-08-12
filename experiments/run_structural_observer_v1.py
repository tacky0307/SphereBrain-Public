from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_observer import StructuralEpisode, StructuralObserver

OUTPUT_DIR = ROOT / "data" / "structural_observer_v1" / "results"
JSON_FILE = OUTPUT_DIR / "structural_observer_v1.json"
CSV_FILE = OUTPUT_DIR / "structural_observer_v1.csv"


def episode(steps, edges):
    return StructuralEpisode.from_lists(steps, edges)


def translated(base: StructuralEpisode, offset: int) -> StructuralEpisode:
    return episode(
        [[node + offset for node in step] for step in base.steps],
        [[(source + offset, target + offset) for source, target in step_edges]
         for step_edges in base.edges_by_step],
    )


def build_patterns() -> dict[str, StructuralEpisode]:
    # Same total node/edge counts where practical; only topology/timing differs.
    chain = episode(
        [[0], [1], [2], [3]],
        [[(0, 1)], [(1, 2)], [(2, 3)], []],
    )
    merge = episode(
        [[0, 1], [2], [3], []],
        [[(0, 2), (1, 2)], [(2, 3)], [], []],
    )
    split = episode(
        [[0], [1, 2], [3], []],
        [[(0, 1), (0, 2)], [(1, 3)], [], []],
    )
    parallel = episode(
        [[0, 2], [1, 3], [], []],
        [[(0, 1), (2, 3)], [], [], []],
    )
    cycle = episode(
        [[0], [1], [2], [0]],
        [[(0, 1)], [(1, 2)], [(2, 0)], []],
    )
    return {
        "chain": chain,
        "merge": merge,
        "split": split,
        "parallel": parallel,
        "cycle": cycle,
    }


def main() -> None:
    observer = StructuralObserver()
    patterns = build_patterns()
    observations = {name: observer.observe(value) for name, value in patterns.items()}

    translated_chain = translated(patterns["chain"], 100)
    translated_merge = translated(patterns["merge"], 200)

    identity_checks = {
        "chain_same_shape_different_ids": {
            "signature_equal": observations["chain"]["canonical_signature"]
            == observer.observe(translated_chain)["canonical_signature"],
            "similarity": observer.cosine_similarity(
                observations["chain"]["feature_vector"],
                observer.observe(translated_chain)["feature_vector"],
            ),
        },
        "merge_same_shape_different_ids": {
            "signature_equal": observations["merge"]["canonical_signature"]
            == observer.observe(translated_merge)["canonical_signature"],
            "similarity": observer.cosine_similarity(
                observations["merge"]["feature_vector"],
                observer.observe(translated_merge)["feature_vector"],
            ),
        },
    }

    repeated_sequence = observer.observe_sequence(
        [patterns["chain"], translated(patterns["chain"], 100), translated(patterns["chain"], 200)]
    )
    non_repeated_sequence = observer.observe_sequence(
        [patterns["chain"], translated(patterns["merge"], 100), translated(patterns["split"], 200)]
    )

    pairwise = []
    names = list(patterns)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            pairwise.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "similarity": observer.cosine_similarity(
                        observations[left_name]["feature_vector"],
                        observations[right_name]["feature_vector"],
                    ),
                    "signature_equal": observations[left_name]["canonical_signature"]
                    == observations[right_name]["canonical_signature"],
                }
            )

    payload = {
        "experiment": "Structural Observer v1",
        "read_only": True,
        "language_free": True,
        "purpose": (
            "Test whether chain, merge, split, parallel, cycle, and repetition "
            "can be distinguished beyond node/edge counts without changing Core."
        ),
        "patterns": observations,
        "identity_checks": identity_checks,
        "repetition": {
            "same_shape_three_times": repeated_sequence,
            "different_shapes_three_times": non_repeated_sequence,
        },
        "pairwise": pairwise,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pattern",
                "node_count",
                "edge_count",
                "sources",
                "sinks",
                "merges",
                "splits",
                "components",
                "max_depth",
                "parallel_width",
                "temporal_overlap",
                "edge_reuse",
                "canonical_signature",
            ]
        )
        for name, observation in observations.items():
            shape = observation["shape"]
            temporal = observation["temporal"]
            writer.writerow(
                [
                    name,
                    shape["node_count"],
                    shape["edge_count"],
                    shape["sources"],
                    shape["sinks"],
                    shape["merges"],
                    shape["splits"],
                    shape["components"],
                    shape["max_depth"],
                    shape["parallel_width"],
                    temporal["temporal_overlap"],
                    temporal["edge_reuse"],
                    observation["canonical_signature"],
                ]
            )

    print(f"JSON: {JSON_FILE}")
    print(f"CSV : {CSV_FILE}")
    print()
    print("ID invariance:")
    for name, check in identity_checks.items():
        print(f"  {name}: signature_equal={check['signature_equal']}, similarity={check['similarity']:.6f}")
    print()
    print("Repetition:")
    print(
        "  same shape: exact_repeat_ratio="
        f"{repeated_sequence['exact_repeat_ratio']:.3f}, "
        f"mean_similarity={repeated_sequence['mean_structural_similarity']:.3f}"
    )
    print(
        "  different shapes: exact_repeat_ratio="
        f"{non_repeated_sequence['exact_repeat_ratio']:.3f}, "
        f"mean_similarity={non_repeated_sequence['mean_structural_similarity']:.3f}"
    )


if __name__ == "__main__":
    main()
