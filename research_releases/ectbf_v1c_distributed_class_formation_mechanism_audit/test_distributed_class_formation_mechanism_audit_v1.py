from __future__ import annotations

import unittest
import numpy as np

import distributed_class_formation_mechanism_audit_config_v1 as cfg
import distributed_class_formation_mechanism_audit_v1 as core
import distributed_temporal_boundary_equivalence_config_v1 as v1b_cfg


class DistributedClassFormationMechanismAuditV1Tests(unittest.TestCase):
    def test_conditional_shuffle_preserves_marginals_and_changes_covariance(self) -> None:
        X, y, q, _world, _report = core.collect_formation(
            cfg.DEV_FAMILY_SEEDS[0], 0, dict(v1b_cfg.SUITES[0])
        )
        shuffled = core.conditional_independent_shuffle(
            X, y, q, seed_parts=("unit", 0)
        )
        self.assertLessEqual(core.conditional_marginal_error(X, shuffled, y, q), 1e-10)
        self.assertGreater(core.offdiagonal_covariance_change(X, shuffled), 0.5)

    def test_subsets_are_cardinality_matched(self) -> None:
        snapshot, _world, report, _qf = core.form_mechanism_snapshot(
            cfg.DEV_FAMILY_SEEDS[0], 0, dict(v1b_cfg.SUITES[0])
        )
        self.assertEqual(len(snapshot.cooperative_indices), cfg.COOPERATIVE_SUBSET_SIZE)
        self.assertEqual(len(snapshot.marginal_indices), cfg.COOPERATIVE_SUBSET_SIZE)
        self.assertTrue(report["subset_sizes_equal"])

    def test_snapshot_roundtrip_and_forbidden_boundary(self) -> None:
        snapshot, _world, report, _qf = core.form_mechanism_snapshot(
            cfg.DEV_FAMILY_SEEDS[0], 1, dict(v1b_cfg.SUITES[1])
        )
        restored = core.MechanismSnapshot.from_json(snapshot.to_json())
        self.assertEqual(restored.to_json(), snapshot.to_json())
        self.assertEqual(report["snapshot_forbidden_token_count"], 0)
        self.assertEqual(report["initial_field_norm"], 0.0)

    def test_single_case_preserves_information_boundary(self) -> None:
        case = core.run_case(
            cfg.DEV_FAMILY_SEEDS[0], 2, dict(v1b_cfg.SUITES[0])
        )
        self.assertTrue(all(case["control_identity"].values()))
        self.assertEqual(case["information_boundary"]["true_boundary_reads"], 0)
        self.assertEqual(
            case["information_boundary"]["true_relation_reads_before_decision"], 0
        )
        self.assertEqual(
            case["information_boundary"]["held_query_formation_updates"], 0
        )

    def test_cooperative_subset_beats_marginal_on_reference_case(self) -> None:
        case = core.run_case(
            cfg.DEV_FAMILY_SEEDS[1], 3, dict(v1b_cfg.SUITES[0])
        )
        self.assertGreater(
            case["methods"][cfg.COOPERATIVE]["relation_accuracy"],
            case["methods"][cfg.MARGINAL]["relation_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
