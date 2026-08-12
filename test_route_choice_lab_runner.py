from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import route_choice_lab as lab
import route_choice_lab_runner as runner


class RouteIdentityTests(unittest.TestCase):
    def candidate(
        self,
        key: str,
        nodes: list[int],
        *,
        evidence: str,
    ) -> lab.Candidate:
        return lab.Candidate(
            key=key,
            nodes=nodes,
            edges=[lab.norm(a, b) for a, b in zip(nodes, nodes[1:])],
            evidence_texts=[evidence],
            source_kinds={"test"},
        )

    def test_reverse_route_has_same_fingerprint(self) -> None:
        forward = self.candidate("forward", [1, 2, 3, 4], evidence="犬は泳ぐ")
        reverse = self.candidate("reverse", [4, 3, 2, 1], evidence="犬は泳ぐ")

        self.assertEqual(
            runner.structural_route_signature(forward),
            runner.structural_route_signature(reverse),
        )
        self.assertEqual(
            runner.route_fingerprint(forward),
            runner.route_fingerprint(reverse),
        )

    def test_different_traversal_is_not_collapsed(self) -> None:
        first = self.candidate("first", [1, 2, 3, 4], evidence="犬は泳ぐ")
        second = self.candidate("second", [1, 3, 2, 4], evidence="犬は泳ぐ")

        unique, removed = runner.deduplicate_routes([first, second])

        self.assertEqual(2, len(unique))
        self.assertEqual(0, removed)
        self.assertNotEqual(
            runner.route_fingerprint(first),
            runner.route_fingerprint(second),
        )

    def test_exact_duplicate_merges_metadata_only_once(self) -> None:
        first = self.candidate("a", [1, 2, 3], evidence="犬は泳ぐ")
        duplicate = self.candidate("b", [3, 2, 1], evidence="犬は水を進む")

        unique, removed = runner.deduplicate_routes([first, duplicate])

        self.assertEqual(1, len(unique))
        self.assertEqual(1, removed)
        self.assertEqual(["a", "b"], unique[0].route_alias_keys)
        self.assertEqual(
            ["犬は泳ぐ", "犬は水を進む"],
            unique[0].evidence_texts,
        )


class EvaluationLayerTests(unittest.TestCase):
    def test_core_and_feedback_scores_are_separated(self) -> None:
        metrics = {
            "strength": 0.8,
            "familiarity": 0.7,
            "bridge_affinity": 0.6,
            "geometry": 0.5,
        }
        no_feedback = runner.mode_scores_from_metrics(
            metrics,
            (0.0, 0.0, 0.0),
            decoy=False,
        )
        positive_feedback = runner.mode_scores_from_metrics(
            metrics,
            (10.0, 0.0, 0.0),
            decoy=False,
        )

        self.assertAlmostEqual(
            no_feedback[runner.MODE_CORE_ONLY],
            positive_feedback[runner.MODE_CORE_ONLY],
        )
        self.assertLess(
            no_feedback[runner.MODE_FEEDBACK_ONLY],
            positive_feedback[runner.MODE_FEEDBACK_ONLY],
        )
        self.assertLess(
            no_feedback[runner.MODE_COMBINED],
            positive_feedback[runner.MODE_COMBINED],
        )


class CoreAuditTests(unittest.TestCase):
    def make_brain(self) -> SimpleNamespace:
        brain = SimpleNamespace()
        brain.node_count = 4
        brain.adjacency = np.zeros((4, 4), dtype=bool)
        brain.weights = np.zeros((4, 4), dtype=float)
        brain.usage = np.zeros((4, 4), dtype=int)
        brain.node_usage = np.zeros(4, dtype=int)
        brain.adjacency[0, 1] = brain.adjacency[1, 0] = True
        brain.weights[0, 1] = brain.weights[1, 0] = 0.5
        return brain

    def test_expected_core_change_is_reported_as_normal(self) -> None:
        before = self.make_brain()
        after = self.make_brain()
        after.weights[0, 1] = after.weights[1, 0] = 0.6
        after.usage[0, 1] = after.usage[1, 0] = 1
        after.node_usage[0] = 1

        audit = runner.build_core_audit(
            before,
            after,
            hash_before="a" * 64,
            hash_after="b" * 64,
            expected_signals={(0, 1): 1.0},
            expected_usage_edges={(0, 1)},
            expected_used_nodes={0},
        )

        self.assertTrue(audit["core_changed"])
        self.assertTrue(audit["audit_ok"])
        self.assertEqual(1, audit["changed_edges"])
        self.assertEqual(1, audit["strengthened_edges"])
        self.assertEqual(0, audit["unexpected_edges_changed"])
        self.assertEqual(1, audit["usage_edges_changed"])
        self.assertEqual(1, audit["node_usage_changed"])

    def test_unexpected_edge_change_is_flagged(self) -> None:
        before = self.make_brain()
        after = self.make_brain()
        before.adjacency[2, 3] = before.adjacency[3, 2] = True
        after.adjacency[2, 3] = after.adjacency[3, 2] = True
        before.weights[2, 3] = before.weights[3, 2] = 0.4
        after.weights[2, 3] = after.weights[3, 2] = 0.5

        audit = runner.build_core_audit(
            before,
            after,
            hash_before="a" * 64,
            hash_after="b" * 64,
            expected_signals={(0, 1): 1.0},
            expected_usage_edges=set(),
            expected_used_nodes=set(),
        )

        self.assertFalse(audit["audit_ok"])
        self.assertEqual(1, audit["unexpected_edges_changed"])
        self.assertTrue(audit["warnings"])


class UiPatchTests(unittest.TestCase):
    def test_current_page_can_be_upgraded_to_v04(self) -> None:
        original_page = lab.PAGE
        try:
            runner.install_research_ui()
            self.assertIn("ROUTE CHOICE LEARNING LAB v0.4", lab.PAGE)
            self.assertIn("Core-only", lab.PAGE)
            self.assertIn("Feedback-only", lab.PAGE)
            self.assertIn("CORE CHANGE AUDIT", lab.PAGE)
            self.assertIn("route_fingerprint", lab.PAGE)
        finally:
            lab.PAGE = original_page


if __name__ == "__main__":
    unittest.main()
