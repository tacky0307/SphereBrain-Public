"""Nonformal contract checks for Issue #332. No diagnostic/holdout seeds."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.experiments.experience_shaped_medium_v0 import implementation as model
from research.experiments.experience_shaped_medium_v0 import aggregate as auditor


class PhysicalMediumContracts(unittest.TestCase):
    def test_geometry_is_undirected_6x6_without_extra_wires(self):
        self.assertEqual(model.NCELLS, 36)
        self.assertEqual(model.NEDGES, 60)
        self.assertEqual(len(set(zip(model.LEFT.tolist(), model.RIGHT.tolist()))), 60)
        self.assertEqual(auditor.NEIGHBORS,
                         list(zip(model.LEFT.tolist(), model.RIGHT.tolist())))

    def test_zero_stimulation_stays_empty(self):
        q = np.zeros(36)
        medium = np.full(60, model.G0)
        for _ in range(32):
            q, medium, ledger = model.step(q, medium, None, 0.0, True)
            self.assertEqual(float(ledger), 0.0)
        np.testing.assert_array_equal(q, np.zeros(36))
        np.testing.assert_array_equal(medium, np.full(60, .08))

    def test_local_flux_ledger_independent(self):
        q = np.zeros(36)
        g = np.full(60, .08)
        for i in range(20):
            where = 12 if i % 2 == 0 else 17
            q1, g1, err = model.step(q, g, where, .93, True)
            q2, g2, err2 = auditor.independent_tick(q, g, where, .93, True)
            self.assertLess(err, 1e-10)
            self.assertLess(err2, 1e-10)
            np.testing.assert_allclose(q1, q2, atol=1e-12, rtol=0)
            np.testing.assert_allclose(g1, g2, atol=1e-12, rtol=0)
            q, g = q1, g1

    def test_control_equalities(self):
        raw = model.run_seed(332002)  # smoke-like, not a holdout
        self.assertLessEqual(raw["metrics"]["M7"], 1e-12)
        self.assertEqual(set(raw["probes"]), set(model.VARIANTS))
        for arm in model.TRAIN_ARMS:
            self.assertLessEqual(raw["training"][arm]["ledger_max"], 1e-10)

    def test_independent_replay_full_physics(self):
        result = model.run_seed(332003)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "seed_332003.json"
            import json
            path.write_text(json.dumps(result), encoding="utf-8")
            audit = auditor.review(path)
            self.assertEqual(audit["c0"], "PASS")
            self.assertLessEqual(audit["max_physical_difference"], 1e-9)

    def test_experience_order_source_matching(self):
        a, b = model.get_amplitudes(332004)
        q_ab = model.train(332004, "PLASTIC_AB", a, b)
        q_ba = model.train(332004, "PLASTIC_BA", a, b)
        self.assertEqual(len(q_ab["sigma"]), 60)
        self.assertEqual(len(q_ba["sigma"]), 60)
        self.assertAlmostEqual(sum(a) + sum(b), sum(b) + sum(a), places=10)


if __name__ == "__main__":
    unittest.main()
