"""Independent physical auditor for Issue #332.

Does NOT import the experiment simulator or its laws. Recomputes source streams,
all training ticks, interventions, ALL future probe states and frozen gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N = 36
E = 60
NEIGHBORS = []
for row in range(6):
    for col in range(5):
        NEIGHBORS.append((row * 6 + col, row * 6 + col + 1))
for row in range(5):
    for col in range(6):
        NEIGHBORS.append((row * 6 + col, (row + 1) * 6 + col))
assert len(NEIGHBORS) == E

ARMS = ("PASSIVE_AB", "PLASTIC_AB", "PLASTIC_BA", "PLASTIC_AA", "PLASTIC_NONE")
VARIANTS = ARMS + ("ERASE", "SHUFFLE", "UNIFORM")
PORTS = {"A": 12, "B": 17, "NOVEL": 2}


def independent_tick(state, medium, at, amount, change):
    """Per-face explicit loop, independently of vectorized Core implementation."""
    old_total = sum(float(x) for x in state)
    x = [float(z) for z in state]
    if at is not None:
        x[at] += float(amount)
    moved = [0.0] * N
    flows = []
    for k, (u, v) in enumerate(NEIGHBORS):
        current = float(medium[k]) * (x[u] - x[v])
        flows.append(current)
        moved[u] -= 0.20 * current
        moved[v] += 0.20 * current
    after = np.asarray([(x[i] + moved[i]) * 0.975 for i in range(N)], dtype=np.float64)
    if change:
        updated = np.asarray([
            np.clip(float(medium[k]) + .045 * abs(flows[k]) *
                    (.34 - float(medium[k])) / .34
                    - .0005 * (float(medium[k]) - .08), .08, .34)
            for k in range(E)
        ], dtype=np.float64)
    else:
        updated = np.asarray(medium, dtype=np.float64).copy()
    error = abs(float(np.sum(after)) - .975 * (old_total + float(amount)))
    if (error > 1e-10 or not np.isfinite(after).all()
            or np.min(after) < -1e-10 or not np.isfinite(updated).all()
            or np.min(updated) < .08 - 1e-12 or np.max(updated) > .34 + 1e-12):
        raise AssertionError(f"Independent physical conservation/bounds failure: {error}")
    return after, updated, error


def independent_train(arm, amp_a, amp_b):
    if arm == "PLASTIC_BA":
        groups = ((17, amp_b), (12, amp_a))
    elif arm == "PLASTIC_AA":
        groups = ((12, amp_a), (12, amp_b))
    elif arm == "PLASTIC_NONE":
        groups = ((None, [0.0] * 64), (None, [0.0] * 64))
    else:
        groups = ((12, amp_a), (17, amp_b))
    plastic = arm != "PASSIVE_AB"
    q, medium = np.zeros(N), np.full(E, .08)
    maximum = 0.0
    ticks = 0
    for location, seq in groups:
        for amplitude in seq:
            for i in range(12):
                amt = float(amplitude) if i == 0 and location is not None else 0.0
                at = location if i == 0 else None
                q, medium, deviation = independent_tick(q, medium, at, amt, plastic)
                maximum = max(maximum, deviation)
                ticks += 1
    for i in range(48):
        q, medium, deviation = independent_tick(q, medium, None, 0.0, plastic)
        maximum = max(maximum, deviation)
        ticks += 1
    assert ticks == 1584
    return q, medium, maximum


def independent_probe(medium, cell):
    q = np.zeros(N)
    trajectory = []
    for t in range(24):
        q, unchanged, error = independent_tick(
            q, medium, cell if t == 0 else None, 1.0 if t == 0 else 0.0, False
        )
        assert error <= 1e-10
        assert np.array_equal(unchanged, medium)
        trajectory.append(q.copy())
    return np.asarray(trajectory)


def close(actual, expected, context):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"{context}: invalid array shape/finite")
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    if delta > 1e-9:
        raise AssertionError(f"{context}: independent discrepancy {delta:.12g}")
    return delta


def review(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    seed = int(raw["seed"])
    assert raw["issue"] == 332 and raw["physics"] == "fixed_grid_no_explicit_W"
    local_rng = np.random.default_rng(seed * 101 + 17)
    amp_a = local_rng.uniform(.85, 1.15, size=64)
    amp_b = local_rng.uniform(.85, 1.15, size=64)
    close(raw["input_amplitudes"]["A"], amp_a, f"seed{seed} source A")
    close(raw["input_amplitudes"]["B"], amp_b, f"seed{seed} source B")
    assert set(raw["training"]) == set(ARMS)
    assert set(raw["variant_sigma"]) == set(VARIANTS)
    assert set(raw["probes"]) == set(VARIANTS)
    q_list, medium_list = {}, {}
    maxdiff = 0.0
    for arm in ARMS:
        q, medium, err = independent_train(arm, amp_a, amp_b)
        q_list[arm], medium_list[arm] = q, medium
        record = raw["training"][arm]
        maxdiff = max(maxdiff,
                      close(record["q_final"], q, f"{arm} final q"),
                      close(record["sigma"], medium, f"{arm} final medium"),
                      abs(float(record["ledger_max"]) - err))
        assert float(record["ledger_max"]) <= 1e-10
    formed = medium_list["PLASTIC_AB"]
    medium_list["ERASE"] = np.full(E, .08)
    perm = np.random.default_rng(seed + 12345).permutation(E)
    medium_list["SHUFFLE"] = formed[perm]
    medium_list["UNIFORM"] = np.full(E, float(np.mean(formed)))
    assert abs(float(np.sum(medium_list["SHUFFLE"])) - float(np.sum(formed))) <= 1e-12
    assert abs(float(np.sum(medium_list["UNIFORM"])) - float(np.sum(formed))) <= 1e-12
    trace = {}
    for arm in VARIANTS:
        maxdiff = max(maxdiff, close(raw["variant_sigma"][arm], medium_list[arm],
                                     f"{arm} restored medium"))
        assert set(raw["probes"][arm]) == set(PORTS)
        trace[arm] = {}
        for name, pos in PORTS.items():
            trace[arm][name] = independent_probe(medium_list[arm], pos)
            maxdiff = max(maxdiff,
                          close(raw["probes"][arm][name], trace[arm][name],
                                f"{arm}.{name} full physical timeline"))
    def distance(x, y, p):
        norm = max(np.linalg.norm(trace["PASSIVE_AB"][p]), 1e-12)
        return float(np.linalg.norm(trace[x][p] - trace[y][p]) / norm)
    metrics = {
        "M0": float(np.linalg.norm(formed - .08) / np.linalg.norm(np.full(E, .08))),
        "M1": distance("PLASTIC_AB", "PASSIVE_AB", "NOVEL"),
        "M2": distance("PLASTIC_AB", "ERASE", "NOVEL"),
        "M3": distance("PLASTIC_AB", "PLASTIC_BA", "NOVEL"),
        "M4": distance("PLASTIC_AB", "SHUFFLE", "NOVEL"),
        "M5": distance("PLASTIC_AB", "UNIFORM", "NOVEL"),
        "M6": distance("PLASTIC_AB", "PASSIVE_AB", "A"),
        "M7": distance("ERASE", "PASSIVE_AB", "NOVEL"),
    }
    assert set(raw["metrics"]) == set(metrics)
    for k, value in metrics.items():
        if abs(float(raw["metrics"][k]) - value) > 1e-9:
            raise AssertionError(f"{seed} metric {k}: independent physics mismatch")
    return {"seed": seed, "c0": "PASS", "max_physical_difference": maxdiff,
            "metrics": metrics}


def aggregate(folder, cohort):
    expected = {
        "smoke": [332001],
        "diagnostic": list(range(332101, 332107)),
        "holdout": list(range(332201, 332207)),
    }[cohort]
    paths = sorted(Path(folder).glob("seed_*.json"))
    records = [review(path) for path in paths]
    seeds = [r["seed"] for r in records]
    if sorted(seeds) != expected or len(seeds) != len(set(seeds)):
        raise AssertionError(f"{cohort} seed mismatch: {seeds} expected {expected}")
    means = {m: float(np.mean([r["metrics"][m] for r in records]))
             for m in [f"M{i}" for i in range(8)]}
    gates = {}
    for i, (limit, needed) in enumerate(
        [(0.10, 5), (0.02, 5), (0.02, 5), (.005, 4),
         (.01, 4), (.005, 4), (.02, 5), (1e-10, 6)]
    ):
        key = f"M{i}"
        passed = sum((r["metrics"][key] <= limit if i == 7 else
                      r["metrics"][key] >= limit) for r in records)
        # Smoke gate counts are descriptive only (1 individual, not a formal cohort).
        gates[f"P{i}"] = {"value": f"{passed}/{len(records)}",
                          "required": needed, "passed": passed >= needed}
    all_pass = all(v["passed"] for v in gates.values()) and len(records) == 6
    verdict = ("MEDIUM_TRACE_CAUSAL_EFFECT_SUPPORTED_IN_FIXED_GRID"
               if all_pass else "PARTIAL_OR_NOT_ESTABLISHED")
    return {"issue": 332, "cohort": cohort, "verdict": verdict,
            "c0": "PASS", "individual": records, "means": means, "gates": gates,
            "qualification": "physical medium trace only, not intelligence or solving",
            "preregistered_contract": "PROTOCOL.md before simulation implementation"}


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--raw", required=True)
    arg.add_argument("--cohort", choices=["smoke", "diagnostic", "holdout"], required=True)
    arg.add_argument("--out", required=True)
    a = arg.parse_args()
    report = aggregate(a.raw, a.cohort)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cohort": a.cohort, "verdict": report["verdict"],
                      "c0": report["c0"], "means": report["means"],
                      "gates": report["gates"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
