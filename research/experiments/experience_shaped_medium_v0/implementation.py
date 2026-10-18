"""Issue #332: minimal local physical medium with no explicit long-lived wires.

Physics and numeric gates frozen in PROTOCOL.md at commit 61d7203.
Do not place any task labels, reward or desired route in this model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIZE = 6
NCELLS = SIZE * SIZE
LEFT = np.asarray(
    [r * SIZE + c for r in range(SIZE) for c in range(SIZE - 1)]
    + [r * SIZE + c for r in range(SIZE - 1) for c in range(SIZE)],
    dtype=np.int64,
)
RIGHT = np.asarray(
    [r * SIZE + c + 1 for r in range(SIZE) for c in range(SIZE - 1)]
    + [(r + 1) * SIZE + c for r in range(SIZE - 1) for c in range(SIZE)],
    dtype=np.int64,
)
NEDGES = len(LEFT)
assert NCELLS == 36 and NEDGES == 60

G0 = 0.08
GMAX = 0.34
ETA = 0.045
DECAY = 0.0005
DT = 0.20
LEAK = 0.025
EVENT_TICKS = 12
REST_TICKS = 48
PROBE_TICKS = 24
PROBES = {"A": 12, "B": 17, "NOVEL": 2}
TRAIN_ARMS = ("PASSIVE_AB", "PLASTIC_AB", "PLASTIC_BA", "PLASTIC_AA", "PLASTIC_NONE")
VARIANTS = TRAIN_ARMS + ("ERASE", "SHUFFLE", "UNIFORM")


def step(q: np.ndarray, g: np.ndarray, cell: int | None, pulse: float, plastic: bool):
    """Apply one source, simultaneous antisymmetric neighbor flux, global leak, local plasticity."""
    x = q.copy()
    if cell is not None:
        x[cell] += pulse
    flux = g * (x[LEFT] - x[RIGHT])
    delta = np.zeros(NCELLS, dtype=np.float64)
    np.add.at(delta, LEFT, -DT * flux)
    np.add.at(delta, RIGHT, DT * flux)
    after = (x + delta) * (1.0 - LEAK)
    if plastic:
        new_g = np.clip(g + ETA * np.abs(flux) * (GMAX - g) / GMAX
                        - DECAY * (g - G0), G0, GMAX)
    else:
        new_g = g
    ledger = abs(float(np.sum(after)) - (1.0 - LEAK) * (float(np.sum(q)) + pulse))
    if (ledger > 1e-10 or not np.all(np.isfinite(after))
            or np.min(after) < -1e-10 or not np.all(np.isfinite(new_g))
            or np.min(new_g) < G0 - 1e-12 or np.max(new_g) > GMAX + 1e-12):
        raise AssertionError("Invalid local medium physics")
    return after, new_g, ledger


def get_amplitudes(seed: int):
    rng = np.random.default_rng(seed * 101 + 17)
    return rng.uniform(.85, 1.15, size=64), rng.uniform(.85, 1.15, size=64)


def train(seed: int, arm: str, a: np.ndarray, b: np.ndarray):
    if arm not in TRAIN_ARMS:
        raise ValueError(arm)
    if arm == "PLASTIC_BA":
        schedule = ((17, b), (12, a))
    elif arm == "PLASTIC_AA":
        schedule = ((12, a), (12, b))
    elif arm == "PLASTIC_NONE":
        schedule = ((None, np.zeros(64)), (None, np.zeros(64)))
    else:
        schedule = ((12, a), (17, b))
    plastic = arm != "PASSIVE_AB"
    q = np.zeros(NCELLS, dtype=np.float64)
    sigma = np.full(NEDGES, G0, dtype=np.float64)
    max_ledger = 0.0
    for cell, amplitudes in schedule:
        for amp in amplitudes:
            for tick in range(EVENT_TICKS):
                q, sigma, ledger = step(q, sigma, cell if tick == 0 else None,
                                        float(amp) if tick == 0 else 0.0, plastic)
                max_ledger = max(max_ledger, ledger)
    for _ in range(REST_TICKS):
        q, sigma, ledger = step(q, sigma, None, 0.0, plastic)
        max_ledger = max(max_ledger, ledger)
    return {"sigma": sigma.tolist(), "q_final": q.tolist(), "ledger_max": max_ledger}


def probe(sigma: np.ndarray, cell: int):
    """Readout only: fresh zero fast state and *frozen* medium."""
    q = np.zeros(NCELLS, dtype=np.float64)
    records = []
    for t in range(PROBE_TICKS):
        q, _, _ = step(q, sigma, cell if t == 0 else None, 1.0 if t == 0 else 0.0, False)
        records.append(q.tolist())
    return records


def calculate_metrics(sigmas: dict, probes: dict):
    def y(arm: str, name: str):
        return np.asarray(probes[arm][name], dtype=np.float64).ravel()
    def distance(first: str, second: str, name: str):
        ref = max(np.linalg.norm(y("PASSIVE_AB", name)), 1e-12)
        return float(np.linalg.norm(y(first, name) - y(second, name)) / ref)
    return {
        "M0": float(np.linalg.norm(sigmas["PLASTIC_AB"] - G0)
                    / np.linalg.norm(np.full(NEDGES, G0))),
        "M1": distance("PLASTIC_AB", "PASSIVE_AB", "NOVEL"),
        "M2": distance("PLASTIC_AB", "ERASE", "NOVEL"),
        "M3": distance("PLASTIC_AB", "PLASTIC_BA", "NOVEL"),
        "M4": distance("PLASTIC_AB", "SHUFFLE", "NOVEL"),
        "M5": distance("PLASTIC_AB", "UNIFORM", "NOVEL"),
        "M6": distance("PLASTIC_AB", "PASSIVE_AB", "A"),
        "M7": distance("ERASE", "PASSIVE_AB", "NOVEL"),
    }


def run_seed(seed: int):
    a, b = get_amplitudes(seed)
    training = {arm: train(seed, arm, a, b) for arm in TRAIN_ARMS}
    sigma = {arm: np.asarray(training[arm]["sigma"], dtype=np.float64)
             for arm in TRAIN_ARMS}
    formed = sigma["PLASTIC_AB"]
    sigma["ERASE"] = np.full(NEDGES, G0, dtype=np.float64)
    sigma["SHUFFLE"] = formed[np.random.default_rng(seed + 12345).permutation(NEDGES)]
    sigma["UNIFORM"] = np.full(NEDGES, np.mean(formed), dtype=np.float64)
    trajectories = {arm: {name: probe(gs, cell) for name, cell in PROBES.items()}
                    for arm, gs in sigma.items()}
    metrics = calculate_metrics(sigma, trajectories)
    return {
        "issue": 332, "seed": seed, "physics": "fixed_grid_no_explicit_W",
        "input_amplitudes": {"A": a.tolist(), "B": b.tolist()},
        "training": training,
        "variant_sigma": {arm: gs.tolist() for arm, gs in sigma.items()},
        "probes": trajectories,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_seed(args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")),
                    encoding="utf-8")
    print(json.dumps({"seed": args.seed, "metrics": result["metrics"],
                      "physical_ledger": {k: v["ledger_max"]
                                          for k, v in result["training"].items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
