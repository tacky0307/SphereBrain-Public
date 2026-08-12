# Core Integration Shadow v2

## Purpose

Measure whether structural context can safely influence real Core candidate edges without modifying the saved brain or live Core arrays.

## Changes from v1

- Removes the false-positive cycle signal when no edges exist.
- Separates edge reuse, local structural repetition, and closed directed cycles.
- Applies structural modulation only when the top baseline candidates are within the configured tie margin.
- Caps modulation relative to the baseline margin.
- Sweeps gains: 0.005, 0.01, 0.02, 0.03, 0.05.
- Runs baseline and structural virtual routes independently to the end.
- Reports route and activated-node divergence.

## Core integration boundary

The real Core is not modified in this experiment. The intended future Core hook is:

1. Compute normal focused propagation candidates.
2. Measure the top-candidate margin.
3. Use structural context only when the margin is below the tie threshold.
4. Bound structural modulation by a fraction of that margin.
5. Keep the feature disabled by default until Shadow v2 identifies a safe gain.

## Required checks

- `brain_file_unchanged`
- `all_core_unchanged`
- `all_runs_repeatable`
- `cycle_false_positive_removed`
- `modulation_always_bounded`
- `tie_gate_only`

Results are written to `data/core_integration_shadow_v2/results/`.
