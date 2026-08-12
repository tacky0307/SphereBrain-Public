# Core Structural Assist Live Observer v1

## Purpose

Observe the real Core Structural Assist feature with arbitrary text while keeping the stored brain untouched.

The observer runs the same text twice:

1. Structural Assist OFF
2. Structural Assist ON

It then creates an HTML report showing the activation timeline and assist trace side by side.

## Safety

The observer always uses:

- `learn=False`
- `noise=0.0`
- no call to `brain.save()`
- separate loaded Core instances for OFF and ON
- SHA-256 verification of `data/brain.json` before and after

## Run

```powershell
git fetch origin
git switch experiment/core-structural-assist-live-observer-v1
git pull
.\run_core_structural_assist_live_observer_v1.bat
```

Enter one sentence when prompted. Empty input uses:

```text
今日は晴れて気持ちいい
```

## Output

```text
data/core_structural_assist_live_observer_v1/results/core_structural_assist_live_observer_v1.html
data/core_structural_assist_live_observer_v1/results/core_structural_assist_live_observer_v1.json
```

The HTML report opens automatically and shows:

- input text and source Node IDs
- number of assist activations
- number of near-zero ties resolved by structure
- OFF and ON active Nodes for every step
- baseline margin and maximum modulation per step
- OFF and ON traversed Edges
- route/history/file-integrity comparisons

## Interpretation

- `構造は待機`: the normal Core margin was not ambiguous.
- `構造補助が作動`: the tie gate opened, but the top candidate did not change.
- `構造が同率を解決`: a near-zero tie was ordered by structural context.

A changed internal ordering does not necessarily change the selected Edge set because the current focused Core can propagate through multiple candidates.
