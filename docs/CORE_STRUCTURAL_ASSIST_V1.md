# Core Structural Assist v1

## Purpose

Integrate the structural assistance validated by Core Integration Shadow v3 into `SphereBrain` while preserving existing behavior by default.

## Default safety

`structural_assist_enabled` defaults to `False`.

When disabled, focused propagation keeps the original candidate order and does not apply structural modulation. Existing `brain.json` files that do not contain structural settings load with assistance disabled.

## Enabled behavior

When enabled, the assist:

- acts only when the top baseline margin is at or below `0.0025`;
- uses `0.02` structural gain;
- classifies margins at or below `1e-8` as near-zero ties;
- caps modulation at `0.00005` absolutely;
- for meaningful margins, also caps modulation at 35% of the baseline margin;
- changes candidate ordering only, never candidate values, weights, usage, learning, or candidate membership.

## Runtime control

```python
brain.set_structural_assist(True)
brain.set_structural_assist(False)
```

The most recent focused propagation trace is available at:

```python
brain.last_structural_assist_trace
```

## Persistence

The assist flag and safety parameters are included in `save()` and restored by `load()`. Old files remain compatible and default to disabled.

## Verification

Run:

```powershell
.\run_core_structural_assist_v1.bat
```

Outputs:

- `data/core_structural_assist_v1/results/core_structural_assist_v1.json`
- `data/core_structural_assist_v1/results/core_structural_assist_v1.csv`

The verification checks default-OFF compatibility, deterministic OFF and ON runs, cap enforcement, strong-decision protection, unchanged routes/nodes/history/final activation, and evidence that the assist actually activates on ties.
