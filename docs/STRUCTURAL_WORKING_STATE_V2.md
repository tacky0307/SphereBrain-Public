# Structural Working State v2

## Purpose

Separate pure structural history from local terminal differences.

## Controls

- Both compared episodes use an identical final suffix.
- Terminal local state is calculated only from that suffix.
- With structural state disabled, terminal states must match exactly.
- Earlier history can reach the terminal only through the ephemeral working state.
- No language, answer labels, route choice, Core mutation, or long-term learning.

## Cases

1. Direct history vs merge history, followed by the same suffix.
2. Repeated-edge history vs non-repeated equal-length history, followed by the same suffix.
3. Node-ID invariance control.

## Success criteria

- `local_conditions_equal = true`
- `without_structure_equal = true`
- History comparison: `structure_history_visible = true`
- ID invariance: structural distance remains zero.

## Run

`run_structural_working_state_v2.bat`

Results are written under `data/structural_working_state_v2/results/`.
