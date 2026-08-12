# Structural Propagation v2

## Purpose

Test whether ephemeral structural history can modulate measurable candidate Edge transmission without semantic labels, correct answers, random candidate vectors, or long-term learning.

## Candidate Edge state

Each candidate is described only by:

- weight
- usage
- recency
- target degree
- direction

The baseline transmission is derived from these local properties. Structural context then adds a weak zero-mean modulation.

## Controls

- Structure disabled: histories must produce the same baseline distribution for identical candidate states.
- Node-ID invariance: identical topology with different node IDs must produce identical propagation.
- Candidate-state swap: swapping all candidate properties must swap the probabilities.
- No random candidate vectors.

## Interpretation

Success means structural history can causally affect the next Edge transmission through actual candidate state. It does not yet establish useful reasoning, correct route selection, learning, or integration into the production Core.

## Run

```powershell
.\run_structural_propagation_v2.bat
```

Outputs:

- `data/structural_propagation_v2/results/structural_propagation_v2.json`
- `data/structural_propagation_v2/results/structural_propagation_v2.csv`
