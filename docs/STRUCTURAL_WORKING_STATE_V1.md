# Structural Working State v1

## Purpose

Test whether a non-linguistic structural context can remain during one activity episode so that the same terminal node can retain a different state depending on how it was reached.

## Boundaries

- no language or Embedding
- no correct-answer labels
- no trainable parameters
- no long-term learning
- no modification of the existing SphereBrain Core
- working state is discarded after each episode

This is a simulator experiment before integration into `brain.py`.

## Working state

At each step, the partial activity topology is observed and converted into a small numeric context containing normalized source, sink, merge, split, component, depth, parallel-width, temporal-overlap, edge-reuse, and cycle signals.

The context is updated with decay and appended to the local state of active nodes. It does not choose a route or prescribe an answer.

## Comparisons

1. Direct arrival versus merge arrival at the same terminal node.
2. One chain versus the same chain repeated three times.
3. Direct-arrival structure with different node IDs.
4. Merge-arrival structure with different node IDs.
5. The same comparisons with structural context disabled.

## Success criteria

- direct and merge arrival differ when structural context is enabled;
- the difference disappears or shrinks when disabled;
- repeated structure differs from a single occurrence;
- changing node IDs does not materially change the structural terminal state.

## Run

```bat
run_structural_working_state_v1.bat
```

Results:

- `data/structural_working_state_v1/results/structural_working_state_v1.json`
- `data/structural_working_state_v1/results/structural_working_state_v1.csv`
