# Structural Puzzle Lab v1

A read-only controlled lab for observing whether bounded Structural Assist changes the order of exactly tied candidate routes after different non-language structural histories.

## Puzzles

- Merge: two paths merge before the candidate branch.
- Repetition: a local cycle/repeated route precedes the candidate branch.
- Missing route: a directional chain precedes forward/backward candidates.

## Safety

- No learning.
- No noise.
- `data/brain.json` is not loaded into the synthetic puzzle state and is never saved.
- The hash of `data/brain.json` is checked before and after.
- Structural modulation remains bounded by the Core Structural Assist absolute cap.

## Controls

Every puzzle is rerun with all Node IDs shifted while preserving the same topology, candidate state, and relative geometry. The selected candidate role must remain unchanged.

## Run

```bat
run_structural_puzzle_lab_v1.bat
```

Results:

- `data/structural_puzzle_lab_v1/results/structural_puzzle_lab_v1.html`
- `data/structural_puzzle_lab_v1/results/structural_puzzle_lab_v1.json`

This lab does not demonstrate semantic understanding, learned puzzle solving, or correctness. It tests whether controlled structural history can resolve a neutral candidate tie in an ID-invariant, bounded manner.
