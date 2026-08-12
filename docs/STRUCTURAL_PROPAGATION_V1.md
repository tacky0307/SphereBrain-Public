# Structural Propagation v1

## Purpose

Test whether an ephemeral, language-free structural history can weakly change a later neutral branch without answer labels, semantic branch meanings, or long-term learning.

## Design

- Structural Working State v2 supplies the terminal structural context.
- The two branch candidates begin with identical baseline activation.
- Candidate context directions are deterministic numerical vectors with no semantic labels.
- Structural modulation is zero-mean and intentionally weak.
- Turning structural context off must return exactly `[0.5, 0.5]`.
- Swapping candidate order must swap the resulting probabilities.
- Renaming Node IDs must not change the result.

## Cases

1. Direct history versus merge history with an identical common suffix.
2. Repeated-edge history versus non-repeated equal-length history with an identical common suffix.
3. Identical direct structure using different Node IDs.

## Success checks

- Neutral branch when structural context is disabled.
- Merge history changes the branch distribution.
- Repetition history changes the branch distribution.
- Node-ID invariance.
- Candidate-order symmetry.

This experiment does not claim correct reasoning or route selection. It only tests whether prior structural form can causally influence subsequent activity distribution.
