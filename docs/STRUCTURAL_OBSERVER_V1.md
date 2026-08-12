# SphereBrain Structural Observer v1

## Purpose

Test whether non-linguistic Core activity can be described as structure rather than only as Node/Edge counts.

This experiment distinguishes:

- chain
- merge
- split
- parallel paths
- cycle
- repeated same structure
- repeated different structures

## Boundary

Structural Observer v1 is read-only.

It does not:

- choose a route
- change Node or Edge weights
- learn a correct answer
- use language labels inside the Core
- feed observations back into propagation

It only receives a temporal directed activity graph and extracts ID-invariant topology and timing features.

## Main features

- source and sink counts
- merge and split counts
- connected-component count
- maximum structural depth
- simultaneous activity width
- temporal overlap
- repeated Edge use
- degree-distribution entropy
- canonical temporal-topology signature

## Important control

The same structure is recreated with completely different Node IDs.

A successful observer should produce:

- the same canonical signature
- feature-vector similarity of 1.0 or extremely close to it

This checks that the observer recognizes shape rather than memorizing Node IDs.

## Repetition control

Two three-episode sequences are compared:

1. chain, chain, chain at different Node IDs
2. chain, merge, split at different Node IDs

The first should have a high exact-repeat ratio. The second should not.

## Run

```bat
run_structural_observer_v1.bat
```

## Outputs

```text
data/structural_observer_v1/results/structural_observer_v1.json
data/structural_observer_v1/results/structural_observer_v1.csv
```

## Interpretation

This version does not prove that the Core itself understands structure.

It tests the prerequisite:

> Can activity topology be represented without language, correct-answer labels, or Node identity?

Only after this succeeds should a later Structural Propagation experiment allow the temporary structural state to affect the next propagation step.
