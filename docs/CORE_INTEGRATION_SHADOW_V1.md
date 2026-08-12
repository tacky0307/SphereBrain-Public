# Core Integration Shadow v1

## Purpose

Test Structural Propagation against the real focused SphereBrain Core without changing route selection, activation, weights, usage, node usage, learning, or brain.json.

## Method

The experiment loads the existing brain.json and reproduces focused propagation with noise=0 and learn=False. For each real candidate Edge, it records the baseline Core score and calculates a small structural modulation beside it.

The shadow score is never fed back into Core. The actual route remains the baseline route.

## Edge features used

- current weight
- usage count, normalized
- target degree
- geometric Edge length
- radial direction derived from Node positions

No semantic label, correct answer, random candidate vector, or trainable parameter is used.

## Safety checks

- SHA-256 of brain.json before and after
- digest of Core arrays before and after every run
- deterministic repeated-run equality
- exact baseline replay comparison against `SphereBrain.propagate(..., noise=0, learn=False)`
- explicit flags that learning, noise, and intervention are disabled

## Main metrics

- mean and maximum distance between baseline and shadow probability distributions
- top-k overlap
- number and ratio of top-candidate changes
- route replay symmetric difference

## Interpretation

A useful shadow gain should create measurable but limited probability changes. Large top-candidate change ratios, low top-k overlap, or concentration in only a few cases indicate that integration would be unsafe.

This experiment does not prove improved reasoning. It only measures how strongly structural context would perturb the current Core if enabled.
