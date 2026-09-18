# Frozen Research Record — 2026-09-18

This file preserves the numerical and provenance boundary for the checkpoint. It is a summary of development records, not a new confirmatory analysis.

## A. Progress-Gated Resumption / Two Keys

Source result commit at publication time: 9d10ff2f266296cecd3e488ee7c548186027d240

| seed | old A-solved B-trial | cautious gate | old post-resume failures | cautious failures | total real trials old → cautious |
|---|---:|---:|---:|---:|---:|
| 33013 | 2 | 10 | 0 | 0 | 13 → 27 |
| 33023 | 6 | 17 | 4 | 0 | 23 → 41 |
| 33037 | 4 | 8 | 2 | 0 | 17 → 23 |

Aggregate post-resume failures: **6 → 0**.  
Aggregate total real trials: **53 → 91**.

No superiority claim.

## B. Flip Learning Loop

v1 recovery reference: 5b4e1ea089f6452e6ec094ca8b50c215b1117f42  
v2 development reference: 56bb03e4c8f0d58248fc6b35784886bc7d4c14b9

Preserved v2 seed-42 session:

- fresh individual, no imported actual receipts;
- target reached in **19** real actions;
- **131 / 162** local conditions known at completion;
- **31** unknown; **0** conflict;
- final action predicted all nine target cells correctly;
- learner source recorded as designed conditional memory, not native route prediction.

The earlier canonical 21-receipt development continuation reported **4384 / 4779 (91.7347%)** cell prediction coverage and **4384 / 4384** correctness among predicted cells.

## C. Route Readout v1

Result commit: b21ba61d9b730b1e79c09fc58d6f59ddcc358090

- experienced-pair recall: **19 / 19**;
- readout disconnected: no predictions;
- transitions zeroed: no predictions;
- target-permutation recalls: **1, 3, 1, 0, 0 / 19**;
- pre-result admitted predictions: **0 / 171** cells;
- route connection changed **0 / 19** action choices in the preserved sequence.

## D. Route Readout v2

Result commit: 50d07e14e449e744e4080671893f01219fe773fb

Generated diagnostic evaluation, 3240 output components per family:

| condition | unary predicted/correct | two-input predicted/correct |
|---|---:|---:|
| learned route-selection structure | 2998 / 2998 | 1298 / 1286 |
| whole-cue-only | 0 / — | 0 / — |
| positive weights binarized | 2998 / 2998 | 1298 / 1286 |
| plain episodic + same rule | 2998 / 2998 | 1298 / 1286 |

The **12** two-input errors were premature constant simplifications where a decisive joint input combination had not yet been observed.

## E. Route Readout v3

Protocol-first commit: 52d43120e098212bd2a1f3532a7085318d3b81d0  
Result commit: 8ddac08ea2efc6ee7a68c71c9a45019356ff6841

Prior-world regression:

- unary/constant: **2677 / 3240** predicted, **2677 / 2677** correct;
- two-input: **3092 / 3240** predicted, **3092 / 3092** correct;
- all 12 prior v2 errors became abstentions.

Prospective acquisition diagnostic after 24 selected experiences:

| family | initial | random-order control | unknown-count | ambiguity-guided |
|---|---:|---:|---:|---:|
| unary/constant | 407/3240 (12.56%) | 1242/3240 (38.33%) | 1611/3240 (49.72%) | 1765/3240 (54.48%) |
| two-input | 616/3240 (19.01%) | 1849/3240 (57.07%) | 2190/3240 (67.59%) | 2207/3240 (68.12%) |

Ambiguity-guided selection exceeded the reproducible random-order control in all ten generated environments. At matched decision points, disconnecting the ambiguity acquisition score changed the selected experiment in **233 / 240** decisions.

However, plain episodic storage with the same ambiguity-family rule matched the route implementation on the reported prediction and selection checks.

## Frozen conclusion

Supported as an engineering/research observation:

- inspectable experience-grounded prediction;
- known-pair native-route association readout;
- restricted partial-structure reuse;
- ambiguity-preserving acquisition.

Not supported:

- general intelligence;
- general puzzle solving;
- unique native-route advantage over ordinary memory with the same higher-level rule;
- promotion of these mechanisms into the canonical public Core.

No next experiment is declared in this checkpoint.
