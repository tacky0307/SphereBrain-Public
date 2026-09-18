# SphereBrain Research Checkpoint — Experience-Grounded Prediction, Puzzle Observation, and Route Readout

**Date:** 2026-09-18  
**Status:** public research checkpoint; not a promoted Core release and not a claim of general intelligence.

This checkpoint freezes a short research line asking whether one experience-shaped individual can learn from interaction, reuse that experience in a human-readable puzzle setting, and eventually let persistent SphereBrain routes contribute to prediction and action.

Useful behaviors were demonstrated, but an equally important attribution boundary emerged: several improvements were caused by explicitly designed predictive or hypothesis-management rules, while a distinctive advantage from SphereBrain's native route dynamics has **not yet been established**.

Feature accumulation is intentionally paused here. The next research line will not be chosen merely because another mechanism can be added.

## Two Keys

The playable Two Keys observation environment exposed retention, cross-task experience, and resumption.

A stricter progress gate reduced post-resumption failures from **6 to 0** across three development worlds, but total real trials increased from **53 to 91** and success was delayed. This separated "the candidate changed" from "there is new positive evidence for retrying the same candidate," but did not establish a globally better policy.

## Flip Learning Loop

A fresh individual learned from real before/action/after experience only.

In one preserved v2 seed-42 session:

- target reached after **19 real actions**;
- **131 / 162** local conditions observed;
- final action predicted to produce the target;
- **9 / 9** final output-cell predictions matched.

Direct prediction came from a **designed conditional memory**, not established native-route reasoning.

## Route Readout v1

Native Core transitions were connected to readout.

After training on the same 19 real transitions, native route readout recalled **19 / 19** experienced input/action pairs. Disconnecting or zeroing the route path removed recall, and target permutation disrupted it.

Before the 19 actual results were revealed, however, the conservative route readout admitted **0 / 171** output-cell predictions and did not change action selection.

This established known-pair route recall, not unseen-input transfer.

## Route Readout v2

A reusable partial-input selection extension produced transfer in generated non-spatial worlds.

But ordinary episodic storage using the **same higher-level learning rule** matched the route-based implementation, and binarizing positive route weights did not change outputs.

Twelve errors exposed premature simplification of still-unobserved joint input conditions.

## Route Readout v3

v3 retained all empirically consistent small alternatives instead of collapsing to the simplest explanation too early.

- all **12** prior v2 errors became abstentions;
- in ten new diagnostic environments, ambiguity-guided acquisition ended with wider prediction coverage than a reproducible random-order control after the same 24 added experiences;
- disconnecting the ambiguity score changed the selected experiment in **233 / 240** matched decisions.

Yet ordinary storage using the same ambiguity-family rule again matched the route-based system. Native route magnitude, competition, or numerical dynamics therefore remain unproven as a distinctive advantage.

## Supported at this checkpoint

- experience-grounded prediction loops can be made inspectable;
- native learned route connections can be read as experienced associations;
- restricted partial-structure reuse can work;
- unresolved alternatives can be preserved;
- discriminating experience can be selected from ambiguity under a restricted model class.

## Not established

- general puzzle solving;
- broad autonomous planning;
- general intelligence;
- universal causal discovery;
- superiority of native SphereBrain routes over ordinary memory using the same higher-level rule;
- a causal role for route-weight magnitude or native competition in the observed transfer;
- generalization beyond the restricted 0/1/2-input model class.

## Why pause here

Continuing to add features without a fixed decision criterion would risk changing the research question from:

> Does experience-shaped internal structure provide a useful and distinctive function?

to:

> What additional feature can make the system do one more thing?

Before another experimental line begins, the project will first define one unresolved question and a continue/stop criterion.

See [README_JA.md](README_JA.md), [RESEARCH_RECORD.md](RESEARCH_RECORD.md), and repository-level [LIMITATIONS.md](../../LIMITATIONS.md).
