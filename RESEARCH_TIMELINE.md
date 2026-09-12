# SphereBrain Research Timeline

This document records the development of SphereBrain as an experimental research project. It is intentionally chronological and includes changes in interpretation, not only successful results.

## Origin — July 2026

SphereBrain began from a simple question: can memory and intelligence be represented less as stored symbolic information and more as the structure of paths formed by experience?

The early prototype used a spherical graph with nodes, edges, activation, weights, persistence, and visualization. Repeated experience reinforced frequently used paths.

The project adopted the working statement:

> **Intelligence is an evolving structure of recognition shaped by experience.**

Early experiments used short language inputs as convenient sources of repeated stimulation while the Core itself was intended to remain numerical and structural.

## Path memory and accumulated experience

Initial experiments explored whether repeated input could create thicker / stronger paths and whether later activity would preferentially reuse them.

This phase established the practical foundation:

- persistent graph state,
- repeated reinforcement,
- activity traces,
- controlled experiments,
- local SQLite / JSON persistence,
- experimental HTML and batch runners.

## Semantic Encoder experiments — late July to early August 2026

A Semantic Encoder v2 line represented inputs using staged components such as subject, relation, and content. This allowed controlled tests of whether related experiences could share parts of the Core's paths.

Important lesson: the Encoder may organize stimulation for experimentation, but the Core should not be assumed to possess the linguistic semantics of those labels.

## Primary Native Learning — v79 to v81

A series of structural puzzle experiments investigated learning, consolidation, recovery, and stability.

- v79 Homeostatic Consolidation produced a strong long-horizon result in the controlled puzzle setting.
- v80 / v80B aligned the experimental motif with native learning behavior.
- v81 promoted validated Native Learning state into the primary `SphereBrain` Core while preserving backward compatibility.

This established that a controlled structural learning mechanism could become part of the Core without relying on semantic answer labels.

## Semantic bridge revalidation — v82 to v84

The research then tested whether separated semantic experiences could form shared internal bridges.

- v82 reproduced concept-sharing and one-shot integration in some conditions, but shared-context bridging did not appear reliably.
- v82B microscope experiments showed that newly shared edges were not yet convincing evidence of a meaningful bridge.
- v82C attributed the apparent shared edges primarily to popular hubs rather than context-linked structure.
- v83 introduced continuous semantic episodes and produced a successful bridge / transfer result in a small controlled setup.
- v84 expanded to multiple seeds and domains. The v83 effect did not reproduce robustly.

This was an important negative result: a promising single experiment was not treated as general validation.

## Bridge stability and failure analysis — v84B to v92B

The next experiments focused less on formation and more on why formed bridges disappeared.

- v84B compared successful and failed trials.
- v85 showed that bridges often formed but rarely stabilized naturally.
- v86 tested homeostatic consolidation and found that indiscriminate protection could make results worse.
- v86B showed that protection was often applied too broadly, especially around hub-like edges.
- v87 delayed and filtered consolidation reduced that damage but did not improve stable outcomes.
- v88 event-triggered consolidation produced a promising small result.
- v89 robustness testing showed that the v88 improvement did not generalize strongly across 100 trials per mode.
- v90 identified post-formation stabilization as the primary bottleneck: many trials formed bridges but later lost stable success.
- v91 and v91B decomposed collapse into candidate, control, similarity, and multi-factor mechanisms. The dominant pattern involved experimental weakening combined with control catching up.
- v92 attempted relative selectivity preservation but rarely intervened.
- v92B showed that a strict non-hub filter blocked almost all unstable bridges from becoming intervention candidates.

The combined lesson was that neither blanket hub protection nor blanket hub exclusion was sufficient.

## Consensus persistence — v93 to v95B

The project shifted from single-edge protection toward distributed structural support.

- v93 measured structural consensus at first bridge formation. First-formation consensus did not meaningfully separate stable from unstable trials.
- v93B measured consensus persistence over time. Stable bridges retained or increased distributed support, while unstable bridges lost it.
- v94 robustness testing expanded to 40 seeds and 8 domains. Consensus persistence remained strongly positive across all domains and many comparable seeds.
- v95 ablated consensus-support edges in previously stable trials. Stable survival fell substantially, but the experiment could not yet show that the effect was specific to consensus rather than generic important-edge damage.
- v95B added matched random and hub-matched controls. Consensus-targeted ablation caused substantially more harm than matched local controls, giving stronger specificity evidence.

This was one of the strongest structural findings of the project to date: stable bridges tended to be supported by persistent, distributed internal structure rather than only by individual strong edges.

## Rescue experiments — v96 to v97

The next question reversed the causal direction: if destroying consensus harms stability, can increasing support rescue unstable bridges?

- v96 strengthened existing consensus-support edges. Rescue occurred in some trials but the improvement over matched local support was small.
- v96B compared rescued and non-rescued trials. Rescue correlated more strongly with growth in support sources and quorum than with weight increase alone. Many failures were consistent with "weight-only" reinforcement.
- v97 attempted to create independent support through additional semantic episodes. The intended shared-context support episodes did not outperform random extra experience; random extra episodes actually rescued slightly more trials and produced greater support / quorum growth.

This rejected a simplistic interpretation that adding more human-semantically similar episodes automatically creates independent structural support.

## Conceptual transition — August 11, 2026

The project then made an explicit conceptual correction.

Human researchers may say that two inputs have "the same meaning," but the Core does not necessarily know that meaning. For the Core, the inputs are numerical stimulation and the resulting internal activity.

The research question was reframed:

> Instead of asking whether the Core reproduces human semantic categories, ask whether large combinations of experience naturally produce reusable internal structures, and only afterward investigate what those structures correspond to.

This moves interpretation from:

```text
human meaning → Core → check whether meaning appeared
```

toward:

```text
experience → Core structure → discover organization → interpret afterward
```

## v98 — Emergent Structural Meaning Microscope

v98 is the first explicit label-blind structural experiment in this direction.

Language is still used as an input stimulus, but human semantic family labels are excluded from:

- structure discovery,
- similarity threshold selection,
- cluster formation.

The experiment first derives signatures from internal nodes, edges, and activation. It then discovers natural groupings from those signatures. Human labels are consulted only after structure discovery as a post-hoc interpretation layer.

v98 also compares cycle 0 against later experience so that encoder baseline organization can be distinguished from experience-driven structural change.

At the time this archive was created, v98 was still computationally running locally and no result had yet been recorded here.

## Current research question

The current central question is:

> **Can large streams of experience create persistent, reusable, internally organized structures before human semantic labels are used to define what those structures mean?**

The next phase should prioritize structural observation, reproducibility, and larger mixed experience streams while keeping human semantic interpretation downstream of Core organization.

<!-- DCFMA_V1_TIMELINE_START -->
## Endogenous temporal consequence line — September 2026

ECTBF v1 removed supplied before/after endpoints; ECTBF v1B rejected a simple interchangeable-member explanation; ECTBF v1C directly tested cooperative covariance. The v1C frozen holdout produced strong average and bootstrap evidence, but missed preregistered worst-case and positive-fraction requirements. The canonical result is preserved as **inconclusive cooperative structure**, not a promoted positive result.
<!-- DCFMA_V1_TIMELINE_END -->
