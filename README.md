# SphereBrain

**SphereBrain** is an open experimental research project investigating whether intelligence can emerge from internal structures formed through experience.

The central hypothesis is not that the Core must contain human-readable meanings, labels, or sentences. Language, images, sound, and other external information can be converted into numerical stimuli, while the Core changes through the activity that those experiences produce.

> **Intelligence may be an evolving structure of recognition shaped by experience.**

SphereBrain therefore studies whether repeatedly experienced activity can form reusable paths, distributed support structures, persistent internal organization, and eventually structures that correspond to what humans later describe as meaning or concepts.

## Architecture

The working direction is:

```text
External experience
        ↓
Encoder / sensory interface
        ↓
SphereBrain Core
        ↓
Decoder / expression interface
```

The Encoder and Decoder may use language-aware systems. The research target is the **Core**: an internal network whose structure changes through experience.

The Core is not intended to store statements such as `dog = animal` as explicit symbolic knowledge. Instead, the project asks whether many different experiences can gradually create internal structures that are reused, differentiated, connected, and stabilized.

## Open Research Archive

SphereBrain is being developed as an open research record. Successes, failures, negative results, abandoned hypotheses, and experimental code are all part of the research history.

Start here:

- [README_JA.md](README_JA.md) — Japanese introduction / 日本語版
- [Natural Law Study — Phase I](docs/NATURAL_LAW_STUDY_PHASE_I.md) — experience-shaped dynamics in an artificial world
- [Natural Law Study — Phase I 日本語版](docs/NATURAL_LAW_STUDY_PHASE_I_JA.md) — 経験によって形づくられる人工世界の動力学
- [PHILOSOPHY.md](PHILOSOPHY.md) — research philosophy and current hypothesis
- [ARCHITECTURE.md](ARCHITECTURE.md) — system boundaries and component roles
- [RESEARCH_TIMELINE.md](RESEARCH_TIMELINE.md) — chronological research history
- [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) — experiment map and major findings
- [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md) — evidence status of representative experiments
- [FILE_AND_RIGHTS_AUDIT.md](FILE_AND_RIGHTS_AUDIT.md) — publication safety, file-size, and provenance audit
- [LIMITATIONS.md](LIMITATIONS.md) — unresolved questions and claim boundaries
- [REPRODUCE.md](REPRODUCE.md) — how to reproduce experiments
- [OPEN_RESEARCH_POLICY.md](OPEN_RESEARCH_POLICY.md) — preservation and publication policy
- [CONTRIBUTING.md](CONTRIBUTING.md) — reproduction, criticism, and contribution guidance
- [AI_CONTRIBUTION.md](AI_CONTRIBUTION.md) — disclosure of human and AI roles

## Current Research Direction — August 2026

Recent experiments moved from direct semantic interpretation toward **label-blind structural observation** and then toward an artificial natural-law model in which experience changes the material-like state of the Core.

The current Natural Law Study asks whether Flow, Landscape deformation, finite structural resources, natural recovery, and local metaplastic consolidation can produce history-dependent behavior before human meaning is assigned. Phase I currently freezes this line through ID1R, including repeated maturation, lifetime differentiation, temporal-spacing effects, intervening-experience dependence, and limited experience-order imprinting. See the [Phase I report](docs/NATURAL_LAW_STUDY_PHASE_I.md).

Earlier experiments used human semantic categories to create controlled conditions and inspect Core behavior. These experiments produced useful findings about bridge formation, stability, consensus persistence, ablation, and rescue. They also revealed an important limitation: human semantic labels can easily become confused with the Core's own internal organization.

The current direction therefore asks a more fundamental question:

> If a large stream of experiences is presented to the Core, will reusable internal organization emerge before humans assign meaning to it?

## Research Principles

1. **Experience before explanation.** Internal structure should be formed by experience, not by injecting the desired answer into the Core.
2. **Core structure before human labels.** Human-readable meaning is an interpretation layer, not assumed to exist inside the Core.
3. **Negative results are results.** Failed hypotheses remain in the repository and research history.
4. **Observation and intervention are separated.** A microscope should not silently change the system it measures.
5. **Reproducibility matters more than a persuasive story.** Experiments should preserve seeds, runners, outputs, and code history wherever practical.
6. **Claims remain proportional to evidence.** A PASS in one experiment validates only the tested condition; it does not establish general intelligence or human-like understanding.

## Historical Prototype Features

The project began with a spherical network prototype exploring:

- 3D spherical network structure
- persistent graph state
- path reinforcement
- continuous activity
- path-based memory
- replay / forgetting / consolidation ideas
- language, audio, and multimodal input experiments

These earlier stages remain part of the project history rather than being replaced by the current interpretation.

## Repository Status

The active experimental line is maintained through versioned experiment runners such as:

```text
experiments/run_core_growth_binding_vXX.py
run_core_growth_binding_vXX.bat
```

Experiment results are written under corresponding `data/core_growth_binding_vXX/` directories when the runner is executed locally.

The production `data/brain.json` is intentionally protected in microscope experiments unless an experiment explicitly states otherwise.

## Research leadership

- **Originator and researcher:** Takio (`たきお`)
- **AI collaborator:** “Mimi,” an OpenAI ChatGPT system that assisted with design discussion, implementation, analysis, and documentation

Scientific and publication responsibility remains human-led. See [AI_CONTRIBUTION.md](AI_CONTRIBUTION.md).

## License

Licensed under the [Apache License 2.0](LICENSE).

## Citation / Historical Record

**Natural Law Study — Phase I** is archived on Zenodo as version `v1.0.0-phase1`.

- DOI: [10.5281/zenodo.21964627](https://doi.org/10.5281/zenodo.21964627)
- GitHub release: [`v1.0.0-phase1`](https://github.com/tacky0307/SphereBrain-Public/releases/tag/v1.0.0-phase1)
- Publication date: 2026-08-16

Please cite the archived Zenodo release when referring to this Phase I milestone. Git commit history and versioned experiment files remain the primary detailed chronology of the ongoing project.
