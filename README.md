# SphereBrain

**SphereBrain** is an open experimental research project investigating whether intelligence can be shaped by internal structures that change through experience.

> **Intelligence may be an evolving structure of recognition shaped by experience.**

The project does not assume that the Core must contain human-readable words, labels, or symbolic statements. External language, images, sound, sensor data, or other information may be converted into numerical stimulation, while the research target is the **Core itself**: a plastic internal network whose structure, temporal state, relational memory, and stability change through experience.

## Current Milestone — Unified Experience Core v2.0

The current public milestone consolidates mechanisms that were previously tested separately into one generic experience-driven Core.

The public implementation is:

- [`spherebrain_core_v2.py`](spherebrain_core_v2.py) — self-contained public research Core
- [`run_public_minimal_demo.py`](run_public_minimal_demo.py) — small acquisition/reversal demonstration
- [`docs/LATEST_CORE_RESEARCH_AUDIT.md`](docs/LATEST_CORE_RESEARCH_AUDIT.md) — what is promoted into Core and what remains experimental
- [`data/public_demo/latest_core_regression.json`](data/public_demo/latest_core_regression.json) — integration/regression result
- [`data/public_demo/latest_core_smoke_adaptation_v1.json`](data/public_demo/latest_core_smoke_adaptation_v1.json) — lightweight adaptation result

### Mechanisms currently promoted into Core

```text
External stimulus
      ↓
Structural propagation / long-term structural memory
      ↓
Current activity
      ├── bounded working state
      └── online directed transition trace
      ↓
Experience signature
      ↓
Fast relational form + slow relational identity
      ↓
Soft relation membership
      ↓
Relation-conditioned distributed value
      ↓
Reliability / maturity / replasticity
      ↓
Updated structural + temporal + relational Core
```

The promoted mechanisms are:

- experience-dependent structural weights and usage;
- bounded short-term / working state;
- online directed transition memory;
- self-organizing relational separation;
- cluster-conditioned distributed consequence value;
- maturity-gated success plasticity;
- failure-driven replasticity;
- fast/slow relational identity continuity across structural drift;
- unified save/load persistence.

These are promoted as **research principles**, not universal mathematical laws. Exact constants and experimental coordinate systems remain open to revision.

## What the current Core has demonstrated

The canonical regression suite for the integrated Core passed **11 / 11 checks**, including structural memory, short-term state, transition trace, relational memory, identity continuity, maturity, replasticity, persistence, read-only scoring, and the absence of a task-answer lookup table.

A lightweight two-state sensorimotor smoke test then showed:

```text
Initial accuracy                         50%
After acquisition                      100%
Accuracy on reversed rule before shift   0%
After reversal training                100%
Old rule after reversal                  0%
```

The reversal reached 100% by the 80-trial checkpoint and remained correct through the end of the 200-trial reversal phase in that smoke test.

This supports a narrow claim: the current Core can acquire and revise a tiny experience-dependent sensorimotor mapping while its promoted memory mechanisms remain active.

## What is NOT established

The current public Core does **not** establish:

- general intelligence;
- human-like understanding;
- solved held-out compositional generalization;
- autonomous planning;
- autonomous next-state prediction / predictive free-run;
- a universal semantic representation;
- a universal biological model of the brain.

In particular, held-out relational generalization remains an open research boundary. The project explicitly preserves negative results and unresolved mechanisms rather than presenting them as solved capabilities.

## Run the minimal demo

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements-core-v2.txt
python run_public_minimal_demo.py
```

The demo provides only states, candidate actions, and scalar success/failure. The Core owns structural learning, working state, transition trace, relational memory, maturity, and replasticity.

## Architecture

The broader working direction remains:

```text
External experience
        ↓
Encoder / sensory interface
        ↓
SphereBrain Core
        ↓
Decoder / expression interface
```

The Encoder and Decoder may use language-aware systems. The Core is intentionally not defined as a store of explicit symbolic statements such as `dog = animal`. The research question is whether reusable internal organization can form through repeated experience and later support recognition, adaptation, transfer, or prediction.

## Open Research Archive

SphereBrain is maintained as an open research record. Successes, failures, negative results, abandoned hypotheses, and experimental code are part of the history.

Start here:

- [README_JA.md](README_JA.md) — Japanese introduction / 日本語版
- [Natural Law Study — Phase I](docs/NATURAL_LAW_STUDY_PHASE_I.md)
- [Natural Law Study — Phase I 日本語版](docs/NATURAL_LAW_STUDY_PHASE_I_JA.md)
- [PHILOSOPHY.md](PHILOSOPHY.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [RESEARCH_TIMELINE.md](RESEARCH_TIMELINE.md)
- [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)
- [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)
- [LIMITATIONS.md](LIMITATIONS.md)
- [REPRODUCE.md](REPRODUCE.md)
- [OPEN_RESEARCH_POLICY.md](OPEN_RESEARCH_POLICY.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [AI_CONTRIBUTION.md](AI_CONTRIBUTION.md)

## Research Principles

1. **Experience before explanation.** Internal structure should be formed by experience, not by injecting the desired answer into the Core.
2. **Core structure before human labels.** Human-readable meaning is an interpretation layer, not assumed to exist inside the Core.
3. **Negative results are results.** Failed hypotheses remain part of the research history.
4. **Observation and intervention are separated.** Measurement should not silently change the system being measured.
5. **Reproducibility matters more than a persuasive story.** Seeds, runners, outputs, and code history are preserved wherever practical.
6. **Claims remain proportional to evidence.** A successful experiment validates only its tested conditions.
7. **Validated principles are promoted into one evolving Core.** New experimental mechanisms remain outside the canonical Core until enough evidence supports promotion.

## Research leadership

- **Originator and researcher:** Takio (`たきお`)
- **AI collaborator:** “Mimi,” an OpenAI ChatGPT system that assisted with design discussion, implementation, analysis, and documentation

Scientific and publication responsibility remains human-led. See [AI_CONTRIBUTION.md](AI_CONTRIBUTION.md).

## License

Licensed under the [Apache License 2.0](LICENSE).

## Citation / Historical Record

### Unified Experience Core v2.0

**Unified Experience Core v2.0** is archived on Zenodo as version `v2.0.0-unified-core`.

- Version DOI: [10.5281/zenodo.22048511](https://doi.org/10.5281/zenodo.22048511)
- Concept DOI for all SphereBrain versions: [10.5281/zenodo.21898610](https://doi.org/10.5281/zenodo.21898610)
- GitHub release: [`v2.0.0-unified-core`](https://github.com/tacky0307/SphereBrain-Public/releases/tag/v2.0.0-unified-core)
- Publication date: 2026-08-21

Please cite the **Version DOI** when referring specifically to the Unified Experience Core v2.0 milestone. Use the **Concept DOI** when referring to SphereBrain across versions and you want the citation to resolve to the latest Zenodo record.

### Natural Law Study — Phase I

**Natural Law Study — Phase I** is archived on Zenodo as version `v1.0.0-phase1`.

- DOI: [10.5281/zenodo.21964627](https://doi.org/10.5281/zenodo.21964627)
- GitHub release: [`v1.0.0-phase1`](https://github.com/tacky0307/SphereBrain-Public/releases/tag/v1.0.0-phase1)
- Publication date: 2026-08-16

Please cite the archived Phase I Zenodo release when referring specifically to that milestone. Git history and versioned experiment files remain the detailed chronology of the ongoing project.


## Research checkpoint — Experience-grounded prediction (2026-09-18)

A new public checkpoint records the human-readable puzzle / prediction line from **Two Keys** through **Flip Learning Loop** and **Route Readout v1–v3**.

The line demonstrated experience-grounded prediction, known-pair native-route recall, restricted partial-structure reuse, and ambiguity-guided experience acquisition. The latest attribution controls also showed that ordinary experience storage with the same higher-level learning rule matched the route-based implementation. A distinctive native SphereBrain route advantage is therefore **not yet established**.

Feature accumulation is intentionally paused while the next single research question and stop/continue criterion are reconsidered.

- [Checkpoint overview](research_releases/2026-09-18_experience_prediction_checkpoint/README.md)
- [Japanese overview](research_releases/2026-09-18_experience_prediction_checkpoint/README_JA.md)
