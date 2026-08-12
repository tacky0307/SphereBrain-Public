# SphereBrain Architecture

## Scope

SphereBrain is an experimental framework for studying whether experience-driven changes in an internal network can become a functional substrate for recognition, memory, selection, and later higher-order organization.

This document describes the research architecture. It does not claim biological fidelity or a completed intelligence system.

## Working system boundary

```text
External experience
        ↓
Encoder / sensory interface
        ↓
Numerical stimulation
        ↓
SphereBrain Core
        ↓
Activity trace / structural measurements
        ↓
Decoder / expression interface
```

The Encoder and Decoder may use language-aware or modality-specific systems. The primary research object is the Core and the way its activity and structure change through experience.

## Encoder

The Encoder converts external inputs into numerical stimulation that the Core can receive.

Depending on the experiment, the Encoder may be deliberately simple, semantically structured, embedding-based, or supplied by another model. Encoder behavior must be reported because organization already present in an Encoder can otherwise be mistaken for learning inside the Core.

The Encoder is not evidence that the Core itself contains words or human semantic labels.

## Core

The Core is a dynamic network containing nodes, edges, activity, weights, and persistence-related state. Versioned experiments vary how routes are selected, reinforced, consolidated, weakened, damaged, restored, or observed.

The Core is intended to learn from activity rather than store explicit facts such as `dog = animal`. Human-readable meaning is treated as an interpretation of measured internal organization, not as a primitive assumed to exist inside the Core.

Important measured quantities include:

- activated nodes and traversed edges;
- activation distributions;
- path reuse and divergence;
- convergence on shared regions;
- persistence across later experience;
- distributed structural support;
- response to ablation, rescue, and Save/Load.

## Trace and observation

A Trace records activity that occurred in the Core. It is an observation record, not automatically a memory store or a source of intelligence.

Microscope experiments should keep observation separate from intervention. If a runner changes Core weights, learning state, Assist settings, or persistence behavior, that intervention must be explicit.

## Reflection, Assist, and consolidation

Historical experiments include several mechanisms that use prior activity or detected structure to influence later learning or selection. These mechanisms are research interventions, not assumed components of biological cognition.

Their effects must be compared against suitable controls. A gain in one controlled condition does not establish a general benefit.

## Decoder

A Decoder translates Core output or measurements into an external form, potentially including language. A capable Decoder can add knowledge and fluency that did not originate in the Core, so experiments must separate:

1. information already present in the input representation;
2. changes attributable to Core experience;
3. interpretation or generation added by the Decoder.

## LLM → Core → LLM direction

One experimental direction uses:

```text
Language
   ↓
LLM or embedding-based Encoder
   ↓
SphereBrain Core
   ↓
LLM Decoder
   ↓
Language output
```

In this arrangement, the LLMs act as sensory and expressive interfaces. The hypothesis under test is whether the Core contributes experience-shaped structure beyond what is already present in the input embedding or language model.

This contribution has not yet been established. Required controls include Core-free baselines, untrained and randomized Core conditions, repeated seeds, and unseen probes.

## Experiment families

The repository contains several historically distinct lines:

- early spherical path-memory prototypes;
- Semantic Encoder experiments;
- structural puzzle and Native Learning experiments;
- semantic bridge formation and stabilization microscopes;
- consensus persistence, ablation, and rescue experiments;
- label-blind structural observation;
- LLM → Core → LLM prototypes.

These lines should not be treated as one continuously validated system. Each versioned runner defines a narrower experimental contract.

## What the architecture currently supports

The codebase supports controlled investigation of activity, reinforcement, persistence, structural change, Save/Load behavior, ablation, rescue, and related measurements.

It does not currently establish:

- human-like understanding;
- general intelligence;
- consciousness;
- biological equivalence;
- clinical validity;
- an advantage over embeddings, LLMs, GNNs, or other baselines.

See [LIMITATIONS.md](LIMITATIONS.md) for the current evidence boundaries and [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) for version-specific results.
