# SphereBrain Limitations and Claim Boundaries

## Current status

SphereBrain is a research hypothesis and preliminary experimental framework. Its results are evidence about specific implementations and controlled conditions, not proof of a general theory of intelligence.

## Central unresolved question

The project has observed experience-dependent structural effects, but it has not yet established that the Core contributes useful organization beyond information already supplied by an Encoder, embedding model, experiment design, or evaluation rule.

A central future test is therefore:

> Does experience inside the Core produce a reproducible change in selection or output that cannot be explained by the input representation alone?

## Known limitations

### Encoder confounding

Semantic structure may already be present in staged encodings or pretrained embeddings. Similar Core activity can therefore reflect similar input representations rather than learning produced by the Core.

Necessary comparisons include:

- input-space similarity;
- Core-free baseline;
- Core present but untrained;
- Core trained through experience;
- randomized or structurally disrupted Core.

### Human-label leakage

Several historical experiments used human semantic categories to design controlled stimuli or evaluate results. These experiments are useful, but they do not show that the Core independently discovered those categories.

The label-blind v98 direction was created to move structure discovery before human interpretation. Its final result is not yet recorded in this archive.

### Reproducibility and statistical scope

Some promising effects appeared in small controlled runs and weakened under more seeds or domains. The repository preserves those failures.

Not every historical experiment has yet been independently reproduced from a clean environment. Some dependency versions, raw results, exact commits, or runtime conditions still require reconciliation.

### Experimental intervention

Assist, consolidation, ablation, rescue, thresholds, and experiment-specific weakening rules can influence the result. An intervention that improves an experiment may be exploiting that experiment's measurement contract rather than revealing a general learning mechanism.

### Structural interpretation

Path overlap, persistent edges, clusters, consensus, or activation similarity are measurements. They are not automatically concepts, meanings, memories, or understanding.

Those words may be used as hypotheses or human-side interpretations only when the measured relationship and alternative explanations are stated.

### Biological interpretation

SphereBrain is inspired by broad ideas such as activity, connectivity, persistence, plasticity, damage, and recovery. It is not a simulation of biological neurons, brain anatomy, synaptic chemistry, development, metabolism, or clinical disease.

Shared vocabulary does not imply a shared mechanism.

### LLM contribution

In LLM → Core → LLM experiments, both interfaces can contribute semantic knowledge, generalization, and fluent output. Decoder quality must not be attributed to the Core without controlled separation.

The comparison between embedding similarity and Core path effects remains incomplete.

### Scale and efficiency

Most results come from small or medium controlled experiments. Computational cost, stability, memory use, and behavior at much larger scales are not yet characterized.

### Evaluation design

Many runners use predefined PASS/NO thresholds. A PASS confirms only that the stated condition was met. Threshold choice, multiple comparisons, post-hoc interpretation, and residual structural confounding can affect conclusions.

### Data and domain coverage

The project has relied heavily on synthetic structural tasks and short language-derived stimuli. It has not demonstrated broad multimodal learning, real-world robustness, safety, fairness, or reliable deployment behavior.

## Claims the project does not make

SphereBrain does not currently claim:

- artificial general intelligence;
- human-like understanding or thought;
- consciousness or subjective experience;
- a complete or accurate model of the brain;
- medical or diagnostic validity;
- a replacement for LLMs;
- superiority to transformers, GNNs, associative memory, reservoir computing, or other established methods;
- a proven universal representation of meaning.

## Publication standard

Public descriptions should distinguish:

```text
Hypothesis
Method
Observed result
Interpretation
Alternative explanation
Limitation
Next falsification test
```

Negative, null, contradictory, and invalidated results remain part of the archive.

## Minimum evidence sought next

Before making a stronger functional claim about the Core, the project should provide:

1. Encoder-only versus Core comparisons;
2. untrained, trained, and randomized Core ablations;
3. repeated runs across at least several seeds;
4. probes not used during training;
5. uncertainty or variation, not only a single score;
6. an observable effect on selection or output;
7. raw results and an exact reproducible commit.

A negative outcome is publishable and informative. The purpose of open research is to test the hypothesis, not to guarantee that it survives.
