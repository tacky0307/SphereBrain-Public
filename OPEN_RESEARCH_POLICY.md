# SphereBrain Open Research Policy

## Purpose

The purpose of this policy is to preserve the existence, chronology, and reproducibility of the SphereBrain research project.

The project should remain inspectable even if later interpretations change, individual experiments fail, or development stops.

## Preservation principles

### 1. Do not erase negative results

Failed hypotheses, NO signals, regressions, and contradictory experiments remain part of the research record.

A later successful experiment does not retroactively convert an earlier failure into a success.

### 2. Separate hypothesis, result, and interpretation

Each important experiment should distinguish:

```text
Hypothesis
Method
Observed result
Interpretation
Limitations
Next question
```

The interpretation may change later. The observed result should remain preserved.

### 3. Keep experimental versions immutable in spirit

Historical versioned runners such as `v94`, `v95B`, or `v98` should not be silently rewritten to produce a newer interpretation.

If a correction is necessary, preserve the reason in Git history and preferably create a clearly named follow-up version.

### 4. Preserve provenance

Important archived results should include or reference:

- repository name,
- branch,
- exact Git commit SHA,
- date,
- runner version,
- seeds,
- raw result JSON,
- relevant environment information,
- interpretation written after the result.

### 5. Distinguish observation from intervention

A microscope experiment that claims to observe natural Core behavior should not silently alter Core weights, learning state, Assist settings, or persistence behavior.

If an intervention is used, it must be explicit in the experiment contract.

### 6. Keep claims proportional to evidence

Use wording such as:

- observed in this setup,
- reproduced across these seeds / domains,
- candidate mechanism,
- not yet established,
- failed to reproduce robustly.

Avoid presenting a local PASS as proof of general intelligence, consciousness, universal semantics, or biological equivalence.

## Open publication strategy

The repository is the living development record. Important milestones should additionally be fixed outside the working Git repository.

Recommended archival layers:

```text
Git commits
    ↓
GitHub milestone release
    ↓
independent DOI / immutable archive
    ↓
optional research registration / manuscript
```

The intent is redundancy: no single account, machine, or service should be the only record that the research existed.

## Suggested milestone archive contents

Each major public archive should include:

- source snapshot,
- research timeline,
- experiment index,
- philosophy / hypothesis document,
- reproduction guide,
- raw results for the milestone,
- checksums where practical,
- release date,
- commit SHA.

## Authorship and collaboration

The research record should document actual contributions without requiring claims of prestige, ownership, or commercial value.

If collaborators contribute code, analysis, experimental design, datasets, or documentation, their contributions should be preserved through Git history, release notes, acknowledgements, or another traceable mechanism where appropriate.

## Commercialization and intellectual property

Open publication and patent strategy can conflict. This repository policy is intended to preserve research history, not to provide legal advice about patent rights.

Before publishing material that may be intended for patent protection, obtain appropriate legal advice if preserving those rights matters.

If the project deliberately chooses open publication over exclusive rights, that decision should be explicit rather than accidental.

## Research continuity

The project should be understandable without relying on any one person's memory.

A future researcher should be able to determine:

1. what SphereBrain was trying to test;
2. how the Core was implemented at a given milestone;
3. which experiments succeeded or failed;
4. which conclusions were justified at the time;
5. what remained unresolved;
6. how to rerun the experiment.

## Current open-research commitment

As of Open Research Archive v1, the intended policy is:

> **Preserve the full path of inquiry — including wrong turns — so that the existence and development of SphereBrain can be independently inspected and reproduced.**
