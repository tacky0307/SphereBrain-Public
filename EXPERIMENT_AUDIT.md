# SphereBrain Experiment Publication Audit

**Audit date:** 2026-08-12 (Asia/Tokyo)  
**Scope:** public repository `tacky0307/SphereBrain-Public`, branch `main`

## Purpose

This audit separates executable experiment definitions, human-readable claims, and raw generated evidence. A result is not treated as fully publication-verified merely because its runner and summary text exist.

## Evidence levels

| Level | Meaning |
|---|---|
| A | Runner, launcher, raw result, and documentation are present and numerically reconciled |
| B | Runner and launcher are present; documentation exists; raw result is not archived in the repository |
| B-R | Level B plus a controlled clean-environment rerun whose local raw output is identified by checksum |
| C | Documentation exists but executable or raw evidence is incomplete |
| D | Incomplete, failed, or not yet run; retained as an open research record |

## Representative experiment check

| Experiment | Runner | Launcher | Raw JSON on publication branch | Current evidence level | Publication treatment |
|---|---:|---:|---:|---|---|
| v94 Consensus Persistence Robustness | Present | Present | Not present | B-R | Positive clean rerun completed; local raw outputs checksummed; archive still pending |
| v95B Ablation Specificity Control | Present | Present | Not present | B-R | Positive specificity clean rerun completed; local raw outputs checksummed; archive still pending |
| v97 Independent Support Path Formation | Present | Present | Not present | B-R | Negative clean rerun reproduced; local raw outputs checksummed; archive still pending |
| v98 Emergent Structural Meaning Microscope | Present | Present | Not present | B-R | Negative hotfix rerun completed; local raw outputs checksummed; archive still pending |

## Controlled rerun evidence (2026-08-12)

A clean Windows environment using Python 3.12.10 and the publication minimal dependencies completed two positive milestones and two negative milestones at commit `faa146c7acc2f6c810642889783d77d5728a133f`.

| Experiment | Observed outcome | Local raw files | SHA-256 status |
|---|---|---:|---|
| v94 | PASS; Domain 8/8; Seed 16/22; Save/Load YES; production brain unchanged | 2 | Recorded in `REPRODUCE.md` |
| v95B | Specificity signal YES; Domain specificity 6/8; Save/Load YES; production brain unchanged | 2 | Recorded in `REPRODUCE.md` |
| v97 | Formation signal NO; Domain 0/8; Save/Load YES; production brain unchanged | 2 | Recorded in `REPRODUCE.md` |
| v98 hotfix | Label-blind signal NO; Objective gain -0.0526; Separation gain -0.0822; Reuse Edge gain +41.2; Save/Load YES; production brain unchanged | 2 | Recorded in `REPRODUCE.md` |

The v97 negative finding is scientifically retained: independent support (14.4%) did not exceed random rescue (16.3%). Completing the runner successfully therefore reproduced the negative experimental conclusion rather than producing a software failure.

This raises v94, v95B, v97, and v98 to B-R, not A. Their payloads are still under the ignored local `data/` tree and have not yet been curated into the publication branch or a release artifact. The checksums permit later identity verification.

## Blocking finding: generated data is excluded

The root `.gitignore` excludes the entire `data/` directory. The checked runners write their generated JSON under version-specific paths such as:

```text
data/core_growth_binding_v94/results/latest_binding_v94.json
data/core_growth_binding_v95b/results/latest_binding_v95b.json
data/core_growth_binding_v97/results/latest_binding_v97.json
data/core_growth_binding_v98/results/latest_binding_v98.json
```

None of these checked result files is present on the publication branch.

Therefore, numerical values currently shown in `EXPERIMENT_INDEX.md` must be described as **recorded run summaries, not yet reconciled against archived raw output**.

Before Open Research Archive v1, one of the following must be completed for each headline result:

1. recover the original generated JSON and archive a reviewed, redistributable copy; or
2. rerun the exact historical runner at a recorded commit and archive the new raw JSON with environment, seed, commit, and checksum metadata.

The original generated result should be preferred when available. A rerun must be labeled as a rerun rather than silently replacing the historical record.

## v98 runner status

The root launcher `run_core_growth_binding_v98.bat` invokes:

```text
experiments/run_core_growth_binding_v98_hotfix.py
```

The hotfix patches two observer helpers from the original v98 module, replacing use of `n_nodes` with `node_count`. Its stated scope leaves training data, seeds, thresholds, clustering, and verdict logic unchanged.

For publication:

- preserve both the original v98 runner and the hotfix;
- document that the observed run must use the hotfix launcher;
- describe the controlled v98 hotfix rerun as a later negative rerun, not as recovered historical evidence;
- retain the exact commit, local file identities, and checksums until the raw files are curated.

## Code-level observations

The checked v94, v95B, and v97 runners include contracts intended to verify:

- production `brain.json` remains unchanged;
- Save/Load behavior for selected measurements;
- Native Learning state is present;
- seed/domain/mode definitions are stored in the output payload.

These are useful safeguards, but the safeguards themselves cannot be confirmed for a historical run without its raw payload or a controlled rerun.

## Publication classification

### Publicly safe now

- experiment source code and launcher scripts;
- architecture and methodology descriptions;
- explicit negative and inconclusive interpretations;
- the fact that v98 required a narrowly scoped observer hotfix.

### Public with qualification

- headline numerical summaries in `EXPERIMENT_INDEX.md`, labeled as recorded results awaiting raw-data reconciliation.

### Blocked from definitive presentation

- any claim that a headline result has been independently reproduced from the current public package;
- any claim that the v98 result is historical or independently archived evidence;
- any statement that Open Research Archive v1 is fully reproducible while the raw result package is absent.

## Required next actions

1. Recover original raw JSON outputs from the local research machine, if available.
2. Preserve them outside the ignored production-state area or add a deliberately curated archival results directory.
3. Record SHA-256 checksums and the producing commit.
4. Reconcile every headline number in `EXPERIMENT_INDEX.md`.
5. ~~Rerun at least one positive milestone and one negative milestone from a clean environment.~~ Completed for v94, v95B, v97, and v98 hotfix on 2026-08-12.
6. Curate all eight checksummed rerun payloads and reconcile their headline fields before promoting B-R to A.

This audit is intentionally conservative. Missing raw evidence does not erase the research history, but it changes how strongly the results may be presented.

