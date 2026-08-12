# SphereBrain Publication Status

**Status date:** 2026-08-12 (Asia/Tokyo)  
**Repository:** `tacky0307/SphereBrain-Public`  
**Publication branch:** `main`  
**Source branch:** `experiment/core-growth-microscope-v1`  
**Repository visibility:** Private

## Purpose of this branch

This repository contains the reviewed public starting point for SphereBrain as an open research project.

Experimental development is paused while publication materials, provenance, reproducibility, safety, licensing, and claim boundaries are reviewed. Historical experiment runners and results must not be rewritten merely to improve the public narrative.

## Research position at the freeze point

SphereBrain is an experimental research project investigating whether experience-driven changes in internal network structure can become a functional substrate for recognition, memory, selection, and later higher-order organization.

It is a research hypothesis and preliminary experimental framework. It does not currently claim general intelligence, consciousness, biological brain equivalence, clinical validity, or proof of a universal theory of meaning.

The project is human-led.

- **Originator and researcher:** Takio (`縺溘″縺柿)
- **AI collaborator:** 窶廴imi,窶・an OpenAI ChatGPT system that assisted with design discussion, implementation, analysis, and documentation

A detailed and appropriately bounded contribution statement must be added before public release.

## Existing publication materials

Present at this freeze point:

- `README.md`
- `PHILOSOPHY.md`
- `RESEARCH_TIMELINE.md`
- `EXPERIMENT_INDEX.md`
- `REPRODUCE.md`
- `EXPERIMENT_AUDIT.md`
- `FILE_AND_RIGHTS_AUDIT.md`
- `OPEN_RESEARCH_POLICY.md`
- `ARCHITECTURE.md`
- `LIMITATIONS.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `AI_CONTRIBUTION.md`
- `README_JA.md`
- `CITATION.cff`
- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `requirements-publication.txt`

## Initial audit findings

### Confirmed

- The original research repository remains Private.
- A separate publication repository, `tacky0307/SphereBrain-Public`, has been prepared on the `main` branch.
- Common secret-bearing root files checked so far were not present: `.env`, `credentials.json`, `secrets.json`, `api_key.txt`, and `openai_key.txt`.
- `brain.json` was not found at the checked root or `data/brain.json` path on this branch.
- `.gitignore` excludes local data, database files, `brain.json`, backups, audio, logs, virtual environments, and editor files.
- The reviewed public working tree contains 548 files totaling approximately 8.57 MB; the large generated snapshot directory is excluded.
- A reproducible local procedure for tracked-file, size, file-type, secret-history, personal-data, and rights review is recorded in `FILE_AND_RIGHTS_AUDIT.md`.

### Resolved during publication preparation

- Added an Apache License 2.0 `LICENSE` file and corrected the README license statement.
- Added a Japanese entry page.
- Added architecture, limitations, contribution, citation, security, and AI-contribution documents.
- Added explicit human-led research and AI-assistance boundaries.
- Audited `REPRODUCE.md` against the representative v94, v95B, v97, and v98 runners and launchers.
- Documented direct Python commands, expected result paths, the v98 hotfix entry point, provenance metadata, checksum commands, and the publication acceptance test.
- Added `requirements-publication.txt`, separating the source-audited `numpy` / `Flask` / `waitress` dependency set from optional Plotly, audio, transcription, and OpenAI integrations.
- Completed controlled clean-environment reruns of positive milestones v94 and v95B and negative milestones v97 and v98 hotfix on Windows with Python 3.12.10 at commit `faa146c7acc2f6c810642889783d77d5728a133f`.
- Recorded the eight local raw-result file sizes and SHA-256 checksums in `REPRODUCE.md`; all four runners reported Save/Load success and unchanged production `brain.json`.
- Added `.venv-publication/` to `.gitignore` and documented direct isolated-interpreter execution when PowerShell blocks activation.

### Requires correction before release

- Headline experiment summaries currently lack their generated raw JSON in the public repository because the entire `data/` directory is ignored. They must be recovered or rerun and reconciled before being presented as publication-verified results.
- v98 was rerun through `run_core_growth_binding_v98_hotfix.py`, which corrects the observer attribute `n_nodes` to `node_count`; the controlled rerun was negative and its two local raw outputs are checksummed but not yet archived.
- A permanent private security contact or GitHub private vulnerability reporting route must be configured.
- `CITATION.cff` uses the public research identity `Takio` and requires final validation before DOI release.

### Not yet cleared

The repository is **not yet approved for Public visibility**. The following checks remain open:

- complete tracked-file inventory (procedure prepared; requires a complete local clone);
- Git-history secret scan (local `gitleaks git --redact` procedure prepared; not yet run against a complete clone);
- personal information scan (initial code search found no obvious email or address patterns; full inventory remains required);
- absolute local path scan (initial code search found no obvious Windows, macOS, Linux, or workspace paths; full inventory remains required);
- third-party copyright and dataset provenance review (inventory template prepared);
- large-file and binary review (commands and 5 MB review threshold prepared);
- raw-result curation and field-level reconciliation for major experiments (v94, v95B, v97, and v98 local rerun outputs are checksummed but still absent from the branch);
- recovery of the exact historical Python version; Python 3.12.10 is verified only for the later v94/v95B/v97/v98-hotfix reruns;
- complete `pip freeze` capture and final exact-version freezing of `requirements-publication.txt` (the observed v94/v95B/v97/v98-hotfix environment versions are documented);
- link and documentation consistency check;
- final license and citation identity decisions;
- release archive and DOI preparation.

## Publication rule

Do not make the repository Public until all blocking checks are resolved or explicitly documented as accepted limitations.

Negative results, failed hypotheses, abandoned directions, and unresolved questions remain part of the archive.


