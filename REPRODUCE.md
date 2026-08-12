# Reproducing SphereBrain Experiments

This document describes the current reproduction procedure for the publication candidate. SphereBrain is still a preliminary research framework. A successful launch does not by itself reproduce a historical result; the generated JSON must also be checked against the archived evidence.

## Current reproduction status

The representative v94, v95B, v97, and v98 runners and Windows launchers are present. Their historical raw JSON outputs are not yet included because the repository-wide `data/` directory is ignored.

Accordingly:

- the source procedure can be inspected and rerun;
- historical numerical summaries are not yet raw-data-reconciled;
- v98 has a controlled negative rerun through the documented hotfix, but its raw payload is not yet archived;
- clean-environment execution has succeeded on Windows for the positive v94 and v95B milestones and the negative v97 and v98 milestones.

See [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md) for the evidence classification.

## Platform and Python

The experiments were developed primarily on Windows and Python. The exact historical Python patch version was not recorded and must be recovered or replaced by a clean-environment compatibility test before Open Research Archive v1.

Controlled reruns on 2026-08-12 verified v94, v95B, v97, and v98 hotfix with Windows and Python 3.12.10. This verifies the current publication-branch procedure for those four runners; it does not establish the exact historical Python version.

The runners themselves are ordinary Python programs and may also work on macOS or Linux, but cross-platform execution has not yet been verified.

## Clone and isolate the environment

Use a fresh clone and the publication branch:

```powershell
git clone https://github.com/tacky0307/SphereBrain-Public.git
cd SphereBrain-Public
git rev-parse HEAD
git status
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt
```

If PowerShell execution policy blocks `Activate.ps1`, do not change system policy merely for reproduction. Invoke the isolated interpreter directly:

```powershell
.\.venv-publication\Scripts\python.exe -m pip install --upgrade pip
.\.venv-publication\Scripts\python.exe -m pip install -r requirements-publication.txt
.\.venv-publication\Scripts\python.exe experiments\run_core_growth_binding_v94.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-publication.txt
```

`requirements-publication.txt` is the source-audited minimal dependency set for the representative Core Growth Binding runners: `numpy`, `Flask`, and `waitress`. The repository-wide `requirements.txt` additionally includes Plotly, audio, transcription, and OpenAI integrations that are not required for these four publication experiments. The minimal environment was execution-validated for v94, v95B, v97, and v98 hotfix on Windows with Python 3.12.10, numpy 2.5.2, Flask 3.1.3, and waitress 3.0.2. A complete `pip freeze` artifact and cross-platform validation remain pending.

## Run representative experiments

Run from the repository root with the virtual environment active.

### v94 窶・Consensus Persistence Robustness

```powershell
python experiments/run_core_growth_binding_v94.py
```

Expected result path:

```text
data/core_growth_binding_v94/results/latest_binding_v94.json
```

### v95B 窶・Ablation Specificity Control

```powershell
python experiments/run_core_growth_binding_v95b.py
```

Expected result path:

```text
data/core_growth_binding_v95b/results/latest_binding_v95b.json
```

### v97 窶・Independent Support Path Formation

```powershell
python experiments/run_core_growth_binding_v97.py
```

Expected result path:

```text
data/core_growth_binding_v97/results/latest_binding_v97.json
```

This negative result must be retained even if a rerun remains negative or produces an unexpected result.

### v98 窶・Emergent Structural Meaning Microscope

Use the hotfix runner:

```powershell
python experiments/run_core_growth_binding_v98_hotfix.py
```

The root launcher `run_core_growth_binding_v98.bat` invokes this hotfix. It replaces observer access to `n_nodes` with `node_count` while leaving the stated training data, seeds, thresholds, clustering, and verdict logic unchanged.

Expected result path:

```text
data/core_growth_binding_v98/results/latest_binding_v98.json
```

Preserve both the original runner and the hotfix. Record that the result was produced through the hotfix. The controlled rerun below completed and produced a negative result; its raw JSON remains local and checksummed.

## Verified clean reruns (2026-08-12)

The following reruns were completed on Windows from commit `faa146c7acc2f6c810642889783d77d5728a133f` using the isolated `.venv-publication` interpreter and Python 3.12.10. The repository documentation was subsequently updated; the producing commit recorded here remains the experiment provenance. They are later controlled reruns, not recovered historical outputs.

### v94 positive milestone

- Runner: `experiments/run_core_growth_binding_v94.py`
- Trials: 320
- Verdict: PASS
- Domain direction: 8/8
- Seed direction: 16/22
- Save/Load: YES
- Production `brain.json`: unchanged
- `latest_binding_v94.json`: 1,509,675 bytes; SHA-256 `54DCF5D4B9E0D6A7032B22C56F9B2AB7F44FC81573D20F687B20F48132586843`
- `consensus_persistence_robustness_roundtrip.json`: 4,093,003 bytes; SHA-256 `1BBDF40CA4380F314EAED3FDCD1C126F73BB14D848C60A7539770992AA0E5CF7`

### v95B positive specificity milestone

- Runner: `experiments/run_core_growth_binding_v95b.py`
- Stable cohort: 29
- Consensus effect: 69.0%
- Difference vs matched random: 24.1%
- Difference vs hub matched: 27.6%
- Random/Hub match coverage: 100.0%
- Domain specificity: 6/8
- Specificity signal: YES
- Core readiness: `consensus_ablation_specificity_candidate`
- Save/Load: YES
- Production `brain.json`: unchanged
- `latest_binding_v95b.json`: 7,685,407 bytes; SHA-256 `6481A357E8D649C359FAF4F7AFD7401E14B5B998D3BD4F02ED973FF657589FC7`
- `consensus_specificity_roundtrip.json`: 4,092,682 bytes; SHA-256 `2785D6FA3E08CC0BF51ED1D846EBE9D8F6B2AD5C11671FB589E272D0C7CF7776`

### v97 negative milestone

- Runner: `experiments/run_core_growth_binding_v97.py`
- Unstable cohort: 118
- Rescue eligible: 104
- Independent support: 14.4%
- Random rescue: 16.3%
- Domain direction: 0/8
- Formation signal: NO
- Core readiness: `independent_support_path_formation_not_established`
- Save/Load: YES
- Production `brain.json`: unchanged
- `latest_binding_v97.json`: 31,401,813 bytes; SHA-256 `B5B333636650D1A5A1698F232D3D6801E7F466A83F78898CC6C522AED98DC3AB`
- `independent_support_path_roundtrip.json`: 4,094,157 bytes; SHA-256 `4CA5BD47428E8C2557F35007C526068704E76CB643BDFC97B57A21BF80AF8FC5`

### v98 negative emergent-organization milestone

- Runner: `experiments/run_core_growth_binding_v98_hotfix.py`
- Seeds: 5
- Objective gain: -0.0526 (positive 1/5)
- Separation gain: -0.0822 (positive 0/5)
- Reuse Edge gain: +41.2 (positive 5/5)
- Label-blind signal: NO
- Post-hoc alignment: NO
- Purity gain: +0.013
- Core readiness: `label_blind_emergent_structural_organization_not_yet_observed`
- Observer processing: PASS
- Save/Load: YES
- Production `brain.json`: unchanged
- `latest_binding_v98.json`: 383,761 bytes; SHA-256 `0121F60A4B11D6D4C98C38168CD466A40233789D463C882C26EF6519BF07766B`
- `emergent_structure_roundtrip.json`: 4,095,764 bytes; SHA-256 `81E05E093328106BA647EBC606A83B69F558FFF48A7A6902A63687DC6CF36B01`

These raw files remain on the local research machine under the ignored `data/` tree. The checksums establish identity for later curation, but the publication branch does not yet contain the raw payloads. Accordingly, these reruns satisfy the execution portion of the acceptance test but not raw-result archival.

## Browser workflow

Each runner starts a local server on `127.0.0.1`, chooses a version-specific available port, and opens a browser page. The experiment normally starts only after the page's run button is pressed.

The Flask endpoint returns after the complete computation. A page displaying `螳溯｡御ｸｭ窶ｦ` may therefore still be working. Check the terminal and operating-system process activity before stopping it.

Do not expose the local server to a public interface.

## Result and provenance record

For every archival rerun, preserve:

- experiment version and title;
- Git branch and exact commit SHA;
- operating system;
- Python version from `python --version`;
- installed packages from `python -m pip freeze`;
- exact command or launcher;
- seed and domain definitions stored by the runner;
- start and completion time;
- generated raw JSON;
- SHA-256 checksum of the JSON;
- whether `data/brain.json` existed before the run;
- reported Save/Load and production-brain checks;
- terminal errors or warnings;
- whether the output is original historical data or a later rerun.

Example checksum on Windows PowerShell:

```powershell
Get-FileHash .\data\core_growth_binding_v94\results\latest_binding_v94.json -Algorithm SHA256
```

Example on macOS or Linux:

```bash
sha256sum data/core_growth_binding_v94/results/latest_binding_v94.json
```

Because `data/` is ignored, generated results will not appear in Git automatically. Copy only reviewed, redistributable archival outputs into the future curated release-results area; do not force-add the whole local data directory.

## Protecting production brain state

The representative runners include checks intended to hash `data/brain.json` before and after execution. Many also save and reload a temporary learned brain inside their result directory.

These safeguards must still be verified from the generated payload. A Save/Load PASS covers only the measurements implemented by that runner and does not prove complete behavioral identity.

Before running any unfamiliar experiment:

1. inspect its `OUT` and `BRAIN_PATH` constants;
2. inspect every call to `save`, `load`, and file-writing methods;
3. back up any local research state outside the repository;
4. confirm that modifying production state is not part of the experiment contract.

## Historical reproduction rules

To reproduce a historical result accurately:

1. use the exact historical commit when it can be identified;
2. use the seed set and domains stored in the runner;
3. do not substitute a later `brain.py` unless performing a declared forward-compatibility test;
4. preserve checkpoints, thresholds, Assist, Native Learning, consolidation, and semantic-encoding settings;
5. record all environment differences;
6. retain failed, negative, and unexpected runs;
7. label a rerun as a rerun rather than replacing the historical record.

## Publication acceptance test

Open Research Archive v1 should not be called cleanly reproducible until all of the following are completed:

- a fresh clone installs without relying on unrecorded local files;
- at least one positive milestone and one negative milestone run to completion;
- their raw JSON files are archived with checksums and commit metadata;
- recorded headline values are reconciled against raw output;
- v98 is retained as a negative hotfix rerun and its raw JSON is archived before any promotion to Level A;
- the exact tested Python and dependency versions are recorded;
- Windows instructions are verified, and any cross-platform claim is limited to platforms actually tested.

The goal is to let another person reconstruct the experimental procedure and evaluate the evidence, not merely open the user interface.



