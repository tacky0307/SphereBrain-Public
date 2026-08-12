# SphereBrain File, Size, and Rights Audit

**Audit date:** 2026-08-12 (Asia/Tokyo)  
**Scope:** public repository `tacky0307/SphereBrain-Public`, branch `main`

## Purpose

This audit separates what has been directly inspected from what still requires a complete local checkout. It covers tracked-file inventory, large files, personal information, secrets, local paths, and third-party material.

## Confirmed facts

- The GitHub repository remains Private.
- GitHub reports a repository size of approximately 15 MB.
- The publication branch contains the project's own source code, experiment runners, launchers, and research documentation.
- The root `.gitignore` excludes:
  - `data/`;
  - database files and `brain.json`;
  - backups;
  - common audio formats;
  - logs;
  - virtual environments and editor metadata.
- The checked publication documents do not claim that third-party datasets, images, audio, model weights, or recordings are included.
- The checked dependency list contains package names only; package use does not by itself establish that third-party assets are redistributed.
- No raw headline experiment JSON is currently archived on the publication branch.

## Important limitation

The connected GitHub inspection path can fetch known files but did not provide a complete recursive tree or a byte-level scan of every historical Git object in this Private repository.

Therefore, the following are **not yet cleared**:

- complete current tracked-file inventory;
- all historical blobs;
- binaries and files whose contents are not searchable as text;
- large-file thresholds;
- deleted secrets or personal data remaining in Git history;
- third-party asset provenance for every historical file.

This is a publication blocker, not evidence that a problem exists.

## Local audit procedure

Run the following from a complete local clone of `tacky0307/SphereBrain-Public` before changing repository visibility.

### 1. Record the exact state

```bash
git switch main
git status --short
git rev-parse HEAD
git ls-files > tracked-files.txt
```

The working tree should be clean. Preserve `tracked-files.txt` as an audit artifact, but review it before deciding whether to publish it.

### 2. Find large tracked files

```bash
git ls-files -z | xargs -0 -r du -b | sort -nr > tracked-file-sizes.txt
```

Review every file above 5 MB. Explicitly justify or remove files containing:

- model weights;
- recordings;
- databases;
- generated visualizations;
- archives;
- executables;
- raw personal material.

### 3. Inventory file types

```bash
git ls-files -z | xargs -0 -r file > tracked-file-types.txt
```

Manually review binaries and formats whose contents are not covered by text search.

### 4. Scan the current branch for sensitive text

Use a recognized scanner where available, then manually review its findings:

```bash
gitleaks detect --no-git --source . --redact
```

Also search for common local paths and identity fields without publishing matched secret values:

```bash
git grep -n -I -E '([A-Za-z]:\\\\Users\\\\|/Users/|/home/|/workspace/|api[_-]?key|secret|password|token|authorization|email|address)'
```

A match is a review lead, not automatically a leak.

### 5. Scan all Git history

```bash
gitleaks git --redact
```

If an actual credential is found:

1. revoke or rotate it first;
2. record only the affected path and remediation, never the secret value;
3. decide whether history rewriting is necessary before publication;
4. re-scan after remediation.

### 6. Review third-party rights and provenance

For every non-source asset or imported dataset, record:

| Path | Creator/source | License or permission | Modifications | Publication decision |
|---|---|---|---|---|
| _To be completed from tracked-file inventory_ |  |  |  |  |

Do not publish an asset merely because it is already in Git. If provenance is uncertain, exclude it from Archive v1 or replace it with a reproducible generation procedure.

## Current classification

### Cleared in principle

- original SphereBrain source code;
- original experiment runners and launchers;
- original research documentation;
- dependency declarations.

This classification remains subject to the complete inventory and history scan.

### Publish only after review

- generated charts and screenshots;
- archived JSON results;
- example databases;
- logs;
- notebooks containing embedded output;
- binary artifacts.

### Exclude unless provenance is documented

- recordings of people;
- externally sourced images or audio;
- third-party datasets;
- model weights;
- copied documents or articles;
- files containing personal information or credentials.

## Completion criteria

This audit may be marked complete only when:

1. `tracked-files.txt`, size inventory, and type inventory have been reviewed;
2. current-tree and history secret scans have completed;
3. all findings have a recorded disposition;
4. third-party materials have provenance and redistribution permission;
5. the reviewed commit SHA is recorded here;
6. a second check confirms that the publication branch still points to that reviewed content.

Until then, repository visibility must remain Private.

