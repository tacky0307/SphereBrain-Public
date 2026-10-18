# Public package scope and publication audit — 2026-10-18

**Scope**: newly added/edited files in the SphereBrain-Public 2026-10-18 experience-shaped-medium checkpoint only. This is not an audit of every historical commit in the entire public repository.

## Selection principles

- Original private development repository NOT mirrored wholesale.
- Selected **physical experiment source** and its preregistered numerical PROTOCOL are copied verbatim, preserving algorithm and gate conditions; model file names: implementation.py, aggregate.py, test_contract.py.
- New documentation is tailored for a public research audience. It does not publish the original full internal working compass or private Issue history, and it does not claim all archived GitHub Actions from the source repository are publicly accessible.
- No unneeded personal files, tokens, environment files, .env, private workflow configuration, customer data, user data, or private scientific experiments are included in the new public package.
- The public software is under the existing repository Apache-2.0 license.

## Public reproducibility

- Formal numerical metrics and scientific limits: [RESULTS_JA.md](../../research/experiments/experience_shaped_medium_v0/RESULTS_JA.md).
- Public third-party reproduction and independent simulation: [PUBLIC_REPRODUCE.md](../../research/experiments/experience_shaped_medium_v0/PUBLIC_REPRODUCE.md).
- [Public CI run 39108570080](https://github.com/tacky0307/SphereBrain-Public/actions/runs/39108570080) passed after re-running both diagnostic and the previously-known formal HOLD6 seeds with a fully separate physical-replay audit. All eight physical gates P0–P7 passed, C0 PASS, and reported M1=0.3117761144512821 agreed with original formal M1=0.311776114451282 within floating-point tolerance.
- This run is a **reproduction of known scientific data, not a fresh formal unseen cohort**. Public results can be recreated on a local checkout.
- The raw simulation files from this public CI are available as its GitHub Actions artifact while the 90-day retention period lasts. After expiration they can be regenerated deterministically with the published Python code.

## Limited safety review

Before publishing, the 16 changed files were scanned through GitHub's text-file interfaces for common token/private-key prefixes and explicit references to the original private repository URL. No matches were found in the changed textual files.

**Important limit**: This lightweight source-string review is not a comprehensive historical secret scan, independent legal audit, or guarantee about every third-party component and prior public commit. The entire existing repository's history was not copied or rechecked for this checkpoint. The general [public research policy](../../OPEN_RESEARCH_POLICY.md) remains in force.

## Scientific attribution

**Proven:** history-dependent artificial medium conductance can change future physical activity in one fixed 6×6 geometry under one preregistered family of synthetic pulses. No semantic memory, arbitrary free wiring, autonomous solving or general intelligence is established.

**Next untested hypothesis:** recovery of an earlier *functional transmission relation* after intervening experience (A→B→A), followed eventually by a truly independent task-solution study where LLM is only the language interface.
