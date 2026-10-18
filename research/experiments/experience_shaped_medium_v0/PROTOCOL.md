# SphereBrain Issue #332 — Experience-Shaped Medium v0
## Numerical preregistration — 2026-10-18, BEFORE science-code implementation

This PROTOCOL must be committed separately before the experiment's simulation code. It is not a measured result. Related documents: RESEARCH_COMPASS_JA.md §29, research/foundations/EXPERIENCE_SHAPED_MEDIUM_COMPASS_2026_10_18_JA.md and its verification plan.

### Scientific question

Can experience modify a medium, with no explicit long-lived wire W, such that future identical input produces causally different physical activity? What are the effects of history order, erasing all medium traces, shuffling their positions, and retaining only the same global amount of conductance? This is **physical substrate research only**, not proof of intelligence, unbounded wiring freedom, problem solving, semantic memory, or LLM superiority. Do not add relation banks, supervised labels or reward to the Core.

### Medium and exact deterministic local physics

A **fixed 6×6 grid**, row-major cells i=0..35, capacity C_i=1; undirected nearest-neighbor edges: all horizontal faces row-major, followed by all vertical faces row-major (60 edges). No explicit W connections and no site creation (W=0). This is a finite proxy for a conductive medium, **not yet a freely morphing sphere**.

Fast state q_i starts at zero. Each undirected medium face e=(i,j) has sigma_e initially sigma0=.08. The following is one synchronous microtick:

1. Apply the external pulse amplitude s>=0 at the selected one cell to q before flows, otherwise s=0. Let x=q_before+s*onehot(cell).
2. For all edges evaluate f_e=sigma_e*(x_i-x_j) at this same pre-update state; make y_i=x_i-dt*sum_outward(f)_i, using paired antisymmetric transfers, dt=.20. Apply leakage q_after=(1-leak)*y with leak=.025.
3. If plasticity enabled, update all sigma_e synchronously from the pre-update f_e: sigma_after=clip(sigma_e+eta*abs(f_e)*(sigma_max-sigma_e)/sigma_max - decay*(sigma_e-sigma0), sigma0, sigma_max), with sigma_max=.34, eta=.045, decay=.0005 per tick. Otherwise sigma unchanged. No answer label, correct path, task-specific sign, stimulus name, or reward influences this update.
4. Independent ledger for every tick: |sum(q_after)-(1-leak)*(sum(q_before)+s)| <=1e-10; all q finite and >=-1e-10, and sigma in [.08,.34], W=0. External stimulation and leakage mean total q is not unconditionally conserved. Sigma growth has no conservation-of-chemical-energy model, and we must NOT claim full material/energy conservation.

Numpy float64 and numpy's deterministic PCG64 RNG. Equal cell geometry/boundary conditions for all methods. Seeds change the external stimulus amplitude streams and independent shuffle, not the Core law.

### Experience and input matching

Each seed independently draws from rng=np.random.default_rng(seed*101+17): a_amp=rng.uniform(.85,1.15,size=64), b_amp=rng.uniform(.85,1.15,size=64). This creates **independent physical input histories** while ensuring paired arms use exactly the SAME 64 amplitudes for each named A/B stimulus. A is injected in cell12 (row2 col0); B is cell17 (row2 col5). Each input event is exactly 12 microticks, injection only at its first tick. After 128 events run 48 additional natural no-stimulation REST ticks with same medium update setting as training. The phases/arms:

- PASSIVE_AB: A64 then B64, plasticity OFF, frozen original medium.
- PLASTIC_AB: A64 then B64, plasticity ON, W absent.
- PLASTIC_BA: B64 then A64, plasticity ON, identical A and B amplitude **multisets** as AB; same total dose but different time ordering.
- PLASTIC_AA: A64 with a_amp then A64 with b_amp, plasticity ON, diagnostic only, not a dose-by-port matched control.
- PLASTIC_NONE: 128×12 microticks without any input, then 48 REST, plasticity ON.
- After PLASTIC_AB has completed, probe-only matched interventions to its sigma array: ERASE = all sigma0, SHUFFLE = permutation of the sigma_e values using rng=np.random.default_rng(seed+12345).permutation(60); UNIFORM = all sigma mean(PLASTIC_AB), preserving exact total sigma. These are NOT independent experience episodes; they are interventions on the same formed medium.

All fast q set exactly to ZERO after training/rest and **before each probe**. Probes use frozen sigma (plasticity OFF), 24 ticks, single injection at probe tick0, amplitude1, one probe at a time from the same cloned sigma/q. Probe cells: A cell12, B cell17, NOVEL cell2 (row0 col2 never stimulated during experience). Store ALL q states for 24×36 and training final sigma (60 values) for each arm. No decoder, hidden target, correct route, or reward.

### Predefined metrics

Let Y_X^P be the vector of 24×36 q states under frozen probe P from X's sigma, and D_P=max(||Y_PASSIVE_AB^P||_2,1e-12). All metrics are dimensionless, per independent seed.

M0 = ||sigma_PLASTIC_AB - sigma0||_2 / ||sigma0||_2.
M1 = ||Y_PLASTIC_AB^NOVEL - Y_PASSIVE_AB^NOVEL||_2/D_NOVEL.
M2 = ||Y_PLASTIC_AB^NOVEL - Y_ERASE^NOVEL||_2/D_NOVEL.
M3 = ||Y_PLASTIC_AB^NOVEL - Y_PLASTIC_BA^NOVEL||_2/D_NOVEL.
M4 = ||Y_PLASTIC_AB^NOVEL - Y_SHUFFLE^NOVEL||_2/D_NOVEL.
M5 = ||Y_PLASTIC_AB^NOVEL - Y_UNIFORM^NOVEL||_2/D_NOVEL.
M6 = ||Y_PLASTIC_AB^A - Y_PASSIVE_AB^A||_2/D_A.
M7 = ||Y_ERASE^NOVEL - Y_PASSIVE_AB^NOVEL||_2/D_NOVEL.

M1/M2/M6 are causal medium-change probes but do not prove beneficial semantics; M3 measures experience-order influence at matched stimulus multiset; M4/M5 assess spatial patterns versus a globally increased sigma; M7 is an equality control.

### Independent audit, cohorts and preregistered gates

Non-scientific smoke seed332001. Scientific diagnostic individuals seed332101..332106. **Fresh untouched formal HOLD** seed332201..332206. Every individual has different randomly drawn A/B pulse amplitudes from independent RNG seeds; statistical independence refers to input streams, not independent law families/geometries. Each intervention per seed has **exact same historical input source and physical starting conditions**.

C0 required: independent auditor must NOT import experiment implementation, and must separately recompute every flow, source, leak and plastic medium update through every training/REST tick, verifying independent conservation, source stream and final q/sigma against raw output to <=1e-9. Recompute every frozen probe trajectory, ERASE/SHUFFLE/UNIFORM and actual probe state arrays to <=1e-9. All six holdouts C0 PASS required before interpreting any physical effect.

Numerical gates (holdout seed count):
- P0 M0>=.10, 5/6.
- P1 M1>=.02, 5/6.
- P2 M2>=.02, 5/6.
- P3 M3>=.005, 4/6.
- P4 M4>=.01, 4/6.
- P5 M5>=.005, 4/6.
- P6 M6>=.02, 5/6.
- P7 M7<=1e-10, 6/6.

All C0 and all P0–P7 PASS => MEDIUM_TRACE_CAUSAL_EFFECT_SUPPORTED_IN_FIXED_GRID (physical only). All C0 PASS and any P FAIL => PARTIAL_OR_NOT_ESTABLISHED. Any physical C0 FAIL => INVALID_PHYSICS (no promotion). The eight gates are correlated and represent a small synthetic family; 5/6 is a reproducibility requirement, not a formal statistical confidence interval.

### Implementation controls and failure handling

Commit PROTOCOL first. Then implement simulator, tests and physically independent audit; publish 4 GitHub Actions workflows (contracts, smoke, diagnostic and untouched HOLD) on push-marker triggers. Use same source law, seeds and pass thresholds throughout and never change physics/thresholds after seeing diagnostic results. Contract failures may be corrected with disclosure prior to formal. Preserve raw per-seed files, independent audits and exact run IDs. Save frozen Japanese official results, design limitations and handoff on experiment branch / Draft PR, do not merge into main. Compare independent probe trajectories and actual physical ledger, not just number of connections. This experiment does NOT answer whether the sphere can solve LLM-specified hard problems.
