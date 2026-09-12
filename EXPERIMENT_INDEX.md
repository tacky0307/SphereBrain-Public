# SphereBrain Experiment Index

This file is a map of the major experimental line. It is not a claim that every PASS establishes the general hypothesis. Each result applies only to the conditions tested by that runner.

The canonical executable records are the versioned files under `experiments/` together with their matching root batch files.

## Structural learning foundation

| Version | Focus | Main result |
|---|---|---|
| v66 | 3x3 structural puzzle learning | repeated experience could move routes toward shorter / preferred structure |
| v69 | relative transition representation | same-delta structural relations generalized without semantic direction labels |
| v74 | paired-seed recovery | temporal condition strongly recovered learned structure; Assist added little |
| v75 | robustness | temporal condition remained strong; restore behavior was imperfect |
| v76 | failure attribution | many failures were `success_seen_but_not_stabilized` |
| v77 | success consolidation | restore performance improved |
| v78 | fixed-gain stability | exposed saturation / rigidity problems |
| v79 | homeostatic consolidation | controlled long-horizon puzzle setting reached 100% phase/final success with no saturation |
| v80 / v80B | native-learning equivalence | motif mismatch identified and then aligned; equivalence restored |
| v81 | primary Native Learning | validated Native Learning state promoted into primary `SphereBrain` Core |

## Semantic bridge formation

| Version | Focus | Main result |
|---|---|---|
| v82 | semantic experience revalidation | concept sharing and novel one-shot integration observed; shared-context Bridge failed |
| v82B | bridge microscope | new shared-edge effects did not establish meaningful context bridge |
| v82C | edge attribution | apparent new shared edges were primarily hub-driven |
| v83 | semantic episode binding | continuous episodes produced bridge and transfer in one small controlled setup |
| v84 | multi-seed / multi-domain robustness | v83 effect did not reproduce robustly; stable domain validation failed |
| v84B | success/failure attribution | success was not explained simply by initially closer action representations |

## Stabilization and collapse

| Version | Focus | Main result |
|---|---|---|
| v85 | bridge stabilization | many bridges formed late but few became stable |
| v86 | homeostatic consolidation | broad protection reduced formation / stability rather than improving it |
| v86B | failure attribution | protection frequently covered wrong / hub-like edges too early; Assist rescued degraded cases |
| v87 | delayed selective consolidation | hub damage removed, but no stable gain over primary |
| v88 | event-triggered consolidation | promising small result; event-triggered+Assist improved one condition |
| v89 | robustness | v88 improvement did not robustly reproduce across 20 seeds × 5 domains |
| v90 | formation failure microscope | stabilization identified as primary bottleneck after many bridges formed at least once |
| v91 | post-bridge stability dynamics | relative features collapsed even when edges themselves remained |
| v91B | multi-factor collapse decomposition | dominant mechanism: experimental weakening plus control catching up |
| v92 | relative selectivity preservation | intervention almost never triggered; no improvement established |
| v92B | event detection microscope | non-hub filter was too strict and blocked most unstable bridges from maturity |

## Distributed structural consensus

| Version | Focus | Main result |
|---|---|---|
| v93 | structural consensus at first Bridge | first-formation consensus did not distinguish stable vs unstable trials |
| v93B | consensus persistence dynamics | stable trials retained / increased consensus; unstable trials lost it |
| v94 | consensus persistence robustness | 40 seeds × 8 domains reproduced a large stable-vs-unstable consensus retention gap |
| v95 | consensus causality ablation | destroying consensus support strongly reduced stable survival, but specificity was not isolated |
| v95B | ablation specificity control | consensus-targeted ablation harmed stability more than matched random and hub-matched local controls |

Notable v95B result from the recorded run:

```text
Stable cohort            29
Sham Stable            100.0%
Consensus Stable        31.0%
Matched random Stable   55.2%
Hub matched Stable      58.6%
Consensus effect        69.0%
vs Random margin        24.1%
vs Hub margin           27.6%
Match coverage          100%
Domain specificity      6/8
```

Interpretation: consensus-support edges appeared more specifically important than equally local matched controls, although residual structural confounding could still remain.

## Rescue and support growth

| Version | Focus | Main result |
|---|---|---|
| v96 | consensus support rescue | direct weight strengthening rescued some unstable trials but only slightly outperformed matched local support |
| v96B | rescue failure attribution | rescued trials were more likely to grow support sources and quorum; weight-only strengthening frequently failed |
| v97 | independent support path formation | human-semantically shared-context extra episodes did not outperform random extra experience |

Recorded v96 result:

```text
Rescue eligible        104
Sham rescue            0.0%
Random rescue          5.8%
Consensus rescue       8.7%
Consensus+Assist       6.7%
vs Random margin       +2.9%
Persistence margin     +1.7%
Domain rescue          4/8
Rescue signal          NO
```

Recorded v96B result:

```text
Rescued                         9
Not rescued                    95
Rescued Support↑             11.1%
Failed Support↑               3.2%
Support difference            8.0%
Rescued Quorum↑              44.4%
Failed Quorum↑               13.7%
Quorum difference            30.8%
Weight-only failure            YES
Support growth signal          YES
```

Recorded v97 result:

```text
Sham rescue                   0.0%
Random rescue                16.3%
Independent support          14.4%
Support+Assist               14.4%
vs Random margin             -1.9%
Persistence margin           -3.4%
Support growth margin        -0.508
Quorum growth margin         -0.203
Domain positive                0/8
Formation signal               NO
```

Interpretation: adding experiences that humans consider semantically related is not equivalent to creating independent structural support inside the Core.

## Label-blind structural observation

| Version | Focus | Status |
|---|---|---|
| v98 | Emergent Structural Meaning Microscope | first experiment explicitly discovering Core organization before consulting human semantic family labels; local run was still executing when Open Research Archive v1 was created |

v98 uses language only as input stimulation. Hidden human-side family labels are excluded from clustering and threshold selection. Structure is first discovered from internal node, edge, and activation signatures. Human labels are consulted only afterward.

## Reading results correctly

A version marked PASS means that the runner's own predefined measurement contract was satisfied. It does **not** mean that SphereBrain as a whole is validated.

Likewise, a NO or failed signal is retained because it constrains the hypothesis and often motivates the next microscope experiment.

The experiment history should therefore be read as a chain of progressively narrower questions rather than a sequence of demonstrations of success.

<!-- DCFMA_V1_PUBLIC_RECORD_START -->
## Continuous-stream consequence mechanism — ECTBF v1 to v1C

| Study | Question | Canonical result |
|---|---|---|
| ECTBF v1 | Which moments should be compared? | plan continuation without formal temporal-boundary recovery |
| ECTBF v1B | Are moment pairs interchangeable? | distributed redundancy, but no reliable equivalence structure |
| **ECTBF v1C** | Does cooperative covariance explain the field? | **strong cooperative signal; formal result remained inconclusive and was not promoted** |

ECTBF v1C passed its structural audit and showed positive mean and bootstrap evidence for cooperative covariance, but failed preregistered worst-case and positive-case criteria. See `research_releases/ectbf_v1c_distributed_class_formation_mechanism_audit/`.
<!-- DCFMA_V1_PUBLIC_RECORD_END -->
