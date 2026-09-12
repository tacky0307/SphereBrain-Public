from __future__ import annotations

import json
from pathlib import Path

import distributed_class_formation_mechanism_audit_config_v1 as cfg

RESULTS = Path("results")
SUMMARY_PATH = RESULTS / "distributed_class_formation_mechanism_audit_v1_formal_summary.json"
AUDIT_PATH = RESULTS / "distributed_class_formation_mechanism_audit_v1_holdout_compact_audit.json"
MANIFEST_PATH = RESULTS / "distributed_class_formation_mechanism_audit_v1_frozen_manifest.json"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    status = summary["formal_status"]
    promoted = summary["candidate_promoted"]
    primary = summary["primary_metrics"]
    paired = summary["paired_mechanism_evidence"]
    mechanism = summary["mechanism_metrics"]
    causal = summary["causal_metrics"]

    report = f"""# ECTBF v1C — Distributed Class Formation Mechanism Audit

## 正式判定

`{status}`

Candidate promotion: **{'approved' if promoted else 'not approved'}**

## 問い

各temporal memberのquery・target条件付き周辺証拠を同一に保ったまま、member間の共分散だけを壊すと、未見streamでのrelation判断と複数step計画継続は低下するか。

## 主要性能

| 指標 | 値 |
|---|---:|
| Full covariance relation accuracy | {primary['full_relation_accuracy']:.6f} |
| Cooperative subset relation accuracy | {primary['cooperative_relation_accuracy']:.6f} |
| Marginal subset relation accuracy | {primary['marginal_relation_accuracy']:.6f} |
| Cooperative complete sequence | {primary['cooperative_complete_sequence']:.6f} |
| Combined complete sequence | {primary['combined_complete_sequence']:.6f} |

## Paired mechanism evidence

| 比較 | 平均差 | 95% bootstrap | Positive fraction |
|---|---:|---:|---:|
| Full − conditional covariance-destroyed | {paired['full_over_conditional_shuffle']['mean']:.6f} | [{paired['full_over_conditional_shuffle']['bootstrap_low']:.6f}, {paired['full_over_conditional_shuffle']['bootstrap_high']:.6f}] | {paired['full_over_conditional_shuffle']['positive_fraction']:.4f} |
| Full − diagonal marginal-only | {paired['full_over_diagonal']['mean']:.6f} | [{paired['full_over_diagonal']['bootstrap_low']:.6f}, {paired['full_over_diagonal']['bootstrap_high']:.6f}] | {paired['full_over_diagonal']['positive_fraction']:.4f} |
| Cooperative subset − marginal subset | {paired['cooperative_over_marginal']['mean']:.6f} | [{paired['cooperative_over_marginal']['bootstrap_low']:.6f}, {paired['cooperative_over_marginal']['bootstrap_high']:.6f}] | {paired['cooperative_over_marginal']['positive_fraction']:.4f} |
| Cooperative subset − same members after covariance destruction | {paired['cooperative_over_destroyed']['mean']:.6f} | [{paired['cooperative_over_destroyed']['bootstrap_low']:.6f}, {paired['cooperative_over_destroyed']['bootstrap_high']:.6f}] | {paired['cooperative_over_destroyed']['positive_fraction']:.4f} |

## Formation audit

- Maximum conditional marginal preservation error: `{mechanism['maximum_conditional_marginal_preservation_error']:.3e}`
- Minimum off-diagonal covariance change ratio: `{mechanism['minimum_covariance_destruction_ratio']:.6f}`
- Minimum cooperative formation R²: `{mechanism['minimum_cooperative_formation_r2']:.6f}`
- Compact audit all pass: `{audit['all_pass']}`

## Causal checks

- Minimum full gain over smoothness-matched null: `{causal['minimum_gain_full_over_smooth_null']:.6f}`
- Field-ablation sequence gap: `{causal['field_ablation_sequence_gap']:.6f}`
- No-maintenance sequence gap: `{causal['no_maintenance_sequence_gap']:.6f}`

## 解釈境界

この結果は、合成continuous-stream・generic ordered-moment vocabulary・二値route・遅延scalar feedback・有限planの範囲に限定される。event vocabulary、目標、報酬、介入、任意長planningをCoreが自律発明したことは示さない。

---

Formal result SHA-256: `{summary['formal_result_sha256']}`  
Frozen source manifest SHA-256: `{summary['frozen_source_manifest_sha256']}`
"""
    (RESULTS / "distributed_class_formation_mechanism_audit_v1_full_report_ja.md").write_text(report, encoding="utf-8")

    review = f"""# ECTBF v1C — Joint Research Review

Formal status: `{status}`

Candidate promotion: **{'approved' if promoted else 'not approved'}**

The result is post-adjudication documentation and does not alter the frozen decision.

The decisive control preserved every temporal member's observed-query / signed-target conditional mean while independently permuting columns inside each stratum. The field therefore lost joint covariance without losing the declared member marginals.

Compact audit all pass: `{audit['all_pass']}`

## Next research boundary

The generic ordered-moment vocabulary is still supplied. A positive v1C result would motivate an event-object study in which cooperative temporal assemblies must form before a pair vocabulary is enumerated.
"""
    analysis_dir = Path("research/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1_REVIEW.md").write_text(review, encoding="utf-8")

    milestone_dir = Path("research/milestones")
    milestone_dir.mkdir(parents=True, exist_ok=True)
    milestone = f"""# ECTBF v1C Milestone

- Formal status: `{status}`
- Candidate promoted: `{promoted}`
- Formal cases: `{summary['case_count']}`
- Formal result: `{summary['formal_result_sha256']}`
- Frozen source: `{summary['frozen_source_manifest_sha256']}`
- Compact audit: `{audit['all_pass']}`
"""
    (milestone_dir / "DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1_MILESTONE.md").write_text(milestone, encoding="utf-8")

    index = {
        "study": cfg.STUDY,
        "formal_status": status,
        "candidate_promoted": promoted,
        "formal_result": "results/distributed_class_formation_mechanism_audit_v1_holdout.json",
        "formal_summary": str(SUMMARY_PATH),
        "compact_audit": str(AUDIT_PATH),
        "frozen_manifest": str(MANIFEST_PATH),
        "full_report_ja": "results/distributed_class_formation_mechanism_audit_v1_full_report_ja.md",
        "review": "research/analysis/DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1_REVIEW.md",
        "milestone": "research/milestones/DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1_MILESTONE.md",
    }
    (RESULTS / "distributed_class_formation_mechanism_audit_v1_result_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    publication = {
        "study": cfg.STUDY,
        "post_adjudication_only": True,
        "formal_status_unchanged": status,
        "candidate_promoted_unchanged": promoted,
        "formal_result_sha256": summary["formal_result_sha256"],
        "freeze_state": manifest["freeze_state"],
        "compact_audit_all_pass": audit["all_pass"],
    }
    (RESULTS / "distributed_class_formation_mechanism_audit_v1_publication_audit.json").write_text(json.dumps(publication, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
