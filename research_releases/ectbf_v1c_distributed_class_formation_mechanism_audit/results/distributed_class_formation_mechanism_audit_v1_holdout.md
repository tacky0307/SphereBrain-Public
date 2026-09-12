# ECTBF v1C — Distributed Class Formation Mechanism Audit / Does the Consequence Field Depend on Cooperative Covariance Rather Than Interchangeable Members?

Phase: `holdout`
Cases: `60`
Source manifest: `273e4df7281288b61fc6e711a6f246f8fafef97c65ce9c87e5266d7db471a268`

## Decision

`DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__INCONCLUSIVE_COOPERATIVE_STRUCTURE`

## Primary methods

| Method | Relation | Minimum | Complete sequence |
|---|---:|---:|---:|
| full_covariance_field | 0.987604 | 0.900000 | 0.960764 |
| cooperative_covariance_subset | 0.979167 | 0.881250 | 0.942708 |
| marginal_score_subset | 0.840313 | 0.625000 | 0.606597 |
| conditional_marginal_preserving_covariance_destroyed | 0.964583 | 0.850000 | 0.901736 |
| diagonal_gram_marginal_only | 0.960417 | 0.843750 | 0.882292 |
| cooperative_members_covariance_destroyed | 0.949583 | 0.837500 | 0.859722 |
| query_factorized_field | 0.968958 | 0.856250 | 0.909722 |
| smoothness_matched_shuffled_credit | 0.448750 | 0.175000 | 0.110069 |

## Paired mechanism contrasts

- `full_over_conditional_shuffle`: mean `0.023021`, 95% bootstrap `[0.015833, 0.030625]`, positive fraction `0.6833`
- `full_over_diagonal`: mean `0.027188`, 95% bootstrap `[0.020313, 0.034479]`, positive fraction `0.7833`
- `cooperative_over_marginal`: mean `0.138854`, 95% bootstrap `[0.115938, 0.162187]`, positive fraction `0.8667`
- `cooperative_over_destroyed`: mean `0.029583`, 95% bootstrap `[0.021977, 0.037398]`, positive fraction `0.7833`
