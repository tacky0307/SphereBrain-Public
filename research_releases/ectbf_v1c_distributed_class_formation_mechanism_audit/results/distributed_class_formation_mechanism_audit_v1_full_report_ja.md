# ECTBF v1C — Distributed Class Formation Mechanism Audit

## 正式判定

`DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__INCONCLUSIVE_COOPERATIVE_STRUCTURE`

Candidate promotion: **not approved**

## 問い

各temporal memberのquery・target条件付き周辺証拠を同一に保ったまま、member間の共分散だけを壊すと、未見streamでのrelation判断と複数step計画継続は低下するか。

## 主要性能

| 指標 | 値 |
|---|---:|
| Full covariance relation accuracy | 0.987604 |
| Cooperative subset relation accuracy | 0.979167 |
| Marginal subset relation accuracy | 0.840313 |
| Cooperative complete sequence | 0.942708 |
| Combined complete sequence | 0.844792 |

## Paired mechanism evidence

| 比較 | 平均差 | 95% bootstrap | Positive fraction |
|---|---:|---:|---:|
| Full − conditional covariance-destroyed | 0.023021 | [0.015833, 0.030625] | 0.6833 |
| Full − diagonal marginal-only | 0.027188 | [0.020313, 0.034479] | 0.7833 |
| Cooperative subset − marginal subset | 0.138854 | [0.115938, 0.162187] | 0.8667 |
| Cooperative subset − same members after covariance destruction | 0.029583 | [0.021977, 0.037398] | 0.7833 |

## Formation audit

- Maximum conditional marginal preservation error: `1.776e-15`
- Minimum off-diagonal covariance change ratio: `0.770582`
- Minimum cooperative formation R²: `0.929490`
- Compact audit all pass: `True`

## Causal checks

- Minimum full gain over smoothness-matched null: `0.287500`
- Field-ablation sequence gap: `0.837153`
- No-maintenance sequence gap: `0.440972`

## 解釈境界

この結果は、合成continuous-stream・generic ordered-moment vocabulary・二値route・遅延scalar feedback・有限planの範囲に限定される。event vocabulary、目標、報酬、介入、任意長planningをCoreが自律発明したことは示さない。

---

Formal result SHA-256: `4dc41edd1eaaa91fb352319d4a1158ebe49b3cd9a292c2bff3f78baec5540ba3`  
Frozen source manifest SHA-256: `273e4df7281288b61fc6e711a6f246f8fafef97c65ce9c87e5266d7db471a268`
