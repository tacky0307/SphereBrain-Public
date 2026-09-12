# ECTBF v1C — Distributed Class Formation Mechanism Audit

## consequence fieldは、交換可能なmemberではなく協調的共分散に依存するのか

**正規の正式判定:** `DISTRIBUTED_CLASS_FORMATION_MECHANISM_AUDIT_V1__INCONCLUSIVE_COOPERATIVE_STRUCTURE`  
**候補昇格:** なし  
**正式holdout:** 60ケース  
**構造監査:** 全項目PASS

この公開記録は、GitHub Actionsで凍結実行された正規結果をそのまま保存します。source、固定threshold、frozen manifest、60ケース完全結果、case表、監査、事後レポートを含みます。

## 支持された部分

- cooperative relation accuracy平均: **0.9792**
- full-field relation accuracy平均: **0.9876**
- marginal-only relation accuracy平均: **0.8403**
- cooperative − marginal平均差: **+0.1389**、bootstrap下限 **+0.1159**
- cooperative − covariance破壊平均差: **+0.0296**、bootstrap下限 **+0.0220**
- full field − conditional shuffle平均差: **+0.0230**、bootstrap下限 **+0.0158**
- field除去によるcomplete-sequence低下: **+0.8372**

宣言済みの合成continuous-stream条件では、協調的共分散が有用であるという強い証拠があります。

## 非昇格の理由

- covariance破壊に対する正のケース率: **0.7833**（必要値 **0.80**）
- cooperative relation accuracyの最悪ケース基準が未達
- full-field relation accuracyの最悪ケース基準が未達

したがって、正しい結論は「協調的構造の可能性は強いが正式には未決着」です。SUPPORTEDへの昇格ではありません。

汎用人工知能、event objectの自律発明、自然言語理解、意識、任意長planning、実世界の因果推論の成立を示す結果ではありません。

正式結果SHA-256: `4dc41edd1eaaa91fb352319d4a1158ebe49b3cd9a292c2bff3f78baec5540ba3`
