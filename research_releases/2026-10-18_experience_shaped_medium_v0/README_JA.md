# SphereBrain 公開研究チェックポイント — Experience-Shaped Medium v0

**公開日：2026-10-18**

## 研究の原点

**「知性とは、経験によって変化し続ける認識の構造である」**。今回、自由に内部活動経路が生じる媒体を作り、その経験によって計算空間そのものが変化するかを、最小の数値物理から検証した。最終的な構想は、**LLMが問題を提起し、SphereBrain自身が内部の活動によって自律的に答えを導くこと**。今回その問題解決能力を実証したわけではない。

## この公開チェックポイントに含むもの

- [公開研究の羅針盤](../../RESEARCH_COMPASS_JA.md)
- [経験によって変質する自由媒体の指針](../../research/foundations/EXPERIENCE_SHAPED_MEDIUM_COMPASS_2026_10_18_JA.md)
- [検証計画](../../research/foundations/EXPERIENCE_SHAPED_MEDIUM_VERIFICATION_PLAN_JA.md)
- [事前登録PROTOCOL](../../research/experiments/experience_shaped_medium_v0/PROTOCOL.md)
- [Core物理実装](../../research/experiments/experience_shaped_medium_v0/implementation.py)
- [独立物理計算監査](../../research/experiments/experience_shaped_medium_v0/aggregate.py)
- [基礎契約テスト](../../research/experiments/experience_shaped_medium_v0/test_contract.py)
- [正式日本語結果](../../research/experiments/experience_shaped_medium_v0/RESULTS_JA.md)
- [公開実行・再現手順](../../research/experiments/experience_shaped_medium_v0/PUBLIC_REPRODUCE.md)
- [次への引継ぎ](../../research/experiments/experience_shaped_medium_v0/HANDOFF_JA.md)

## 2026-10-18 公式観測

**MEDIUM_TRACE_CAUSAL_EFFECT_SUPPORTED_IN_FIXED_GRID**。固定36セル/60近傍相互作用で、経験によって媒体の伝導性が変化し、その局所的な配置と経験順序が**同一の未来刺激に対する活動に因果的効果**を与えた。元の研究側の事前登録固定・試運転・診断6seed・未使用正式HOLD6seed・独立フラックス／収支／将来probe監査に従い全8ゲートが成立した。

正式HOLD6平均：M0=1.9017747152、M1=0.3117761145、M2=0.3117761145、M3=0.0253941591、M4=0.1166421750、M5=0.0857756515、M6=0.4363467417、M7=0.0。**数値は物理波形の距離で、予測正解率ではない。**

元の未使用正式holdoutはすでに実施済みであるため、公開GitHub Actionsはその**再現実験**。元の非公開実行への閲覧アクセスがなくても、公開コードと監査器から同じ数値・判定を再計算できる。GitHub ReleaseやZenodo DOIを新たに発行したことは意味しない。

## 正しく理解してほしい限界

この媒体は実際にどんな任意経路でも描ける球体ではなく、**接続関係が決められた有限2D格子**。媒体の可塑性規則は人間が定義した。意味記憶の再現、長寿命の自由回路形成、解候補が正しいと判断する機能、LLMからの難問の自律解決を確認したものではない。従来の公開研究や失敗した比較実験を後付けで成功扱いしない。

## 次の試験（未着手）

**Functional Route Reconstitution v1** — 以前Aで得た伝達機能がBの経験を挟んでも再形成されるか。次の実験は今回の正式結果・ゲート・原法則を上書きせず、別版の事前登録を必要とする。

Apache-2.0ライセンスは本公開リポジトリの[LICENSE](../../LICENSE)に従う。研究成果の独立再現・引用では、この公開コードの正確なコミットと実行環境を記録すること。
