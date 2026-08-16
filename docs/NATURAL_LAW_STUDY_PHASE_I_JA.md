# SphereBrain Natural Law Study — Phase I

## 経験によって形づくられる人工世界の動力学

この文書は、SphereBrain Natural Law Study の最初の大きな到達点を固定する公開記録です。

現在の問いは、Core がすでに言語を理解するか、人間の認知を再現するかではありません。より手前の問いを扱います。

> 経験が人工ネットワーク内部の物理的な状態を変えるとき、人間側が意味を与える前に、どのような履歴依存のDynamicsが自然に現れるのか。

Phase I は、初期のNatural Law実験からInternal Dynamics StudyのID1Rまでを対象とします。

## 研究姿勢

SphereBrain Coreは、記号知識の保存庫ではなく、人工的な動力学世界として扱います。

現在の基本像は次の通りです。

```text
Experience
   ↓
Activity / Motion
   ↓
Flow
   ↓
Landscape deformation
   ↓
Consolidation / recovery timescale
   ↓
Future Activity
```

ここでまとめる実験では、意味ラベル、Memory lookup、Reward target、Context ID、Sequence ID、TimestampなどをCoreへ直接与えていません。

## 意図的に与えたもの

Natural Law Studyでは、以下の最小限の物理的要素を使いました。

- Node / Edge：人工世界の構造的な骨格
- Activity / Motion：その瞬間の活動状態
- Flow：状態差とconductanceから生じる局所的な流れ
- Landscape deformation：Flowによって変形するEdgeの物性
- finite shared structural resource：有限の構造資源
- natural recovery / decay：自然回復・自然減衰
- 後半では、local Flowだけによって形成され、その場所のRecoveryを遅くするConsolidation

Consolidationは局所的な持続変数ですが、Memory label、経験回数カウンター、意味カテゴリ、Reward signalではありません。

## 与えていないもの

Phase Iの主要実験では、以下を導入していません。

- Core内部の意味情報
- 明示的なMemory record / retrieval
- 正解targetやReward learning
- 重要度ラベル
- Natural Lawが参照する反復回数カウンター
- Context ID
- Sequence ID
- Lawが解釈するTimestamp

「こちらが入れた法則」と「その法則の上で観測された現象」を分けることを重視しています。

## 主な観測結果

### 1. Flowは未来のDynamicsを変えるLandscapeを形成できる

経験によって生じたFlowはLandscapeを変形し、Transient activityを消した後でも、その変形したLandscapeが後続Dynamicsを変えました。

これはpersistent structureによるhistory dependenceを支持しますが、それだけでMemoryやCognitionを意味しません。

### 2. 同時刺激は単独刺激の足し算ではないWhole-Stateを生み得る

2つの刺激を同時に入れたとき、Plasticな世界では、そのWhole-StateとRest後Landscapeが単独刺激2つを足しただけでは説明できない場合がありました。

テストした15ペアすべてでinteraction-level non-additivityが観測され、Plasticity OFFのControlでは重ね合わせが数値精度内で成立しました。

この結果は、「関係」を明示的なRelation labelとして保存しなくても、世界内部の相互作用から関係らしき差分が生じる可能性を示します。

### 3. 反復経験は成熟と飽和を生み得る

同じPair experienceを繰り返すと、Landscapeと対応するWhole-State responseは、テスト条件内で安定または緩やかにしか変化しない領域へ近づきました。

最初のRepeated Pair Maturation Auditでは、15/15ペアが `saturating_or_converging` に分類されました。

ここではMemoryではなく、**maturation（成熟）** と呼びます。

### 4. 単一時間尺度のRecoveryでは、成熟した履歴も新しい履歴も消えた

以前のRecovery lawでは、長いNatural Restによって一度だけの新規経験だけでなく、十分に成熟したLandscapeも消えていきました。

この否定結果から、「強く変形したこと」だけでは長期的な履歴保持に十分ではないことが分かりました。

### 5. Metaplasticityにより経験寿命の異なる領域が現れた

そこで、最小限のLocal Consolidationを導入しました。

```text
local Flow
   ↓
local Consolidation
   ↓
local natural recovery が遅くなる
```

30点の gain × consolidation-decay sweepでは、観察用の分類として、

- forgetting：17 / 30
- transient memory：8 / 30
- long-lived structure：5 / 30

が現れました。

これは数学的なPhase transitionを証明する分類ではありません。保持時間・保持率が大きく異なる領域が存在したことを表す観察上の区分です。

### 6. 反復回数を数えなくても経験寿命が変わった

反復回数を細かく変えた実験では、Long-rest retentionは多くの場合、特定回数で突然ONになるのではなく、反復に応じて連続的に増加し、その後飽和する形を示しました。

境界付近10世界を調べたID1Oでは、保守的な `sharp_lifetime_transition_signal` は0/10でした。

つまり現在のLawでは、「5回だから保存する」のではなく、Flow-driven consolidation、Consolidation decay、反復Exposureの相互作用によって経験寿命が変わっています。

### 7. 同じ経験回数でも、いつ経験したかで成熟度が変わった

A+Bを10回経験させ、経験間Restだけを変えたID1Pでは、4世界中3世界でTemporal spacing effectが観測されました。

現在のMetaplastic dynamicsでは、経験間隔を長くしすぎると、次の経験が来る前にLandscape / Consolidationが減衰し、成熟が弱くなる場合がありました。

Natural LawはClockやSchedule labelを受け取っていません。時間の影響は、時間経過中に世界自身が変化することで生じています。

### 8. 空白時間と、その時間の中で別の経験をしたことは同じではなかった

最初のA+B、最後のA+B、総経過時間を揃えたうえで、途中をRestだけにする条件と、別Experienceを挟む条件を比較しました。

3/3世界で、途中の経験を変えると最後のA+B responseが変わりました。

テストした世界では、中間Experienceの影響は一貫して `A+C > C+D > E+F` の順でした。

これはSemantic context understandingではなく、原始的なHistory / Context dependenceとして解釈しています。

### 9. 経験順序が一部の組み合わせで現在状態へ刻まれた

ID1Rでは、最初と最後のA+B、Experienceの種類・回数・Spacing・総時間を揃え、途中の2経験の順番だけを反転しました。

3世界 × 3逆順比較で、

- 9 comparisons
- 3 supported order effects
- 3世界すべてに少なくとも1つのorder-sensitive comparison
- 3世界すべてで支持された組は `C+D ↔ E+F`

という結果でした。

これは、テストしたMaterial dynamicsの中で一部のExperience operationが可換でなくなったことを意味します。

```text
F(B, F(A, S)) != F(A, F(B, S))
```

ただし、これはSequence understandingやTemporal reasoningではなく、**Experience-order imprint**と表現します。

## 重要な否定結果と修正

Phase Iでは、成功結果だけでなく、否定された仮説も公開対象とします。

- 初期Free-run predictionでは、Native threshold dynamics下でPredictive free-run transitionは支持されませんでした。
- v103CではSubthreshold predictive structureは支持されませんでした。
- v103DではNative storageは対称で、General temporal directionalityは支持されませんでした。
- ID1Hでは、累積Flow cosineを近づけただけでは、入力位置を分離したときLandscape responseを再現できませんでした。
- ID1Lでは、当時のSingle-timescale lawのもとで、新しい経験を十分繰り返すと以前の成熟状態はほぼ上書きされました。
- ID1Mでは長いRestによって古い成熟履歴も一度のNovel experienceもBlank方向へ消えました。
- ID1Oでは鋭い反復回数Thresholdは確認できませんでした。
- ID1Rの順序効果は普遍ではなく、9比較中3比較のみでした。

これらの否定結果が、後のNatural Law modelの設計変更を導いています。

## 現在の解釈

Phase I終了時点での最も有用なWorking interpretationは次のものです。

> 現在のSphereBrain stateは、過去の出来事を保存した一覧ではない。その人工世界が履歴によってどう変えられたか、そのMaterial consequenceそのものである。

この見方では、

- 過去のFlowがLandscapeを変える
- 反復FlowがLandscapeのRecovery timescaleを変える
- 時間経過が現在のMaterial stateを変える
- 同じ時間でも途中に別Experienceがあると違う状態になる
- 一部のExperience sequenceは非可換になる
- そのため同じ現在Stimulusでも、異なる履歴の世界では異なるResponseが生じる

となります。

Memory、Context、Sequence dependenceをそれぞれ明示的なSymbolic moduleとして実装しなくても、それらの原型に見える現象がMaterial historyから生じる可能性を示しています。

## Phase Iでまだ示していないこと

Phase Iは以下を証明していません。

- Language understanding
- Core内部のSemantic concept
- Free recall
- Reliable prediction
- General temporal directionality
- Causal reasoning
- Episodic memory
- Human-like cognition
- General intelligence
- Biological brainとの等価性

また、現在の結果はテストしたArtificial worlds、Topology、Parameter、Measurement procedureに限定されます。

## なぜ翻訳を先に決めないのか

現在のSphereBrainに、人間側がAやBの意味を与える必要はありません。

まず、この人工世界自身がどのようなRegularityを持つのかを観察します。

将来、安定した内部現象が、Relation、Context、Prediction、Categoryなどの外界構造と一貫して対応することが確認できた場合、その観測結果からTranslation layerを設計できます。

LLMなどのLanguage-aware systemは、将来的にEncoder / Decoderとして人間世界とのBridgeになり得ますが、Core自体はその独自のNon-linguistic dynamicsを保つことを想定しています。

## Phase Iの境界

この文書では、**Natural Law Study — Phase I** をID1Rまでで一区切りとします。

次のPhaseでは、ここまでの結果を守るのではなく、一般性を壊しに行く必要があります。

優先的な問いは、

- TopologyやWorld sizeを変えても同じ現象が出るか
- なぜ一部のExperience pairだけがOrder-sensitiveなのか
- 複数のMature historyは共存できるか
- Long-lived LandscapeがExternal stimulusなしのInternal Dynamicsを生み得るか
- Consolidationのどの部分が本当に必要で、どこまで削れるか
- Internal stateのRegularity自身からExternal meaningへのTranslation方法を見つけられるか

です。

Phase Iは完成した知性理論ではなく、**人工自然の第一観測マイルストーン**です。
