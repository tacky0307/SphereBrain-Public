# Core Growth Microscope v1

## 目的

3地点のP/E世界を、行動課題ではなくCore診断用の最小世界へ変える。

この段階では、Coreに答えを出させない。動かさない。学習させない。

観察する問いは次だけ。

> PとEの存在と絶対位置だけを入力したとき、Core内部にどのような活動差が生まれるか。

## Coreへ入れる情報

- Pという独立入力チャネル
- Pの絶対位置（左・中央・右）
- Eという独立入力チャネル
- Eの絶対位置（左・中央・右）

各入力Nodeの初期活性はすべて1.0で統一する。

## Coreへ入れない情報

- Pから見たEの方向
- Eまでの距離
- 近い／遠い
- 左右関係
- 正解行動
- 行動候補
- 移動可能性
- 目的
- 報酬
- 教師
- 最短経路
- 未通過優先
- ループ回避

## 使用する現在技術

- 実 `brain.py` のFocused propagation
- 実 `brain.json` の構造を読み込んだコピーCore
- learning OFF
- noise OFF
- brain.json保存なし
- Structural Assist OFF本線
- Structural Assist ON Shadow比較
- Activation history
- Traversed Edge
- Structural Assist trace
- 同条件再実行による再現性確認
- P/E位置交換対照
- brain.json SHA-256不変確認

## なぜ行動を作らないか

行動候補や正解を用意すると、Encoder、Decoder、候補列挙、教師規則のどこかに答えが混入する。

まず、Coreが9種類の世界状態を異なる活動構造として保持できるかを確認する。

それが成立しない場合、行動学習以前に不足している機能を特定する。

## 実行

```bat
run_core_growth_microscope_v1.bat
```

ブラウザ:

```text
http://127.0.0.1:5034
```

## 保存結果

```text
data/core_growth_microscope_v1/results/latest_observation.json
```

## 次の判断

v1の結果から次を切り分ける。

1. 異なる世界状態でも活動がほぼ同じ
   - 入力分離、初期刺激、Core伝播の識別能力が不足
2. 活動は異なるがすぐ消える
   - 状態保持、残響、再帰、収束が不足
3. 活動差はあるが経験へ残せない
   - 可塑性と状態表現の結合が不足
4. 経験へ残るが再利用できない
   - 想起、補完、行動Portへの接続が不足
5. Structural Assistだけで差が出る
   - 通常伝播と構造利用の接続が不足

この順で、Coreへ追加すべき機能を決める。
