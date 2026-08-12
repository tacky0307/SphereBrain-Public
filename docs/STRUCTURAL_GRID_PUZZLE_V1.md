# Structural Grid Puzzle v1

## 目的

過去の SphereWorld Puzzle の 3×3 盤面を、現在の Structural Assist 実装で再構成する。

盤面:

```text
P . .
# # .
. . G
```

- P: 開始位置 / 現在位置
- G: ゴール
- #: 通行不可
- 上下左右移動

## 比較

各手で同じ候補集合と同じ基礎スコアを用い、以下を同時表示する。

- Structural Assist OFF の候補順位
- Structural Assist ON の候補順位
- tie gate
- baseline margin
- top candidate changed
- absolute modulation
- 通過Node / Edge履歴

## 固定条件

- 学習OFF
- noise OFF
- brain.json保存なし
- 候補集合不変
- 候補値不変
- Edge weight不変
- 候補順位のみStructural Assistが変更可能

## 実行

```bat
run_structural_grid_puzzle_v1.bat
```

ブラウザで `http://127.0.0.1:5033` を開く。

## 出力

各移動後の状態を次へ保存する。

```text
data/structural_grid_puzzle_v1/results/structural_grid_puzzle_v1.json
```
