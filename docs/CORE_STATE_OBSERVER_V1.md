# Core State Observer v1

## 目的

Transformer全体をCoreへ入れず、複数要素間の関係を同じ計算段階で並列に捉える原理だけを、読み取り専用の観測へ利用する。

Observerは次を行わない。

- 経路選択
- 正解方向への誘導
- 学習
- Node / Edgeの変更
- Coreへの再入力
- 言語処理

## 構成

```text
文章
↓
LLM Embedding
↓
SphereBrain Core（従来どおり伝播）
↓
SignalResult / Core構造
↓
Core State Observer（読み取り専用）
↓
全体状態ベクトル・Node間関係
```

## 並列観測する特徴

活動した各Nodeについて、球体内座標、Node使用回数、接続重量、今回の経路内次数、活動した時間段階、入力Nodeかどうか、最終活動値を数値特徴にする。

全Nodeを行列Xとして、固定・非学習の自己関係計算を行う。

```text
relation = softmax(X X^T / sqrt(d))
context  = relation X
```

これは答えを作るAttentionではない。すでに起きたCore活動の各部分が、全体の中でどのような関係にあるかを読むための計算である。

## 比較指標

1. Embedding similarity: LLM入力側の意味類似度
2. Route similarity: 従来のNode・Edge重なり
3. Observer similarity: Node配置、使用履歴、時間段階、経路内関係を含む全体状態の類似度

## 実行

```bash
git switch experiment/core-state-observer-v1
git pull
```

`run_core_state_observer_v1.bat` を実行する。

## 出力

```text
data/core_state_observer_v1/results/core_state_observer_v1.json
data/core_state_observer_v1/results/core_state_observer_v1.csv
```

## v1の成功条件

- Coreの伝播結果を一切変更しない
- Observerを外してもCoreは同じように動く
- 単純な重なり率だけでは見えない構造差を観測できる
- 無関係入力と類似・曖昧入力の差を、Embeddingだけに依存せず記述できる

この段階ではObserver出力をDecoderやReflectionへ戻さない。
