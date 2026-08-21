# SphereBrain

**SphereBrain**は、経験によって形成され続ける内部構造が、認識・記憶・選択、さらに高次の組織化の基盤になり得るかを検証する公開研究です。

> **知性とは、経験によって変化し続ける認識の構造である。**

SphereBrainは、言葉や「犬は動物」といった命題をCore内部へそのまま保存することを目標にしていません。言語・画像・音声・センサ情報などは数値刺激へ変換でき、研究対象であるCoreは、その経験が生じさせた活動によって経路・時間状態・関係記憶・可塑性を変化させます。

## 現在の到達点 — Unified Experience Core v2.0

これまで別々の実験で検証してきた一般原理のうち、因果的・回帰的にCoreへ昇格してよいと判断したものを、ひとつの経験駆動Coreへ統合しました。

公開版の入口は以下です。

- [`spherebrain_core_v2.py`](spherebrain_core_v2.py) — 1ファイルで動く公開研究Core
- [`run_public_minimal_demo.py`](run_public_minimal_demo.py) — 学習とルール反転の最小デモ
- [`docs/LATEST_CORE_RESEARCH_AUDIT.md`](docs/LATEST_CORE_RESEARCH_AUDIT.md) — Coreへ採用したもの／まだ研究中のもの
- [`data/public_demo/latest_core_regression.json`](data/public_demo/latest_core_regression.json) — 統合回帰テスト結果
- [`data/public_demo/latest_core_smoke_adaptation_v1.json`](data/public_demo/latest_core_smoke_adaptation_v1.json) — 公開前の軽量適応テスト結果

現在Coreへ正式採用しているのは、次の原理です。

```text
外部刺激
   ↓
構造伝播・長期構造記憶
   ↓
現在活動
   ├─ 有限のworking state
   └─ online directed transition trace
   ↓
経験signature
   ↓
fast relational form + slow relational identity
   ↓
関係ごとのsoft membership
   ↓
cluster-conditioned distributed value
   ↓
reliability / maturity / replasticity
   ↓
構造・時間・関係状態を更新
```

具体的には、

- 経験で変わるedge weight・edge usage・node usage
- 有限の短期／working state
- 「どう来たか」を逐次保持するonline transition trace
- 自己組織的なrelation separation
- relationごとに分離された分散価値記憶
- 安定経験で可塑性を弱めるmaturity
- 失敗や環境変化で再び学習可能にするreplasticity
- 構造が変わっても同じ経験関係を保つfast/slow identity continuity
- 上記を一体として保存するsave/load

を一つのCoreへ統合しています。

## 現在確認できていること

統合Coreの標準回帰テストは **11 / 11項目すべてPASS** しました。確認対象は、長期構造記憶、short-term state、transition trace、relation memory、identity continuity、maturity、replasticity、save/load、read-only評価が学習しないこと、task answer lookupをCoreが持たないことなどです。

また、公開前の2状態sensorimotor smoke testでは、

```text
初期精度                         50%
最初の規則を学習後              100%
規則反転直前の新規則精度           0%
反転学習後の新規則精度           100%
反転後の旧規則精度                 0%
```

となりました。反転学習では60 trial時点で50%、80 trial時点で100%へ到達し、その後200 trialまで正しい状態を維持しました。

これは、**小規模な条件では、Coreが経験した行動関係を獲得し、環境変化後に旧規則を手放して新しい規則へ適応できた**ことを示します。

## まだ証明していないこと

現時点では、以下を達成済みとは扱いません。

- 汎用人工知能
- 人間のような理解
- 未経験組み合わせへの一般化の完成
- 自律的planning
- 自律的な次状態予測／predictive free-run
- 普遍的な意味表現
- 生物学的脳の再現

特に、full Unified Coreでのheld-out relational generalizationはまだ成立していません。失敗や否定結果も研究成果として保存し、未解決部分を「できたこと」に含めない方針です。

## 最小デモの実行

Python 3.10以降を推奨します。

```bash
python -m pip install -r requirements-core-v2.txt
python run_public_minimal_demo.py
```

外部側は状態・候補action・成功／失敗だけを与えます。構造学習、working state、transition trace、relation memory、maturity、replasticityの更新順はCore内部が担当します。

## 基本構成

```text
外部の経験
    ↓
Encoder／感覚インターフェース
    ↓
数値刺激
    ↓
SphereBrain Core
    ↓
活動・構造
    ↓
Decoder／表現インターフェース
```

EncoderとDecoderには言語を扱うAIを利用できます。しかし、研究対象は経験によって変化するCoreです。Encoderに元から含まれる意味構造やDecoderが加えた表現能力を、Coreの知性と混同しないための比較が必要です。

## 研究原則

1. 答えをCoreへ注入せず、経験を先に置く
2. 人間の意味ラベルより先にCoreの構造を観測する
3. 成功だけでなく、失敗・否定的結果・未確認結果を残す
4. 観測と介入を分ける
5. 説得力のある物語より再現可能性を優先する
6. 主張を得られた証拠の範囲に限定する
7. 因果的に支持された一般原理だけを、ひとつの進化するCoreへ昇格する

## 資料

- [README.md](README.md) — 英語版
- [docs/LATEST_CORE_RESEARCH_AUDIT.md](docs/LATEST_CORE_RESEARCH_AUDIT.md) — 最新Core監査
- [PHILOSOPHY.md](PHILOSOPHY.md) — 研究思想
- [ARCHITECTURE.md](ARCHITECTURE.md) — 設計と役割分担
- [RESEARCH_TIMELINE.md](RESEARCH_TIMELINE.md) — 研究史
- [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) — 実験索引
- [LIMITATIONS.md](LIMITATIONS.md) — 未証明事項と主張の境界
- [REPRODUCE.md](REPRODUCE.md) — 再現方法
- [OPEN_RESEARCH_POLICY.md](OPEN_RESEARCH_POLICY.md) — 公開研究方針
- [AI_CONTRIBUTION.md](AI_CONTRIBUTION.md) — AI協力の開示

## 研究者とAI協力

- 発案者・研究者：Takio（たきお）
- 共同設計・実装支援：対話型AI「みみ」（OpenAI ChatGPT）

研究判断と公開責任は、たきおが担います。AIの支援は、設計議論、実装、分析、文書化に及びますが、AIの出力そのものは研究結果の正しさを保証しません。

## 参加について

再現、批判、反証、別解釈、比較実験、脳科学上の指摘を歓迎します。参加方法は[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。
