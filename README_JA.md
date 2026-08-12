# SphereBrain

**SphereBrain**は、経験によって形成され続ける内部構造が、認識・記憶・選択、さらに高次の組織化の基盤になり得るかを検証する公開研究です。

> **知性とは、経験によって変化し続ける認識の構造である。**

SphereBrainは、言葉や「犬は動物」といった命題をCore内部へそのまま保存することを目標にしていません。言語・画像・音声などの外部情報は数値刺激へ変換され、Coreでは、その経験が生じさせた活動によって経路や内部状態が変化します。

## 現在の位置づけ

SphereBrainは、**研究仮説・予備的実験段階**です。

現時点では、汎用人工知能、意識、人間のような理解、生物学的な脳の再現、医学的有効性、LLMや既存手法に対する優位性を主張しません。

この研究が公開するのは完成した答えではなく、検証可能な問い、コード、観測結果、失敗、反証、未解決事項です。

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

## 資料

- [README.md](README.md) — 英語版の入口
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

本リポジトリは公開準備中です。公開版が固定されるまでは、記載内容や構成が変わる可能性があります。
