from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline
import run_llm_core_in_out_demo as base


def decoder_output(input_text: str, order_name: str, metrics: dict, adapter: base.CachedAdapter) -> str:
    """Form a standalone sentence from Core state instead of explaining the state."""
    instructions = (
        "あなたはSphereBrain Coreの数値状態を、日本語の文章そのものへ変換するDecoderです。"
        "解説、考察、判定報告、会話への返答は書かないでください。"
        "『Coreは』『入力は』『〜と受け取った』『〜と感じた』『〜を示している』など、"
        "観測者の説明文は禁止です。"
        "出力は、そのCore状態から直接生まれた独立した日本語の一文だけにしてください。"
        "入力への返事になっていなくても構いません。詩的、素朴、少し不完全でも構いませんが、"
        "できるだけ文として読める形にしてください。"
        "入力文は語彙と題材の手掛かりとしてのみ使い、一般知識で内容を補足しないでください。"
        "晴れ・雨・曖昧の強さ、両者の差、活動規模を文章の重心、断定の強さ、混ざり方へ反映してください。"
        "数値、経験順序名、内部用語を本文に書かないでください。"
        "出力は必ず一文のみです。"
    )
    payload = {
        "source_text_for_vocabulary_only": input_text,
        "core_state": {
            "sunny_affinity": round(metrics["sunny_affinity"], 4),
            "rainy_affinity": round(metrics["rainy_affinity"], 4),
            "ambiguous_affinity": round(metrics["ambiguous_affinity"], 4),
            "sunny_minus_rainy": round(metrics["sunny_minus_rainy"], 4),
            "bridge_strength": round(metrics["bridge_strength"], 4),
            "activity_nodes": metrics["node_count"],
            "activity_edges": metrics["edge_count"],
            "top_experience_group": metrics["top_group"],
        },
        "required_output": {
            "type": "standalone_sentence",
            "not_analysis": True,
            "not_conversation_reply": True,
            "sentence_count": 1,
        },
        "examples_of_form_only": [
            "雨の名残を抱えた空に、細い日差しがほどけていく。",
            "明るさと雨の気配が、まだ同じ空に残っている。",
        ],
    }
    response = adapter.client.responses.create(
        model=pipeline.DECODER_MODEL,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
    )
    text = response.output_text.strip()
    return text.splitlines()[0].strip() if text else ""


base.decoder_output = decoder_output
base.PAGE = (
    base.PAGE
    .replace("3つのCoreで解釈する", "3つのCoreから文章を形成する")
    .replace(
        "DecoderにはCore指標を渡します。差が小さいときは、無理に異なる文章を作らないよう制約しています。",
        "Decoderは解説や返答をせず、Core指標から独立した一文を形成します。会話として成立しなくても構いません。",
    )
    .replace(
        "Coreを観測し、Decoderで言葉へ戻しています…",
        "Coreを観測し、数値状態から文章を形成しています…",
    )
    .replace(
        "同じ入力を、経験順序の異なる3つのCoreへ通し、人間が読める言葉へ戻します。",
        "同じ入力を3つのCoreへ通し、それぞれの数値状態から一文を形成します。",
    )
)


if __name__ == "__main__":
    base.main()
