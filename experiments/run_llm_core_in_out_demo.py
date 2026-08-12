from __future__ import annotations

import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Iterable

import numpy as np
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

APP_DATA = ROOT / "data" / "llm_core_in_out_demo"
CACHE_FILE = APP_DATA / "embedding_cache.json"
STATE_FILE = APP_DATA / "demo_state.json"

SUNNY = [
    "今日は晴れて気持ちいい",
    "青い空が広がって爽やかだ",
    "暖かな日差しが心地よい",
]
RAINY = [
    "今日は雨で肌寒い",
    "暗い空から雨が降っている",
    "冷たい雨で気分が沈む",
]
AMBIGUOUS = [
    "今日は天気が変わりやすい",
    "晴れたり雨が降ったりしている",
    "空模様が落ち着かない",
]

ORDERS = {
    "晴れ→雨→曖昧": [("晴れ", SUNNY), ("雨", RAINY), ("曖昧", AMBIGUOUS)],
    "雨→晴れ→曖昧": [("雨", RAINY), ("晴れ", SUNNY), ("曖昧", AMBIGUOUS)],
    "曖昧→晴れ→雨": [("曖昧", AMBIGUOUS), ("晴れ", SUNNY), ("雨", RAINY)],
}

SAMPLES = [
    "雨上がりの空に日が差している",
    "雨の日でも気分は明るい",
    "空は暗いが雨は降っていない",
]

TRAIN_REPEATS = 20
LOCK = threading.RLock()
STATUS = {"ready": False, "running": False, "message": "未準備", "error": ""}


class CachedAdapter(pipeline.OpenAIAdapter):
    def __init__(self) -> None:
        super().__init__()
        APP_DATA.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            try:
                self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def embed(self, text: str) -> list[float]:
        clean = text.strip()
        if clean in self.cache:
            return list(self.cache[clean])
        value = super().embed(clean)
        self.cache[clean] = value
        CACHE_FILE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return value


def configure_core(path: Path) -> None:
    pipeline.DATA = path
    pipeline.BRAIN_FILE = path / "brain.json"
    pipeline.DB_FILE = path / "experiences.db"
    pipeline.PROJECTION_FILE = path / "projection.npy"
    pipeline.PROJECTION_SEED = 20260804


def observe(text: str, adapter: CachedAdapter) -> dict:
    embedding, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(sources, steps=14, threshold=0.18, noise=0.0, learn=False)
    return {
        "nodes": sorted(result.activated_nodes),
        "edges": [list(edge) for edge in result.traversed_edges],
        "node_count": len(result.activated_nodes),
        "edge_count": len(result.traversed_edges),
    }


def as_edge_set(route: dict) -> set[tuple[int, int]]:
    return {tuple(edge) for edge in route["edges"]}


def jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def route_overlap(left: dict, right: dict) -> float:
    node_score = jaccard(left["nodes"], right["nodes"])
    edge_score = jaccard(as_edge_set(left), as_edge_set(right))
    return 0.35 * node_score + 0.65 * edge_score


def mean_overlap(route: dict, references: list[dict]) -> float:
    return float(np.mean([route_overlap(route, ref) for ref in references])) if references else 0.0


def core_path(order_name: str) -> Path:
    safe = order_name.replace("→", "_")
    return APP_DATA / "cores" / safe


def prepare_demo() -> None:
    with LOCK:
        if STATUS["running"]:
            return
        STATUS.update({"running": True, "error": "", "message": "準備を開始しました"})

    try:
        adapter = CachedAdapter()
        state = {"version": 1, "orders": {}, "train_repeats": TRAIN_REPEATS}
        all_reference_texts = {"晴れ": SUNNY, "雨": RAINY, "曖昧": AMBIGUOUS}

        for order_index, (order_name, sequence) in enumerate(ORDERS.items(), start=1):
            with LOCK:
                STATUS["message"] = f"Core {order_index}/3 を形成中: {order_name}"
            path = core_path(order_name)
            configure_core(path)
            pipeline.reset_experiment()
            for stage_name, texts in sequence:
                with LOCK:
                    STATUS["message"] = f"{order_name}: {stage_name}経験を追加中"
                for text in texts:
                    pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)

            references = {}
            for group_name, texts in all_reference_texts.items():
                references[group_name] = [observe(text, adapter) for text in texts]

            state["orders"][order_name] = {
                "path": str(path.relative_to(ROOT)),
                "references": references,
                "experience_order": [name for name, _ in sequence],
            }

        APP_DATA.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        with LOCK:
            STATUS.update({"ready": True, "running": False, "message": "3つの経験Coreを準備しました"})
    except Exception as exc:
        with LOCK:
            STATUS.update({"ready": False, "running": False, "message": "準備に失敗しました", "error": str(exc)})


def load_state() -> dict:
    if not STATE_FILE.exists():
        raise RuntimeError("先に『3つのCoreを準備』を実行してください。")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def decoder_output(input_text: str, order_name: str, metrics: dict, adapter: CachedAdapter) -> str:
    instructions = (
        "あなたはSphereBrain Coreの状態を人間の言葉へ翻訳するDecoderです。"
        "入力文の一般的な解説をしてはいけません。"
        "与えられたCore指標の差だけを根拠に、このCoreが入力をどう受け取ったかを日本語1文で表してください。"
        "Coreにない出来事、感情、因果を追加しないでください。"
        "差が小さいときは無理に個性を作らず、曖昧さを残してください。"
        "数値や経験順序名を回答本文に書かないでください。"
    )
    payload = {
        "input": input_text,
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
        "output_rule": "入力に含まれる要素のうち、Core状態が強く示した側へ表現の重心を置く。",
    }
    response = adapter.client.responses.create(
        model=pipeline.DECODER_MODEL,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
    )
    return response.output_text.strip()


def analyze_input(text: str) -> list[dict]:
    state = load_state()
    adapter = CachedAdapter()
    results = []
    for order_name, info in state["orders"].items():
        configure_core(ROOT / info["path"])
        route = observe(text, adapter)
        refs = info["references"]
        sunny = mean_overlap(route, refs["晴れ"])
        rainy = mean_overlap(route, refs["雨"])
        ambiguous = mean_overlap(route, refs["曖昧"])
        affinities = {"晴れ": sunny, "雨": rainy, "曖昧": ambiguous}
        ranked = sorted(affinities.items(), key=lambda item: item[1], reverse=True)
        metrics = {
            "sunny_affinity": sunny,
            "rainy_affinity": rainy,
            "ambiguous_affinity": ambiguous,
            "sunny_minus_rainy": sunny - rainy,
            "bridge_strength": ambiguous - abs(sunny - rainy) * 0.5,
            "node_count": route["node_count"],
            "edge_count": route["edge_count"],
            "top_group": ranked[0][0],
        }
        output = decoder_output(text, order_name, metrics, adapter)
        results.append({
            "order": order_name,
            "output": output,
            "metrics": {
                "晴れ親和性": round(sunny * 100, 1),
                "雨親和性": round(rainy * 100, 1),
                "曖昧親和性": round(ambiguous * 100, 1),
                "晴れ−雨": round((sunny - rainy) * 100, 1),
                "活動Node": route["node_count"],
                "通過Edge": route["edge_count"],
            },
        })
    return results


PAGE = r"""
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SphereBrain IN → Core → OUT</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--card:#12233d;--line:#315177;--text:#f5f7fb;--muted:#9fc1e8;--accent:#ef9654}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,"Yu Gothic UI",sans-serif}
main{max-width:1450px;margin:auto;padding:28px}.hero{margin-bottom:24px}.hero h1{font-size:30px;margin:0 0 8px}.hero p{color:var(--muted);margin:0}
.panel{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px;margin-bottom:22px}
textarea{width:100%;min-height:110px;background:#081522;color:var(--text);border:1px solid #46709d;border-radius:14px;padding:16px;font-size:18px;resize:vertical}
button{border:0;border-radius:12px;padding:13px 20px;font-weight:700;font-size:16px;cursor:pointer;background:var(--accent);color:white;margin:8px 8px 0 0}
button.secondary{background:#28496d}.samples button{font-size:14px;padding:9px 13px}.status{color:var(--muted);margin-top:10px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px}.card h2{font-size:19px;margin:0 0 13px;color:#85cdfa}.out{font-size:20px;line-height:1.65;min-height:105px;background:#091726;padding:15px;border-radius:12px}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:14px}.metric{background:#0a192b;border:1px solid #2f5277;border-radius:10px;padding:9px}.metric b{display:block;font-size:18px}.metric span{color:var(--muted);font-size:12px}.note{font-size:13px;color:var(--muted);margin-top:14px}.hidden{display:none}@media(max-width:950px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="hero"><h1>SphereBrain IN → Core → OUT</h1><p>同じ入力を、経験順序の異なる3つのCoreへ通し、人間が読める言葉へ戻します。</p></div>
<div class="panel">
  <button id="prepare" class="secondary">3つのCoreを準備</button>
  <span id="status" class="status">状態を確認中…</span>
</div>
<div class="panel">
  <label for="input"><b>IN</b></label>
  <textarea id="input">雨上がりの空に日が差している</textarea>
  <div class="samples">
    {% for sample in samples %}<button class="secondary sample" data-text="{{sample}}">{{sample}}</button>{% endfor %}
  </div>
  <button id="run">3つのCoreで解釈する</button>
  <div class="note">DecoderにはCore指標を渡します。差が小さいときは、無理に異なる文章を作らないよう制約しています。</div>
</div>
<div id="loading" class="panel hidden">Coreを観測し、Decoderで言葉へ戻しています…</div>
<div id="results" class="grid"></div>
</main>
<script>
const statusEl=document.getElementById('status');
async function refresh(){const r=await fetch('/api/status');const s=await r.json();statusEl.textContent=s.message+(s.error?'：'+s.error:'');document.getElementById('run').disabled=!s.ready||s.running;document.getElementById('prepare').disabled=s.running;if(s.running)setTimeout(refresh,1500)}
refresh();
document.getElementById('prepare').onclick=async()=>{await fetch('/api/prepare',{method:'POST'});refresh()};
document.querySelectorAll('.sample').forEach(b=>b.onclick=()=>document.getElementById('input').value=b.dataset.text);
document.getElementById('run').onclick=async()=>{const text=document.getElementById('input').value.trim();if(!text)return;document.getElementById('loading').classList.remove('hidden');document.getElementById('results').innerHTML='';try{const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});const data=await r.json();if(!r.ok)throw new Error(data.error||'失敗');for(const item of data.results){const card=document.createElement('section');card.className='card';let metrics='';for(const [k,v] of Object.entries(item.metrics))metrics+=`<div class="metric"><span>${k}</span><b>${v}${k.includes('親和')||k==='晴れ−雨'?'%':''}</b></div>`;card.innerHTML=`<h2>${item.order}</h2><div class="out">${item.output}</div><div class="metrics">${metrics}</div>`;document.getElementById('results').appendChild(card)}}catch(e){document.getElementById('results').innerHTML=`<div class="panel">${e.message}</div>`}finally{document.getElementById('loading').classList.add('hidden')}};
</script></body></html>
"""

app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string(PAGE, samples=SAMPLES)


@app.get("/api/status")
def api_status():
    if STATE_FILE.exists() and not STATUS["running"]:
        STATUS["ready"] = True
        if STATUS["message"] == "未準備":
            STATUS["message"] = "準備済み"
    return jsonify(STATUS)


@app.post("/api/prepare")
def api_prepare():
    if not STATUS["running"]:
        threading.Thread(target=prepare_demo, daemon=True).start()
    return jsonify({"started": True})


@app.post("/api/analyze")
def api_analyze():
    try:
        payload = request.get_json(force=True)
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("入力文が空です。")
        with LOCK:
            results = analyze_input(text)
        return jsonify({"input": text, "results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def main() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        STATUS.update({"ready": True, "message": "準備済み"})
    print("SphereBrain IN-Core-OUT demo")
    print("http://127.0.0.1:5082")
    app.run(host="127.0.0.1", port=5082, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
