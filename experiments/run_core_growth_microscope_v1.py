from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

HOST = "127.0.0.1"
PORT = 5034
BRAIN_PATH = ROOT / "data" / "brain.json"
OUT = ROOT / "data" / "core_growth_microscope_v1" / "results"
POSITIONS = ["左", "中央", "右"]


class UniformInputBrain(SphereBrain):
    """Current Core with equal activation for every raw sensor channel."""

    def _initial_activation(self, source_nodes, context_nodes):
        sources = list(source_nodes)
        activation = np.zeros(self.node_count, dtype=float)
        for node in sources:
            activation[node] = 1.0
        if context_nodes:
            for node in context_nodes:
                activation[node] = max(activation[node], 0.34)
        return sources, activation


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def load_core() -> UniformInputBrain:
    if BRAIN_PATH.exists():
        loaded = SphereBrain.load(BRAIN_PATH)
        brain = UniformInputBrain(
            node_count=loaded.node_count,
            neighbors_per_node=loaded.neighbors_per_node,
            seed=loaded.seed,
            learning_rate=loaded.learning_rate,
            decay_rate=loaded.decay_rate,
            propagation_mode=loaded.propagation_mode,
            signal_decay=loaded.signal_decay,
            max_branches=loaded.max_branches,
            max_active_per_step=loaded.max_active_per_step,
            max_total_active_nodes=loaded.max_total_active_nodes,
            structural_assist_enabled=False,
        )
        for name in ("positions", "adjacency", "weights", "usage", "node_usage"):
            setattr(brain, name, copy.deepcopy(getattr(loaded, name)))
        return brain
    return UniformInputBrain(structural_assist_enabled=False)


def allocate_ports(brain: SphereBrain) -> dict[str, list[int]]:
    labels = ["entity:P", "entity:E"] + [f"position:{p}" for p in POSITIONS]
    used: set[int] = set()
    ports: dict[str, list[int]] = {}
    for label in labels:
        candidates = brain.text_to_sources(f"raw-sensor/{label}", count=8)
        chosen = []
        for node in candidates:
            if node not in used:
                chosen.append(node)
                used.add(node)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for node in range(brain.node_count):
                if node not in used:
                    chosen.append(node)
                    used.add(node)
                if len(chosen) == 2:
                    break
        ports[label] = chosen
    return ports


CORE = load_core()
PORTS = allocate_ports(CORE)
BEFORE_HASH = sha(BRAIN_PATH)


def raw_sources(player: str, other: str) -> list[int]:
    # Only atomic facts: entity identity and absolute position.
    return (
        PORTS["entity:P"]
        + PORTS[f"position:{player}"]
        + PORTS["entity:E"]
        + PORTS[f"position:{other}"]
    )


def summarize(result, traces: list[dict]) -> dict:
    final = np.asarray(result.final_activation, dtype=float)
    active_final = np.flatnonzero(final > 0).tolist()
    return {
        "source_nodes": result.source_nodes,
        "activated_nodes": result.activated_nodes,
        "activated_node_count": len(result.activated_nodes),
        "traversed_edges": [list(edge) for edge in result.traversed_edges],
        "traversed_edge_count": len(result.traversed_edges),
        "activation_history": result.activation_history,
        "steps_survived": max(0, len(result.activation_history) - 1),
        "final_active_nodes": active_final,
        "final_energy": float(final.sum()),
        "structural_assist_trace": traces,
        "assist_activations": sum(1 for x in traces if x.get("tie_gate_active")),
        "assist_rank_changes": sum(1 for x in traces if x.get("top_candidate_changed")),
    }


def run_once(player: str, other: str, assist: bool) -> dict:
    brain = copy.deepcopy(CORE)
    brain.set_structural_assist(assist)
    result = brain.propagate(
        raw_sources(player, other),
        steps=18,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )
    return summarize(result, brain.last_structural_assist_trace)


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def observe(player: str, other: str) -> dict:
    off1 = run_once(player, other, False)
    off2 = run_once(player, other, False)
    on = run_once(player, other, True)
    swapped = run_once(other, player, False)
    payload = {
        "experiment": "Core Growth Microscope v1",
        "world": {"P": player, "E": other},
        "input_contract": {
            "included": ["Pという入力チャネル", "Pの絶対位置", "Eという入力チャネル", "Eの絶対位置"],
            "excluded": ["相対方向", "距離", "正解行動", "移動可能性", "目的", "報酬", "教師", "行動候補", "最短経路"],
            "source_strength": 1.0,
        },
        "raw_ports": PORTS,
        "raw_source_nodes": raw_sources(player, other),
        "core_off": off1,
        "core_on_shadow": on,
        "controls": {
            "repeatability_jaccard": jaccard(off1["activated_nodes"], off2["activated_nodes"]),
            "assist_path_jaccard": jaccard(off1["activated_nodes"], on["activated_nodes"]),
            "role_swap_jaccard": jaccard(off1["activated_nodes"], swapped["activated_nodes"]),
            "off_repeat_exact": off1["activation_history"] == off2["activation_history"],
        },
        "brain_file_unchanged": BEFORE_HASH == sha(BRAIN_PATH),
        "decision": None,
        "movement": None,
        "learning": False,
        "noise": 0.0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_observation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Microscope v1</title><style>
:root{--bg:#09111e;--panel:#17253c;--panel2:#0d1828;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--blue:#8ed8ff;--orange:#ffad67;--green:#91efb0}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1250px;margin:auto;padding:34px 22px 70px}h1{font-size:clamp(34px,5vw,62px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.7}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cell{min-height:160px;background:#213653;border:1px solid #466486;border-radius:18px;padding:16px;text-align:center}.tokens{font-size:48px;font-weight:900;margin-top:20px}.p{color:var(--blue)}.e{color:var(--orange)}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;margin-top:18px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.contract{display:grid;grid-template-columns:1fr 1fr;gap:14px}.box{background:var(--panel2);border-radius:14px;padding:17px}.yes{color:var(--green)}.no{color:#ff9fa7}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);border-radius:14px;padding:16px}.metric b{display:block;font-size:25px;margin-top:6px}.raw{white-space:pre-wrap;max-height:430px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:800px){.controls,.contract,.metrics{grid-template-columns:1fr}.world{grid-template-columns:1fr}}
</style></head><body><main><h1>Core Growth Microscope</h1><p class="lead">答えを出させない。動かさない。PとEの存在と絶対位置だけをCoreへ入れ、Coreが何を形成できて、何が足りないかを観察する。</p><section class="panel"><h2>最小世界</h2><div id="world" class="world"></div><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="observeCore()">Coreを観察</button></div></section><section class="panel"><h2>入力契約</h2><div class="contract"><div class="box yes"><b>Coreへ入れる</b><div>Pチャネル / Pの絶対位置 / Eチャネル / Eの絶対位置</div></div><div class="box no"><b>Coreへ入れない</b><div>方向 / 距離 / 正解 / 行動 / 移動可能性 / 目的 / 報酬 / 教師 / 最短経路</div></div></div></section><section class="panel"><h2>観測結果</h2><div id="metrics" class="metrics"></div><h3>Core生データ</h3><pre id="raw" class="raw">まだ観察していません。</pre></section></main><script>
function draw(){const p=document.getElementById('p').value,e=document.getElementById('e').value;document.getElementById('world').innerHTML=['左','中央','右'].map(x=>`<div class="cell"><b>${x}</b><div class="tokens">${p===x?'<span class="p">P</span> ':''}${e===x?'<span class="e">E</span>':''}</div></div>`).join('')}document.getElementById('p').onchange=draw;document.getElementById('e').onchange=draw;draw();async function observeCore(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=`<div class="metric">活動Node<b>${d.core_off.activated_node_count}</b></div><div class="metric">通過Edge<b>${d.core_off.traversed_edge_count}</b></div><div class="metric">持続Step<b>${d.core_off.steps_survived}</b></div><div class="metric">再現性<b>${d.controls.repeatability_jaccard.toFixed(3)}</b></div><div class="metric">役割交換との重なり<b>${d.controls.role_swap_jaccard.toFixed(3)}</b></div><div class="metric">Assistとの重なり<b>${d.controls.assist_path_jaccard.toFixed(3)}</b></div><div class="metric">Assist作動<b>${d.core_on_shadow.assist_activations}</b></div><div class="metric">brain.json<b>${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/observe")
def api_observe():
    data = request.get_json(silent=True) or {}
    player = str(data.get("player", "左"))
    other = str(data.get("other", "右"))
    if player not in POSITIONS or other not in POSITIONS:
        return jsonify({"error": "位置が不正です。"}), 400
    return jsonify(observe(player, other))


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Microscope v1: http://{HOST}:{PORT}")
    print("No decision / no movement / no teacher / learning OFF / noise OFF")
    serve(app, host=HOST, port=PORT)
