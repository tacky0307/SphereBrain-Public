from __future__ import annotations

import copy
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

import run_core_growth_microscope_v1 as base

HOST = "127.0.0.1"
PORT = 5035
OUT = ROOT / "data" / "core_growth_binding_v1" / "results"
POSITIONS = base.POSITIONS


def summarize(result, traces: list[dict]) -> dict:
    final = np.asarray(result.final_activation, dtype=float)
    return {
        "source_nodes": list(result.source_nodes),
        "activated_nodes": list(result.activated_nodes),
        "activated_node_count": len(result.activated_nodes),
        "traversed_edges": [list(edge) for edge in result.traversed_edges],
        "traversed_edge_count": len(result.traversed_edges),
        "activation_history": [list(step) for step in result.activation_history],
        "steps_survived": max(0, len(result.activation_history) - 1),
        "final_active_nodes": np.flatnonzero(final > 0).tolist(),
        "final_energy": float(final.sum()),
        "assist_activations": sum(1 for x in traces if x.get("tie_gate_active")),
        "assist_rank_changes": sum(1 for x in traces if x.get("top_candidate_changed")),
        "structural_assist_trace": traces,
    }


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def edge_jaccard(a: list[list[int]], b: list[list[int]]) -> float:
    sa = {tuple(sorted(x)) for x in a}
    sb = {tuple(sorted(x)) for x in b}
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def propagate(brain, sources, *, context=None, steps=10):
    result = brain.propagate(
        sources,
        context_nodes=context or None,
        steps=steps,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )
    return summarize(result, list(brain.last_structural_assist_trace))


def entity_nodes(entity: str) -> list[int]:
    return base.PORTS[f"entity:{entity}"]


def position_nodes(position: str) -> list[int]:
    return base.PORTS[f"position:{position}"]


def simultaneous(player: str, other: str) -> dict:
    brain = copy.deepcopy(base.CORE)
    brain.set_structural_assist(False)
    result = propagate(
        brain,
        entity_nodes("P") + position_nodes(player) + entity_nodes("E") + position_nodes(other),
        steps=18,
    )
    return {"mode": "simultaneous_set", "result": result}


def sequential_no_echo(player: str, other: str) -> dict:
    brain = copy.deepcopy(base.CORE)
    brain.set_structural_assist(False)
    stages = [
        ("P", entity_nodes("P")),
        (player, position_nodes(player)),
        ("BOUNDARY", []),
        ("E", entity_nodes("E")),
        (other, position_nodes(other)),
    ]
    outputs = []
    all_nodes: set[int] = set()
    all_edges: set[tuple[int, int]] = set()
    for label, sources in stages:
        if label == "BOUNDARY":
            outputs.append({"label": label, "cleared": True})
            continue
        stage = propagate(brain, sources, steps=8)
        outputs.append({"label": label, "result": stage})
        all_nodes.update(stage["activated_nodes"])
        all_edges.update(tuple(sorted(edge)) for edge in stage["traversed_edges"])
    return {
        "mode": "sequential_no_echo",
        "stages": outputs,
        "combined_activated_nodes": sorted(all_nodes),
        "combined_traversed_edges": [list(edge) for edge in sorted(all_edges)],
    }


def binding_window(player: str, other: str) -> dict:
    brain = copy.deepcopy(base.CORE)
    brain.set_structural_assist(False)

    # Window A: P -> player position. P's final activity remains as short-term context.
    p_stage = propagate(brain, entity_nodes("P"), steps=8)
    p_echo = p_stage["final_active_nodes"]
    p_pos_stage = propagate(brain, position_nodes(player), context=p_echo, steps=10)

    # Boundary: discard all Window A context before opening Window B.
    boundary = {
        "type": "hard_boundary",
        "cleared_context_nodes": p_echo,
        "next_context_nodes": [],
    }

    # Window B: E -> other position, independently from Window A.
    e_stage = propagate(brain, entity_nodes("E"), steps=8)
    e_echo = e_stage["final_active_nodes"]
    e_pos_stage = propagate(brain, position_nodes(other), context=e_echo, steps=10)

    window_a_nodes = sorted(set(p_stage["activated_nodes"]) | set(p_pos_stage["activated_nodes"]))
    window_b_nodes = sorted(set(e_stage["activated_nodes"]) | set(e_pos_stage["activated_nodes"]))
    window_a_edges = sorted({tuple(sorted(x)) for x in p_stage["traversed_edges"] + p_pos_stage["traversed_edges"]})
    window_b_edges = sorted({tuple(sorted(x)) for x in e_stage["traversed_edges"] + e_pos_stage["traversed_edges"]})

    return {
        "mode": "binding_window",
        "window_a": {
            "sequence": ["P", player],
            "entity_stage": p_stage,
            "echo_nodes": p_echo,
            "position_stage": p_pos_stage,
            "combined_activated_nodes": window_a_nodes,
            "combined_traversed_edges": [list(x) for x in window_a_edges],
        },
        "boundary": boundary,
        "window_b": {
            "sequence": ["E", other],
            "entity_stage": e_stage,
            "echo_nodes": e_echo,
            "position_stage": e_pos_stage,
            "combined_activated_nodes": window_b_nodes,
            "combined_traversed_edges": [list(x) for x in window_b_edges],
        },
        "combined_activated_nodes": sorted(set(window_a_nodes) | set(window_b_nodes)),
        "combined_traversed_edges": [list(x) for x in sorted(set(window_a_edges) | set(window_b_edges))],
        "cross_window_node_overlap": jaccard(window_a_nodes, window_b_nodes),
    }


def observe(player: str, other: str) -> dict:
    swapped_player, swapped_other = other, player

    current_sim = simultaneous(player, other)
    swapped_sim = simultaneous(swapped_player, swapped_other)
    current_seq = sequential_no_echo(player, other)
    swapped_seq = sequential_no_echo(swapped_player, swapped_other)
    current_bind = binding_window(player, other)
    swapped_bind = binding_window(swapped_player, swapped_other)
    repeat_bind = binding_window(player, other)

    payload = {
        "experiment": "Core Growth Binding v1",
        "world": {"P": player, "E": other},
        "swapped_world": {"P": swapped_player, "E": swapped_other},
        "input_contract": {
            "included": ["P identity", "E identity", "absolute position", "input order", "short-term echo", "hard episode boundary"],
            "excluded": ["relative direction", "distance", "correct action", "goal", "reward", "teacher", "movement", "shortest path"],
            "learning": False,
            "noise": 0.0,
        },
        "simultaneous": current_sim,
        "sequential_no_echo": current_seq,
        "binding_window": current_bind,
        "controls": {
            "simultaneous_role_swap_node_jaccard": jaccard(
                current_sim["result"]["activated_nodes"], swapped_sim["result"]["activated_nodes"]
            ),
            "sequential_role_swap_node_jaccard": jaccard(
                current_seq["combined_activated_nodes"], swapped_seq["combined_activated_nodes"]
            ),
            "binding_role_swap_node_jaccard": jaccard(
                current_bind["combined_activated_nodes"], swapped_bind["combined_activated_nodes"]
            ),
            "binding_role_swap_edge_jaccard": edge_jaccard(
                current_bind["combined_traversed_edges"], swapped_bind["combined_traversed_edges"]
            ),
            "binding_repeat_node_jaccard": jaccard(
                current_bind["combined_activated_nodes"], repeat_bind["combined_activated_nodes"]
            ),
            "binding_repeat_edge_jaccard": edge_jaccard(
                current_bind["combined_traversed_edges"], repeat_bind["combined_traversed_edges"]
            ),
            "brain_file_unchanged": base.BEFORE_HASH == base.sha(base.BRAIN_PATH),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_observation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v1</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--blue:#8ed8ff;--orange:#ffad67;--green:#91efb0;--purple:#c6a7ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1320px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.7}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cell{min-height:145px;background:#213653;border:1px solid #466486;border-radius:18px;padding:16px;text-align:center}.tokens{font-size:48px;font-weight:900;margin-top:18px}.p{color:var(--blue)}.e{color:var(--orange)}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;margin-top:18px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.flow{font-family:ui-monospace,Consolas,monospace;background:var(--panel2);padding:18px;border-radius:14px;font-size:20px;color:var(--purple)}.methods{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.method{background:var(--panel2);padding:18px;border-radius:15px}.method b{display:block;font-size:22px;margin-bottom:8px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:27px;margin-top:6px}.good{color:var(--green)}.raw{white-space:pre-wrap;max-height:480px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:850px){.controls,.methods,.metrics{grid-template-columns:1fr}.world{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding</h1><p class="lead">同じ事実を、同時集合・時間順・残響＋時間窓＋境界の3方式でCoreへ渡し、Pと位置／Eと位置を区別できるか比較する。正解・行動・報酬・教師は一切ない。</p><section class="panel"><h2>最小世界</h2><div id="world" class="world"></div><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="observeCore()">3方式を比較</button></div></section><section class="panel"><h2>Binding Window</h2><div id="flow" class="flow">P → 左 ｜ E → 右</div><p class="lead">Pの最終活動を短期contextとして位置入力へ残し、境界で完全に消去してからE側を開始する。</p></section><section class="panel"><h2>比較方式</h2><div class="methods"><div class="method"><b>① 同時集合</b>P + 左 + E + 右</div><div class="method"><b>② 時間順・残響なし</b>P → 左 ｜ E → 右<br>各入力は独立</div><div class="method"><b>③ Binding Window</b>P → 左 ｜ E → 右<br>窓内だけ残響あり</div></div></section><section class="panel"><h2>役割交換との重なり</h2><div id="metrics" class="metrics"></div><h3>Core生データ</h3><pre id="raw" class="raw">まだ比較していません。</pre></section></main><script>
function draw(){const p=document.getElementById('p').value,e=document.getElementById('e').value;document.getElementById('world').innerHTML=['左','中央','右'].map(x=>`<div class="cell"><b>${x}</b><div class="tokens">${p===x?'<span class="p">P</span> ':''}${e===x?'<span class="e">E</span>':''}</div></div>`).join('');document.getElementById('flow').textContent=`P → ${p} ｜ E → ${e}`}document.getElementById('p').onchange=draw;document.getElementById('e').onchange=draw;draw();async function observeCore(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const c=d.controls;document.getElementById('metrics').innerHTML=`<div class="metric">同時集合<b>${c.simultaneous_role_swap_node_jaccard.toFixed(3)}</b></div><div class="metric">時間順・残響なし<b>${c.sequential_role_swap_node_jaccard.toFixed(3)}</b></div><div class="metric">Binding Node<b class="good">${c.binding_role_swap_node_jaccard.toFixed(3)}</b></div><div class="metric">Binding Edge<b class="good">${c.binding_role_swap_edge_jaccard.toFixed(3)}</b></div><div class="metric">Binding再現 Node<b>${c.binding_repeat_node_jaccard.toFixed(3)}</b></div><div class="metric">Binding再現 Edge<b>${c.binding_repeat_edge_jaccard.toFixed(3)}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v1: http://{HOST}:{PORT}")
    print("Simultaneous vs sequential vs binding window / learning OFF / noise OFF")
    serve(app, host=HOST, port=PORT)
