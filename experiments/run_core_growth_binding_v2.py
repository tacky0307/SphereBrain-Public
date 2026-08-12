from __future__ import annotations

import copy
import json
import sys
import threading
import types
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
PORT = 5036
OUT = ROOT / "data" / "core_growth_binding_v2" / "results"
POSITIONS = base.POSITIONS
ECHO_STRENGTH = 0.68
ECHO_NODE_LIMIT = 8


def clone_core(*, assist: bool, echo_strength: float = ECHO_STRENGTH):
    brain = copy.deepcopy(base.CORE)
    brain.set_structural_assist(assist)

    def initial_activation(self, source_nodes, context_nodes):
        sources = list(source_nodes)
        activation = np.zeros(self.node_count, dtype=float)
        for node in sources:
            activation[node] = 1.0
        if context_nodes:
            for node in context_nodes:
                activation[node] = max(activation[node], echo_strength)
        return sources, activation

    brain._initial_activation = types.MethodType(initial_activation, brain)
    return brain


def entity_nodes(entity: str) -> list[int]:
    return list(base.PORTS[f"entity:{entity}"])


def position_nodes(position: str) -> list[int]:
    return list(base.PORTS[f"position:{position}"])


def edge_set(result) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in result.traversed_edges}


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


def propagate(brain, sources, *, context=None, steps=10):
    result = brain.propagate(
        sources,
        context_nodes=context or None,
        steps=steps,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )
    return result, summarize(result, list(brain.last_structural_assist_trace))


def top_echo_nodes(result, limit: int = ECHO_NODE_LIMIT) -> list[int]:
    final = np.asarray(result.final_activation, dtype=float)
    active = np.flatnonzero(final > 0)
    if active.size:
        order = active[np.argsort(-final[active])]
        return [int(x) for x in order[:limit]]
    history = list(result.activation_history or [])
    return [int(x) for x in (history[-1] if history else [])[:limit]]


def jaccard(values_a, values_b) -> float:
    a, b = set(values_a), set(values_b)
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def build_binding(entity: str, position: str, *, assist: bool) -> dict:
    # Entity activity creates a short-lived echo.
    entity_brain = clone_core(assist=assist)
    entity_result, entity_summary = propagate(entity_brain, entity_nodes(entity), steps=8)
    echo_nodes = top_echo_nodes(entity_result)

    # Position-only control: what this position produces without the entity echo.
    alone_brain = clone_core(assist=assist)
    alone_result, alone_summary = propagate(alone_brain, position_nodes(position), steps=10)

    # Bound condition: same position input while the entity echo is still active.
    bound_brain = clone_core(assist=assist)
    bound_result, bound_summary = propagate(
        bound_brain,
        position_nodes(position),
        context=echo_nodes,
        steps=10,
    )

    entity_active = set(entity_result.activated_nodes)
    alone_active = set(alone_result.activated_nodes)
    bound_active = set(bound_result.activated_nodes)
    alone_edges = edge_set(alone_result)
    bound_edges = edge_set(bound_result)
    echo_set = set(echo_nodes)

    interaction_nodes = sorted(bound_active - alone_active)
    suppressed_nodes = sorted(alone_active - bound_active)
    shared_with_entity = sorted(bound_active & entity_active)
    interaction_edges = sorted(bound_edges - alone_edges)
    suppressed_edges = sorted(alone_edges - bound_edges)
    echo_origin_edges = sorted(edge for edge in bound_edges if edge[0] in echo_set or edge[1] in echo_set)

    return {
        "binding_id": f"{entity}@{position}",
        "entity": entity,
        "position": position,
        "echo_strength": ECHO_STRENGTH,
        "echo_nodes": echo_nodes,
        "entity_stage": entity_summary,
        "position_alone_control": alone_summary,
        "position_with_echo": bound_summary,
        "temporary_binding_state": {
            "shared_with_entity_nodes": shared_with_entity,
            "interaction_nodes_added_by_echo": interaction_nodes,
            "nodes_suppressed_by_echo": suppressed_nodes,
            "interaction_edges_added_by_echo": [list(x) for x in interaction_edges],
            "edges_suppressed_by_echo": [list(x) for x in suppressed_edges],
            "echo_origin_edges": [list(x) for x in echo_origin_edges],
            "echo_changed_path": bool(interaction_nodes or suppressed_nodes or interaction_edges or suppressed_edges),
        },
    }


def run_scene(player: str, other: str, *, assist: bool) -> dict:
    # Hard boundary is represented by two independent binding builds.
    p_binding = build_binding("P", player, assist=assist)
    e_binding = build_binding("E", other, assist=assist)
    return {
        "scene": {"P": player, "E": other},
        "assist_enabled": assist,
        "bindings": [p_binding, e_binding],
        "boundary": {
            "type": "hard_boundary",
            "window_a": p_binding["binding_id"],
            "window_b": e_binding["binding_id"],
            "carried_context_nodes": [],
        },
        "assist_activity": {
            "tie_gate_activations": sum(
                item["position_with_echo"]["assist_activations"] for item in (p_binding, e_binding)
            ),
            "rank_changes": sum(
                item["position_with_echo"]["assist_rank_changes"] for item in (p_binding, e_binding)
            ),
        },
    }


def binding_compare(scene_a: dict, scene_b: dict) -> dict:
    comparisons = []
    for bind_a, bind_b in zip(scene_a["bindings"], scene_b["bindings"]):
        state_a = bind_a["temporary_binding_state"]
        state_b = bind_b["temporary_binding_state"]
        comparisons.append({
            "role": bind_a["entity"],
            "binding_a": bind_a["binding_id"],
            "binding_b": bind_b["binding_id"],
            "bound_node_jaccard": jaccard(
                bind_a["position_with_echo"]["activated_nodes"],
                bind_b["position_with_echo"]["activated_nodes"],
            ),
            "bound_edge_jaccard": jaccard(
                [tuple(x) for x in bind_a["position_with_echo"]["traversed_edges"]],
                [tuple(x) for x in bind_b["position_with_echo"]["traversed_edges"]],
            ),
            "interaction_node_jaccard": jaccard(
                state_a["interaction_nodes_added_by_echo"],
                state_b["interaction_nodes_added_by_echo"],
            ),
            "interaction_edge_jaccard": jaccard(
                [tuple(x) for x in state_a["interaction_edges_added_by_echo"]],
                [tuple(x) for x in state_b["interaction_edges_added_by_echo"]],
            ),
        })
    return {
        "per_binding": comparisons,
        "mean_bound_node_jaccard": average([x["bound_node_jaccard"] for x in comparisons]),
        "mean_bound_edge_jaccard": average([x["bound_edge_jaccard"] for x in comparisons]),
        "mean_interaction_node_jaccard": average([x["interaction_node_jaccard"] for x in comparisons]),
        "mean_interaction_edge_jaccard": average([x["interaction_edge_jaccard"] for x in comparisons]),
    }


def observe(player: str, other: str) -> dict:
    current_off = run_scene(player, other, assist=False)
    swapped_off = run_scene(other, player, assist=False)
    repeated_off = run_scene(player, other, assist=False)
    current_on = run_scene(player, other, assist=True)

    role_swap = binding_compare(current_off, swapped_off)
    repeatability = binding_compare(current_off, repeated_off)
    assist_effect = binding_compare(current_off, current_on)

    payload = {
        "experiment": "Core Growth Binding v2",
        "world": {"P": player, "E": other},
        "swapped_world": {"P": other, "E": player},
        "input_contract": {
            "included": [
                "entity identity",
                "absolute position",
                "short-term echo",
                "binding window",
                "hard boundary",
                "temporary binding state",
            ],
            "excluded": [
                "relative direction",
                "distance",
                "correct action",
                "movement",
                "goal",
                "reward",
                "teacher",
                "shortest path",
            ],
            "learning": False,
            "noise": 0.0,
            "echo_strength": ECHO_STRENGTH,
        },
        "current_off": current_off,
        "swapped_off": swapped_off,
        "current_on_shadow": current_on,
        "controls": {
            "role_swap": role_swap,
            "repeatability": repeatability,
            "assist_effect": assist_effect,
            "brain_file_unchanged": base.BEFORE_HASH == base.sha(base.BRAIN_PATH),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_observation_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v2</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--blue:#8ed8ff;--orange:#ffad67;--green:#91efb0;--purple:#c6a7ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1320px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.7}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cell{min-height:145px;background:#213653;border:1px solid #466486;border-radius:18px;padding:16px;text-align:center}.tokens{font-size:48px;font-weight:900;margin-top:18px}.p{color:var(--blue)}.e{color:var(--orange)}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;margin-top:18px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.flow{font-family:ui-monospace,Consolas,monospace;background:var(--panel2);padding:18px;border-radius:14px;font-size:20px;color:var(--purple)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:27px;margin-top:6px}.raw{white-space:pre-wrap;max-height:520px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}.note{background:var(--panel2);padding:16px;border-radius:14px;color:var(--muted)}@media(max-width:850px){.controls,.metrics{grid-template-columns:1fr}.world{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v2</h1><p class="lead">P側とE側を別々の一時Bindingとして保持し、残響が位置単独経路を本当に変えたかを見る。Structural Assistは現行実装のままShadow比較する。</p><section class="panel"><h2>最小世界</h2><div id="world" class="world"></div><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="observeCore()">一時結合を観察</button></div></section><section class="panel"><h2>一時結合</h2><div id="flow" class="flow">Binding A: P → 左　｜　Binding B: E → 右</div><p class="lead">残響強度 0.68。各Bindingで「位置単独」と「対象残響あり」を比較し、追加・抑制されたNode/Edgeを保持する。境界では前Bindingのcontextを次へ渡さない。</p></section><section class="panel"><h2>役割交換との比較</h2><div id="metrics" class="metrics"></div><div class="note">値が1.000なら同じ。1.000より下がれば、P左/E右とP右/E左の一時結合状態を区別した可能性がある。再現性は1.000が理想。</div><h3>Core生データ</h3><pre id="raw" class="raw">まだ観察していません。</pre></section></main><script>
function draw(){const p=document.getElementById('p').value,e=document.getElementById('e').value;document.getElementById('world').innerHTML=['左','中央','右'].map(x=>`<div class="cell"><b>${x}</b><div class="tokens">${p===x?'<span class="p">P</span> ':''}${e===x?'<span class="e">E</span>':''}</div></div>`).join('');document.getElementById('flow').textContent=`Binding A: P → ${p}　｜　Binding B: E → ${e}`}document.getElementById('p').onchange=draw;document.getElementById('e').onchange=draw;draw();async function observeCore(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const rs=d.controls.role_swap,rep=d.controls.repeatability,ae=d.controls.assist_effect;document.getElementById('metrics').innerHTML=`<div class="metric">役割交換 Node<b>${rs.mean_bound_node_jaccard.toFixed(3)}</b></div><div class="metric">役割交換 Edge<b>${rs.mean_bound_edge_jaccard.toFixed(3)}</b></div><div class="metric">相互作用Node<b>${rs.mean_interaction_node_jaccard.toFixed(3)}</b></div><div class="metric">相互作用Edge<b>${rs.mean_interaction_edge_jaccard.toFixed(3)}</b></div><div class="metric">再現 Node<b>${rep.mean_bound_node_jaccard.toFixed(3)}</b></div><div class="metric">再現 Edge<b>${rep.mean_bound_edge_jaccard.toFixed(3)}</b></div><div class="metric">Assist差 Node<b>${ae.mean_bound_node_jaccard.toFixed(3)}</b></div><div class="metric">Assist順位変更<b>${d.current_on_shadow.assist_activity.rank_changes}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v2: http://{HOST}:{PORT}")
    print("Temporary bindings / current Structural Assist shadow / learning OFF / noise OFF")
    serve(app, host=HOST, port=PORT)
