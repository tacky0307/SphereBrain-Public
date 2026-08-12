from __future__ import annotations

import copy
import json
import socket
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

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v12 as v12
import run_core_growth_binding_v15 as v15
import run_core_growth_binding_v17 as v17

HOST = "127.0.0.1"
START_PORT = 5053
OUT = ROOT / "data" / "core_growth_binding_v18" / "results"
POSITIONS = v3.POSITIONS
WINDOW = 4
DECAY = 0.95
GAP = 0
THRESHOLD = 0.18


def choose_port(start: int) -> int:
    for port in range(start, start + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("利用可能なローカルポートが見つかりません。")


PORT = choose_port(START_PORT)


def edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def edge_set(edges) -> set[tuple[int, int]]:
    return {edge_key(a, b) for a, b in edges}


def trace_from_snapshots(values: list[dict], node_count: int, mode: str, *, remove_step_age: int | None = None, remove_node: int | None = None) -> np.ndarray:
    result = np.zeros(node_count, dtype=float)
    recent = values[-WINDOW:]
    for age, snapshot in enumerate(reversed(recent)):
        if remove_step_age is not None and age == remove_step_age:
            continue
        factor = DECAY ** (age + GAP + 1)
        for node_text, value in snapshot.items():
            node = int(node_text)
            if remove_node is not None and node == remove_node:
                continue
            contribution = float(value) * factor
            if mode == "max":
                result[node] = max(result[node], contribution)
            else:
                result[node] = min(1.0, result[node] + contribution)
    return result


def run_position(entity: str, position: str, trace: np.ndarray, reference: dict) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    initial = trace.copy()
    for node in v3.position_nodes(position):
        initial[int(node)] = max(initial[int(node)], 1.0)
    state = v15.propagate_state(brain, initial, v15.POSITION_STEPS)
    traversed = edge_set(state["traversed_edges"])
    chain = reference.get("chain", [])
    replayed = [tuple(item["edge"]) for item in chain if tuple(item["edge"]) in traversed]
    return {
        "replayed_edge_count": len(replayed),
        "chain_edge_count": len(chain),
        "replay_ratio": 0.0 if not chain else len(replayed) / len(chain),
        "replayed_edges": [list(x) for x in replayed],
        "activated_node_count": len(state["activated_nodes"]),
        "traversed_edge_count": len(state["traversed_edges"]),
    }


def analyze_mode(entity: str, position: str, mode: str) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    entity_state = v15.propagate_state(brain, v15.initial_sources(brain, v3.entity_nodes(entity)), v15.ENTITY_STEPS)
    values = entity_state["activation_values"]
    reference = v12.binding_reference(entity, position)
    baseline_trace = trace_from_snapshots(values, brain.node_count, mode)
    baseline = run_position(entity, position, baseline_trace, reference)

    step_ablation = []
    for age in range(min(WINDOW, len(values))):
        trace = trace_from_snapshots(values, brain.node_count, mode, remove_step_age=age)
        result = run_position(entity, position, trace, reference)
        step_ablation.append({
            "removed_age": age,
            "removed_label": f"{age} step ago" if age else "latest step",
            "replay_ratio": result["replay_ratio"],
            "replay_drop": baseline["replay_ratio"] - result["replay_ratio"],
            "entry_activation": 0.0 if reference.get("entry") is None else float(trace[int(reference["entry"]["source"])]),
            "result": result,
        })

    active_nodes = np.flatnonzero(baseline_trace > 0).tolist()
    node_ablation = []
    for node in active_nodes:
        trace = trace_from_snapshots(values, brain.node_count, mode, remove_node=int(node))
        result = run_position(entity, position, trace, reference)
        node_ablation.append({
            "node": int(node),
            "trace_value": float(baseline_trace[node]),
            "replay_ratio": result["replay_ratio"],
            "replay_drop": baseline["replay_ratio"] - result["replay_ratio"],
            "result": result,
        })
    node_ablation.sort(key=lambda row: (row["replay_drop"], row["trace_value"]), reverse=True)

    essential_nodes = [row for row in node_ablation if row["replay_drop"] > 0]
    redundant_nodes = [row for row in node_ablation if row["replay_drop"] == 0]
    essential_steps = [row for row in step_ablation if row["replay_drop"] > 0]

    entry_source = None if reference.get("entry") is None else int(reference["entry"]["source"])
    return {
        "mode": mode,
        "settings": {"window": WINDOW, "decay": DECAY, "gap": GAP},
        "baseline": baseline,
        "trace_node_count": len(active_nodes),
        "trace_energy": float(baseline_trace.sum()),
        "entry_source": entry_source,
        "entry_trace_value": 0.0 if entry_source is None else float(baseline_trace[entry_source]),
        "step_ablation": step_ablation,
        "essential_steps": essential_steps,
        "node_ablation": node_ablation,
        "essential_nodes": essential_nodes,
        "redundant_node_count": len(redundant_nodes),
        "top_essential_nodes": essential_nodes[:20],
    }


def diagnose(entity: str, position: str) -> dict:
    return {
        "entity": entity,
        "position": position,
        "max_mode": analyze_mode(entity, position, "max"),
        "capped_sum_mode": analyze_mode(entity, position, "capped_sum"),
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v18",
        "world": {"P": player, "E": other},
        "purpose": "Ablate temporal-trace steps and nodes to identify which recent activity components are necessary for Binding-chain replay.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "hand_selected_trace_nodes": False,
            "core_file_modified": False,
        },
        "diagnostics": {"P": diagnose("P", player), "E": diagnose("E", other)},
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v18.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v18</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:780px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v18</h1><p class="lead">Eの成功した短期Traceを、Step単位・Node単位で外し、再生に必要な成分と冗長な成分を分解する。Pにも同じ診断を適用する。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Traceを分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ分解していません。</pre></section></main><script>
function cards(label,d){const m=d.max_mode,s=d.capped_sum_mode;return `<div class="metric">${label} max再生率<b>${m.baseline.replay_ratio.toFixed(3)}</b></div><div class="metric">${label} max必須Step<b>${m.essential_steps.length}</b></div><div class="metric">${label} max必須Node<b>${m.essential_nodes.length}</b></div><div class="metric">${label} max冗長Node<b>${m.redundant_node_count}</b></div><div class="metric">${label} sum再生率<b>${s.baseline.replay_ratio.toFixed(3)}</b></div><div class="metric">${label} sum必須Step<b>${s.essential_steps.length}</b></div><div class="metric">${label} sum必須Node<b class="${s.essential_nodes.length?'good':'warn'}">${s.essential_nodes.length}</b></div><div class="metric">${label} sum冗長Node<b>${s.redundant_node_count}</b></div>`}
async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=cards('P',d.diagnostics.P)+cards('E',d.diagnostics.E)+`<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v18: http://{HOST}:{PORT}")
    print("Temporal trace step/node ablation / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
