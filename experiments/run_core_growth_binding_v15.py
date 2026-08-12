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
import run_core_growth_binding_v5 as v5

HOST = "127.0.0.1"
START_PORT = 5050
OUT = ROOT / "data" / "core_growth_binding_v15" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
ENTITY_STEPS = 8
POSITION_STEPS = 10
RESIDUAL_RATES = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
GAP_STEPS = [0, 1, 2, 3]


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


def propagate_state(brain, activation: np.ndarray, steps: int) -> dict:
    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed: set[tuple[int, int]] = set()
    history = [sorted(activated_nodes)]
    values = [{str(int(i)): float(activation[i]) for i in np.flatnonzero(activation > 0)}]

    for _ in range(steps):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break
        candidates: dict[int, tuple[float, int]] = {}
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            for idx in best_indices:
                target = int(neighbors[idx])
                signal = float(scores[idx]) * brain.signal_decay
                if signal < THRESHOLD:
                    continue
                previous = candidates.get(target)
                if previous is None or signal > previous[0]:
                    candidates[target] = (signal, int(source))
        if not candidates:
            break
        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining = max(0, brain.max_total_active_nodes - len(activated_nodes))
        selected = []
        new_count = 0
        for target, payload in ranked:
            is_new = target not in activated_nodes
            if is_new and new_count >= remaining:
                continue
            selected.append((target, payload))
            if is_new:
                new_count += 1
            if len(selected) >= min(brain.max_active_per_step, len(ranked)):
                break
        next_activation = np.zeros(brain.node_count, dtype=float)
        for target, (signal, source) in selected:
            next_activation[target] = max(next_activation[target], signal)
            traversed.add(edge_key(source, target))
        active_now = np.flatnonzero(next_activation > 0).tolist()
        if not active_now:
            break
        activated_nodes.update(active_now)
        history.append(active_now)
        values.append({str(int(i)): float(next_activation[i]) for i in active_now})
        activation = next_activation

    return {
        "final_activation": activation,
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(e) for e in sorted(traversed)],
        "history": history,
        "activation_values": values,
    }


def initial_sources(brain, nodes) -> np.ndarray:
    activation = np.zeros(brain.node_count, dtype=float)
    for node in nodes:
        activation[int(node)] = 1.0
    return activation


def natural_binding(entity: str, position: str, residual_rate: float, gap_steps: int) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    entity_state = propagate_state(brain, initial_sources(brain, v3.entity_nodes(entity)), ENTITY_STEPS)

    residual = np.asarray(entity_state["final_activation"], dtype=float).copy()
    residual *= float(residual_rate)
    for _ in range(gap_steps):
        residual *= float(residual_rate)
        residual[residual < 1e-12] = 0.0

    position_initial = residual.copy()
    for node in v3.position_nodes(position):
        position_initial[int(node)] = max(position_initial[int(node)], 1.0)

    position_state = propagate_state(brain, position_initial, POSITION_STEPS)
    all_nodes = set(entity_state["activated_nodes"]) | set(position_state["activated_nodes"])
    all_edges = edge_set(entity_state["traversed_edges"]) | edge_set(position_state["traversed_edges"])

    specific = set(v5.binding_components(v3.base.CORE, entity, position)["binding_only_edges"])
    replayed = all_edges & specific
    return {
        "entity": entity,
        "position": position,
        "residual_rate": residual_rate,
        "gap_steps": gap_steps,
        "residual_node_count": int(np.count_nonzero(residual > 0)),
        "residual_energy": float(residual.sum()),
        "residual_max": float(residual.max()) if residual.size else 0.0,
        "entity_state": {k: v for k, v in entity_state.items() if k != "final_activation"},
        "position_state": {k: v for k, v in position_state.items() if k != "final_activation"},
        "combined_node_count": len(all_nodes),
        "combined_edge_count": len(all_edges),
        "specific_edge_count": len(specific),
        "specific_edges_replayed": len(replayed),
        "specific_replay_ratio": 0.0 if not specific else len(replayed) / len(specific),
        "replayed_specific_edges": [list(e) for e in sorted(replayed)],
    }


def diagnose(entity: str, position: str) -> dict:
    baseline = natural_binding(entity, position, 0.0, 0)
    rows = []
    for gap in GAP_STEPS:
        for rate in RESIDUAL_RATES:
            row = natural_binding(entity, position, rate, gap)
            row["node_growth_vs_no_residual"] = row["combined_node_count"] - baseline["combined_node_count"]
            row["edge_growth_vs_no_residual"] = row["combined_edge_count"] - baseline["combined_edge_count"]
            rows.append(row)

    first_any = next((r for r in rows if r["specific_edges_replayed"] > 0), None)
    first_full = next((r for r in rows if r["specific_edge_count"] > 0 and r["specific_replay_ratio"] == 1.0), None)
    safe_full = next((r for r in rows if r["specific_edge_count"] > 0 and r["specific_replay_ratio"] == 1.0 and r["node_growth_vs_no_residual"] <= 10), None)
    return {
        "entity": entity,
        "position": position,
        "baseline": baseline,
        "first_any_replay": first_any,
        "first_full_replay": first_full,
        "first_safe_full_replay": safe_full,
        "rows": rows,
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v15",
        "world": {"P": player, "E": other},
        "purpose": "Test whether a generic decaying residual of the full Core activation state can replace hand-selected artificial echo nodes.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "hand_selected_echo_nodes": False,
            "residual_rule": "all previous final activations multiplied by one common residual rate",
            "core_file_modified": False,
        },
        "settings": {
            "threshold": THRESHOLD,
            "residual_rates": RESIDUAL_RATES,
            "gap_steps": GAP_STEPS,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v15.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v15</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:22px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:760px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v15</h1><p class="lead">特定Nodeを選ぶ人工echoを使わず、直前のCore全activationを一律に減衰させて次の入力へ残す。Binding経路の再生と活動の広がりを同時に測る。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">自然残響を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ検証していません。</pre></section></main><script>
function fmt(x){if(!x)return 'なし';return `rate ${Number(x.residual_rate).toFixed(2)} / gap ${x.gap_steps} / Node増 ${x.node_growth_vs_no_residual}`};async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;document.getElementById('metrics').innerHTML=`<div class="metric">P初回再生<b class="${P.first_any_replay?'good':'warn'}">${fmt(P.first_any_replay)}</b></div><div class="metric">P完全再生<b class="${P.first_full_replay?'good':'warn'}">${fmt(P.first_full_replay)}</b></div><div class="metric">P安全完全再生<b class="${P.first_safe_full_replay?'good':'warn'}">${fmt(P.first_safe_full_replay)}</b></div><div class="metric">P基準Node<b>${P.baseline.combined_node_count}</b></div><div class="metric">E初回再生<b class="${E.first_any_replay?'good':'warn'}">${fmt(E.first_any_replay)}</b></div><div class="metric">E完全再生<b class="${E.first_full_replay?'good':'warn'}">${fmt(E.first_full_replay)}</b></div><div class="metric">E安全完全再生<b class="${E.first_safe_full_replay?'good':'warn'}">${fmt(E.first_safe_full_replay)}</b></div><div class="metric">E基準Node<b>${E.baseline.combined_node_count}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v15: http://{HOST}:{PORT}")
    print("Generic decaying residual activation / no hand-selected echo / no learning")
    serve(app, host=HOST, port=PORT)
