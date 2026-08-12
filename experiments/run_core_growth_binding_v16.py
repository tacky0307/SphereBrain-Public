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

HOST = "127.0.0.1"
START_PORT = 5051
OUT = ROOT / "data" / "core_growth_binding_v16" / "results"
POSITIONS = v3.POSITIONS
REQUIRED_ENTRY = 0.35


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


def inspect_condition(entity: str, position: str, rate: float, gap: int, entry: dict, chain: list[dict]) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    entity_state = v15.propagate_state(
        brain,
        v15.initial_sources(brain, v3.entity_nodes(entity)),
        v15.ENTITY_STEPS,
    )

    residual = np.asarray(entity_state["final_activation"], dtype=float).copy()
    residual *= float(rate)
    for _ in range(gap):
        residual *= float(rate)
        residual[residual < 1e-12] = 0.0

    entry_source = int(entry["source"])
    entry_residual = float(residual[entry_source])
    total_energy = float(residual.sum())
    active_count = int(np.count_nonzero(residual > 0))
    concentration = 0.0 if total_energy <= 0 else entry_residual / total_energy

    position_initial = residual.copy()
    for node in v3.position_nodes(position):
        position_initial[int(node)] = max(position_initial[int(node)], 1.0)

    position_state = v15.propagate_state(brain, position_initial, v15.POSITION_STEPS)
    traversed = edge_set(position_state["traversed_edges"])
    chain_edges = [tuple(item["edge"]) for item in chain]
    replayed = sum(1 for edge in chain_edges if edge in traversed)

    max_entry_seen = entry_residual
    max_entry_step = -1
    for step_index, values in enumerate(position_state["activation_values"]):
        value = float(values.get(str(entry_source), 0.0))
        if value > max_entry_seen:
            max_entry_seen = value
            max_entry_step = step_index

    return {
        "residual_rate": rate,
        "gap_steps": gap,
        "entry_source": entry_source,
        "entry_residual_before_position": entry_residual,
        "entry_gap_to_required": max(0.0, REQUIRED_ENTRY - entry_residual),
        "entry_reaches_required": entry_residual >= REQUIRED_ENTRY,
        "total_residual_energy": total_energy,
        "residual_active_node_count": active_count,
        "entry_energy_concentration": concentration,
        "max_entry_activation_during_position": max_entry_seen,
        "max_entry_activation_step": max_entry_step,
        "chain_edge_count": len(chain_edges),
        "replayed_chain_edge_count": replayed,
        "chain_replay_ratio": 0.0 if not chain_edges else replayed / len(chain_edges),
        "combined_node_count": len(set(entity_state["activated_nodes"]) | set(position_state["activated_nodes"])),
        "combined_edge_count": len(edge_set(entity_state["traversed_edges"]) | traversed),
    }


def diagnose(entity: str, position: str) -> dict:
    reference = v12.binding_reference(entity, position)
    entry = reference.get("entry")
    if entry is None:
        return {"entity": entity, "position": position, "entry": None, "rows": []}

    rows = []
    for gap in v15.GAP_STEPS:
        for rate in v15.RESIDUAL_RATES:
            rows.append(inspect_condition(entity, position, rate, gap, entry, reference["chain"]))

    best_entry = max(rows, key=lambda row: row["entry_residual_before_position"], default=None)
    best_concentration = max(rows, key=lambda row: row["entry_energy_concentration"], default=None)
    first_required = next((row for row in rows if row["entry_reaches_required"]), None)
    first_replay = next((row for row in rows if row["replayed_chain_edge_count"] > 0), None)

    return {
        "entity": entity,
        "position": position,
        "entry": entry,
        "required_entry_activation": REQUIRED_ENTRY,
        "binding_chain_edge_count": len(reference["chain"]),
        "best_entry_condition": best_entry,
        "best_concentration_condition": best_concentration,
        "first_required_entry_condition": first_required,
        "first_chain_replay_condition": first_replay,
        "rows": rows,
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v16",
        "world": {"P": player, "E": other},
        "purpose": "Directly measure how much generic natural residual activation remains at each Binding entry node.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "hand_selected_echo_nodes": False,
            "core_behavior_changed": False,
        },
        "settings": {
            "required_entry_activation": REQUIRED_ENTRY,
            "residual_rates": v15.RESIDUAL_RATES,
            "gap_steps": v15.GAP_STEPS,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v16.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v16</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:22px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:760px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v16</h1><p class="lead">自然残響の各条件で、Binding入口Nodeに実際どれだけactivationが残るか、全残響エネルギーの何割が入口へ集中するかを直接測る。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">入口残響を測る</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ測定していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function cond(x){if(!x)return 'なし';return `rate ${Number(x.residual_rate).toFixed(2)} / gap ${x.gap_steps}`}function cards(label,d){if(!d.entry)return `<div class="metric">${label}<b>入口なし</b></div>`;const b=d.best_entry_condition,c=d.best_concentration_condition;return `<div class="metric">${label}最大入口残響<b class="${b&&b.entry_reaches_required?'good':'warn'}">${b?f(b.entry_residual_before_position):'なし'}</b></div><div class="metric">${label}最大入口条件<b>${cond(b)}</b></div><div class="metric">${label}その時の総残響<b>${b?f(b.total_residual_energy):'なし'}</b></div><div class="metric">${label}入口集中率<b class="blue">${b?f(b.entry_energy_concentration):'なし'}</b></div><div class="metric">${label}最高集中条件<b>${cond(c)}</b></div><div class="metric">${label}0.35到達<b class="${d.first_required_entry_condition?'good':'warn'}">${cond(d.first_required_entry_condition)}</b></div><div class="metric">${label}初回連鎖再生<b class="${d.first_chain_replay_condition?'good':'warn'}">${cond(d.first_chain_replay_condition)}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=cards('P',d.diagnostics.P)+cards('E',d.diagnostics.E)+`<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v16: http://{HOST}:{PORT}")
    print("Direct natural-residual entry measurement / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
