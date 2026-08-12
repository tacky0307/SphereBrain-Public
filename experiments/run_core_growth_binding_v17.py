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
import run_core_growth_binding_v12 as v12
import run_core_growth_binding_v15 as v15

HOST = "127.0.0.1"
START_PORT = 5052
OUT = ROOT / "data" / "core_growth_binding_v17" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
REQUIRED_ENTRY = 0.35
WINDOWS = [2, 4, 6, 8]
DECAYS = [0.50, 0.70, 0.85, 0.95]
MODES = ["max", "capped_sum"]
GAPS = [0, 1, 2]


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


def temporal_trace(values: list[dict], node_count: int, window: int, decay: float, mode: str, gap: int) -> np.ndarray:
    result = np.zeros(node_count, dtype=float)
    recent = values[-window:]
    for age, snapshot in enumerate(reversed(recent)):
        factor = float(decay) ** (age + gap + 1)
        for node_text, value in snapshot.items():
            node = int(node_text)
            contribution = float(value) * factor
            if mode == "max":
                result[node] = max(result[node], contribution)
            else:
                result[node] = min(1.0, result[node] + contribution)
    return result


def replay_ratio(traversed: set[tuple[int, int]], chain: list[dict]) -> tuple[int, int, float]:
    total = len(chain)
    replayed = sum(1 for item in chain if tuple(item["edge"]) in traversed)
    return replayed, total, 0.0 if total == 0 else replayed / total


def run_case(entity: str, position: str, window: int, decay: float, mode: str, gap: int, reference: dict) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    entity_state = v15.propagate_state(
        brain,
        v15.initial_sources(brain, v3.entity_nodes(entity)),
        v15.ENTITY_STEPS,
    )
    trace = temporal_trace(
        entity_state["activation_values"], brain.node_count, window, decay, mode, gap
    )

    entry = reference.get("entry")
    entry_source = None if entry is None else int(entry["source"])
    entry_before = 0.0 if entry_source is None else float(trace[entry_source])

    position_initial = trace.copy()
    for node in v3.position_nodes(position):
        position_initial[int(node)] = max(position_initial[int(node)], 1.0)
    position_state = v15.propagate_state(brain, position_initial, v15.POSITION_STEPS)

    traversed = edge_set(position_state["traversed_edges"])
    replayed, total, ratio = replay_ratio(traversed, reference.get("chain", []))
    entry_during = entry_before
    if entry_source is not None:
        for snapshot in position_state["activation_values"]:
            entry_during = max(entry_during, float(snapshot.get(str(entry_source), 0.0)))

    baseline = v15.natural_binding(entity, position, 0.0, 0)
    combined_nodes = set(entity_state["activated_nodes"]) | set(position_state["activated_nodes"])

    return {
        "entity": entity,
        "position": position,
        "window": window,
        "decay": decay,
        "mode": mode,
        "gap": gap,
        "trace_node_count": int(np.count_nonzero(trace > 0)),
        "trace_energy": float(trace.sum()),
        "trace_max": float(trace.max()) if trace.size else 0.0,
        "entry_activation_before_position": entry_before,
        "entry_activation_during_position": entry_during,
        "entry_gap_to_required": max(0.0, REQUIRED_ENTRY - entry_before),
        "entry_reached_required": entry_before >= REQUIRED_ENTRY,
        "replayed_chain_edges": replayed,
        "chain_edge_count": total,
        "chain_replay_ratio": ratio,
        "combined_node_count": len(combined_nodes),
        "node_growth_vs_no_trace": len(combined_nodes) - baseline["combined_node_count"],
        "traversed_edge_count": len(position_state["traversed_edges"]),
    }


def diagnose(entity: str, position: str) -> dict:
    reference = v12.binding_reference(entity, position)
    rows = []
    for mode in MODES:
        for gap in GAPS:
            for window in WINDOWS:
                for decay in DECAYS:
                    rows.append(run_case(entity, position, window, decay, mode, gap, reference))

    first_entry = next((r for r in rows if r["entry_reached_required"]), None)
    first_any = next((r for r in rows if r["replayed_chain_edges"] > 0), None)
    first_full = next((r for r in rows if r["chain_edge_count"] > 0 and r["chain_replay_ratio"] == 1.0), None)
    safe_full = next((r for r in rows if r["chain_edge_count"] > 0 and r["chain_replay_ratio"] == 1.0 and r["node_growth_vs_no_trace"] <= 10), None)
    best_entry = max(rows, key=lambda r: r["entry_activation_before_position"])
    best_ratio = max(rows, key=lambda r: (r["chain_replay_ratio"], -r["node_growth_vs_no_trace"]))

    return {
        "entity": entity,
        "position": position,
        "reference_entry": reference.get("entry"),
        "reference_chain_edge_count": len(reference.get("chain", [])),
        "first_entry_reached": first_entry,
        "first_any_replay": first_any,
        "first_full_replay": first_full,
        "first_safe_full_replay": safe_full,
        "best_entry_case": best_entry,
        "best_replay_case": best_ratio,
        "rows": rows,
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v17",
        "world": {"P": player, "E": other},
        "purpose": "Build a generic short-term temporal trace from recent activation history without selecting specific nodes.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "hand_selected_trace_nodes": False,
            "core_file_modified": False,
        },
        "settings": {
            "threshold": THRESHOLD,
            "required_entry_activation": REQUIRED_ENTRY,
            "windows": WINDOWS,
            "decays": DECAYS,
            "modes": MODES,
            "gaps": GAPS,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v17.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v17</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:760px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v17</h1><p class="lead">特定Nodeを選ばず、直近の活動履歴全体から時間減衰付き短期Traceを作る。最大値方式と上限付き加算方式を比較し、入口activation・Binding連鎖再生・活動拡大を測る。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">短期Traceを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ検証していません。</pre></section></main><script>
function fmt(x){if(!x)return 'なし';return `${x.mode} / W${x.window} / d${Number(x.decay).toFixed(2)} / g${x.gap} / Node増${x.node_growth_vs_no_trace}`}
function cards(label,d){return `<div class="metric">${label}入口0.35到達<b class="${d.first_entry_reached?'good':'warn'}">${fmt(d.first_entry_reached)}</b></div><div class="metric">${label}初回再生<b class="${d.first_any_replay?'good':'warn'}">${fmt(d.first_any_replay)}</b></div><div class="metric">${label}完全再生<b class="${d.first_full_replay?'good':'warn'}">${fmt(d.first_full_replay)}</b></div><div class="metric">${label}安全完全再生<b class="${d.first_safe_full_replay?'good':'warn'}">${fmt(d.first_safe_full_replay)}</b></div><div class="metric">${label}最大入口<b>${Number(d.best_entry_case.entry_activation_before_position).toFixed(6)}</b></div><div class="metric">${label}最高再生率<b>${Number(d.best_replay_case.chain_replay_ratio).toFixed(6)}</b></div>`}
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
    print(f"Core Growth Binding v17: http://{HOST}:{PORT}")
    print("Temporal residual trace / no hand-selected nodes / no learning")
    serve(app, host=HOST, port=PORT)
