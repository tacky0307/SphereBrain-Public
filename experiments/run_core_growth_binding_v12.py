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
import run_core_growth_binding_v8 as v8
import run_core_growth_binding_v10 as v10

HOST = "127.0.0.1"
START_PORT = 5047
OUT = ROOT / "data" / "core_growth_binding_v12" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
ACTIVATION_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.68]
MAX_STEPS = 12


def choose_port(start: int) -> int:
    for port in range(start, start + 30):
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


def binding_reference(entity: str, position: str) -> dict:
    trace = v10.run_detailed(entity, position, threshold=THRESHOLD, binding=True)
    specific = sorted(v8.binding_specific_edges(entity, position))
    if not specific:
        return {"trace": trace, "specific_edges": [], "entry": None, "chain": []}

    best = None
    chosen_edge = None
    for edge in specific:
        row = v10.best_record(trace, edge)
        if row is not None and (best is None or row["signal"] > best["signal"]):
            best = row
            chosen_edge = edge

    if chosen_edge is None or best is None:
        return {"trace": trace, "specific_edges": [list(x) for x in specific], "entry": None, "chain": []}

    chain = v10.ordered_binding_chain(trace, chosen_edge)
    return {
        "trace": trace,
        "specific_edges": [list(x) for x in specific],
        "entry": {
            "edge": list(chosen_edge),
            "source": int(best["source"]),
            "target": int(best["target"]),
            "binding_source_activation": float(best["source_activation"]),
            "binding_signal": float(best["signal"]),
            "binding_step": int(best["step"]),
        },
        "chain": chain,
    }


def run_with_entry_activation(entity: str, entry_source: int, forced_activation: float) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    source_nodes = list(v3.entity_nodes(entity))
    activation = np.zeros(brain.node_count, dtype=float)
    for node in source_nodes:
        activation[node] = 1.0
    activation[entry_source] = max(float(activation[entry_source]), float(forced_activation))

    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed_edges: set[tuple[int, int]] = set()
    step_records = []

    for step_index in range(MAX_STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break

        candidates: dict[int, tuple[float, int]] = {}
        local_records = []
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            local_top = {int(neighbors[i]) for i in best_indices}
            for idx, raw_target in enumerate(neighbors):
                target = int(raw_target)
                signal = float(scores[idx]) * brain.signal_decay
                is_local_top = target in local_top
                row = {
                    "source": int(source),
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "signal": signal,
                    "is_local_top": is_local_top,
                    "passes_threshold": signal >= THRESHOLD,
                }
                local_records.append(row)
                if not is_local_top or signal < THRESHOLD:
                    continue
                previous = candidates.get(target)
                if previous is None or signal > previous[0]:
                    candidates[target] = (signal, int(source))

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
        selected = []
        new_nodes = 0
        for target, payload in ranked:
            is_new = target not in activated_nodes
            if is_new and new_nodes >= remaining_capacity:
                continue
            selected.append((target, payload))
            if is_new:
                new_nodes += 1
            if len(selected) >= min(brain.max_active_per_step, len(ranked)):
                break

        next_activation = np.zeros(brain.node_count, dtype=float)
        accepted = []
        for target, (signal, source) in selected:
            next_activation[target] = max(next_activation[target], signal)
            accepted.append(edge_key(source, target))
            traversed_edges.add(edge_key(source, target))

        accepted_set = set(accepted)
        candidate_set = {edge_key(source, target) for target, (_, source) in candidates.items()}
        for row in local_records:
            key = tuple(row["edge"])
            row["became_candidate"] = key in candidate_set
            row["accepted"] = key in accepted_set

        active_now = np.flatnonzero(next_activation > 0).tolist()
        step_records.append({
            "step": step_index,
            "active_sources": [int(x) for x in active_sources],
            "edge_records": local_records,
            "accepted_edges": [list(x) for x in sorted(accepted_set)],
            "active_now": active_now,
        })
        if not active_now:
            break
        activated_nodes.update(active_now)
        activation = next_activation

    return {
        "entity": entity,
        "entry_source": entry_source,
        "forced_activation": forced_activation,
        "threshold": THRESHOLD,
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed_edges)],
        "step_records": step_records,
    }


def chain_replay(trace: dict, chain: list[dict]) -> dict:
    traversed = edge_set(trace["traversed_edges"])
    rows = []
    replayed = 0
    first_missing = None
    for index, item in enumerate(chain):
        edge = tuple(item["edge"])
        selected = edge in traversed
        if selected:
            replayed += 1
        elif first_missing is None:
            first_missing = {"chain_index": index, "binding_step": item["step"], "edge": list(edge)}
        rows.append({"chain_index": index, "binding_step": item["step"], "edge": list(edge), "selected": selected})
    return {
        "chain_edge_count": len(chain),
        "replayed_edge_count": replayed,
        "replay_ratio": 0.0 if not chain else replayed / len(chain),
        "first_missing": first_missing,
        "rows": rows,
    }


def diagnose(entity: str, position: str) -> dict:
    reference = binding_reference(entity, position)
    entry = reference["entry"]
    if entry is None:
        return {
            "entity": entity,
            "position": position,
            "reference": reference,
            "rows": [],
            "first_entry_hit": None,
            "first_full_chain_replay": None,
        }

    rows = []
    for activation in ACTIVATION_SWEEP:
        trace = run_with_entry_activation(entity, entry["source"], activation)
        replay = chain_replay(trace, reference["chain"])
        entry_selected = tuple(entry["edge"]) in edge_set(trace["traversed_edges"])
        rows.append({
            "forced_activation": activation,
            "entry_edge_selected": entry_selected,
            "chain_replay": replay,
            "activated_node_count": len(trace["activated_nodes"]),
            "traversed_edge_count": len(trace["traversed_edges"]),
            "trace": trace,
        })

    first_entry = next((row for row in rows if row["entry_edge_selected"]), None)
    first_full = next((row for row in rows if row["chain_replay"]["replay_ratio"] == 1.0 and row["chain_replay"]["chain_edge_count"] > 0), None)
    return {
        "entity": entity,
        "position": position,
        "reference": reference,
        "rows": rows,
        "first_entry_hit": first_entry,
        "first_full_chain_replay": first_full,
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v12",
        "world": {"P": player, "E": other},
        "purpose": "Measure the minimum restored entry activation needed to restart and fully replay a Binding chain.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "puzzle_specific_rules": False,
            "artificial_entry_activation_for_diagnosis": True,
        },
        "settings": {
            "threshold": THRESHOLD,
            "activation_sweep": ACTIVATION_SWEEP,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v12.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v12</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:23px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:760px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v12</h1><p class="lead">Binding入口activationを段階的に復元し、連鎖開始と完全再生に必要な最小値を測る。これは診断用の人工activationであり、Coreの機能として実装したものではない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Sweepする</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ測定していません。</pre></section></main><script>
function val(x){return x?Number(x.forced_activation).toFixed(2):'なし'}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;document.getElementById('metrics').innerHTML=`<div class="metric">P入口初回通過<b class="${P.first_entry_hit?'good':'warn'}">${val(P.first_entry_hit)}</b></div><div class="metric">P完全連鎖<b class="${P.first_full_chain_replay?'good':'warn'}">${val(P.first_full_chain_replay)}</b></div><div class="metric">P形成時入口<b>${P.reference.entry?Number(P.reference.entry.binding_source_activation).toFixed(6):'なし'}</b></div><div class="metric">P連鎖Edge数<b>${P.reference.chain.length}</b></div><div class="metric">E入口初回通過<b class="${E.first_entry_hit?'good':'warn'}">${val(E.first_entry_hit)}</b></div><div class="metric">E完全連鎖<b class="${E.first_full_chain_replay?'good':'warn'}">${val(E.first_full_chain_replay)}</b></div><div class="metric">E形成時入口<b>${E.reference.entry?Number(E.reference.entry.binding_source_activation).toFixed(6):'なし'}</b></div><div class="metric">E連鎖Edge数<b>${E.reference.chain.length}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v12: http://{HOST}:{PORT}")
    print("Entry activation sweep / diagnostic only / no learning / no weight changes")
    serve(app, host=HOST, port=PORT)
