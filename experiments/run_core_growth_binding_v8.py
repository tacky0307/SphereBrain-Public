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
START_PORT = 5043
OUT = ROOT / "data" / "core_growth_binding_v8" / "results"
POSITIONS = v3.POSITIONS
THRESHOLDS = [0.18, 0.17, 0.16, 0.15, 0.14]
PERSISTENCE = [0.00, 0.08, 0.16, 0.24, 0.32]
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


def edge_set(edges):
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def edge_jaccard(a, b):
    sa, sb = edge_set(a), edge_set(b)
    union = sa | sb
    return 1.0 if not union else len(sa & sb) / len(union)


def binding_refs(entity: str):
    return {
        p: v3.make_binding(copy.deepcopy(v3.base.CORE), entity, p, learn=False)
        for p in POSITIONS
    }


def binding_specific_edges(entity: str, position: str):
    return set(v5.binding_components(v3.base.CORE, entity, position)["binding_only_edges"])


def run_probe(entity: str, *, threshold: float, persistence: float):
    brain = copy.deepcopy(v3.base.CORE)
    sources = list(v3.entity_nodes(entity))
    activation = np.zeros(brain.node_count, dtype=float)
    for node in sources:
        activation[node] = 1.0

    activated_nodes = set(sources)
    traversed_edges = set()
    history = [sorted(sources)]
    step_records = []

    for step in range(MAX_STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break

        candidates = {}
        local_seen = []
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            for local_index in best_indices:
                target = int(neighbors[local_index])
                value = float(scores[local_index]) * brain.signal_decay
                local_seen.append((int(source), target, value))
                if value < threshold:
                    continue
                previous = candidates.get(target)
                if previous is None or value > previous[0]:
                    candidates[target] = (value, int(source))

        retained = activation * float(persistence)
        retained[retained < threshold] = 0.0
        next_activation = retained.copy()
        accepted = []

        if candidates:
            ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
            remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
            step_limit = min(brain.max_active_per_step, len(ranked))
            selected = []
            new_nodes = 0
            for target, payload in ranked:
                is_new = target not in activated_nodes
                if is_new and new_nodes >= remaining_capacity:
                    continue
                selected.append((target, payload))
                if is_new:
                    new_nodes += 1
                if len(selected) >= step_limit:
                    break
            for target, (value, source) in selected:
                if value >= threshold:
                    next_activation[target] = max(next_activation[target], value)
                    accepted.append((source, target))
                    traversed_edges.add(tuple(sorted((source, target))))

        active_now = np.flatnonzero(next_activation > 0).tolist()
        step_records.append({
            "step": step,
            "active_sources": [int(x) for x in active_sources],
            "local_top_edges": [[a, b, v] for a, b, v in local_seen],
            "candidate_count": len(candidates),
            "accepted_edges": [list(x) for x in accepted],
            "retained_nodes": np.flatnonzero(retained > 0).tolist(),
            "active_now": active_now,
        })
        if not active_now:
            break
        activated_nodes.update(active_now)
        history.append(active_now)
        activation = next_activation

    return {
        "entity": entity,
        "threshold": threshold,
        "persistence": persistence,
        "activated_nodes": sorted(activated_nodes),
        "activated_node_count": len(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed_edges)],
        "traversed_edge_count": len(traversed_edges),
        "steps": len(step_records),
        "history": history,
        "step_records": step_records,
    }


def score_probe(probe, refs, target):
    scores = {}
    for position, ref in refs.items():
        scores[position] = {
            "node": jaccard(probe["activated_nodes"], ref["bound_nodes"]),
            "edge": edge_jaccard(probe["traversed_edges"], ref["bound_edges"]),
        }
    target_node = scores[target]["node"]
    other_node = max(v["node"] for k, v in scores.items() if k != target)
    winner = max(scores, key=lambda p: scores[p]["node"])
    return {
        "scores": scores,
        "target_node_margin": target_node - other_node,
        "winner": winner,
    }


def diagnose(entity: str, position: str):
    refs = binding_refs(entity)
    specific = binding_specific_edges(entity, position)
    rows = []

    for threshold in THRESHOLDS:
        probe = run_probe(entity, threshold=threshold, persistence=0.0)
        traversed = edge_set(probe["traversed_edges"])
        rows.append({
            "mode": "threshold",
            "threshold": threshold,
            "persistence": 0.0,
            "probe": probe,
            "specific_edge_count": len(specific),
            "specific_edges_selected": len(traversed & specific),
            "specific_edge_recall_ratio": 0.0 if not specific else len(traversed & specific) / len(specific),
            "selectivity": score_probe(probe, refs, position),
        })

    for persistence in PERSISTENCE:
        probe = run_probe(entity, threshold=0.18, persistence=persistence)
        traversed = edge_set(probe["traversed_edges"])
        rows.append({
            "mode": "persistence",
            "threshold": 0.18,
            "persistence": persistence,
            "probe": probe,
            "specific_edge_count": len(specific),
            "specific_edges_selected": len(traversed & specific),
            "specific_edge_recall_ratio": 0.0 if not specific else len(traversed & specific) / len(specific),
            "selectivity": score_probe(probe, refs, position),
        })

    threshold_hit = next((r for r in rows if r["mode"] == "threshold" and r["specific_edges_selected"] > 0), None)
    persistence_hit = next((r for r in rows if r["mode"] == "persistence" and r["specific_edges_selected"] > 0), None)

    return {
        "entity": entity,
        "target_position": position,
        "binding_specific_edges": [list(x) for x in sorted(specific)],
        "threshold_first_hit": threshold_hit,
        "persistence_first_hit": persistence_hit,
        "rows": rows,
    }


def observe(player: str, other: str):
    payload = {
        "experiment": "Core Growth Binding v8",
        "world": {"P": player, "E": other},
        "purpose": "Generic recall diagnosis: compare lower threshold with short-term activity persistence.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "puzzle_specific_rules": False,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v8.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v8</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:23px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:680px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v8</h1><p class="lead">Binding固有Edgeが、閾値を下げることで通るのか、活動を短く保持することで通るのかを比較する。学習・重み変更・新規Edge・パズル専用規則はない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Sweepする</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ実行していません。</pre></section></main><script>
function hit(x,kind){if(!x)return 'なし';return kind==='t'?`threshold ${x.threshold}`:`persistence ${x.persistence}`};async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;document.getElementById('metrics').innerHTML=`<div class="metric">P 閾値初回通過<b class="${P.threshold_first_hit?'good':'warn'}">${hit(P.threshold_first_hit,'t')}</b></div><div class="metric">P 持続初回通過<b class="${P.persistence_first_hit?'good':'warn'}">${hit(P.persistence_first_hit,'p')}</b></div><div class="metric">E 閾値初回通過<b class="${E.threshold_first_hit?'good':'warn'}">${hit(E.threshold_first_hit,'t')}</b></div><div class="metric">E 持続初回通過<b class="${E.persistence_first_hit?'good':'warn'}">${hit(E.persistence_first_hit,'p')}</b></div><div class="metric">P固有Edge数<b>${P.binding_specific_edges.length}</b></div><div class="metric">E固有Edge数<b>${E.binding_specific_edges.length}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v8: http://{HOST}:{PORT}")
    print("Threshold sweep vs short-term persistence / no learning / no weight changes")
    serve(app, host=HOST, port=PORT)
