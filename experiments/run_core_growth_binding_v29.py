from __future__ import annotations

import copy
import json
import socket
import sys
import threading
import webbrowser
from collections import deque
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v27 as v27

HOST = "127.0.0.1"
START_PORT = 5075
OUT = ROOT / "data" / "core_growth_binding_v29" / "results"
POSITIONS = list(v3.POSITIONS)
MAX_GRAPH_DISTANCE = 4


def choose_port(start: int) -> int:
    for port in range(start, start + 40):
        if port in {5060, 5061}:
            continue
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


def adjacency_sets(brain) -> list[set[int]]:
    return [set(int(x) for x in np.flatnonzero(brain.adjacency[i])) for i in range(brain.node_count)]


def step_nodes(trace: dict) -> list[set[int]]:
    rows = []
    for step in trace.get("steps", []):
        rows.append(set(int(x) for x in step.get("active_sources", [])))
    return rows


def shortest_graph_distance(adj: list[set[int]], starts: set[int], goals: set[int], limit: int = MAX_GRAPH_DISTANCE) -> dict:
    if not starts or not goals:
        return {"distance": None, "start": None, "goal": None, "path": []}
    overlap = starts & goals
    if overlap:
        node = min(overlap)
        return {"distance": 0, "start": node, "goal": node, "path": [node]}

    queue = deque()
    previous: dict[int, int | None] = {}
    origin: dict[int, int] = {}
    for start in sorted(starts):
        queue.append((start, 0))
        previous[start] = None
        origin[start] = start

    found = None
    while queue:
        node, depth = queue.popleft()
        if depth >= limit:
            continue
        for nxt in adj[node]:
            if nxt in previous:
                continue
            previous[nxt] = node
            origin[nxt] = origin[node]
            if nxt in goals:
                found = nxt
                queue.clear()
                break
            queue.append((nxt, depth + 1))

    if found is None:
        return {"distance": None, "start": None, "goal": None, "path": []}

    path = []
    cursor: int | None = found
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return {
        "distance": len(path) - 1,
        "start": origin[found],
        "goal": found,
        "path": path,
    }


def contact_rows(echo_trace: dict, position_trace: dict) -> dict:
    brain = v3.base.CORE
    adj = adjacency_sets(brain)
    echo_steps = step_nodes(echo_trace)
    pos_steps = step_nodes(position_trace)
    max_steps = max(len(echo_steps), len(pos_steps))
    rows = []

    best = None
    for e_step in range(len(echo_steps)):
        for p_step in range(len(pos_steps)):
            time_gap = abs(e_step - p_step)
            if time_gap > 1:
                continue
            e_nodes = echo_steps[e_step]
            p_nodes = pos_steps[p_step]
            same = sorted(e_nodes & p_nodes)
            adjacent_pairs = sorted(
                (a, b)
                for a in e_nodes
                for b in p_nodes
                if b in adj[a]
            )
            shared_neighbor_rows = []
            for a in e_nodes:
                for b in p_nodes:
                    shared = adj[a] & adj[b]
                    if shared:
                        shared_neighbor_rows.append({
                            "echo_node": a,
                            "position_node": b,
                            "shared_neighbors": sorted(shared),
                        })
            distance = shortest_graph_distance(adj, e_nodes, p_nodes)
            row = {
                "echo_step": e_step,
                "position_step": p_step,
                "time_gap": time_gap,
                "same_nodes": same,
                "adjacent_pairs": [list(x) for x in adjacent_pairs],
                "shared_neighbor_pairs": shared_neighbor_rows,
                "minimum_graph_distance": distance,
            }
            rows.append(row)

            score = (
                0 if same else
                1 if adjacent_pairs else
                2 if shared_neighbor_rows else
                distance["distance"] if distance["distance"] is not None else 999
            )
            candidate = (score, time_gap, e_step, p_step, row)
            if best is None or candidate[:4] < best[:4]:
                best = candidate

    best_row = None if best is None else best[4]
    has_same = any(row["time_gap"] == 0 and row["same_nodes"] for row in rows)
    has_adjacent = any(row["time_gap"] == 0 and row["adjacent_pairs"] for row in rows)
    has_shared_neighbor = any(row["time_gap"] == 0 and row["shared_neighbor_pairs"] for row in rows)
    has_temporal = any(
        row["time_gap"] <= 1 and (
            row["same_nodes"] or row["adjacent_pairs"] or row["shared_neighbor_pairs"] or
            (row["minimum_graph_distance"]["distance"] is not None and row["minimum_graph_distance"]["distance"] <= 2)
        )
        for row in rows
    )
    return {
        "same_node_same_step": has_same,
        "adjacent_same_step": has_adjacent,
        "shared_neighbor_same_step": has_shared_neighbor,
        "temporal_contact_within_one_step": has_temporal,
        "best_contact": best_row,
        "rows": rows,
    }


def diagnose_position(position: str) -> dict:
    echo_only = v27.run_live(position=None, include_echo=True)
    position_only = v27.run_live(position=position, include_echo=False)
    combined = v27.run_live(position=position, include_echo=True)
    contact = contact_rows(echo_only, position_only)

    combined_edges = {edge_key(*edge) for edge in combined["traversed_edges"]}
    echo_edges = {edge_key(*edge) for edge in echo_only["traversed_edges"]}
    position_edges = {edge_key(*edge) for edge in position_only["traversed_edges"]}
    interaction_edges = combined_edges - echo_edges - position_edges

    has_contact = any([
        contact["same_node_same_step"],
        contact["adjacent_same_step"],
        contact["shared_neighbor_same_step"],
        contact["temporal_contact_within_one_step"],
    ])
    if interaction_edges:
        verdict = "interaction_edge_formed"
    elif has_contact:
        verdict = "contact_exists_but_not_integrated"
    else:
        verdict = "no_contact_candidate"

    return {
        "position": position,
        "contact": contact,
        "interaction_edge_count": len(interaction_edges),
        "interaction_edges": [list(x) for x in sorted(interaction_edges)],
        "verdict": verdict,
        "counts": {
            "echo_steps": len(echo_only["steps"]),
            "position_steps": len(position_only["steps"]),
            "combined_edges": len(combined_edges),
            "echo_edges": len(echo_edges),
            "position_edges": len(position_edges),
        },
    }


def observe() -> dict:
    reports = {position: diagnose_position(position) for position in POSITIONS}
    contact_positions = [p for p, row in reports.items() if row["verdict"] == "contact_exists_but_not_integrated"]
    no_contact_positions = [p for p, row in reports.items() if row["verdict"] == "no_contact_candidate"]
    interactions = [p for p, row in reports.items() if row["interaction_edge_count"] > 0]

    if interactions:
        overall = "interaction_already_present"
    elif len(contact_positions) == len(POSITIONS):
        overall = "integration_mechanism_missing"
    elif len(no_contact_positions) == len(POSITIONS):
        overall = "path_contact_missing"
    else:
        overall = "mixed_contact_availability"

    payload = {
        "experiment": "Core Growth Binding v29",
        "purpose": "Measure whether E-residual and position pathways become spatially and temporally close enough to interact before adding a Binding mechanism.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "positions": reports,
        "summary": {
            "positions_with_contact_but_no_integration": contact_positions,
            "positions_without_contact_candidate": no_contact_positions,
            "positions_with_interaction_edges": interactions,
            "overall_verdict": overall,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v29.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v29</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v29</h1><p class="lead">E残響経路と位置経路が、同一Node・隣接・共通neighbor・1Step差以内で接触可能かを測り、統合機構不足か経路配置不足かを判定する。</p><section class="panel"><div class="controls"><button id="run">接触可能性を解剖</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),rows=Object.values(d.positions);document.getElementById('metrics').innerHTML=rows.map(r=>{const c=r.contact,b=c.best_contact,dist=b&&b.minimum_graph_distance?b.minimum_graph_distance.distance:null;return `<div class="metric">${r.position} 同一Node<b>${yn(c.same_node_same_step)}</b></div><div class="metric">${r.position} 隣接<b>${yn(c.adjacent_same_step)}</b></div><div class="metric">${r.position} 共通neighbor<b>${yn(c.shared_neighbor_same_step)}</b></div><div class="metric">${r.position} 1Step差接触<b>${yn(c.temporal_contact_within_one_step)}</b></div><div class="metric">${r.position} 最短距離<b>${dist===null?'なし':dist}</b></div><div class="metric">${r.position} 判定<b>${r.verdict}</b></div>`}).join('')+`<div class="metric">総合判定<b class="blue">${d.summary.overall_verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/observe")
def api_observe():
    return jsonify(observe())


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v29: http://{HOST}:{PORT}")
    print("Path contact-potential diagnostics / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
