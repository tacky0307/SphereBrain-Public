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
from flask import Flask, jsonify, request
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v12 as v12
import run_core_growth_binding_v15 as v15
import run_core_growth_binding_v19 as v19
import run_core_growth_binding_v20 as v20

HOST = "127.0.0.1"
START_PORT = 5056
OUT = ROOT / "data" / "core_growth_binding_v21" / "results"
POSITIONS = v3.POSITIONS


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


def shortest_path(brain, start: int, goal: int) -> list[int]:
    queue = deque([start])
    previous: dict[int, int | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            break
        for nxt in np.flatnonzero(brain.adjacency[node]):
            nxt = int(nxt)
            if nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    if goal not in previous:
        return []
    path: list[int] = []
    cursor: int | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return list(reversed(path))


def path_metrics(brain, nodes: list[int], source_activation: float) -> dict:
    edges = []
    signal = float(source_activation)
    for a, b in zip(nodes, nodes[1:]):
        weight = float(brain.weights[a, b])
        distance = float(np.linalg.norm(brain.positions[a] - brain.positions[b]))
        signal *= weight * float(brain.signal_decay)
        edges.append({
            "edge": [a, b],
            "weight": weight,
            "distance": distance,
            "signal_after_edge": signal,
        })
    return {
        "nodes": nodes,
        "edge_count": len(edges),
        "edges": edges,
        "first_edge_weight": None if not edges else edges[0]["weight"],
        "final_signal_estimate": signal,
        "total_distance": sum(e["distance"] for e in edges),
    }


def replay_single(entity: str, position: str, full_trace: np.ndarray, node: int, reference: dict) -> dict:
    trace = np.zeros_like(full_trace)
    trace[node] = full_trace[node]
    report = v19.replay_with_trace(entity, position, trace, reference)
    report["node"] = int(node)
    report["trace_activation"] = float(full_trace[node])
    return report


def candidate_report(entity: str, position: str, trace: np.ndarray, metadata: dict[int, dict], node: int, reference: dict) -> dict:
    brain = v3.base.CORE
    entry = reference.get("entry")
    entry_source = None if entry is None else int(entry["source"])
    path = [] if entry_source is None else shortest_path(brain, node, entry_source)
    replay = replay_single(entity, position, trace, node, reference)
    meta = metadata[node]
    metrics = path_metrics(brain, path, float(trace[node]))
    entry_candidate = metrics["final_signal_estimate"] >= 0.18 if metrics["edge_count"] > 0 else False
    return {
        "node": int(node),
        "trace_activation": float(trace[node]),
        "appearance_count": int(meta["appearance_count"]),
        "newest_age": int(meta["newest_age"]),
        "degree": int(np.count_nonzero(brain.adjacency[node])),
        "distance_to_entry": None if entry_source is None else float(np.linalg.norm(brain.positions[node] - brain.positions[entry_source])),
        "path_to_entry": metrics,
        "estimated_entry_candidate": bool(entry_candidate),
        "replay": replay,
    }


def similarity_score(candidate: dict, key_roles: list[dict]) -> float:
    if not key_roles:
        return 999.0
    target_activation = float(np.mean([x["trace_activation"] for x in key_roles]))
    target_count = float(np.mean([x["appearance_count"] for x in key_roles]))
    target_degree = float(np.mean([x["degree"] for x in key_roles]))
    target_distance = float(np.mean([x["distance_to_entry"] for x in key_roles if x["distance_to_entry"] is not None]))
    return (
        abs(candidate["trace_activation"] - target_activation)
        + 0.10 * abs(candidate["appearance_count"] - target_count)
        + 0.03 * abs(candidate["degree"] - target_degree)
        + 0.50 * abs((candidate["distance_to_entry"] or 0.0) - target_distance)
    )


def diagnose(player: str, other: str) -> dict:
    base = v20.diagnose(player, other)
    e_roles = base["E"]["roles"]

    p_data = v19.grouped_ablation("P", player)
    p_trace, _ = v19.build_trace("P")
    p_meta = {int(row["node"]): row for row in p_data["node_metadata"]}
    p_reference = v12.binding_reference("P", player)

    reports = [
        candidate_report("P", player, p_trace, p_meta, int(row["node"]), p_reference)
        for row in base["P"]["structurally_similar_candidates"]
    ]
    for row in reports:
        row["similarity_to_e_keys"] = similarity_score(row, e_roles)

    reports.sort(key=lambda x: (x["similarity_to_e_keys"], x["node"]))

    key_summary = []
    for role in e_roles:
        key_summary.append({
            "node": role["node"],
            "trace_activation": role["trace_activation"],
            "appearance_count": role["appearance_count"],
            "degree": role["degree"],
            "distance_to_entry": role["distance_to_entry"],
            "path_edge_count": role["path_to_entry"]["edge_count"],
            "first_edge_weight": None if not role["path_to_entry"]["edges"] else role["path_to_entry"]["edges"][0]["weight"],
            "estimated_final_signal": role["path_to_entry"]["propagation_product"] * role["trace_activation"],
            "replay_ratio": role["replay"]["ratio"],
        })

    return {
        "E_keys": key_summary,
        "P_candidates": reports,
        "comparison": {
            "p_candidate_count": len(reports),
            "p_candidates_replaying_any": sum(1 for x in reports if x["replay"]["ratio"] > 0),
            "p_candidates_replaying_full": sum(1 for x in reports if x["replay"]["ratio"] == 1.0),
            "p_candidates_reaching_entry_estimate": sum(1 for x in reports if x["estimated_entry_candidate"]),
            "best_candidate": None if not reports else reports[0],
        },
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v21",
        "world": {"P": player, "E": other},
        "purpose": "Compare E Recall Keys 80/450 with the five closest P-side structural candidates under the same path and replay diagnostics.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "diagnostics": diagnose(player, other),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v21.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v21</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:840px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v21</h1><p class="lead">E Key 80 / 450と、P側の構造類似候補5Nodeを同じ基準で比較する。Trace量・反復回数・入口接続・Edge強度・推定信号・単独再生率を診断する。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Keyと候補を比較</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function keyCard(k,i){return `<div class="metric">E Key ${i+1}<b class="blue">Node ${k.node}</b></div><div class="metric">E Key ${i+1} activation<b>${f(k.trace_activation)}</b></div><div class="metric">E Key ${i+1} 入口Edge<b>${k.path_edge_count}</b></div><div class="metric">E Key ${i+1} 推定信号<b>${f(k.estimated_final_signal)}</b></div>`}function pCard(c,i){return `<div class="metric">P候補${i+1}<b class="blue">Node ${c.node}</b></div><div class="metric">P候補${i+1} activation<b>${f(c.trace_activation)}</b></div><div class="metric">P候補${i+1} 入口Edge<b>${c.path_to_entry.edge_count}</b></div><div class="metric">P候補${i+1} 再生率<b class="${c.replay.ratio>0?'good':'warn'}">${f(c.replay.ratio)}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json(),x=d.diagnostics;document.getElementById('metrics').innerHTML=x.E_keys.map(keyCard).join('')+x.P_candidates.map(pCard).join('')+`<div class="metric">P入口到達候補<b>${x.comparison.p_candidates_reaching_entry_estimate}</b></div><div class="metric">P一部再生候補<b>${x.comparison.p_candidates_replaying_any}</b></div><div class="metric">P完全再生候補<b>${x.comparison.p_candidates_replaying_full}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v21: http://{HOST}:{PORT}")
    print("E Recall Keys vs P structural candidates / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
