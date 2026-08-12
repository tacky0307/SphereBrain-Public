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

HOST = "127.0.0.1"
START_PORT = 5055
OUT = ROOT / "data" / "core_growth_binding_v20" / "results"
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
    path = []
    cursor: int | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return list(reversed(path))


def path_report(brain, nodes: list[int]) -> dict:
    edges = []
    total_distance = 0.0
    weight_product = 1.0
    for a, b in zip(nodes, nodes[1:]):
        weight = float(brain.weights[a, b])
        distance = float(np.linalg.norm(brain.positions[a] - brain.positions[b]))
        edges.append({"edge": [a, b], "weight": weight, "distance": distance})
        total_distance += distance
        weight_product *= weight * float(brain.signal_decay)
    return {
        "nodes": nodes,
        "edge_count": len(edges),
        "edges": edges,
        "total_distance": total_distance,
        "propagation_product": weight_product,
    }


def single_key_replay(entity: str, position: str, trace: np.ndarray, key: int, reference: dict) -> dict:
    kept = np.zeros_like(trace)
    kept[key] = trace[key]
    brain = copy.deepcopy(v3.base.CORE)
    initial = kept.copy()
    for node in v3.position_nodes(position):
        initial[int(node)] = max(initial[int(node)], 1.0)
    state = v15.propagate_state(brain, initial, v15.POSITION_STEPS)
    traversed = edge_set(state["traversed_edges"])
    chain = reference.get("chain", [])
    rows = []
    for item in chain:
        edge = tuple(item["edge"])
        rows.append({"edge": list(edge), "selected": edge in traversed})
    replayed = sum(1 for row in rows if row["selected"])
    return {
        "key": key,
        "trace_activation": float(trace[key]),
        "replayed": replayed,
        "total": len(rows),
        "ratio": 0.0 if not rows else replayed / len(rows),
        "chain_rows": rows,
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "activated_nodes": state["activated_nodes"],
    }


def metadata_map(rows: list[dict]) -> dict[int, dict]:
    return {int(row["node"]): row for row in rows}


def key_role(entity: str, position: str, key: int, trace: np.ndarray, metadata: dict[int, dict], reference: dict) -> dict:
    brain = v3.base.CORE
    entry = reference.get("entry")
    entry_source = None if entry is None else int(entry["source"])
    path = [] if entry_source is None else shortest_path(brain, key, entry_source)
    replay = single_key_replay(entity, position, trace, key, reference)
    row = metadata[key]
    return {
        "node": key,
        "trace_activation": float(trace[key]),
        "appearance_count": row["appearance_count"],
        "newest_age": row["newest_age"],
        "oldest_age": row["oldest_age"],
        "appearances": row["appearances"],
        "degree": int(np.count_nonzero(brain.adjacency[key])),
        "distance_to_entry": None if entry_source is None else float(np.linalg.norm(brain.positions[key] - brain.positions[entry_source])),
        "path_to_entry": path_report(brain, path),
        "replay": replay,
    }


def p_candidates(p_data: dict, e_roles: list[dict], limit: int = 5) -> list[dict]:
    if not e_roles:
        return []
    target_activation = float(np.mean([r["trace_activation"] for r in e_roles]))
    target_count = float(np.mean([r["appearance_count"] for r in e_roles]))
    target_degree = float(np.mean([r["degree"] for r in e_roles]))
    rows = []
    for meta in p_data["node_metadata"]:
        node = int(meta["node"])
        degree = int(np.count_nonzero(v3.base.CORE.adjacency[node]))
        score = (
            abs(float(meta["trace_activation"]) - target_activation)
            + 0.10 * abs(float(meta["appearance_count"]) - target_count)
            + 0.03 * abs(float(degree) - target_degree)
        )
        rows.append({
            "node": node,
            "structural_distance": score,
            "trace_activation": meta["trace_activation"],
            "appearance_count": meta["appearance_count"],
            "newest_age": meta["newest_age"],
            "degree": degree,
        })
    return sorted(rows, key=lambda x: (x["structural_distance"], x["node"]))[:limit]


def diagnose(player: str, other: str) -> dict:
    p = v19.grouped_ablation("P", player)
    e = v19.grouped_ablation("E", other)
    e_trace, _ = v19.build_trace("E")
    p_trace, _ = v19.build_trace("P")
    e_reference = v12.binding_reference("E", other)
    e_meta = metadata_map(e["node_metadata"])

    minimum_sets = e["exact_minimum"].get("minimum_sets", [])
    key_nodes = sorted({int(report["kept_nodes"][0]) for report in minimum_sets if len(report.get("kept_nodes", [])) == 1})
    roles = [key_role("E", other, node, e_trace, e_meta, e_reference) for node in key_nodes]

    route_overlap = None
    if len(roles) >= 2:
        a = edge_set(roles[0]["replay"]["traversed_edges"])
        b = edge_set(roles[1]["replay"]["traversed_edges"])
        union = a | b
        route_overlap = 1.0 if not union else len(a & b) / len(union)

    all_e_nodes = {int(x) for x in np.flatnonzero(e_trace > 0)}
    remove_both = v19.evaluate_keep("E", other, e_trace, all_e_nodes - set(key_nodes), e_reference)

    return {
        "E": {
            "position": other,
            "key_nodes": key_nodes,
            "key_count": len(key_nodes),
            "roles": roles,
            "route_edge_jaccard": route_overlap,
            "both_keys_removed": remove_both,
            "pair_failure_matches_keys": any(set(row["removed"]) == set(key_nodes) for row in e.get("pair_failures", [])) if len(key_nodes) == 2 else False,
        },
        "P": {
            "position": player,
            "baseline_ratio": p["baseline"]["ratio"],
            "trace_node_count": p["trace_node_count"],
            "structurally_similar_candidates": p_candidates(p, roles),
            "trace_energy": float(p_trace.sum()),
        },
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v20",
        "world": {"P": player, "E": other},
        "purpose": "Display and compare the two alternative minimum Recall Key nodes and their structural roles.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
        },
        "diagnostics": diagnose(player, other),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v20.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v20</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:820px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v20</h1><p class="lead">Eの2つの代替Recall Keyを直接表示し、活動履歴・入口までの経路・単独再生経路・冗長性を比較する。Core本体は変更しない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Keyを比較</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function role(r,i){return `<div class="metric">Key ${i+1} Node<b class="blue">${r.node}</b></div><div class="metric">Key ${i+1} activation<b>${f(r.trace_activation)}</b></div><div class="metric">Key ${i+1} 活動回数<b>${r.appearance_count}</b></div><div class="metric">Key ${i+1} 入口距離<b>${f(r.distance_to_entry)}</b></div><div class="metric">Key ${i+1} 経路Edge<b>${r.path_to_entry.edge_count}</b></div><div class="metric">Key ${i+1} 再生率<b class="good">${f(r.replay.ratio)}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json(),E=d.diagnostics.E,P=d.diagnostics.P;document.getElementById('metrics').innerHTML=E.roles.map(role).join('')+`<div class="metric">Key経路重なり<b>${f(E.route_edge_jaccard)}</b></div><div class="metric">両Key削除後<b class="${E.both_keys_removed.ratio===0?'good':'warn'}">再生率 ${f(E.both_keys_removed.ratio)}</b></div><div class="metric">破綻Pair一致<b>${E.pair_failure_matches_keys?'YES':'NO'}</b></div><div class="metric">P類似候補数<b>${P.structurally_similar_candidates.length}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v20: http://{HOST}:{PORT}")
    print("Recall Key role comparison / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
