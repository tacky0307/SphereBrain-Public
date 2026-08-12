from __future__ import annotations

import copy
import json
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
import run_core_growth_binding_v5 as v5

HOST = "127.0.0.1"
PORT = 5040
OUT = ROOT / "data" / "core_growth_binding_v6" / "results"
POSITIONS = v3.POSITIONS
REPEATS = v3.TRAIN_REPEATS
EPS = 1e-12


def edge_set(edges):
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def adjacency_from_edges(edges):
    graph: dict[int, set[int]] = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def shortest_edge_path(edges, starts, goals):
    graph = adjacency_from_edges(edges)
    starts = set(int(x) for x in starts)
    goals = set(int(x) for x in goals)
    queue = deque()
    previous: dict[int, int | None] = {}
    for node in starts:
        queue.append(node)
        previous[node] = None
    found = None
    while queue:
        node = queue.popleft()
        if node in goals:
            found = node
            break
        for nxt in graph.get(node, ()):
            if nxt not in previous:
                previous[nxt] = node
                queue.append(nxt)
    if found is None:
        return set()
    nodes = []
    cursor = found
    while cursor is not None:
        nodes.append(cursor)
        cursor = previous[cursor]
    nodes.reverse()
    return {tuple(sorted((a, b))) for a, b in zip(nodes, nodes[1:])}


def fixed_strategy_edges(entity: str, position: str):
    components = v5.binding_components(v3.base.CORE, entity, position)
    binding = components["binding"]
    bound = set(components["bound_edges"])
    specific = set(components["binding_only_edges"])

    specific_nodes = {node for edge in specific for node in edge}
    predecessor = {
        edge for edge in bound
        if edge in specific or edge[0] in specific_nodes or edge[1] in specific_nodes
    }

    source_nodes = set(binding["entity_stage"]["source_nodes"])
    bridge = set(specific)
    if specific_nodes:
        bridge |= shortest_edge_path(bound, source_nodes, specific_nodes)

    return {
        "specific_only": specific,
        "specific_plus_one_hop": predecessor,
        "short_bridge": bridge,
        "reference": binding,
        "bound_edges": bound,
        "entity_source_nodes": sorted(source_nodes),
    }


def reinforce_fixed(brain, selected_edges):
    before = brain.weights.copy()
    for _ in range(REPEATS):
        v5.reinforce_selected(brain, selected_edges)
    delta = np.asarray(brain.weights - before, dtype=float)
    rows, cols = np.triu_indices(brain.node_count, k=1)
    changed = {
        (int(a), int(b))
        for a, b in zip(rows, cols)
        if abs(float(delta[a, b])) > EPS
    }
    return changed


def references(entity: str):
    return {
        position: v3.make_binding(copy.deepcopy(v3.base.CORE), entity, position, learn=False)
        for position in POSITIONS
    }


def probe_report(brain, entity: str, refs, target: str):
    report = v5.probe(brain, entity, refs)
    return {
        "probe": report["probe"],
        "scores": report["scores"],
        "node_selectivity": v5.selectivity(report["scores"], target, "node"),
        "edge_selectivity": v5.selectivity(report["scores"], target, "edge"),
    }


def diagnose(entity: str, position: str):
    refs = references(entity)
    baseline = probe_report(v3.base.CORE, entity, refs, position)
    strategy_edges = fixed_strategy_edges(entity, position)
    strategies = {}

    for name in ("specific_only", "specific_plus_one_hop", "short_bridge"):
        selected = set(strategy_edges[name])
        trained = copy.deepcopy(v3.base.CORE)
        changed = reinforce_fixed(trained, selected)
        after = probe_report(trained, entity, refs, position)
        strategies[name] = {
            "selected_edges": [list(x) for x in sorted(selected)],
            "selected_edge_count": len(selected),
            "changed_edge_count": len(changed),
            "all_changes_within_selected": changed <= selected,
            "path_reaches_specific": bool(selected & strategy_edges["specific_only"]),
            "probe": after,
            "node_margin_delta": (
                after["node_selectivity"]["margin"]
                - baseline["node_selectivity"]["margin"]
            ),
            "edge_margin_delta": (
                after["edge_selectivity"]["margin"]
                - baseline["edge_selectivity"]["margin"]
            ),
        }

    return {
        "entity": entity,
        "trained_position": position,
        "baseline": baseline,
        "reference": {
            "binding_only_edge_count": len(strategy_edges["specific_only"]),
            "bound_edge_count": len(strategy_edges["bound_edges"]),
            "entity_source_nodes": strategy_edges["entity_source_nodes"],
        },
        "strategies": strategies,
    }


def observe(player: str, other: str):
    p = diagnose("P", player)
    e = diagnose("E", other)
    payload = {
        "experiment": "Core Growth Binding v6",
        "world": {"P": player, "E": other},
        "purpose": "Compare memory-mark edges with progressively larger existing recall paths.",
        "learning_contract": {
            "core_formula_changed": False,
            "new_edges_created": False,
            "selected_paths_fixed_from_initial_binding": True,
            "strategies": {
                "specific_only": "Only edges unique to simultaneous Binding.",
                "specific_plus_one_hop": "Specific edges plus adjacent bound edges.",
                "short_bridge": "Existing shortest path from entity input nodes to the specific Binding region, plus specific edges.",
            },
            "teacher": None,
            "reward": None,
            "correct_action": None,
        },
        "diagnostics": {"P": p, "E": e},
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v6.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v6</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:23px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:650px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v6</h1><p class="lead">Bindingの印だけ、手前1段を含む経路、主体入力からBinding領域までの短い橋を比較する。新しいEdgeは作らず、実際にBinding時に通った既存Edgeだけを現行式で強化する。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">比較する</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ比較していません。</pre></section></main><script>
function f(x){return Number(x).toFixed(6)}function card(entity,label,s,target){const ok=s.probe.node_selectivity.winner===target;return `<div class="metric">${entity} ${label} Edge数<b>${s.selected_edge_count}</b></div><div class="metric">${entity} ${label} 選択性<b class="${ok?'good':'warn'}">${f(s.probe.node_selectivity.margin)} / ${s.probe.node_selectivity.winner}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;document.getElementById('metrics').innerHTML=card('P','固有',P.strategies.specific_only,P.trained_position)+card('P','前1段',P.strategies.specific_plus_one_hop,P.trained_position)+card('P','短い橋',P.strategies.short_bridge,P.trained_position)+card('E','固有',E.strategies.specific_only,E.trained_position)+card('E','前1段',E.strategies.specific_plus_one_hop,E.trained_position)+card('E','短い橋',E.strategies.short_bridge,E.trained_position)+`<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v6: http://{HOST}:{PORT}")
    print("Existing-edge recall path comparison / no teacher / no reward")
    serve(app, host=HOST, port=PORT)
