from __future__ import annotations

import copy
import json
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

HOST = "127.0.0.1"
PORT = 5039
OUT = ROOT / "data" / "core_growth_binding_v5" / "results"
POSITIONS = v3.POSITIONS
REPEATS = v3.TRAIN_REPEATS
EPS = 1e-12


def edge_set(edges):
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def reinforce_selected(brain, edges):
    """Use the current Core reinforcement equation, but only on selected Binding edges.

    This experiment intentionally does not call Core._reinforce(), because that method
    also decays every connected edge. The strengthening equation itself is unchanged:
    w <- w + learning_rate * (1 - w).
    """
    selected = sorted(edge_set(edges))
    touched_nodes = set()
    for a, b in selected:
        current = float(brain.weights[a, b])
        updated = current + brain.learning_rate * (1.0 - current)
        brain.weights[a, b] = updated
        brain.weights[b, a] = updated
        brain.usage[a, b] += 1
        brain.usage[b, a] += 1
        touched_nodes.update((a, b))
    for node in touched_nodes:
        brain.node_usage[node] += 1
    brain.weights = np.clip(brain.weights, 0.0, 1.0)
    return selected


def binding_components(brain, entity, position):
    binding = v3.make_binding(copy.deepcopy(brain), entity, position, learn=False)
    entity_only = v3.propagate(
        copy.deepcopy(brain), v3.entity_nodes(entity), learn=False, steps=8
    )
    position_only = v3.propagate(
        copy.deepcopy(brain), v3.position_nodes(position), learn=False, steps=10
    )
    bound = edge_set(binding["bound_edges"])
    entity_edges = edge_set(entity_only["traversed_edges"])
    position_edges = edge_set(position_only["traversed_edges"])
    binding_only = bound - entity_edges - position_edges
    return {
        "binding": binding,
        "bound_edges": bound,
        "entity_edges": entity_edges,
        "position_edges": position_edges,
        "binding_only_edges": binding_only,
    }


def train_normal(entity, position):
    brain = copy.deepcopy(v3.base.CORE)
    logs = []
    for repeat in range(REPEATS):
        before = brain.weights.copy()
        binding = v3.make_binding(brain, entity, position, learn=True)
        changed = int(np.count_nonzero(np.abs(brain.weights - before) > EPS) // 2)
        logs.append({
            "repeat": repeat + 1,
            "changed_edges": changed,
            "interaction_edges_observed": len(binding["interaction_edges"]),
        })
    return brain, logs


def train_limited(entity, position):
    brain = copy.deepcopy(v3.base.CORE)
    logs = []
    all_selected = set()
    for repeat in range(REPEATS):
        components = binding_components(brain, entity, position)
        selected = reinforce_selected(brain, components["binding_only_edges"])
        all_selected.update(selected)
        logs.append({
            "repeat": repeat + 1,
            "bound_edges": len(components["bound_edges"]),
            "entity_edges": len(components["entity_edges"]),
            "position_edges": len(components["position_edges"]),
            "selected_binding_only_edges": len(selected),
        })
    return brain, logs, all_selected


def changed_edges(before, after):
    delta = np.asarray(after.weights - before.weights, dtype=float)
    rows, cols = np.triu_indices(before.node_count, k=1)
    changed = []
    for a, b in zip(rows, cols):
        value = float(delta[a, b])
        if abs(value) > EPS:
            changed.append((int(a), int(b), value))
    return changed


def references(entity):
    return {
        position: v3.make_binding(
            copy.deepcopy(v3.base.CORE), entity, position, learn=False
        )
        for position in POSITIONS
    }


def probe(brain, entity, refs):
    result = v3.propagate(
        copy.deepcopy(brain), v3.entity_nodes(entity), learn=False, steps=10
    )
    scores = {}
    for position, ref in refs.items():
        scores[position] = {
            "node": v3.jaccard(result["activated_nodes"], ref["bound_nodes"]),
            "edge": v3.edge_jaccard(result["traversed_edges"], ref["bound_edges"]),
        }
    return {"probe": result, "scores": scores}


def selectivity(scores, target, key):
    target_value = float(scores[target][key])
    others = {
        position: float(value[key])
        for position, value in scores.items()
        if position != target
    }
    best_other_position = max(others, key=others.get)
    best_other = others[best_other_position]
    winner = max(scores, key=lambda position: float(scores[position][key]))
    return {
        "target": target_value,
        "best_other": best_other,
        "best_other_position": best_other_position,
        "margin": target_value - best_other,
        "winner": winner,
    }


def diagnose(entity, position):
    refs = references(entity)
    baseline = probe(v3.base.CORE, entity, refs)

    normal_brain, normal_log = train_normal(entity, position)
    limited_brain, limited_log, selected = train_limited(entity, position)

    normal_probe = probe(normal_brain, entity, refs)
    limited_probe = probe(limited_brain, entity, refs)

    normal_changes = changed_edges(v3.base.CORE, normal_brain)
    limited_changes = changed_edges(v3.base.CORE, limited_brain)
    normal_set = {(a, b) for a, b, _ in normal_changes}
    limited_set = {(a, b) for a, b, _ in limited_changes}

    initial_components = binding_components(v3.base.CORE, entity, position)
    target_binding_only = initial_components["binding_only_edges"]

    return {
        "entity": entity,
        "trained_position": position,
        "baseline": baseline,
        "normal": {
            "training_log": normal_log,
            "unique_changed_edges": len(normal_set),
            "target_binding_only_coverage": (
                0.0 if not target_binding_only
                else len(normal_set & target_binding_only) / len(target_binding_only)
            ),
            "outside_target_share": (
                0.0 if not normal_set
                else len(normal_set - target_binding_only) / len(normal_set)
            ),
            "probe": normal_probe,
            "node_selectivity": selectivity(normal_probe["scores"], position, "node"),
            "edge_selectivity": selectivity(normal_probe["scores"], position, "edge"),
        },
        "limited": {
            "training_log": limited_log,
            "selected_unique_edges": len(selected),
            "unique_changed_edges": len(limited_set),
            "all_changes_within_selected": limited_set <= selected,
            "target_binding_only_coverage": (
                0.0 if not target_binding_only
                else len(limited_set & target_binding_only) / len(target_binding_only)
            ),
            "outside_target_share": (
                0.0 if not limited_set
                else len(limited_set - target_binding_only) / len(limited_set)
            ),
            "probe": limited_probe,
            "node_selectivity": selectivity(limited_probe["scores"], position, "node"),
            "edge_selectivity": selectivity(limited_probe["scores"], position, "edge"),
        },
        "baseline_node_selectivity": selectivity(baseline["scores"], position, "node"),
        "baseline_edge_selectivity": selectivity(baseline["scores"], position, "edge"),
        "initial_binding_only_edge_count": len(target_binding_only),
        "normal_vs_limited_changed_edge_jaccard": (
            1.0 if not (normal_set | limited_set)
            else len(normal_set & limited_set) / len(normal_set | limited_set)
        ),
    }


def observe(player, other):
    payload = {
        "experiment": "Core Growth Binding v5",
        "world": {"P": player, "E": other},
        "purpose": "Compare normal broad learning with experiment-side Binding-limited learning.",
        "learning_contract": {
            "core_learning_formula_changed": False,
            "normal": "Current Core learning through every traversed edge plus global decay.",
            "limited": "Same per-edge strengthening equation, applied only to edges that appear in Binding but in neither entity-only nor position-only propagation.",
            "correct_action": None,
            "reward": None,
            "teacher": None,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v5.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v5</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1450px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:24px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:620px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v5</h1><p class="lead">通常学習とBinding限定学習を比較する。Binding限定学習は、主体単独・位置単独では通らず、両者を同じ時間窓で重ねた時だけ現れたEdgeだけを強化する。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">比較する</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ比較していません。</pre></section></main><script>
function n(x){return Number(x).toFixed(6)}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;document.getElementById('metrics').innerHTML=`<div class="metric">P Binding固有Edge<b>${P.initial_binding_only_edge_count}</b></div><div class="metric">P 通常変化Edge<b>${P.normal.unique_changed_edges}</b></div><div class="metric">P 限定変化Edge<b class="blue">${P.limited.unique_changed_edges}</b></div><div class="metric">P 限定範囲外<b class="${P.limited.outside_target_share===0?'good':'warn'}">${n(P.limited.outside_target_share)}</b></div><div class="metric">P通常 選択性<b>${n(P.normal.node_selectivity.margin)} / ${P.normal.node_selectivity.winner}</b></div><div class="metric">P限定 選択性<b class="${P.limited.node_selectivity.winner===P.trained_position?'good':'warn'}">${n(P.limited.node_selectivity.margin)} / ${P.limited.node_selectivity.winner}</b></div><div class="metric">E Binding固有Edge<b>${E.initial_binding_only_edge_count}</b></div><div class="metric">E 限定変化Edge<b class="blue">${E.limited.unique_changed_edges}</b></div><div class="metric">E通常 選択性<b>${n(E.normal.node_selectivity.margin)} / ${E.normal.node_selectivity.winner}</b></div><div class="metric">E限定 選択性<b class="${E.limited.node_selectivity.winner===E.trained_position?'good':'warn'}">${n(E.limited.node_selectivity.margin)} / ${E.limited.node_selectivity.winner}</b></div><div class="metric">限定式内一致<b>${P.limited.all_changes_within_selected&&E.limited.all_changes_within_selected?'YES':'NO'}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v5: http://{HOST}:{PORT}")
    print("Normal learning vs Binding-limited learning / no teacher / no reward")
    serve(app, host=HOST, port=PORT)
