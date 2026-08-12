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
PORT = 5038
OUT = ROOT / "data" / "core_growth_binding_v4" / "results"
POSITIONS = v3.POSITIONS
REPEATS = v3.TRAIN_REPEATS
DELTA_EPS = 1e-12


def edge_set(edges):
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def weight_delta_report(before, after, reference_sets: dict[str, set[tuple[int, int]]]):
    delta = np.asarray(after.weights - before.weights, dtype=float)
    rows, cols = np.triu_indices(before.node_count, k=1)
    changed = []
    for a, b in zip(rows, cols):
        value = float(delta[a, b])
        if abs(value) > DELTA_EPS:
            changed.append((int(a), int(b), value))
    changed_edges = {(a, b) for a, b, _ in changed}
    positive = [value for _, _, value in changed if value > 0]
    negative = [value for _, _, value in changed if value < 0]
    categories = {}
    covered: set[tuple[int, int]] = set()
    for name, refs in reference_sets.items():
        overlap = changed_edges & refs
        covered |= overlap
        categories[name] = {
            "changed_edge_count": len(overlap),
            "reference_edge_count": len(refs),
            "coverage": 0.0 if not refs else len(overlap) / len(refs),
            "changed_share": 0.0 if not changed_edges else len(overlap) / len(changed_edges),
        }
    outside = changed_edges - covered
    return {
        "unique_changed_edge_count": len(changed_edges),
        "positive_edge_count": len(positive),
        "negative_edge_count": len(negative),
        "max_increase": max(positive, default=0.0),
        "max_decrease": min(negative, default=0.0),
        "mean_abs_change": 0.0 if not changed else float(np.mean([abs(v) for _, _, v in changed])),
        "sum_abs_change": float(sum(abs(v) for _, _, v in changed)),
        "categories": categories,
        "outside_reference_count": len(outside),
        "outside_reference_share": 0.0 if not changed_edges else len(outside) / len(changed_edges),
        "top_changes": [
            {"a": a, "b": b, "delta": value}
            for a, b, value in sorted(changed, key=lambda item: abs(item[2]), reverse=True)[:60]
        ],
        "changed_edges": [list(edge) for edge in sorted(changed_edges)],
    }


def binding_references(entity: str):
    refs = {}
    for position in POSITIONS:
        binding = v3.make_binding(copy.deepcopy(v3.base.CORE), entity, position, learn=False)
        refs[position] = binding
    return refs


def probe_similarities(brain, entity: str, refs):
    probe = v3.propagate(copy.deepcopy(brain), v3.entity_nodes(entity), learn=False, steps=10)
    similarities = {}
    for position, ref in refs.items():
        similarities[position] = {
            "node": v3.jaccard(probe["activated_nodes"], ref["bound_nodes"]),
            "edge": v3.edge_jaccard(probe["traversed_edges"], ref["bound_edges"]),
        }
    return {"probe": probe, "similarities": similarities}


def selectivity(similarities, target: str, key: str):
    target_value = float(similarities[target][key])
    others = [float(value[key]) for position, value in similarities.items() if position != target]
    best_other = max(others, default=0.0)
    return {
        "target": target_value,
        "best_other": best_other,
        "margin": target_value - best_other,
        "winner": max(similarities, key=lambda position: similarities[position][key]),
    }


def train_entity_only(entity: str):
    brain = copy.deepcopy(v3.base.CORE)
    for _ in range(REPEATS):
        v3.propagate(brain, v3.entity_nodes(entity), learn=True, steps=8)
    return brain


def train_position_only(position: str):
    brain = copy.deepcopy(v3.base.CORE)
    for _ in range(REPEATS):
        v3.propagate(brain, v3.position_nodes(position), learn=True, steps=10)
    return brain


def train_binding(entity: str, position: str):
    brain = copy.deepcopy(v3.base.CORE)
    logs = []
    for repeat in range(REPEATS):
        binding = v3.make_binding(brain, entity, position, learn=True)
        logs.append({
            "repeat": repeat + 1,
            "bound_node_count": len(binding["bound_nodes"]),
            "bound_edge_count": len(binding["bound_edges"]),
            "interaction_node_count": len(binding["interaction_nodes"]),
            "interaction_edge_count": len(binding["interaction_edges"]),
        })
    return brain, logs


def overlap_report(a_edges, b_edges):
    a, b = edge_set(a_edges), edge_set(b_edges)
    union = a | b
    return {
        "intersection": len(a & b),
        "union": len(union),
        "jaccard": 1.0 if not union else len(a & b) / len(union),
        "only_a": len(a - b),
        "only_b": len(b - a),
    }


def diagnose(entity: str, position: str):
    refs = binding_references(entity)
    target_ref = refs[position]
    entity_ref = v3.propagate(copy.deepcopy(v3.base.CORE), v3.entity_nodes(entity), learn=False, steps=8)
    position_ref = v3.propagate(copy.deepcopy(v3.base.CORE), v3.position_nodes(position), learn=False, steps=10)
    reference_sets = {
        "target_binding": edge_set(target_ref["bound_edges"]),
        "target_interaction": edge_set(target_ref["interaction_edges"]),
        "entity_path": edge_set(entity_ref["traversed_edges"]),
        "position_path": edge_set(position_ref["traversed_edges"]),
    }

    before_probe = probe_similarities(v3.base.CORE, entity, refs)
    entity_brain = train_entity_only(entity)
    position_brain = train_position_only(position)
    binding_brain, training_log = train_binding(entity, position)

    entity_delta = weight_delta_report(v3.base.CORE, entity_brain, reference_sets)
    position_delta = weight_delta_report(v3.base.CORE, position_brain, reference_sets)
    binding_delta = weight_delta_report(v3.base.CORE, binding_brain, reference_sets)

    after_probe = probe_similarities(binding_brain, entity, refs)
    entity_after_probe = probe_similarities(entity_brain, entity, refs)
    position_after_probe = probe_similarities(position_brain, entity, refs)

    return {
        "entity": entity,
        "trained_position": position,
        "repeats": REPEATS,
        "references": {
            "target_binding": {
                "bound_node_count": len(target_ref["bound_nodes"]),
                "bound_edge_count": len(target_ref["bound_edges"]),
                "interaction_node_count": len(target_ref["interaction_nodes"]),
                "interaction_edge_count": len(target_ref["interaction_edges"]),
            },
        },
        "training_log": training_log,
        "weight_changes": {
            "entity_only": entity_delta,
            "position_only": position_delta,
            "binding": binding_delta,
        },
        "changed_edge_overlap": {
            "binding_vs_entity": overlap_report(binding_delta["changed_edges"], entity_delta["changed_edges"]),
            "binding_vs_position": overlap_report(binding_delta["changed_edges"], position_delta["changed_edges"]),
            "entity_vs_position": overlap_report(entity_delta["changed_edges"], position_delta["changed_edges"]),
        },
        "recall": {
            "before": before_probe,
            "after_binding_training": after_probe,
            "after_entity_only": entity_after_probe,
            "after_position_only": position_after_probe,
            "node_selectivity_before": selectivity(before_probe["similarities"], position, "node"),
            "node_selectivity_after": selectivity(after_probe["similarities"], position, "node"),
            "edge_selectivity_before": selectivity(before_probe["similarities"], position, "edge"),
            "edge_selectivity_after": selectivity(after_probe["similarities"], position, "edge"),
        },
    }


def observe(player: str, other: str):
    p = diagnose("P", player)
    e = diagnose("E", other)
    payload = {
        "experiment": "Core Growth Binding v4",
        "world": {"P": player, "E": other},
        "purpose": "Separate broad reinforcement from binding-specific learning and test recall selectivity.",
        "settings": {
            "repeats": REPEATS,
            "echo_strength": v3.ECHO_STRENGTH,
            "echo_limit": v3.ECHO_LIMIT,
            "noise": 0.0,
            "structural_assist_modified": False,
        },
        "diagnostics": {"P": p, "E": e},
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v4.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v4</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1400px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:25px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:560px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v4</h1><p class="lead">Binding学習・主体単独学習・位置単独学習を分離し、変化したユニークEdgeと想起の位置選択性を比較する。正解行動・目的・報酬・教師はない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">診断する</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;const pm0=P.recall.node_selectivity_before,pm1=P.recall.node_selectivity_after,em0=E.recall.node_selectivity_before,em1=E.recall.node_selectivity_after;document.getElementById('metrics').innerHTML=`<div class="metric">P Binding変化Edge<b>${P.weight_changes.binding.unique_changed_edge_count}</b></div><div class="metric">P 範囲外割合<b>${P.weight_changes.binding.outside_reference_share.toFixed(3)}</b></div><div class="metric">P選択性 前→後<b>${pm0.margin.toFixed(3)} → ${pm1.margin.toFixed(3)}</b></div><div class="metric">P想起勝者<b class="${pm1.winner===P.trained_position?'good':'warn'}">${pm1.winner}</b></div><div class="metric">E Binding変化Edge<b>${E.weight_changes.binding.unique_changed_edge_count}</b></div><div class="metric">E 範囲外割合<b>${E.weight_changes.binding.outside_reference_share.toFixed(3)}</b></div><div class="metric">E選択性 前→後<b>${em0.margin.toFixed(3)} → ${em1.margin.toFixed(3)}</b></div><div class="metric">E想起勝者<b class="${em1.winner===E.trained_position?'good':'warn'}">${em1.winner}</b></div><div class="metric">P Binding/主体重なり<b>${P.changed_edge_overlap.binding_vs_entity.jaccard.toFixed(3)}</b></div><div class="metric">P Binding/位置重なり<b>${P.changed_edge_overlap.binding_vs_position.jaccard.toFixed(3)}</b></div><div class="metric">E Binding/主体重なり<b>${E.changed_edge_overlap.binding_vs_entity.jaccard.toFixed(3)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v4: http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
