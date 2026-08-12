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

import run_core_growth_microscope_v1 as base

HOST = "127.0.0.1"
PORT = 5037
OUT = ROOT / "data" / "core_growth_binding_v3" / "results"
POSITIONS = base.POSITIONS
ECHO_STRENGTH = 0.68
ECHO_LIMIT = 8
TRAIN_REPEATS = 12


def jaccard(a, b):
    sa, sb = set(a), set(b)
    u = sa | sb
    return 1.0 if not u else len(sa & sb) / len(u)


def edge_jaccard(a, b):
    sa = {tuple(sorted(x)) for x in a}
    sb = {tuple(sorted(x)) for x in b}
    u = sa | sb
    return 1.0 if not u else len(sa & sb) / len(u)


def summarize(result, traces):
    final = np.asarray(result.final_activation, dtype=float)
    return {
        "source_nodes": list(result.source_nodes),
        "activated_nodes": list(result.activated_nodes),
        "traversed_edges": [list(x) for x in result.traversed_edges],
        "activation_history": [list(x) for x in result.activation_history],
        "final_active_nodes": np.flatnonzero(final > 0).tolist(),
        "final_energy": float(final.sum()),
        "assist_activations": sum(1 for x in traces if x.get("tie_gate_active")),
        "assist_rank_changes": sum(1 for x in traces if x.get("top_candidate_changed")),
    }


def entity_nodes(entity):
    return base.PORTS[f"entity:{entity}"]


def position_nodes(position):
    return base.PORTS[f"position:{position}"]


def propagate(brain, sources, *, context=None, learn=False, assist=False, steps=10):
    brain.set_structural_assist(assist)
    # Temporarily raise context activity so the echo can actually propagate.
    original = brain._initial_activation

    def initial(source_nodes, context_nodes):
        src = list(source_nodes)
        activation = np.zeros(brain.node_count, dtype=float)
        for node in src:
            activation[node] = 1.0
        if context_nodes:
            for node in context_nodes:
                activation[node] = max(activation[node], ECHO_STRENGTH)
        return src, activation

    brain._initial_activation = initial
    try:
        result = brain.propagate(
            sources,
            context_nodes=context or None,
            steps=steps,
            threshold=0.18,
            noise=0.0,
            learn=learn,
        )
    finally:
        brain._initial_activation = original
    return summarize(result, list(brain.last_structural_assist_trace))


def make_binding(brain, entity, position, *, learn=False, assist=False):
    entity_stage = propagate(brain, entity_nodes(entity), learn=learn, assist=assist, steps=8)
    echo = entity_stage["final_active_nodes"][:ECHO_LIMIT]
    position_alone = propagate(copy.deepcopy(brain), position_nodes(position), learn=False, assist=False, steps=10)
    bound_stage = propagate(
        brain,
        position_nodes(position),
        context=echo,
        learn=learn,
        assist=assist,
        steps=10,
    )
    bn, pn = set(bound_stage["activated_nodes"]), set(position_alone["activated_nodes"])
    be = {tuple(sorted(x)) for x in bound_stage["traversed_edges"]}
    pe = {tuple(sorted(x)) for x in position_alone["traversed_edges"]}
    entity_nodes_set = set(entity_stage["activated_nodes"])
    return {
        "label": f"{entity}@{position}",
        "entity": entity,
        "position": position,
        "echo_nodes": echo,
        "entity_stage": entity_stage,
        "position_alone": position_alone,
        "bound_stage": bound_stage,
        "bound_nodes": sorted(bn),
        "bound_edges": [list(x) for x in sorted(be)],
        "interaction_nodes": sorted(bn - pn),
        "interaction_edges": [list(x) for x in sorted(be - pe)],
        "echo_to_new_edges": [list(x) for x in sorted(
            x for x in be if (x[0] in set(echo) or x[1] in set(echo)) and x not in pe
        )],
        "entity_bound_overlap": jaccard(entity_nodes_set, bn),
    }


def specificity_matrix():
    rows = []
    bindings = {}
    for entity in ("P", "E"):
        for position in POSITIONS:
            b = make_binding(copy.deepcopy(base.CORE), entity, position)
            bindings[b["label"]] = b
    labels = list(bindings)
    for a in labels:
        row = {"binding": a}
        for b in labels:
            row[b] = {
                "node": jaccard(bindings[a]["bound_nodes"], bindings[b]["bound_nodes"]),
                "edge": edge_jaccard(bindings[a]["bound_edges"], bindings[b]["bound_edges"]),
            }
        rows.append(row)
    return {"labels": labels, "rows": rows, "bindings": bindings}


def boundary_test(player, other):
    brain = copy.deepcopy(base.CORE)
    a = make_binding(brain, "P", player)
    # Hard boundary: no context from A is passed into B.
    b = make_binding(brain, "E", other)
    a_nodes = set(a["bound_nodes"])
    b_entity = set(b["entity_stage"]["activated_nodes"])
    b_bound = set(b["bound_nodes"])
    leaked = sorted((a_nodes & b_bound) - b_entity)
    return {
        "binding_a": a,
        "boundary": {"type": "hard", "passed_context_nodes": []},
        "binding_b": b,
        "cross_window_node_jaccard": jaccard(a["bound_nodes"], b["bound_nodes"]),
        "possible_leaked_nodes": leaked,
        "possible_leak_count": len(leaked),
    }


def recall_test(entity, position):
    untrained = copy.deepcopy(base.CORE)
    baseline_probe = propagate(copy.deepcopy(untrained), entity_nodes(entity), learn=False, steps=10)

    trained = copy.deepcopy(base.CORE)
    training_log = []
    for i in range(TRAIN_REPEATS):
        b = make_binding(trained, entity, position, learn=True)
        training_log.append({
            "repeat": i + 1,
            "interaction_nodes": len(b["interaction_nodes"]),
            "interaction_edges": len(b["interaction_edges"]),
        })

    trained_probe = propagate(copy.deepcopy(trained), entity_nodes(entity), learn=False, steps=10)
    target_reference = make_binding(copy.deepcopy(base.CORE), entity, position, learn=False)
    other_refs = {
        p: make_binding(copy.deepcopy(base.CORE), entity, p, learn=False)
        for p in POSITIONS if p != position
    }
    return {
        "entity": entity,
        "trained_position": position,
        "repeats": TRAIN_REPEATS,
        "training_log": training_log,
        "baseline_probe": baseline_probe,
        "trained_probe": trained_probe,
        "target_similarity_before": {
            "node": jaccard(baseline_probe["activated_nodes"], target_reference["bound_nodes"]),
            "edge": edge_jaccard(baseline_probe["traversed_edges"], target_reference["bound_edges"]),
        },
        "target_similarity_after": {
            "node": jaccard(trained_probe["activated_nodes"], target_reference["bound_nodes"]),
            "edge": edge_jaccard(trained_probe["traversed_edges"], target_reference["bound_edges"]),
        },
        "other_position_similarity_after": {
            p: {
                "node": jaccard(trained_probe["activated_nodes"], ref["bound_nodes"]),
                "edge": edge_jaccard(trained_probe["traversed_edges"], ref["bound_edges"]),
            }
            for p, ref in other_refs.items()
        },
        "weight_change": {
            "max_abs": float(np.max(np.abs(trained.weights - base.CORE.weights))),
            "changed_edge_count": int(np.count_nonzero(np.abs(trained.weights - base.CORE.weights) > 1e-12) // 2),
        },
    }


def assist_shadow(player, other):
    off_a = make_binding(copy.deepcopy(base.CORE), "P", player, assist=False)
    on_a = make_binding(copy.deepcopy(base.CORE), "P", player, assist=True)
    off_b = make_binding(copy.deepcopy(base.CORE), "E", other, assist=False)
    on_b = make_binding(copy.deepcopy(base.CORE), "E", other, assist=True)
    return {
        "P": {
            "node_jaccard": jaccard(off_a["bound_nodes"], on_a["bound_nodes"]),
            "edge_jaccard": edge_jaccard(off_a["bound_edges"], on_a["bound_edges"]),
            "rank_changes": on_a["bound_stage"]["assist_rank_changes"],
        },
        "E": {
            "node_jaccard": jaccard(off_b["bound_nodes"], on_b["bound_nodes"]),
            "edge_jaccard": edge_jaccard(off_b["bound_edges"], on_b["bound_edges"]),
            "rank_changes": on_b["bound_stage"]["assist_rank_changes"],
        },
    }


def observe(player, other):
    payload = {
        "experiment": "Core Growth Binding v3",
        "world": {"P": player, "E": other},
        "settings": {
            "echo_strength": ECHO_STRENGTH,
            "echo_limit": ECHO_LIMIT,
            "train_repeats": TRAIN_REPEATS,
            "learning_in_specificity": False,
            "noise": 0.0,
            "structural_assist_modified": False,
        },
        "specificity": specificity_matrix(),
        "boundary": boundary_test(player, other),
        "recall": {
            "P": recall_test("P", player),
            "E": recall_test("E", other),
        },
        "structural_assist_shadow": assist_shadow(player, other),
        "brain_file_unchanged": base.BEFORE_HASH == base.sha(base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v3.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v3</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1350px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:27px;margin-top:6px}.good{color:var(--green)}.raw{white-space:pre-wrap;max-height:520px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:850px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v3</h1><p class="lead">Binding固有性、境界分離、反復後の再利用を順番に検証する。正解行動・目的・報酬・教師はない。Structural Assistは現行のままShadow比較のみ。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">検証する</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ検証していません。</pre></section></main><script>
async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const rp=d.recall.P,re=d.recall.E,as=d.structural_assist_shadow;document.getElementById('metrics').innerHTML=`<div class="metric">境界漏れNode<b>${d.boundary.possible_leak_count}</b></div><div class="metric">P想起Node 前→後<b>${rp.target_similarity_before.node.toFixed(3)} → ${rp.target_similarity_after.node.toFixed(3)}</b></div><div class="metric">E想起Node 前→後<b>${re.target_similarity_before.node.toFixed(3)} → ${re.target_similarity_after.node.toFixed(3)}</b></div><div class="metric">P学習Edge数<b>${rp.weight_change.changed_edge_count}</b></div><div class="metric">E学習Edge数<b>${re.weight_change.changed_edge_count}</b></div><div class="metric">Assist P差<b>${as.P.node_jaccard.toFixed(3)}</b></div><div class="metric">Assist E差<b>${as.E.node_jaccard.toFixed(3)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v3: http://{HOST}:{PORT}")
    serve(app, host=HOST, port=PORT)
