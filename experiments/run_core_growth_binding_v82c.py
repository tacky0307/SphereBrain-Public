from __future__ import annotations

import hashlib
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
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v82 as v82
import run_core_growth_binding_v82b as v82b

HOST = "127.0.0.1"
START_PORT = 5131
OUT = ROOT / "data" / "core_growth_binding_v82c" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v82b.CHECKPOINTS

BIRD = v82b.BIRD
PLANE = v82b.PLANE
BIRD_SKY = v82.StructuredInput("鳥", "場所", "空")
PLANE_SKY = v82.StructuredInput("飛行機", "場所", "空")
BIRD_FOREST = v82.StructuredInput("鳥", "場所", "森")
PLANE_AIRPORT = v82.StructuredInput("飛行機", "場所", "空港")


def choose_port(start: int) -> int:
    for port in range(start, start + 50):
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


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def sig(brain: SphereBrain, item: v82.StructuredInput) -> dict:
    return v82b.signature(brain, item)


def shared_action_edges(brain: SphereBrain) -> set[tuple[int, int]]:
    return sig(brain, BIRD)["edges"] & sig(brain, PLANE)["edges"]


def sky_signature(brain: SphereBrain) -> dict:
    left = sig(brain, BIRD_SKY)
    right = sig(brain, PLANE_SKY)
    return {
        "nodes": left["nodes"] | right["nodes"],
        "edges": left["edges"] | right["edges"],
        "shared_nodes": left["nodes"] & right["nodes"],
        "shared_edges": left["edges"] & right["edges"],
    }


def control_context_signature(brain: SphereBrain) -> dict:
    left = sig(brain, BIRD_FOREST)
    right = sig(brain, PLANE_AIRPORT)
    return {"nodes": left["nodes"] | right["nodes"], "edges": left["edges"] | right["edges"]}


def min_hops(brain: SphereBrain, starts: set[int], targets: set[int], cap: int = 8) -> int | None:
    if not starts or not targets:
        return None
    if starts & targets:
        return 0
    q = deque((int(x), 0) for x in starts)
    seen = set(starts)
    while q:
        node, depth = q.popleft()
        if depth >= cap:
            continue
        for nxt in np.flatnonzero(brain.adjacency[node]).tolist():
            nxt = int(nxt)
            if nxt in seen:
                continue
            if nxt in targets:
                return depth + 1
            seen.add(nxt)
            q.append((nxt, depth + 1))
    return None


def node_degree(brain: SphereBrain, node: int) -> int:
    return int(np.count_nonzero(brain.adjacency[int(node)]))


def usage_percentile(brain: SphereBrain, value: float) -> float:
    vals = np.asarray(brain.node_usage, dtype=float)
    return float(np.mean(vals <= float(value))) if vals.size else 0.0


def annotate_edge(
    brain: SphereBrain,
    edge: tuple[int, int],
    *,
    first_cycle: int | None,
    sky: dict,
    control_shared: set[tuple[int, int]],
    control_context: dict,
) -> dict:
    a, b = [int(x) for x in edge]
    endpoints = {a, b}
    in_sky_edge = edge in sky["edges"]
    endpoint_in_sky = bool(endpoints & sky["nodes"])
    endpoint_in_shared_sky = bool(endpoints & sky["shared_nodes"])
    hops = min_hops(brain, endpoints, set(sky["nodes"]))
    control_seen = edge in control_shared
    control_context_seen = edge in control_context["edges"]
    usage_a = float(brain.node_usage[a])
    usage_b = float(brain.node_usage[b])
    percentile = max(usage_percentile(brain, usage_a), usage_percentile(brain, usage_b))
    degree = max(node_degree(brain, a), node_degree(brain, b))

    context_linked = (
        not control_seen
        and (in_sky_edge or endpoint_in_shared_sky or (hops is not None and hops <= 1))
    )
    popular_like = (percentile >= 0.90 or degree >= max(12, brain.neighbors_per_node * 2)) and not context_linked
    if context_linked:
        attribution = "context_linked_bridge_candidate"
    elif popular_like:
        attribution = "popular_hub_like"
    elif control_seen or control_context_seen:
        attribution = "control_shared_or_context_generic"
    else:
        attribution = "unexplained_new_shared_edge"

    return {
        "edge": [a, b],
        "first_cycle": first_cycle,
        "in_sky_edge": in_sky_edge,
        "endpoint_in_sky": endpoint_in_sky,
        "endpoint_in_shared_sky": endpoint_in_shared_sky,
        "min_hops_to_sky": hops,
        "control_action_shared_seen": control_seen,
        "control_context_seen": control_context_seen,
        "node_usage": [usage_a, usage_b],
        "node_usage_percentile_max": percentile,
        "degree": [node_degree(brain, a), node_degree(brain, b)],
        "degree_max": degree,
        "attribution": attribution,
    }


def run_attribution() -> dict:
    experimental = v82.clean_primary()
    control = v82.clean_primary()
    exp_baseline = shared_action_edges(experimental)
    ctrl_baseline = shared_action_edges(control)

    exp_first: dict[tuple[int, int], int] = {}
    ctrl_first: dict[tuple[int, int], int] = {}
    cycle_rows = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            exp_shared = shared_action_edges(experimental)
            ctrl_shared = shared_action_edges(control)
            exp_new = exp_shared - exp_baseline
            ctrl_new = ctrl_shared - ctrl_baseline
            for edge in exp_new:
                exp_first.setdefault(edge, cycle)
            for edge in ctrl_new:
                ctrl_first.setdefault(edge, cycle)
            sky = sky_signature(experimental)
            ctrl_ctx = control_context_signature(control)
            cycle_rows.append({
                "cycle": cycle,
                "experimental_new_shared_edges": [list(x) for x in sorted(exp_new)],
                "control_new_shared_edges": [list(x) for x in sorted(ctrl_new)],
                "experimental_new_count": len(exp_new),
                "control_new_count": len(ctrl_new),
                "effect": len(exp_new) - len(ctrl_new),
                "sky_internal_nodes": len(sky["nodes"]),
                "sky_shared_edges": len(sky["shared_edges"]),
            })
        if cycle < max(CHECKPOINTS):
            v82.learn_items(experimental, v82b.EXP_CURRICULUM)
            v82.learn_items(control, v82b.CTRL_CURRICULUM)

    final_exp_shared = shared_action_edges(experimental)
    final_ctrl_shared = shared_action_edges(control)
    final_exp_new = final_exp_shared - exp_baseline
    final_ctrl_new = final_ctrl_shared - ctrl_baseline
    sky = sky_signature(experimental)
    ctrl_ctx = control_context_signature(control)

    attributed = [
        annotate_edge(
            experimental,
            edge,
            first_cycle=exp_first.get(edge),
            sky=sky,
            control_shared=final_ctrl_shared,
            control_context=ctrl_ctx,
        )
        for edge in sorted(final_exp_new)
    ]

    return {
        "experimental": experimental,
        "control": control,
        "cycle_rows": cycle_rows,
        "baseline_exp_shared": [list(x) for x in sorted(exp_baseline)],
        "baseline_ctrl_shared": [list(x) for x in sorted(ctrl_baseline)],
        "final_exp_new": [list(x) for x in sorted(final_exp_new)],
        "final_ctrl_new": [list(x) for x in sorted(final_ctrl_new)],
        "attributed_edges": attributed,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    result = run_attribution()
    edges = result["attributed_edges"]
    context_candidates = [x for x in edges if x["attribution"] == "context_linked_bridge_candidate"]
    popular = [x for x in edges if x["attribution"] == "popular_hub_like"]
    generic = [x for x in edges if x["attribution"] == "control_shared_or_context_generic"]
    unexplained = [x for x in edges if x["attribution"] == "unexplained_new_shared_edge"]

    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "bridge_edge_attribution_roundtrip.json"
    learned = result["experimental"]
    before_edges = sorted(shared_action_edges(learned))
    learned.save(temp)
    loaded = SphereBrain.load(temp)
    after_edges = sorted(shared_action_edges(loaded))
    saveload_equal = before_edges == after_edges
    production_unchanged = before_hash == file_hash(BRAIN_PATH)

    if context_candidates:
        verdict = "at_least_one_new_shared_edge_is_structurally_linked_to_the_shared_sky_context"
        readiness = "semantic_bridge_seed_candidate_observed"
        next_step = "test_context_linked_edge_causality_by_selective_ablation_and_episode_linking"
    elif popular:
        verdict = "new_shared_edges_are_better_explained_by_hub_usage_than_shared_context_linkage"
        readiness = "bridge_effect_looks_hub_driven"
        next_step = "compare_hub_normalized_semantic_paths_before_encoder_or_core_changes"
    else:
        verdict = "new_shared_edges_exist_but_no_specific_shared_context_attribution_was_observed"
        readiness = "bridge_edge_origin_unresolved"
        next_step = "trace_stage_to_stage_context_carryover_at_edge_level"

    payload = {
        "experiment": "Core Growth Binding v82C — Bridge Edge Attribution",
        "contract": {
            "primary_core_modified": False,
            "semantic_encoder_modified": False,
            "source_experiment": "v82B Shared Context Bridge Microscope",
            "checkpoints": CHECKPOINTS,
            "direct_input_nodes_excluded": True,
            "production_brain_json_saved": False,
            "attribution_is_observational_not_causal": True,
        },
        "cycle_rows": result["cycle_rows"],
        "attributed_edges": edges,
        "summary": {
            "final_new_shared_edge_count": len(edges),
            "context_linked_candidate_count": len(context_candidates),
            "popular_hub_like_count": len(popular),
            "control_generic_count": len(generic),
            "unexplained_count": len(unexplained),
            "context_linked_candidate_observed": bool(context_candidates),
            "earliest_context_candidate_cycle": min((x["first_cycle"] for x in context_candidates if x["first_cycle"] is not None), default=None),
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode"),
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": saveload_equal and production_unchanged and len(edges) > 0,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    (OUT / "latest_binding_v82c.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v82C</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1050px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v82C：Bridge Edge Attribution</h1><p class="lead">v82Bで増えた新規共有Edgeを1本ずつ追い、共有文脈「空」に近いBridge候補なのか、人気Nodeへの合流なのか、Controlにも現れる一般的な共有なのかを観察する。</p><section class="panel"><div class="controls"><button id="run">新規Edgeの正体を追う</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Edge別 Attribution</h2><pre id="edges" class="raw">未実行</pre></section><section class="panel"><h2>Cycle別</h2><pre id="cycles" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function cyc(v){return v===null||v===undefined?'—':v}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('新規共有Edge',s.final_new_shared_edge_count),metric('Context-linked候補',s.context_linked_candidate_count,s.context_linked_candidate_observed?'good':'warn'),metric('候補最短cycle',cyc(s.earliest_context_candidate_cycle)),metric('Popular hub型',s.popular_hub_like_count),metric('Control/Generic',s.control_generic_count),metric('未説明',s.unexplained_count),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution観測',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('edges').textContent=JSON.stringify(d.attributed_edges,null,2);document.getElementById('cycles').textContent=JSON.stringify(d.cycle_rows,null,2)}document.getElementById('run').onclick=run;</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    try:
        return jsonify(observe())
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v82C: http://{HOST}:{PORT}")
    print("Bridge Edge Attribution / Primary Core変更なし / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
