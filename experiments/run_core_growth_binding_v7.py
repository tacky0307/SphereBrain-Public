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
START_PORT = 5042
OUT = ROOT / "data" / "core_growth_binding_v7" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18


def edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def choose_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
                return port
            except OSError:
                continue
    raise RuntimeError("利用可能なポートが見つかりません。")


def trace_focused(brain, sources, *, context=None, context_strength=0.68, steps=12):
    brain = copy.deepcopy(brain)
    brain.set_structural_assist(False)
    activation = np.zeros(brain.node_count, dtype=float)
    for node in sources:
        activation[int(node)] = 1.0
    for node in context or []:
        activation[int(node)] = max(activation[int(node)], context_strength)

    activated = set(np.flatnonzero(activation > 0).tolist())
    history = [sorted(activated)]
    step_rows = []
    all_edges = set()

    for step_index in range(steps):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break

        candidates = {}
        local_rows = []
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            best_set = {int(neighbors[i]) for i in best_indices}
            for idx, target in enumerate(neighbors):
                raw = float(scores[idx]) * brain.signal_decay
                in_local_top = int(target) in best_set
                accepted_local = bool(in_local_top and raw >= THRESHOLD)
                local_rows.append({
                    "source": int(source),
                    "target": int(target),
                    "edge": list(edge_key(source, target)),
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "score": raw,
                    "in_local_top": in_local_top,
                    "above_threshold": raw >= THRESHOLD,
                    "accepted_local": accepted_local,
                })
                if not accepted_local:
                    continue
                previous = candidates.get(int(target))
                if previous is None or raw > previous[0]:
                    candidates[int(target)] = (raw, int(source))

        if not candidates:
            step_rows.append({"step": step_index, "active_sources": active_sources.tolist(), "local": local_rows, "ranked": [], "selected": []})
            break

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining = max(0, brain.max_total_active_nodes - len(activated))
        step_limit = min(brain.max_active_per_step, len(ranked))
        selected = []
        new_selected = 0
        for target, payload in ranked:
            is_new = target not in activated
            if is_new and new_selected >= remaining:
                continue
            selected.append((target, payload))
            if is_new:
                new_selected += 1
            if len(selected) >= step_limit:
                break

        next_activation = np.zeros(brain.node_count, dtype=float)
        selected_rows = []
        for rank, (target, (value, source)) in enumerate(selected, start=1):
            clipped = float(np.clip(value, 0.0, 1.0))
            if clipped < THRESHOLD:
                continue
            next_activation[target] = max(next_activation[target], clipped)
            edge = edge_key(source, target)
            all_edges.add(edge)
            selected_rows.append({"rank": rank, "source": source, "target": target, "edge": list(edge), "score": clipped})

        step_rows.append({
            "step": step_index,
            "active_sources": active_sources.tolist(),
            "local": local_rows,
            "ranked": [
                {"rank": rank, "target": int(target), "source": int(payload[1]), "edge": list(edge_key(payload[1], target)), "score": float(payload[0])}
                for rank, (target, payload) in enumerate(ranked, start=1)
            ],
            "selected": selected_rows,
        })

        active_now = np.flatnonzero(next_activation > 0).tolist()
        if not active_now:
            break
        activated.update(active_now)
        history.append(active_now)
        activation = next_activation
        if len(activated) >= brain.max_total_active_nodes:
            break

    return {
        "sources": list(map(int, sources)),
        "context": list(map(int, context or [])),
        "activated_nodes": sorted(activated),
        "traversed_edges": [list(x) for x in sorted(all_edges)],
        "activation_history": history,
        "steps": step_rows,
    }


def first_divergence(a_history, b_history):
    length = max(len(a_history), len(b_history))
    for index in range(length):
        a = set(a_history[index]) if index < len(a_history) else set()
        b = set(b_history[index]) if index < len(b_history) else set()
        if a != b:
            return {"step": index, "only_a": sorted(a - b), "only_b": sorted(b - a), "shared": sorted(a & b)}
    return None


def inspect_edges(trace, target_edges):
    target_edges = {edge_key(*edge) for edge in target_edges}
    report = {}
    for edge in sorted(target_edges):
        rows = []
        ever_source_active = False
        ever_neighbor_seen = False
        ever_local_top = False
        ever_above_threshold = False
        ever_candidate = False
        ever_selected = False
        for step in trace["steps"]:
            local_matches = [row for row in step["local"] if tuple(row["edge"]) == edge]
            ranked_matches = [row for row in step["ranked"] if tuple(row["edge"]) == edge]
            selected_matches = [row for row in step["selected"] if tuple(row["edge"]) == edge]
            if local_matches:
                ever_neighbor_seen = True
                ever_source_active = True
                ever_local_top |= any(row["in_local_top"] for row in local_matches)
                ever_above_threshold |= any(row["above_threshold"] for row in local_matches)
            ever_candidate |= bool(ranked_matches)
            ever_selected |= bool(selected_matches)
            if local_matches or ranked_matches or selected_matches:
                rows.append({
                    "step": step["step"],
                    "local": local_matches,
                    "ranked": ranked_matches,
                    "selected": selected_matches,
                })
        if ever_selected:
            status = "selected"
        elif ever_candidate:
            status = "candidate_but_not_selected"
        elif ever_local_top and not ever_above_threshold:
            status = "local_top_but_below_threshold"
        elif ever_neighbor_seen:
            status = "neighbor_seen_but_not_local_top"
        else:
            status = "not_visible_from_active_sources"
        report[f"{edge[0]}-{edge[1]}"] = {
            "edge": list(edge),
            "status": status,
            "ever_neighbor_seen": ever_neighbor_seen,
            "ever_local_top": ever_local_top,
            "ever_above_threshold": ever_above_threshold,
            "ever_candidate": ever_candidate,
            "ever_selected": ever_selected,
            "steps": rows,
        }
    return report


def diagnose_pair(entity: str, position: str):
    components = v5.binding_components(v3.base.CORE, entity, position)
    binding = components["binding"]
    specific = sorted(components["binding_only_edges"])
    entity_sources = v3.entity_nodes(entity)
    position_sources = v3.position_nodes(position)
    echo = binding["echo_nodes"]

    solo = trace_focused(v3.base.CORE, entity_sources, steps=12)
    position_alone = trace_focused(v3.base.CORE, position_sources, steps=12)
    bound = trace_focused(v3.base.CORE, position_sources, context=echo, steps=12)

    return {
        "pair": f"{entity}@{position}",
        "binding_specific_edges": [list(x) for x in specific],
        "entity_solo": solo,
        "position_alone": position_alone,
        "bound_stage": bound,
        "entity_vs_bound_first_divergence": first_divergence(solo["activation_history"], bound["activation_history"]),
        "specific_edge_visibility_in_entity_solo": inspect_edges(solo, specific),
        "specific_edge_visibility_in_position_alone": inspect_edges(position_alone, specific),
        "specific_edge_visibility_in_bound_stage": inspect_edges(bound, specific),
    }


def summarize_status(report):
    counts = {}
    for item in report.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def observe(player: str, other: str):
    p = diagnose_pair("P", player)
    e = diagnose_pair("E", other)
    payload = {
        "experiment": "Core Growth Binding v7",
        "world": {"P": player, "E": other},
        "purpose": "Generic diagnosis of whether Binding-specific edges are visible, competitive, or absent during cue-only recall.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "structural_assist": False,
            "weights_changed": False,
            "new_edges_created": False,
            "puzzle_specific_rules": False,
        },
        "diagnostics": {"P": p, "E": e},
        "summary": {
            "P": summarize_status(p["specific_edge_visibility_in_entity_solo"]),
            "E": summarize_status(e["specific_edge_visibility_in_entity_solo"]),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v7.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v7</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1450px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:22px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:650px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v7</h1><p class="lead">単独手掛かり時にBinding固有Edgeが、候補に出て負けるのか、threshold未満なのか、活動源から見えないのかをStep単位で診断する。学習・補正・パズル専用ルールは一切ない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">診断する</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function fmt(x){return x===null||x===undefined?'なし':JSON.stringify(x)}function cards(entity,diag,sum){const div=diag.entity_vs_bound_first_divergence;return `<div class="metric">${entity} 固有Edge数<b>${diag.binding_specific_edges.length}</b></div><div class="metric">${entity} 最初の分岐Step<b>${div?div.step:'なし'}</b></div><div class="metric">${entity} 単独時Status<b class="${sum.selected?'good':'warn'}">${fmt(sum)}</b></div><div class="metric">${entity} brain保護<b class="good">不変</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=cards('P',d.diagnostics.P,d.summary.P)+cards('E',d.diagnostics.E,d.summary.E);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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


if __name__ == "__main__":
    port = choose_port(START_PORT)
    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{port}")).start()
    print(f"Core Growth Binding v7: http://{HOST}:{port}")
    serve(app, host=HOST, port=port)
