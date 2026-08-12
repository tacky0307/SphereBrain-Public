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
import run_core_growth_binding_v8 as v8

HOST = "127.0.0.1"
START_PORT = 5045
OUT = ROOT / "data" / "core_growth_binding_v10" / "results"
POSITIONS = v3.POSITIONS
BASE_THRESHOLD = 0.18
LOW_THRESHOLD = 0.16
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


def edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def edge_set(edges) -> set[tuple[int, int]]:
    return {edge_key(a, b) for a, b in edges}


def run_detailed(
    entity: str,
    position: str | None,
    *,
    threshold: float,
    binding: bool,
) -> dict:
    brain = copy.deepcopy(v3.base.CORE)

    if binding:
        entity_stage = v3.propagate(
            copy.deepcopy(brain), v3.entity_nodes(entity), learn=False, steps=8
        )
        echo_nodes = list(entity_stage["final_active_nodes"][: v3.ECHO_LIMIT])
        source_nodes = list(v3.position_nodes(str(position)))
        context_nodes = echo_nodes
    else:
        entity_stage = None
        echo_nodes = []
        source_nodes = list(v3.entity_nodes(entity))
        context_nodes = []

    activation = np.zeros(brain.node_count, dtype=float)
    for node in source_nodes:
        activation[node] = 1.0
    for node in context_nodes:
        activation[node] = max(activation[node], v3.ECHO_STRENGTH)

    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed_edges: set[tuple[int, int]] = set()
    history = [sorted(activated_nodes)]
    step_records = []

    for step_index in range(MAX_STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break

        candidates: dict[int, tuple[float, int]] = {}
        local_records = []

        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            raw_scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(raw_scores, -branch_count)[-branch_count:]
            best_local = {int(neighbors[i]) for i in best_indices}

            for idx, target_raw in enumerate(neighbors):
                target = int(target_raw)
                signal = float(raw_scores[idx]) * brain.signal_decay
                is_local_top = target in best_local
                record = {
                    "source": int(source),
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "signal_decay": float(brain.signal_decay),
                    "signal": signal,
                    "is_local_top": is_local_top,
                    "passes_threshold": bool(signal >= threshold),
                }
                local_records.append(record)
                if not is_local_top or signal < threshold:
                    continue
                previous = candidates.get(target)
                if previous is None or signal > previous[0]:
                    candidates[target] = (signal, int(source))

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

        next_activation = np.zeros(brain.node_count, dtype=float)
        accepted = []
        for target, (signal, source) in selected:
            next_activation[target] = max(next_activation[target], signal)
            accepted.append((source, target))
            traversed_edges.add(edge_key(source, target))

        accepted_set = {edge_key(a, b) for a, b in accepted}
        candidate_set = {edge_key(source, target) for target, (_, source) in candidates.items()}
        for record in local_records:
            key = tuple(record["edge"])
            record["became_candidate"] = key in candidate_set
            record["accepted"] = key in accepted_set

        active_now = np.flatnonzero(next_activation > 0).tolist()
        step_records.append({
            "step": step_index,
            "active_sources": [int(x) for x in active_sources],
            "active_values": {str(int(x)): float(activation[x]) for x in active_sources},
            "edge_records": local_records,
            "accepted_edges": [list(edge_key(a, b)) for a, b in accepted],
            "active_now": active_now,
        })
        if not active_now:
            break
        activated_nodes.update(active_now)
        history.append(active_now)
        activation = next_activation

    return {
        "entity": entity,
        "position": position,
        "binding": binding,
        "threshold": threshold,
        "source_nodes": source_nodes,
        "echo_nodes": echo_nodes,
        "entity_stage": entity_stage,
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed_edges)],
        "history": history,
        "step_records": step_records,
    }


def records_for_edge(trace: dict, edge: tuple[int, int]) -> list[dict]:
    rows = []
    for step in trace["step_records"]:
        for record in step["edge_records"]:
            if tuple(record["edge"]) == edge:
                rows.append({"step": step["step"], **record})
    return rows


def best_record(trace: dict, edge: tuple[int, int]) -> dict | None:
    rows = records_for_edge(trace, edge)
    return max(rows, key=lambda row: row["signal"], default=None)


def classify_record(record: dict | None) -> str:
    if record is None:
        return "not_visible_from_active_sources"
    if record.get("accepted"):
        return "selected"
    if record.get("became_candidate"):
        return "candidate_but_not_selected"
    if record.get("is_local_top") and not record.get("passes_threshold"):
        return "local_top_but_below_threshold"
    if not record.get("is_local_top"):
        return "neighbor_seen_but_not_local_top"
    return "unclassified"


def ordered_binding_chain(binding_trace: dict, start_edge: tuple[int, int]) -> list[dict]:
    accepted_by_step = [
        [tuple(edge) for edge in step["accepted_edges"]]
        for step in binding_trace["step_records"]
    ]
    start_step = None
    for index, edges in enumerate(accepted_by_step):
        if start_edge in edges:
            start_step = index
            break
    if start_step is None:
        return []

    chain = [{"step": start_step, "edge": list(start_edge)}]
    frontier = set(start_edge)
    for step_index in range(start_step + 1, len(accepted_by_step)):
        touching = [edge for edge in accepted_by_step[step_index] if frontier & set(edge)]
        if not touching:
            break
        chosen = sorted(touching)[0]
        chain.append({"step": step_index, "edge": list(chosen)})
        frontier = set(chosen)
    return chain


def diagnose_entity(entity: str, position: str) -> dict:
    specific = sorted(v8.binding_specific_edges(entity, position))
    single_018 = run_detailed(entity, None, threshold=BASE_THRESHOLD, binding=False)
    single_016 = run_detailed(entity, None, threshold=LOW_THRESHOLD, binding=False)
    bound_018 = run_detailed(entity, position, threshold=BASE_THRESHOLD, binding=True)

    edge_reports = []
    for edge in specific:
        single_best_018 = best_record(single_018, edge)
        single_best_016 = best_record(single_016, edge)
        bound_best = best_record(bound_018, edge)
        chain = ordered_binding_chain(bound_018, edge)

        chain_comparison = []
        first_stop = None
        for chain_item in chain:
            chain_edge = tuple(chain_item["edge"])
            single_record = best_record(single_016, chain_edge)
            status = classify_record(single_record)
            row = {
                "binding_step": chain_item["step"],
                "edge": list(chain_edge),
                "single_0_16_status": status,
                "single_0_16_best": single_record,
            }
            chain_comparison.append(row)
            if first_stop is None and status != "selected":
                first_stop = row

        edge_reports.append({
            "specific_edge": list(edge),
            "single_0_18_best": single_best_018,
            "single_0_18_status": classify_record(single_best_018),
            "single_0_16_best": single_best_016,
            "single_0_16_status": classify_record(single_best_016),
            "binding_0_18_best": bound_best,
            "binding_chain": chain,
            "chain_comparison_at_0_16": chain_comparison,
            "first_chain_stop_at_0_16": first_stop,
        })

    nearest = None
    for report in edge_reports:
        row = report["single_0_18_best"]
        if row is not None and (nearest is None or row["signal"] > nearest["signal"]):
            nearest = row

    return {
        "entity": entity,
        "position": position,
        "specific_edge_count": len(specific),
        "specific_edges": [list(x) for x in specific],
        "nearest_single_signal_at_0_18": None if nearest is None else nearest["signal"],
        "nearest_single_step_at_0_18": None if nearest is None else nearest["step"],
        "edge_reports": edge_reports,
        "single_0_18_summary": {
            "activated_node_count": len(single_018["activated_nodes"]),
            "traversed_edge_count": len(single_018["traversed_edges"]),
        },
        "single_0_16_summary": {
            "activated_node_count": len(single_016["activated_nodes"]),
            "traversed_edge_count": len(single_016["traversed_edges"]),
        },
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v10",
        "world": {"P": player, "E": other},
        "purpose": "Locate pre-entry signal decay and post-entry chain failure without changing Core behavior.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "puzzle_specific_rules": False,
        },
        "diagnostics": {
            "P": diagnose_entity("P", player),
            "E": diagnose_entity("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v10.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v10</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:22px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:760px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v10</h1><p class="lead">P側は固有Edgeへ届く前の信号低下を、E側は固有Edge通過後の連鎖停止点をStep単位で診断する。学習・重み変更・新規Edge・専用規則はない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">連鎖を調べる</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function text(x){return x===null||x===undefined?'なし':String(x)}function stop(d){const r=d.edge_reports[0];if(!r)return '固有Edgeなし';const s=r.first_chain_stop_at_0_16;return s?`${s.single_0_16_status} / Step ${s.binding_step}`:'連鎖停止なし'}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E;document.getElementById('metrics').innerHTML=`<div class="metric">P 最接近Step<b>${text(P.nearest_single_step_at_0_18)}</b></div><div class="metric">P 最接近信号<b>${P.nearest_single_signal_at_0_18===null?'なし':Number(P.nearest_single_signal_at_0_18).toFixed(6)}</b></div><div class="metric">P 0.16連鎖停止<b class="warn">${stop(P)}</b></div><div class="metric">E 0.16連鎖停止<b class="warn">${stop(E)}</b></div><div class="metric">P 0.18→0.16 Node<b>${P.single_0_18_summary.activated_node_count} → ${P.single_0_16_summary.activated_node_count}</b></div><div class="metric">E 0.18→0.16 Node<b>${E.single_0_18_summary.activated_node_count} → ${E.single_0_16_summary.activated_node_count}</b></div><div class="metric">P固有Edge数<b>${P.specific_edge_count}</b></div><div class="metric">E固有Edge数<b>${E.specific_edge_count}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v10: http://{HOST}:{PORT}")
    print("Pre-entry decay and post-entry chain diagnosis / no learning / no weight changes")
    serve(app, host=HOST, port=PORT)
