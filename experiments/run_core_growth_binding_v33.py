from __future__ import annotations

import copy
import json
import socket
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v27 as v27
import run_core_growth_binding_v29 as v29

HOST = "127.0.0.1"
START_PORT = 5079
OUT = ROOT / "data" / "core_growth_binding_v33" / "results"
POSITION = "左"
THRESHOLD = 0.18
MAX_STEPS = 10
ASSIST_GAIN = 0.02
TIE_MARGIN = 0.0025
ABSOLUTE_CAP = 0.00005


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


@dataclass(frozen=True)
class BindingContact:
    echo_node: int
    position_node: int
    neighbor: int
    echo_step: int
    position_step: int
    time_gap: int
    graph_distance: int


@dataclass
class CrossLineageBindingState:
    echo_nodes: set[int]
    position_nodes: set[int]
    contacts: list[BindingContact]
    active_until_step: int

    def affinity(self, source: int, target: int, step: int, adj: list[set[int]]) -> float:
        if step > self.active_until_step:
            return 0.0
        score = 0.0
        for contact in self.contacts:
            proximity = 1.0 / max(1, contact.graph_distance)
            simultaneity = 1.0 if contact.time_gap == 0 else 0.5
            if target == contact.neighbor:
                score = max(score, proximity * simultaneity)
            elif source in {contact.echo_node, contact.position_node} and target in adj[contact.neighbor]:
                score = max(score, 0.5 * proximity * simultaneity)
            elif target in adj[contact.neighbor]:
                score = max(score, 0.25 * proximity * simultaneity)
        return score


def build_binding_state(position: str) -> dict:
    echo_trace = v27.run_live(position=None, include_echo=True)
    position_trace = v27.run_live(position=position, include_echo=False)
    contact = v29.contact_rows(echo_trace, position_trace)
    adj = adjacency_sets(v3.base.CORE)
    contacts: list[BindingContact] = []
    for row in contact["rows"]:
        if row["time_gap"] > 1:
            continue
        for item in row["shared_neighbor_pairs"]:
            contacts.append(BindingContact(
                echo_node=int(item["echo_node"]),
                position_node=int(item["position_node"]),
                neighbor=int(item["shared_neighbors"][0]),
                echo_step=int(row["echo_step"]),
                position_step=int(row["position_step"]),
                time_gap=int(row["time_gap"]),
                graph_distance=2,
            ))
    unique = {}
    for item in contacts:
        unique[(item.echo_node, item.position_node, item.neighbor, item.echo_step, item.position_step)] = item
    contacts = list(unique.values())
    state = CrossLineageBindingState(
        echo_nodes=set(int(x) for x in echo_trace["echo_nodes"]),
        position_nodes=set(int(x) for x in position_trace["position_sources"]),
        contacts=contacts,
        active_until_step=2,
    )
    return {
        "state": state,
        "echo_trace": echo_trace,
        "position_trace": position_trace,
        "contact_diagnostic": contact,
        "adj": adj,
    }


def initial_activation(position: str) -> np.ndarray:
    brain = v3.base.CORE
    activation = np.zeros(brain.node_count, dtype=float)
    for node in v27.entity_echo_nodes():
        activation[int(node)] = max(activation[int(node)], v3.ECHO_STRENGTH)
    for node in v3.position_nodes(position):
        activation[int(node)] = max(activation[int(node)], 1.0)
    return activation


def run_mode(position: str, mode: str, state: CrossLineageBindingState, adj: list[set[int]]) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    activation = initial_activation(position)
    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed: set[tuple[int, int]] = set()
    trace = []

    for step in range(MAX_STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break
        target_candidates: dict[int, list[dict]] = {}
        all_rows = []
        for source_raw in active_sources:
            source = int(source_raw)
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            top_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            local_top = {int(neighbors[i]) for i in top_indices}
            for idx, target_raw in enumerate(neighbors):
                target = int(target_raw)
                base_signal = float(scores[idx]) * float(brain.signal_decay)
                is_top = target in local_top
                passes = base_signal >= THRESHOLD
                affinity = state.affinity(source, target, step, adj) if mode != "baseline" else 0.0
                modulation = 0.0
                if mode == "binding_state_assist" and affinity > 0:
                    modulation = min(ABSOLUTE_CAP, ASSIST_GAIN * affinity)
                row = {
                    "source": source,
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "base_signal": base_signal,
                    "affinity": affinity,
                    "modulation": modulation,
                    "rank_signal": base_signal + modulation,
                    "local_top": is_top,
                    "passes_threshold": passes,
                }
                all_rows.append(row)
                if not is_top or not passes:
                    continue
                target_candidates.setdefault(target, []).append(row)

        ranked = []
        for target, rows in target_candidates.items():
            rows.sort(key=lambda r: r["rank_signal"], reverse=True)
            winner = rows[0]
            if len(rows) > 1:
                baseline_sorted = sorted(rows, key=lambda r: r["base_signal"], reverse=True)
                baseline_margin = baseline_sorted[0]["base_signal"] - baseline_sorted[1]["base_signal"]
            else:
                baseline_margin = None
            winner = dict(winner)
            winner["baseline_margin"] = baseline_margin
            winner["tie_gate_active"] = bool(
                mode == "binding_state_assist"
                and baseline_margin is not None
                and baseline_margin <= TIE_MARGIN
            )
            ranked.append(winner)

        ranked.sort(key=lambda r: r["rank_signal"], reverse=True)
        selected = ranked[: min(brain.max_active_per_step, len(ranked))]
        next_activation = np.zeros(brain.node_count, dtype=float)
        selected_edges = []
        for row in selected:
            next_activation[row["target"]] = max(next_activation[row["target"]], row["base_signal"])
            key = edge_key(row["source"], row["target"])
            traversed.add(key)
            selected_edges.append(list(key))
        trace.append({
            "step": step,
            "active_sources": [int(x) for x in active_sources],
            "selected_edges": selected_edges,
            "selected_rows": selected,
            "binding_state_active": step <= state.active_until_step,
            "binding_contact_count": len(state.contacts),
        })
        if not selected:
            break
        activated_nodes.update(np.flatnonzero(next_activation > 0).tolist())
        activation = next_activation

    return {
        "mode": mode,
        "activated_nodes": sorted(int(x) for x in activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "trace": trace,
    }


def edge_set(rows) -> set[tuple[int, int]]:
    return {edge_key(*row) for row in rows}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def observe() -> dict:
    built = build_binding_state(POSITION)
    state: CrossLineageBindingState = built["state"]
    adj = built["adj"]
    baseline = run_mode(POSITION, "baseline", state, adj)
    state_only = run_mode(POSITION, "binding_state_only", state, adj)
    assisted = run_mode(POSITION, "binding_state_assist", state, adj)

    baseline_edges = edge_set(baseline["traversed_edges"])
    state_edges = edge_set(state_only["traversed_edges"])
    assist_edges = edge_set(assisted["traversed_edges"])
    assist_only_edges = assist_edges - baseline_edges
    strong_override_count = 0
    tie_gate_count = 0
    modulation_count = 0
    for step in assisted["trace"]:
        for row in step["selected_rows"]:
            if row.get("modulation", 0.0) > 0:
                modulation_count += 1
            if row.get("tie_gate_active"):
                tie_gate_count += 1
            margin = row.get("baseline_margin")
            if margin is not None and margin > TIE_MARGIN and row.get("modulation", 0.0) > 0:
                strong_override_count += 1

    payload = {
        "experiment": "Core Growth Binding v33",
        "purpose": "Test a temporary Cross-Lineage Binding State built only from concurrent E-residual and position pathway structure, then compare baseline, state-only, and state-plus-Structural-Assist propagation.",
        "position": POSITION,
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "candidate_set_changed": False,
            "puzzle_specific_answer_rule": False,
            "core_file_modified": False,
        },
        "binding_state": {
            "echo_node_count": len(state.echo_nodes),
            "position_node_count": len(state.position_nodes),
            "contact_count": len(state.contacts),
            "active_until_step": state.active_until_step,
            "contacts": [item.__dict__ for item in state.contacts],
        },
        "runs": {
            "baseline": baseline,
            "binding_state_only": state_only,
            "binding_state_assist": assisted,
        },
        "comparison": {
            "baseline_vs_state_node_jaccard": jaccard(set(baseline["activated_nodes"]), set(state_only["activated_nodes"])),
            "baseline_vs_state_edge_jaccard": jaccard(baseline_edges, state_edges),
            "baseline_vs_assist_node_jaccard": jaccard(set(baseline["activated_nodes"]), set(assisted["activated_nodes"])),
            "baseline_vs_assist_edge_jaccard": jaccard(baseline_edges, assist_edges),
            "assist_only_edge_count": len(assist_only_edges),
            "assist_only_edges": [list(x) for x in sorted(assist_only_edges)],
            "modulation_count": modulation_count,
            "tie_gate_count": tie_gate_count,
            "strong_override_count": strong_override_count,
            "state_only_changed_route": state_edges != baseline_edges,
            "assist_changed_route": assist_edges != baseline_edges,
        },
        "diagnostics": {
            "contact": built["contact_diagnostic"],
            "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v33.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v33</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v33</h1><p class="lead">Cross-Lineage Binding Stateを一時保持し、通常伝播・状態保持のみ・状態＋Structural Assistを比較する。新規Edge、weight変更、正解規則は使わない。</p><section class="panel"><div class="controls"><button id="run">Binding Stateを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return Number(x).toFixed(6)}function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),c=d.comparison,s=d.binding_state;document.getElementById('metrics').innerHTML=`<div class="metric">Binding接触数<b>${s.contact_count}</b></div><div class="metric">State保持Step<b>${s.active_until_step}</b></div><div class="metric">通常vsState Node<b>${f(c.baseline_vs_state_node_jaccard)}</b></div><div class="metric">通常vsState Edge<b>${f(c.baseline_vs_state_edge_jaccard)}</b></div><div class="metric">通常vsAssist Node<b>${f(c.baseline_vs_assist_node_jaccard)}</b></div><div class="metric">通常vsAssist Edge<b>${f(c.baseline_vs_assist_edge_jaccard)}</b></div><div class="metric">Stateのみ経路変更<b>${yn(c.state_only_changed_route)}</b></div><div class="metric">Assist経路変更<b>${yn(c.assist_changed_route)}</b></div><div class="metric">AssistのみEdge<b>${c.assist_only_edge_count}</b></div><div class="metric">変調作動<b>${c.modulation_count}</b></div><div class="metric">tie gate作動<b>${c.tie_gate_count}</b></div><div class="metric">強判断上書き<b class="${c.strong_override_count===0?'good':'warn'}">${c.strong_override_count}</b></div><div class="metric">brain.json<b class="good">${d.diagnostics.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v33: http://{HOST}:{PORT}")
    print("Cross-Lineage Binding State / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
