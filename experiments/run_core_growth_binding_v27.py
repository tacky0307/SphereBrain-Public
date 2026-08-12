from __future__ import annotations

import copy
import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v25 as v25

HOST = "127.0.0.1"
START_PORT = 5073
OUT = ROOT / "data" / "core_growth_binding_v27" / "results"
POSITIONS = list(v3.POSITIONS)
THRESHOLD = 0.18
STEPS = 10


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


def entity_echo_nodes() -> list[int]:
    brain = copy.deepcopy(v3.base.CORE)
    stage = v3.propagate(brain, v3.entity_nodes("E"), learn=False, assist=False, steps=8)
    return [int(x) for x in stage["final_active_nodes"][:v3.ECHO_LIMIT]]


def run_live(*, position: str | None, include_echo: bool) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    activation = np.zeros(brain.node_count, dtype=float)
    echoes = entity_echo_nodes() if include_echo else []
    for node in echoes:
        activation[node] = max(activation[node], v3.ECHO_STRENGTH)
    position_sources = []
    if position is not None:
        position_sources = [int(x) for x in v3.position_nodes(position)]
        for node in position_sources:
            activation[node] = max(activation[node], 1.0)

    traversed: set[tuple[int, int]] = set()
    steps = []
    for step_index in range(STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break
        candidates: dict[int, tuple[float, int]] = {}
        records = []
        for source_raw in active_sources:
            source = int(source_raw)
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            local_top = {int(neighbors[i]) for i in best_indices}
            for idx, target_raw in enumerate(neighbors):
                target = int(target_raw)
                signal = float(scores[idx]) * float(brain.signal_decay)
                row = {
                    "edge": list(edge_key(source, target)),
                    "source": source,
                    "target": target,
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "signal": signal,
                    "local_top": target in local_top,
                    "passes_threshold": signal >= THRESHOLD,
                }
                records.append(row)
                if not row["local_top"] or not row["passes_threshold"]:
                    continue
                previous = candidates.get(target)
                if previous is None or signal > previous[0]:
                    candidates[target] = (signal, source)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        selected = ranked[: min(brain.max_active_per_step, len(ranked))]
        next_activation = np.zeros(brain.node_count, dtype=float)
        accepted = set()
        for target, (signal, source) in selected:
            next_activation[target] = max(next_activation[target], signal)
            accepted.add(edge_key(source, target))
            traversed.add(edge_key(source, target))
        for row in records:
            row["accepted"] = tuple(row["edge"]) in accepted
        steps.append({
            "step": step_index,
            "active_sources": [int(x) for x in active_sources],
            "active_values": {str(int(x)): float(activation[x]) for x in active_sources},
            "accepted_edges": [list(x) for x in sorted(accepted)],
            "records": records,
        })
        if not accepted:
            break
        activation = next_activation

    return {
        "position": position,
        "include_echo": include_echo,
        "echo_nodes": echoes,
        "position_sources": position_sources,
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "steps": steps,
    }


def edge_observation(trace: dict, edge: tuple[int, int]) -> dict:
    selected = []
    visible = []
    for step in trace["steps"]:
        for row in step["records"]:
            if tuple(row["edge"]) != edge:
                continue
            item = {"step": step["step"], **row, "active_sources": step["active_sources"]}
            visible.append(item)
            if row["accepted"]:
                selected.append(item)
    best = max(visible, key=lambda x: x["signal"], default=None)
    first = selected[0] if selected else None
    return {
        "visible": bool(visible),
        "selected": bool(selected),
        "first_selected": first,
        "best_observation": best,
    }


def predecessor_edges(trace: dict, source_node: int, before_step: int | None) -> list[list[int]]:
    if before_step is None or before_step <= 0:
        return []
    prior = trace["steps"][before_step - 1]
    return [edge for edge in prior["accepted_edges"] if source_node in edge]


def diagnose() -> dict:
    decomposition = v25.decompose()
    common = [tuple(x) for x in decomposition["groups"]["all_three_common"]]
    if not common:
        return {"common_edge_found": False, "decomposition": decomposition}
    edge = common[0]
    source, target = edge

    echo_only = run_live(position=None, include_echo=True)
    no_input = run_live(position=None, include_echo=False)
    position_only = {p: run_live(position=p, include_echo=False) for p in POSITIONS}
    combined = {p: run_live(position=p, include_echo=True) for p in POSITIONS}

    def report(trace: dict) -> dict:
        obs = edge_observation(trace, edge)
        first = obs["first_selected"]
        obs["predecessor_edges_into_source"] = predecessor_edges(
            trace, source, None if first is None else int(first["step"])
        )
        return obs

    echo_report = report(echo_only)
    position_reports = {p: report(t) for p, t in position_only.items()}
    combined_reports = {p: report(t) for p, t in combined.items()}

    combined_selected_all = all(r["selected"] for r in combined_reports.values())
    position_selected_any = any(r["selected"] for r in position_reports.values())
    second_stimulus = (not echo_report["selected"]) and (not position_selected_any) and combined_selected_all

    predecessor_sets = {
        p: {tuple(x) for x in r["predecessor_edges_into_source"]}
        for p, r in combined_reports.items()
    }
    different_predecessors = len({tuple(sorted(s)) for s in predecessor_sets.values()}) > 1
    common_hub = combined_selected_all and different_predecessors
    echo_residual = echo_report["selected"]

    if echo_residual:
        verdict = "E_residual"
    elif second_stimulus and common_hub:
        verdict = "second_stimulus_via_common_hub"
    elif second_stimulus:
        verdict = "second_stimulus_detection"
    elif common_hub:
        verdict = "common_hub"
    else:
        verdict = "unresolved"

    return {
        "common_edge_found": True,
        "common_edge": list(edge),
        "source_node": source,
        "target_node": target,
        "echo_nodes": echo_only["echo_nodes"],
        "echo_only": echo_report,
        "position_only": position_reports,
        "combined": combined_reports,
        "predecessor_sets": {p: [list(x) for x in sorted(s)] for p, s in predecessor_sets.items()},
        "hypotheses": {
            "E_residual": echo_residual,
            "second_stimulus_detection": second_stimulus,
            "common_hub": common_hub,
            "different_position_predecessors": different_predecessors,
        },
        "verdict": verdict,
        "traces": {
            "no_input": no_input,
            "echo_only": echo_only,
            "position_only": position_only,
            "combined": combined,
        },
        "decomposition": decomposition,
    }


def observe() -> dict:
    payload = {
        "experiment": "Core Growth Binding v27",
        "purpose": "Determine whether the one cross-position common Binding edge is an E residual path, a generic second-stimulus response, or a shared convergence hub.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "diagnostics": diagnose(),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v27.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v27</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v27</h1><p class="lead">3位置に共通した1Edgeを、E残響だけ・位置だけ・E残響＋位置で実伝播させ、残響経路・第二刺激反応・共通ハブのどれかを判定する。</p><section class="panel"><div class="controls"><button id="run">共通Edgeを解剖</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),x=d.diagnostics,h=x.hypotheses;if(!x.common_edge_found){document.getElementById('metrics').textContent='共通Edgeが見つかりません。';return}const combined=Object.entries(x.combined).map(([p,r])=>`<div class="metric">${p} 結合時通過<b class="${r.selected?'good':'warn'}">${yn(r.selected)}</b></div><div class="metric">${p} 通過Step<b>${r.first_selected?r.first_selected.step:'なし'}</b></div>`).join('');document.getElementById('metrics').innerHTML=`<div class="metric">共通Edge<b class="blue">${x.common_edge.join(' ↔ ')}</b></div><div class="metric">E残響だけで通過<b>${yn(x.echo_only.selected)}</b></div><div class="metric">E残響説<b class="${h.E_residual?'good':'warn'}">${yn(h.E_residual)}</b></div><div class="metric">第二刺激検知説<b class="${h.second_stimulus_detection?'good':'warn'}">${yn(h.second_stimulus_detection)}</b></div><div class="metric">共通ハブ説<b class="${h.common_hub?'good':'warn'}">${yn(h.common_hub)}</b></div><div class="metric">位置別前経路差<b>${yn(h.different_position_predecessors)}</b></div><div class="metric">判定<b>${x.verdict}</b></div>${combined}<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v27: http://{HOST}:{PORT}")
    print("Common Binding edge origin diagnostic / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
