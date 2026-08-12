from __future__ import annotations

import copy
import json
import math
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
import run_core_growth_binding_v27 as v27
import run_core_growth_binding_v43 as v43

HOST = "127.0.0.1"
START_PORT = 5090
OUT = ROOT / "data" / "core_growth_binding_v44" / "results"
POSITIONS = ["左", "中央", "右"]
THRESHOLD = 0.18
STEPS = 10
WINDOW = 1

CONDITIONS = [
    ("baseline", 1.00, 1.00),
    ("echo_0.97", 0.97, 1.00),
    ("echo_1.03", 1.03, 1.00),
    ("position_0.97", 1.00, 0.97),
    ("position_1.03", 1.00, 1.03),
    ("common_0.97", 0.97, 0.97),
    ("common_1.03", 1.03, 1.03),
]


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


def edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def run_live_scaled(*, position: str | None, include_echo: bool, echo_scale: float, position_scale: float) -> dict:
    """v27と同じ伝播を、初期activationだけ倍率変更して実行する。"""
    brain = copy.deepcopy(v3.base.CORE)
    activation = np.zeros(brain.node_count, dtype=float)
    echoes = v27.entity_echo_nodes() if include_echo else []
    for node in echoes:
        activation[int(node)] = max(
            activation[int(node)], float(v3.ECHO_STRENGTH) * float(echo_scale)
        )

    position_sources: list[int] = []
    if position is not None:
        position_sources = [int(x) for x in v3.position_nodes(position)]
        for node in position_sources:
            activation[node] = max(activation[node], 1.0 * float(position_scale))

    traversed: set[tuple[int, int]] = set()
    steps: list[dict] = []
    for step_index in range(STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break
        candidates: dict[int, tuple[float, int]] = {}
        records: list[dict] = []
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
        accepted: set[tuple[int, int]] = set()
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
        "echo_scale": float(echo_scale),
        "position_scale": float(position_scale),
        "echo_nodes": [int(x) for x in echoes],
        "position_sources": position_sources,
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "steps": steps,
    }


def adjacency_sets() -> list[set[int]]:
    brain = v3.base.CORE
    return [set(int(x) for x in np.flatnonzero(brain.adjacency[i])) for i in range(brain.node_count)]


def records_to_target(trace: dict, step_index: int, target: int) -> list[dict]:
    if step_index < 0 or step_index >= len(trace.get("steps", [])):
        return []
    return [
        row for row in trace["steps"][step_index].get("records", [])
        if int(row["target"]) == int(target)
    ]


def first_contact_event(echo_trace: dict, position_trace: dict, position: str) -> dict | None:
    """v36のContact Event条件を、scaled traceから直接再検出する。"""
    adj = adjacency_sets()
    candidates: list[dict] = []
    echo_steps = echo_trace.get("steps", [])
    pos_steps = position_trace.get("steps", [])

    for es, estep in enumerate(echo_steps):
        echo_nodes = [int(x) for x in estep.get("active_sources", [])]
        for ps, pstep in enumerate(pos_steps):
            if abs(es - ps) > 1:
                continue
            pos_nodes = [int(x) for x in pstep.get("active_sources", [])]
            for e in echo_nodes:
                for p in pos_nodes:
                    shared = sorted(adj[e] & adj[p])
                    for neighbor in shared:
                        erows = [r for r in records_to_target(echo_trace, es, neighbor) if int(r["source"]) == e]
                        prows = [r for r in records_to_target(position_trace, ps, neighbor) if int(r["source"]) == p]
                        if not erows or not prows:
                            continue
                        er = max(erows, key=lambda r: float(r["signal"]))
                        pr = max(prows, key=lambda r: float(r["signal"]))
                        if float(er["signal"]) >= THRESHOLD or float(pr["signal"]) >= THRESHOLD:
                            continue
                        candidates.append({
                            "position": position,
                            "created_step": max(es, ps),
                            "echo_step": es,
                            "position_step": ps,
                            "echo_node": e,
                            "position_node": p,
                            "shared_neighbor": int(neighbor),
                            "time_gap": abs(es - ps),
                            "graph_distance": 2,
                            "echo_signal": float(er["signal"]),
                            "position_signal": float(pr["signal"]),
                            "ttl": 2,
                        })

    if not candidates:
        return None
    candidates.sort(key=lambda r: (
        int(r["time_gap"]),
        -(float(r["echo_signal"]) + float(r["position_signal"])),
        int(r["shared_neighbor"]),
    ))
    return candidates[0]


def make_scaled_report(position: str, echo_scale: float, position_scale: float) -> dict:
    echo_trace = run_live_scaled(
        position=None, include_echo=True, echo_scale=echo_scale, position_scale=1.0
    )
    position_trace = run_live_scaled(
        position=position, include_echo=False, echo_scale=1.0, position_scale=position_scale
    )
    combined_trace = run_live_scaled(
        position=position, include_echo=True, echo_scale=echo_scale, position_scale=position_scale
    )
    event = first_contact_event(echo_trace, position_trace, position)
    return {
        "position": position,
        "echo_scale": float(echo_scale),
        "position_scale": float(position_scale),
        "event_count": 0 if event is None else 1,
        "event_formed": event is not None,
        "events": [] if event is None else [event],
        "traces": {
            "echo_only": echo_trace,
            "position_only": position_trace,
            "combined": combined_trace,
        },
    }


def route_edges(trace: dict) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in trace.get("traversed_edges", [])}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def condition_runs(position: str) -> list[dict]:
    rows = []
    for name, echo_scale, position_scale in CONDITIONS:
        report = make_scaled_report(position, echo_scale, position_scale)
        identity = v43.relative_identity(report, WINDOW) if report["event_formed"] else None
        rows.append({
            "condition": name,
            "echo_scale": echo_scale,
            "position_scale": position_scale,
            "event_formed": report["event_formed"],
            "identity": identity,
            "report": report,
        })
    return rows


def identity_distance(a: list[float], b: list[float]) -> float:
    return v43.hamming_distance(a, b)


def separation(left: list[list[float]], center: list[list[float]]) -> dict:
    within = []
    for rows in (left, center):
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                within.append(identity_distance(a, b))
    between = [identity_distance(a, b) for a in left for b in center]
    max_within = max(within, default=0.0)
    min_between = min(between, default=0.0)
    return {
        "max_same_position_distance": max_within,
        "min_left_center_distance": min_between,
        "separation_margin": min_between - max_within,
        "separated": bool(left and center and min_between > max_within),
    }


def summarize_position(rows: list[dict]) -> dict:
    baseline = next(row for row in rows if row["condition"] == "baseline")
    baseline_edges = route_edges(baseline["report"]["traces"]["combined"])
    identities = [row["identity"] for row in rows if row["identity"] is not None]
    event_count = sum(1 for row in rows if row["event_formed"])
    route_jaccards = {}
    identity_to_baseline = {}
    for row in rows:
        edges = route_edges(row["report"]["traces"]["combined"])
        route_jaccards[row["condition"]] = jaccard(baseline_edges, edges)
        identity_to_baseline[row["condition"]] = (
            None if row["identity"] is None or baseline["identity"] is None
            else identity_distance(baseline["identity"], row["identity"])
        )
    return {
        "event_formed_count": event_count,
        "condition_count": len(rows),
        "event_stable_all_conditions": event_count == len(rows),
        "unique_identity_count": len({tuple(x) for x in identities}),
        "identity_stable_all_conditions": bool(identities) and len({tuple(x) for x in identities}) == 1 and len(identities) == len(rows),
        "minimum_route_jaccard_vs_baseline": min(route_jaccards.values(), default=1.0),
        "route_jaccards_vs_baseline": route_jaccards,
        "identity_distance_to_baseline": identity_to_baseline,
    }


def observe() -> dict:
    runs = {position: condition_runs(position) for position in POSITIONS}
    summaries = {position: summarize_position(rows) for position, rows in runs.items()}

    left_ids = [row["identity"] for row in runs["左"] if row["identity"] is not None]
    center_ids = [row["identity"] for row in runs["中央"] if row["identity"] is not None]
    sep = separation(left_ids, center_ids)

    right_absent = all(not row["event_formed"] for row in runs["右"])
    left_complete = summaries["左"]["event_stable_all_conditions"]
    center_complete = summaries["中央"]["event_stable_all_conditions"]
    same_position_stable = summaries["左"]["identity_stable_all_conditions"] and summaries["中央"]["identity_stable_all_conditions"]

    if left_complete and center_complete and same_position_stable and sep["separated"] and right_absent:
        verdict = "relative_context_identity_survives_live_input_perturbation"
        next_step = "candidate_for_core_short_term_relative_context_state"
    elif left_complete and center_complete and sep["separated"] and right_absent:
        verdict = "relative_context_identity_separates_live_runs_but_changes_within_position"
        next_step = "refine_relative_bands_before_core_integration"
    elif not left_complete or not center_complete:
        verdict = "contact_event_itself_not_stable_under_live_input_perturbation"
        next_step = "stabilize_contact_event_detection_before_identity_integration"
    else:
        verdict = "relative_context_identity_not_robust_in_live_propagation"
        next_step = "do_not_integrate_into_core_yet"

    compact_runs = {}
    for position, rows in runs.items():
        compact_runs[position] = [
            {
                "condition": row["condition"],
                "echo_scale": row["echo_scale"],
                "position_scale": row["position_scale"],
                "event_formed": row["event_formed"],
                "identity": row["identity"],
                "combined_edges": row["report"]["traces"]["combined"]["traversed_edges"],
            }
            for row in rows
        ]

    payload = {
        "experiment": "Core Growth Binding v44",
        "purpose": "Rerun Core propagation under ±3% live echo/position input-strength changes, rebuild Contact Events and 1-step Relative Context Identity from the resulting traces, and test whether identity survives actual propagation changes.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_perturbation": True,
            "posthoc_feature_perturbation": False,
        },
        "conditions": [
            {"name": name, "echo_scale": e, "position_scale": p}
            for name, e, p in CONDITIONS
        ],
        "identity": {
            "source": "v43 Relative Context Identity",
            "window": WINDOW,
            "definition": "ordering / coarse dominance / step trend / relative topology; no Node IDs, position labels, absolute event signal, TTL, or raw continuous context values",
        },
        "position_summaries": summaries,
        "runs": compact_runs,
        "left_center_separation": sep,
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent_all_conditions": right_absent,
            "left_identity_stable_all_conditions": summaries["左"]["identity_stable_all_conditions"],
            "center_identity_stable_all_conditions": summaries["中央"]["identity_stable_all_conditions"],
            "same_position_identity_stable": same_position_stable,
            "left_center_separated": sep["separated"],
            "separation_margin": sep["separation_margin"],
            "minimum_left_route_jaccard": summaries["左"]["minimum_route_jaccard_vs_baseline"],
            "minimum_center_route_jaccard": summaries["中央"]["minimum_route_jaccard_vs_baseline"],
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v44.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v44</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v44</h1><p class="lead">E残響・位置入力の初期activationを±3%変えてCoreを実際に再伝播し、Contact Eventと1Step Relative Context Identityを毎回作り直す。後処理だけの摂動ではない。</p><section class="panel"><div class="controls"><button id="run">Live Perturbationを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Live Perturbation 生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,p=d.position_summaries,sep=d.left_center_separation;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event全条件<b class="${s.left_event_all_conditions?'good':'warn'}">${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b class="${s.center_event_all_conditions?'good':'warn'}">${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent_all_conditions)}</b></div><div class="metric">左 Identity固定<b class="${s.left_identity_stable_all_conditions?'good':'warn'}">${yn(s.left_identity_stable_all_conditions)}</b></div><div class="metric">中央 Identity固定<b class="${s.center_identity_stable_all_conditions?'good':'warn'}">${yn(s.center_identity_stable_all_conditions)}</b></div><div class="metric">左中央 分離<b class="${s.left_center_separated?'good':'warn'}">${yn(s.left_center_separated)}</b></div><div class="metric">分離margin<b class="${s.separation_margin>0?'good':'warn'}">${f(s.separation_margin)}</b></div><div class="metric">同位置最大距離<b>${f(sep.max_same_position_distance)}</b></div><div class="metric">異位置最小距離<b>${f(sep.min_left_center_distance)}</b></div><div class="metric">左 最小route Jaccard<b>${f(s.minimum_left_route_jaccard)}</b></div><div class="metric">中央 最小route Jaccard<b>${f(s.minimum_center_route_jaccard)}</b></div><div class="metric">次段階<b class="blue">${s.next_step}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v44: http://{HOST}:{PORT}")
    print("Relative Context Identity live perturbation / real Core reruns / no learning")
    serve(app, host=HOST, port=PORT)
