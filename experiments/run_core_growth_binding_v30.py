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
import run_core_growth_binding_v27 as v27
import run_core_growth_binding_v29 as v29

HOST = "127.0.0.1"
START_PORT = 5076
OUT = ROOT / "data" / "core_growth_binding_v30" / "results"
TARGET_POSITIONS = ["左", "中央"]
THRESHOLD = 0.18


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


def adjacency_sets(brain) -> list[set[int]]:
    return [set(int(x) for x in np.flatnonzero(brain.adjacency[i])) for i in range(brain.node_count)]


def records_to_target(trace: dict, step_index: int, target: int) -> list[dict]:
    if step_index < 0 or step_index >= len(trace.get("steps", [])):
        return []
    step = trace["steps"][step_index]
    rows = []
    for record in step.get("records", []):
        if int(record["target"]) != int(target):
            continue
        rows.append({"step": step_index, **record})
    return sorted(rows, key=lambda row: row["signal"], reverse=True)


def active_value(trace: dict, step_index: int, node: int) -> float:
    if step_index < 0 or step_index >= len(trace.get("steps", [])):
        return 0.0
    values = trace["steps"][step_index].get("active_values", {})
    return float(values.get(str(int(node)), 0.0))


def selected_into_target(trace: dict, step_index: int, target: int) -> list[dict]:
    return [row for row in records_to_target(trace, step_index, target) if row.get("accepted")]


def candidate_rows(position: str) -> dict:
    brain = v3.base.CORE
    adj = adjacency_sets(brain)
    echo_trace = v27.run_live(position=None, include_echo=True)
    position_trace = v27.run_live(position=position, include_echo=False)
    combined_trace = v27.run_live(position=position, include_echo=True)

    echo_steps = v29.step_nodes(echo_trace)
    pos_steps = v29.step_nodes(position_trace)
    candidates = []

    for echo_step, echo_nodes in enumerate(echo_steps):
        for position_step, position_nodes in enumerate(pos_steps):
            time_gap = abs(echo_step - position_step)
            if time_gap > 1:
                continue
            for echo_node in sorted(echo_nodes):
                for position_node in sorted(position_nodes):
                    shared = sorted(adj[echo_node] & adj[position_node])
                    for neighbor in shared:
                        echo_records = [
                            row for row in records_to_target(echo_trace, echo_step, neighbor)
                            if int(row["source"]) == echo_node
                        ]
                        position_records = [
                            row for row in records_to_target(position_trace, position_step, neighbor)
                            if int(row["source"]) == position_node
                        ]
                        if not echo_records or not position_records:
                            continue
                        er = echo_records[0]
                        pr = position_records[0]
                        simultaneous = time_gap == 0
                        combined_step = max(echo_step, position_step)
                        combined_records = records_to_target(combined_trace, combined_step, neighbor)
                        accepted = selected_into_target(combined_trace, combined_step, neighbor)
                        max_signal = max(er["signal"], pr["signal"])
                        sum_signal = er["signal"] + pr["signal"]
                        capped_sum = min(1.0, sum_signal)
                        weaker = min(er["signal"], pr["signal"])
                        stronger_lineage = "echo" if er["signal"] >= pr["signal"] else "position"
                        both_threshold = er["passes_threshold"] and pr["passes_threshold"]
                        either_threshold = er["passes_threshold"] or pr["passes_threshold"]
                        candidates.append({
                            "position": position,
                            "neighbor": int(neighbor),
                            "echo_node": int(echo_node),
                            "position_node": int(position_node),
                            "echo_step": echo_step,
                            "position_step": position_step,
                            "time_gap": time_gap,
                            "simultaneous": simultaneous,
                            "echo_signal": float(er["signal"]),
                            "position_signal": float(pr["signal"]),
                            "echo_weight": float(er["weight"]),
                            "position_weight": float(pr["weight"]),
                            "echo_local_top": bool(er["local_top"]),
                            "position_local_top": bool(pr["local_top"]),
                            "echo_passes_threshold": bool(er["passes_threshold"]),
                            "position_passes_threshold": bool(pr["passes_threshold"]),
                            "both_pass_threshold": bool(both_threshold),
                            "either_pass_threshold": bool(either_threshold),
                            "max_signal": float(max_signal),
                            "sum_signal_diagnostic": float(sum_signal),
                            "capped_sum_diagnostic": float(capped_sum),
                            "weaker_signal_discarded_if_same_target": float(weaker),
                            "stronger_lineage": stronger_lineage,
                            "combined_step_checked": combined_step,
                            "combined_records_to_neighbor": combined_records,
                            "combined_accepted_into_neighbor": accepted,
                            "combined_neighbor_selected": bool(accepted),
                            "combined_selected_signal": None if not accepted else float(accepted[0]["signal"]),
                            "cross_lineage_pair_present": True,
                            "current_core_preserves_both_lineages": False,
                        })

    # Deduplicate exact source/neighbor/step tuples.
    unique = {}
    for row in candidates:
        key = (
            row["echo_node"], row["position_node"], row["neighbor"],
            row["echo_step"], row["position_step"],
        )
        unique[key] = row
    candidates = list(unique.values())
    candidates.sort(key=lambda row: (
        row["time_gap"],
        -int(row["both_pass_threshold"]),
        -row["sum_signal_diagnostic"],
        row["neighbor"],
    ))

    simultaneous_rows = [row for row in candidates if row["simultaneous"]]
    temporal_rows = [row for row in candidates if not row["simultaneous"]]
    best = candidates[0] if candidates else None

    if best is None:
        verdict = "no_common_neighbor_signal_pair"
    elif best["simultaneous"] and best["both_pass_threshold"]:
        verdict = "simultaneous_dual_signal_collapsed_to_single_target_value"
    elif best["simultaneous"]:
        verdict = "simultaneous_contact_but_one_or_both_signals_subthreshold"
    else:
        verdict = "temporal_near_contact_without_memory_bridge"

    return {
        "position": position,
        "candidate_count": len(candidates),
        "simultaneous_candidate_count": len(simultaneous_rows),
        "temporal_candidate_count": len(temporal_rows),
        "both_threshold_candidate_count": sum(1 for row in candidates if row["both_pass_threshold"]),
        "combined_selected_candidate_count": sum(1 for row in candidates if row["combined_neighbor_selected"]),
        "best_candidate": best,
        "verdict": verdict,
        "candidates": candidates,
        "traces": {
            "echo_only": echo_trace,
            "position_only": position_trace,
            "combined": combined_trace,
        },
    }


def observe() -> dict:
    reports = {position: candidate_rows(position) for position in TARGET_POSITIONS}
    any_simultaneous_dual = any(
        row["best_candidate"] is not None
        and row["best_candidate"]["simultaneous"]
        and row["best_candidate"]["both_pass_threshold"]
        for row in reports.values()
    )
    any_temporal_only = any(
        row["best_candidate"] is not None and not row["best_candidate"]["simultaneous"]
        for row in reports.values()
    )

    if any_simultaneous_dual:
        overall = "cross_lineage_signals_reach_same_neighbor_but_core_keeps_one_value"
    elif any_temporal_only:
        overall = "signals_reach_common_neighbor_at_different_steps_without_short_term_bridge"
    else:
        overall = "common_neighbor_exists_structurally_but_effective_dual_signal_not_confirmed"

    payload = {
        "experiment": "Core Growth Binding v30",
        "purpose": "Inspect left/center shared-neighbor candidates and determine whether E-residual and position signals truly arrive, and how Core collapses or discards them.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "diagnostic_sum_only": True,
            "puzzle_specific_adjustment": False,
        },
        "positions": reports,
        "summary": {
            "overall_verdict": overall,
            "current_target_aggregation": "max signal per target",
            "cross_lineage_identity_preserved": False,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v30.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v30</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v30</h1><p class="lead">左・中央の共通neighborへ、E残響側と位置側の信号が本当に届くかを調べる。signal・threshold・local top・採用結果を比較し、Coreの最大値集約で何が失われるかを診断する。</p><section class="panel"><div class="controls"><button id="run">共通neighborを解剖</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),rows=Object.values(d.positions);document.getElementById('metrics').innerHTML=rows.map(r=>{const b=r.best_candidate;if(!b)return `<div class="metric">${r.position} 候補<b class="warn">なし</b></div>`;return `<div class="metric">${r.position} 共通neighbor<b class="blue">Node ${b.neighbor}</b></div><div class="metric">${r.position} E側signal<b>${f(b.echo_signal)}</b></div><div class="metric">${r.position} 位置側signal<b>${f(b.position_signal)}</b></div><div class="metric">${r.position} 同時到着<b>${yn(b.simultaneous)}</b></div><div class="metric">${r.position} 両方threshold通過<b>${yn(b.both_pass_threshold)}</b></div><div class="metric">${r.position} 現行max<b>${f(b.max_signal)}</b></div><div class="metric">${r.position} 診断用sum<b>${f(b.sum_signal_diagnostic)}</b></div><div class="metric">${r.position} 弱い信号消失量<b>${f(b.weaker_signal_discarded_if_same_target)}</b></div><div class="metric">${r.position} 結合時neighbor採用<b>${yn(b.combined_neighbor_selected)}</b></div><div class="metric">${r.position} 判定<b>${r.verdict}</b></div>`}).join('')+`<div class="metric">総合判定<b class="blue">${d.summary.overall_verdict}</b></div><div class="metric">系統識別保持<b class="warn">${yn(d.summary.cross_lineage_identity_preserved)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v30: http://{HOST}:{PORT}")
    print("Shared-neighbor dual-signal diagnostic / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
