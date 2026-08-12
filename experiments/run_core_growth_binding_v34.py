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
import run_core_growth_binding_v33 as v33

HOST = "127.0.0.1"
START_PORT = 5080
OUT = ROOT / "data" / "core_growth_binding_v34" / "results"
POSITION = "左"
THRESHOLD = v33.THRESHOLD
MAX_STEPS = v33.MAX_STEPS
TIE_MARGIN = v33.TIE_MARGIN


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


def audited_run(position: str, state, adj) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    activation = v33.initial_activation(position)
    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed: set[tuple[int, int]] = set()
    steps = []
    audit_events = []

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
                affinity = float(state.affinity(source, target, step, adj))
                modulation = 0.0
                if affinity > 0:
                    modulation = min(v33.ABSOLUTE_CAP, v33.ASSIST_GAIN * affinity)
                row = {
                    "source": source,
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "base_signal": base_signal,
                    "affinity": affinity,
                    "modulation": modulation,
                    "rank_signal": base_signal + modulation,
                    "local_top": target in local_top,
                    "passes_threshold": base_signal >= THRESHOLD,
                }
                all_rows.append(row)
                if row["local_top"] and row["passes_threshold"]:
                    target_candidates.setdefault(target, []).append(row)

        baseline_winners = []
        assisted_winners = []
        target_audits = []
        for target, rows in target_candidates.items():
            baseline_sorted = sorted(rows, key=lambda r: r["base_signal"], reverse=True)
            assisted_sorted = sorted(rows, key=lambda r: r["rank_signal"], reverse=True)
            baseline_winner = dict(baseline_sorted[0])
            assisted_winner = dict(assisted_sorted[0])
            margin = None
            if len(baseline_sorted) > 1:
                margin = baseline_sorted[0]["base_signal"] - baseline_sorted[1]["base_signal"]
            tie_gate = margin is not None and margin <= TIE_MARGIN
            winner_changed = (
                baseline_winner["source"] != assisted_winner["source"]
                or baseline_winner["target"] != assisted_winner["target"]
            )
            audit = {
                "step": step,
                "target": int(target),
                "candidate_count": len(rows),
                "baseline_margin": margin,
                "tie_gate_active": tie_gate,
                "baseline_winner": baseline_winner,
                "assisted_winner": assisted_winner,
                "winner_changed": winner_changed,
                "modulation_applied_outside_tie_gate": bool(
                    assisted_winner["modulation"] > 0 and not tie_gate
                ),
                "candidates": [
                    {
                        **dict(row),
                        "baseline_rank": i + 1,
                        "assisted_rank": next(
                            j + 1 for j, x in enumerate(assisted_sorted)
                            if x["source"] == row["source"] and x["target"] == row["target"]
                        ),
                    }
                    for i, row in enumerate(baseline_sorted)
                ],
            }
            target_audits.append(audit)
            baseline_winners.append(baseline_winner)
            assisted_winners.append(assisted_winner)
            if any(row["modulation"] > 0 for row in rows):
                audit_events.append(audit)

        baseline_ranked = sorted(baseline_winners, key=lambda r: r["base_signal"], reverse=True)
        assisted_ranked = sorted(assisted_winners, key=lambda r: r["rank_signal"], reverse=True)
        limit = min(brain.max_active_per_step, len(assisted_ranked))
        baseline_selected = baseline_ranked[:limit]
        assisted_selected = assisted_ranked[:limit]
        baseline_set = {(r["source"], r["target"]) for r in baseline_selected}
        assisted_set = {(r["source"], r["target"]) for r in assisted_selected}

        next_activation = np.zeros(brain.node_count, dtype=float)
        selected_edges = []
        for row in assisted_selected:
            next_activation[row["target"]] = max(next_activation[row["target"]], row["base_signal"])
            key = edge_key(row["source"], row["target"])
            traversed.add(key)
            selected_edges.append(list(key))

        steps.append({
            "step": step,
            "active_sources": [int(x) for x in active_sources],
            "target_audits": target_audits,
            "baseline_selected": [list(edge_key(r["source"], r["target"])) for r in baseline_selected],
            "assisted_selected": selected_edges,
            "selected_set_changed": baseline_set != assisted_set,
            "binding_state_active": step <= state.active_until_step,
        })
        if not assisted_selected:
            break
        activated_nodes.update(np.flatnonzero(next_activation > 0).tolist())
        activation = next_activation

    return {
        "activated_nodes": sorted(int(x) for x in activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "steps": steps,
        "audit_events": audit_events,
    }


def observe() -> dict:
    built = v33.build_binding_state(POSITION)
    state = built["state"]
    audited = audited_run(POSITION, state, built["adj"])

    events = audited["audit_events"]
    outside_gate = [e for e in events if e["modulation_applied_outside_tie_gate"]]
    winner_changes = [e for e in events if e["winner_changed"]]
    selected_set_changes = [s for s in audited["steps"] if s["selected_set_changed"]]
    true_strong_overrides = [
        e for e in events
        if e["winner_changed"]
        and e["baseline_margin"] is not None
        and e["baseline_margin"] > TIE_MARGIN
    ]

    if true_strong_overrides:
        verdict = "true_strong_decision_override_detected"
    elif outside_gate:
        verdict = "modulation_applied_outside_tie_gate_but_no_decision_changed"
    else:
        verdict = "tie_gate_behavior_consistent"

    payload = {
        "experiment": "Core Growth Binding v34",
        "purpose": "Audit every Binding-State assist modulation and determine whether v33 strong-override counts reflect actual winner/selection changes or modulation merely applied outside the tie gate.",
        "position": POSITION,
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "diagnostic_only": True,
            "core_file_modified": False,
        },
        "binding_state": {
            "contact_count": len(state.contacts),
            "active_until_step": state.active_until_step,
        },
        "summary": {
            "audit_event_count": len(events),
            "modulation_outside_tie_gate_count": len(outside_gate),
            "target_winner_change_count": len(winner_changes),
            "selected_set_change_count": len(selected_set_changes),
            "true_strong_override_count": len(true_strong_overrides),
            "v33_counting_rule": "margin > tie_margin and selected winner modulation > 0",
            "v34_strict_rule": "winner actually changed while baseline margin > tie_margin",
            "verdict": verdict,
        },
        "audit_events": events,
        "outside_tie_gate_events": outside_gate,
        "winner_change_events": winner_changes,
        "true_strong_override_events": true_strong_overrides,
        "run": audited,
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v34.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v34</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v34</h1><p class="lead">v33の全変調を監査し、tie gate外の変調・候補winner変化・選択集合変化・本当の強判断上書きを分離する。機能や値は変更しない。</p><section class="panel"><div class="controls"><button id="run">Assist変調を監査</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>監査生データ</h2><pre id="raw" class="raw">まだ監査していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary;document.getElementById('metrics').innerHTML=`<div class="metric">監査変調件数<b>${s.audit_event_count}</b></div><div class="metric">tie外変調<b class="${s.modulation_outside_tie_gate_count?'warn':'good'}">${s.modulation_outside_tie_gate_count}</b></div><div class="metric">target winner変更<b>${s.target_winner_change_count}</b></div><div class="metric">選択集合変更Step<b>${s.selected_set_change_count}</b></div><div class="metric">真の強判断上書き<b class="${s.true_strong_override_count?'warn':'good'}">${s.true_strong_override_count}</b></div><div class="metric">判定<b class="blue">${s.verdict}</b></div><div class="metric">Binding接触数<b>${d.binding_state.contact_count}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v34: http://{HOST}:{PORT}")
    print("Binding Assist audit / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
