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
START_PORT = 5081
OUT = ROOT / "data" / "core_growth_binding_v35" / "results"
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


def edge_set(rows) -> set[tuple[int, int]]:
    return {edge_key(*row) for row in rows}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def initial_activation(position: str) -> np.ndarray:
    brain = v3.base.CORE
    activation = np.zeros(brain.node_count, dtype=float)
    for node in v33.v27.entity_echo_nodes():
        activation[int(node)] = max(activation[int(node)], v3.ECHO_STRENGTH)
    for node in v3.position_nodes(position):
        activation[int(node)] = max(activation[int(node)], 1.0)
    return activation


def run_mode(
    position: str,
    mode: str,
    state: v33.CrossLineageBindingState,
    adj: list[set[int]],
) -> dict:
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

            raw_scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            top_indices = np.argpartition(raw_scores, -branch_count)[-branch_count:]
            local_top = {int(neighbors[i]) for i in top_indices}

            for idx, target_raw in enumerate(neighbors):
                target = int(target_raw)
                base_signal = float(raw_scores[idx]) * float(brain.signal_decay)
                is_top = target in local_top
                passes = base_signal >= THRESHOLD
                affinity = state.affinity(source, target, step, adj) if mode != "baseline" else 0.0
                row = {
                    "source": source,
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "base_signal": base_signal,
                    "affinity": affinity,
                    "local_top": is_top,
                    "passes_threshold": passes,
                }
                all_rows.append(row)
                if not is_top or not passes:
                    continue
                target_candidates.setdefault(target, []).append(row)

        ranked_targets = []
        target_audits = []

        for target, rows in target_candidates.items():
            baseline_sorted = sorted(rows, key=lambda r: r["base_signal"], reverse=True)
            baseline_winner = baseline_sorted[0]
            baseline_margin = (
                baseline_sorted[0]["base_signal"] - baseline_sorted[1]["base_signal"]
                if len(baseline_sorted) > 1
                else None
            )
            tie_gate_active = bool(
                mode == "binding_state_assist"
                and baseline_margin is not None
                and baseline_margin <= TIE_MARGIN
            )

            modulated_rows = []
            for row in rows:
                modulation = 0.0
                if tie_gate_active and row["affinity"] > 0.0:
                    modulation = min(ABSOLUTE_CAP, ASSIST_GAIN * row["affinity"])
                updated = dict(row)
                updated["modulation"] = modulation
                updated["rank_signal"] = row["base_signal"] + modulation
                updated["baseline_margin"] = baseline_margin
                updated["tie_gate_active"] = tie_gate_active
                modulated_rows.append(updated)

            modulated_rows.sort(key=lambda r: r["rank_signal"], reverse=True)
            winner = modulated_rows[0]
            winner_changed = (
                winner["source"] != baseline_winner["source"]
                or winner["target"] != baseline_winner["target"]
            )
            winner = dict(winner)
            winner["baseline_winner_source"] = int(baseline_winner["source"])
            winner["winner_changed_within_target"] = winner_changed
            ranked_targets.append(winner)
            target_audits.append({
                "target": int(target),
                "baseline_margin": baseline_margin,
                "tie_gate_active": tie_gate_active,
                "baseline_winner_source": int(baseline_winner["source"]),
                "modulated_winner_source": int(winner["source"]),
                "winner_changed": bool(winner_changed),
                "candidate_count": len(rows),
                "modulation_count": sum(1 for r in modulated_rows if r["modulation"] > 0.0),
                "rows": modulated_rows,
            })

        baseline_ranked = sorted(
            [
                {
                    "source": int(sorted(rows, key=lambda r: r["base_signal"], reverse=True)[0]["source"]),
                    "target": int(target),
                    "base_signal": float(sorted(rows, key=lambda r: r["base_signal"], reverse=True)[0]["base_signal"]),
                }
                for target, rows in target_candidates.items()
            ],
            key=lambda r: r["base_signal"],
            reverse=True,
        )
        baseline_selected = baseline_ranked[: min(brain.max_active_per_step, len(baseline_ranked))]
        baseline_selected_edges = {
            edge_key(r["source"], r["target"]) for r in baseline_selected
        }

        ranked_targets.sort(key=lambda r: r["rank_signal"], reverse=True)
        selected = ranked_targets[: min(brain.max_active_per_step, len(ranked_targets))]
        selected_edges = {edge_key(r["source"], r["target"]) for r in selected}

        next_activation = np.zeros(brain.node_count, dtype=float)
        for row in selected:
            next_activation[row["target"]] = max(next_activation[row["target"]], row["base_signal"])
            traversed.add(edge_key(row["source"], row["target"]))

        trace.append({
            "step": step,
            "active_sources": [int(x) for x in active_sources],
            "binding_state_active": step <= state.active_until_step,
            "binding_contact_count": len(state.contacts),
            "target_audits": target_audits,
            "selected_rows": selected,
            "baseline_selected_edges": [list(x) for x in sorted(baseline_selected_edges)],
            "selected_edges": [list(x) for x in sorted(selected_edges)],
            "selection_set_changed": selected_edges != baseline_selected_edges,
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


def observe() -> dict:
    built = v33.build_binding_state(POSITION)
    state: v33.CrossLineageBindingState = built["state"]
    adj = built["adj"]

    baseline = run_mode(POSITION, "baseline", state, adj)
    state_only = run_mode(POSITION, "binding_state_only", state, adj)
    assisted = run_mode(POSITION, "binding_state_assist", state, adj)

    baseline_edges = edge_set(baseline["traversed_edges"])
    state_edges = edge_set(state_only["traversed_edges"])
    assist_edges = edge_set(assisted["traversed_edges"])

    tie_target_count = 0
    tie_inside_modulation_count = 0
    tie_outside_modulation_count = 0
    target_winner_change_count = 0
    selection_change_step_count = 0
    true_strong_override_count = 0
    audit_events = []

    for step in assisted["trace"]:
        if step["selection_set_changed"]:
            selection_change_step_count += 1
        for target_audit in step["target_audits"]:
            if target_audit["tie_gate_active"]:
                tie_target_count += 1
            if target_audit["winner_changed"]:
                target_winner_change_count += 1
            margin = target_audit["baseline_margin"]
            for row in target_audit["rows"]:
                if row["modulation"] > 0.0:
                    if target_audit["tie_gate_active"]:
                        tie_inside_modulation_count += 1
                    else:
                        tie_outside_modulation_count += 1
            if (
                target_audit["winner_changed"]
                and margin is not None
                and margin > TIE_MARGIN
            ):
                true_strong_override_count += 1
            audit_events.append({"step": step["step"], **target_audit})

    if tie_outside_modulation_count > 0:
        verdict = "tie_gate_implementation_failed"
    elif true_strong_override_count > 0:
        verdict = "true_strong_decision_override_detected"
    elif tie_inside_modulation_count == 0:
        verdict = "true_tie_gate_safe_but_no_eligible_modulation"
    elif target_winner_change_count > 0 or selection_change_step_count > 0:
        verdict = "tie_only_decision_change_detected"
    else:
        verdict = "tie_only_modulation_applied_but_route_unchanged"

    payload = {
        "experiment": "Core Growth Binding v35",
        "purpose": "Apply Cross-Lineage Binding affinity only after baseline target margins are computed, and only inside the true tie gate.",
        "position": POSITION,
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "candidate_set_changed": False,
            "tie_gate_controls_modulation": True,
            "core_file_modified": False,
        },
        "binding_state": {
            "contact_count": len(state.contacts),
            "active_until_step": state.active_until_step,
            "contacts": [item.__dict__ for item in state.contacts],
        },
        "runs": {
            "baseline": baseline,
            "binding_state_only": state_only,
            "true_tie_gated_assist": assisted,
        },
        "comparison": {
            "baseline_vs_state_node_jaccard": jaccard(set(baseline["activated_nodes"]), set(state_only["activated_nodes"])),
            "baseline_vs_state_edge_jaccard": jaccard(baseline_edges, state_edges),
            "baseline_vs_assist_node_jaccard": jaccard(set(baseline["activated_nodes"]), set(assisted["activated_nodes"])),
            "baseline_vs_assist_edge_jaccard": jaccard(baseline_edges, assist_edges),
            "state_only_changed_route": state_edges != baseline_edges,
            "assist_changed_route": assist_edges != baseline_edges,
            "assist_only_edges": [list(x) for x in sorted(assist_edges - baseline_edges)],
        },
        "audit": {
            "tie_target_count": tie_target_count,
            "tie_inside_modulation_count": tie_inside_modulation_count,
            "tie_outside_modulation_count": tie_outside_modulation_count,
            "target_winner_change_count": target_winner_change_count,
            "selection_change_step_count": selection_change_step_count,
            "true_strong_override_count": true_strong_override_count,
            "events": audit_events,
            "verdict": verdict,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v35.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v35</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v35</h1><p class="lead">通常scoreでtarget内marginを計算した後、真のtie targetだけCross-Lineage Binding affinityを適用する。tie外変調・強判断上書き・経路差を監査する。</p><section class="panel"><div class="controls"><button id="run">True Tie Gateを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>監査生データ</h2><pre id="raw" class="raw">まだ検証していません。</pre></section></main><script>
function f(x){return Number(x).toFixed(6)}function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),a=d.audit,c=d.comparison;document.getElementById('metrics').innerHTML=`<div class="metric">Binding接触数<b>${d.binding_state.contact_count}</b></div><div class="metric">tie target数<b>${a.tie_target_count}</b></div><div class="metric">tie内変調<b>${a.tie_inside_modulation_count}</b></div><div class="metric">tie外変調<b class="${a.tie_outside_modulation_count===0?'good':'warn'}">${a.tie_outside_modulation_count}</b></div><div class="metric">target winner変更<b>${a.target_winner_change_count}</b></div><div class="metric">選択集合変更Step<b>${a.selection_change_step_count}</b></div><div class="metric">真の強判断上書き<b class="${a.true_strong_override_count===0?'good':'warn'}">${a.true_strong_override_count}</b></div><div class="metric">Assist経路変更<b>${yn(c.assist_changed_route)}</b></div><div class="metric">通常vsAssist Node<b>${f(c.baseline_vs_assist_node_jaccard)}</b></div><div class="metric">通常vsAssist Edge<b>${f(c.baseline_vs_assist_edge_jaccard)}</b></div><div class="metric">判定<b class="blue">${a.verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v35: http://{HOST}:{PORT}")
    print("True tie-gated Binding Assist / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
