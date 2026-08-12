from __future__ import annotations

import copy
import hashlib
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
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v61 as v61
import run_core_growth_binding_v62 as v62
from behavioral_shadow_assist import BehavioralShadowAssistConfig, BoundedBehavioralShadowAssist
from core_shadow_state import attach_shadow_state

HOST = "127.0.0.1"
START_PORT = 5110
OUT = ROOT / "data" / "core_growth_binding_v63" / "results"
REPLAY_N = 20
CONTROL_N = 10
TOL = 1e-12


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


def array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def structural_hashes(brain) -> dict[str, str]:
    return {
        "weights": array_hash(brain.weights),
        "adjacency": array_hash(brain.adjacency.astype(np.uint8)),
        "usage": array_hash(brain.usage),
        "node_usage": array_hash(brain.node_usage),
    }


def no_assist_trace() -> dict:
    return {
        "enabled": False,
        "shadow_gate_open": False,
        "tie_gate_active": False,
        "baseline_margin": None,
        "top_candidate_changed": False,
        "absolute_modulation": 0.0,
        "meaningful_relative_ratio": 0.0,
        "candidate_values_changed": False,
        "threshold_crossing_possible": False,
        "winner_forced_by_shadow": False,
    }


class TargetStepAssist:
    """Permit the v61 assist at one observed propagation step only."""

    def __init__(self, target_step: int) -> None:
        self.target_step = int(target_step)
        self.call_index = 0
        self.inner = BoundedBehavioralShadowAssist(
            BehavioralShadowAssistConfig(
                enabled=True,
                minimum_confidence=0.90,
                tie_margin=v61.TIE_MARGIN,
                relative_cap_ratio=v61.REL_CAP,
                absolute_cap=v61.ABS_CAP,
            )
        )

    def reorder(self, brain, ranked, history, edges_by_step):
        step = self.call_index
        self.call_index += 1
        if step != self.target_step:
            trace = no_assist_trace()
            trace["target_step"] = self.target_step
            trace["observed_step"] = step
            return list(ranked), trace
        reordered, trace = self.inner.reorder(brain, ranked, history, edges_by_step)
        trace["target_step"] = self.target_step
        trace["observed_step"] = step
        return reordered, trace


def initial_activation(brain, position: str, echo_scale: float, position_scale: float) -> np.ndarray:
    activation = np.zeros(brain.node_count, dtype=float)
    for node in v27.entity_echo_nodes():
        activation[int(node)] = max(
            activation[int(node)], float(v3.ECHO_STRENGTH) * float(echo_scale)
        )
    for node in v3.position_nodes(position):
        activation[int(node)] = max(activation[int(node)], float(position_scale))
    return activation


def replay_trace(
    *,
    position: str,
    echo_scale: float,
    position_scale: float,
    state=None,
    target_step: int | None = None,
) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    if state is not None:
        attach_shadow_state(brain, copy.deepcopy(state))
    before_hash = structural_hashes(brain)
    assist = TargetStepAssist(target_step) if target_step is not None else None

    activation = initial_activation(brain, position, echo_scale, position_scale)
    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed: set[tuple[int, int]] = set()
    history = [sorted(activated_nodes)]
    edges_by_step: list[list[tuple[int, int]]] = []
    step_rows = []

    for step_index in range(v44.STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break

        candidates: dict[int, tuple[float, int]] = {}
        for source_raw in active_sources:
            source = int(source_raw)
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            for local_index in best_indices:
                target = int(neighbors[local_index])
                signal = float(scores[local_index]) * float(brain.signal_decay)
                if signal < v44.THRESHOLD:
                    continue
                previous = candidates.get(target)
                if previous is None or signal > previous[0]:
                    candidates[target] = (signal, source)

        if not candidates:
            break

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        baseline_top = int(ranked[0][0]) if ranked else None
        baseline_second = int(ranked[1][0]) if len(ranked) > 1 else None
        baseline_margin = None if len(ranked) < 2 else float(ranked[0][1][0] - ranked[1][1][0])

        trace = no_assist_trace()
        if assist is not None:
            ranked, trace = assist.reorder(brain, ranked, history, edges_by_step)
        assisted_top = int(ranked[0][0]) if ranked else None

        selected = ranked[: min(brain.max_active_per_step, len(ranked))]
        next_activation = np.zeros(brain.node_count, dtype=float)
        accepted: list[tuple[int, int]] = []
        for target, (signal, source) in selected:
            if signal < v44.THRESHOLD:
                continue
            next_activation[target] = max(next_activation[target], float(signal))
            edge = tuple(sorted((int(source), int(target))))
            accepted.append(edge)
            traversed.add(edge)

        step_rows.append({
            "step": step_index,
            "candidate_count": len(candidates),
            "baseline_top": baseline_top,
            "baseline_second": baseline_second,
            "baseline_margin": baseline_margin,
            "assisted_top": assisted_top,
            "accepted_edges": [list(x) for x in accepted],
            "assist": trace,
        })

        if not accepted:
            break
        active_now = np.flatnonzero(next_activation > 0).tolist()
        if not active_now:
            break
        activated_nodes.update(active_now)
        history.append(active_now)
        edges_by_step.append(accepted)
        activation = next_activation

    after_hash = structural_hashes(brain)
    return {
        "steps": step_rows,
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "final_active_nodes": np.flatnonzero(activation > 0).tolist(),
        "structure_unchanged": before_hash == after_hash,
    }


def target_row(report: dict, step: int) -> dict | None:
    for row in report.get("steps", []):
        if int(row["step"]) == int(step):
            return row
    return None


def replay_case(case: dict, state, *, control: bool = False) -> dict:
    kwargs = {
        "position": case["position"],
        "echo_scale": float(case["echo_scale"]),
        "position_scale": float(case["position_scale"]),
    }
    baseline = replay_trace(**kwargs)
    assisted = replay_trace(**kwargs, state=state, target_step=int(case["step"]))
    bstep = target_row(baseline, case["step"])
    astep = target_row(assisted, case["step"])
    reproduced = bool(
        bstep is not None
        and bstep.get("baseline_margin") is not None
        and abs(float(bstep["baseline_margin"]) - float(case["margin"])) <= TOL
    )
    trace = {} if astep is None else astep.get("assist", {})
    top_changed = bool(astep is not None and bstep is not None and astep["assisted_top"] != bstep["baseline_top"])
    route_changed = baseline["traversed_edges"] != assisted["traversed_edges"]
    final_reconverged = baseline["final_active_nodes"] == assisted["final_active_nodes"]
    strong_margin = float(case["margin"]) > v61.TIE_MARGIN + TOL
    return {
        "case": case,
        "control": control,
        "reproduced_exact_margin": reproduced,
        "baseline_target_step": bstep,
        "assisted_target_step": astep,
        "assist_gate_open": bool(trace.get("shadow_gate_open")),
        "assist_active": bool(trace.get("tie_gate_active")),
        "top_candidate_changed": top_changed,
        "route_changed": route_changed,
        "final_active_reconverged": final_reconverged,
        "strong_margin_control": strong_margin,
        "strong_decision_overridden": bool(top_changed and strong_margin),
        "absolute_cap_respected": float(trace.get("absolute_modulation", 0.0)) <= v61.ABS_CAP + 1e-15,
        "relative_cap_respected": float(trace.get("meaningful_relative_ratio", 0.0)) <= v61.REL_CAP + 1e-12,
        "candidate_values_changed": bool(trace.get("candidate_values_changed", False)),
        "threshold_crossing_possible": bool(trace.get("threshold_crossing_possible", False)),
        "winner_forced_by_shadow": bool(trace.get("winner_forced_by_shadow", False)),
        "baseline_structure_unchanged": baseline["structure_unchanged"],
        "assisted_structure_unchanged": assisted["structure_unchanged"],
    }


def suppression_case(case: dict, state, *, drift: bool = False, low_confidence: bool = False) -> dict:
    altered = copy.deepcopy(state)
    if drift:
        altered.__dict__["drift_suspected"] = True
    if low_confidence:
        altered.__dict__["evidence_confidence"] = 0.50
    report = replay_trace(
        position=case["position"],
        echo_scale=float(case["echo_scale"]),
        position_scale=float(case["position_scale"]),
        state=altered,
        target_step=int(case["step"]),
    )
    row = target_row(report, case["step"])
    trace = {} if row is None else row.get("assist", {})
    return {
        "drift": drift,
        "low_confidence": low_confidence,
        "assist_active": bool(trace.get("tie_gate_active")),
        "top_candidate_changed": bool(trace.get("top_candidate_changed")),
        "suppressed": not bool(trace.get("tie_gate_active")) and not bool(trace.get("top_candidate_changed")),
    }


def observe() -> dict:
    print("v63: reproducing v53 Shadow states and v62 boundary scan...", flush=True)
    source = v53.observe()
    _, states = v61.build_shadow_states(source)
    shadows = v62.shadow_meta(source)

    rows = []
    for position in v62.POSITIONS:
        for echo_scale in v62.SCALES:
            for position_scale in v62.SCALES:
                rows.extend(v62.scan_trace(position, echo_scale, position_scale, shadows[position]))

    eligible = sorted(
        [r for r in rows if r["would_be_v61_eligible"] and r["position"] in states],
        key=lambda r: (float(r["margin"]), r["position"], float(r["echo_scale"]), float(r["position_scale"]), int(r["step"])),
    )[:REPLAY_N]
    controls = sorted(
        [r for r in rows if r["position"] in states and float(r["margin"]) > v61.TIE_MARGIN],
        key=lambda r: (-float(r["margin"]), r["position"], int(r["step"])),
    )[:CONTROL_N]

    print(f"v63: replaying {len(eligible)} observed eligible boundaries...", flush=True)
    replayed = [replay_case(case, states[case["position"]]) for case in eligible]
    print(f"v63: replaying {len(controls)} strong-margin controls...", flush=True)
    control_reports = [replay_case(case, states[case["position"]], control=True) for case in controls]

    suppression = {}
    if eligible:
        first = eligible[0]
        suppression = {
            "drift": suppression_case(first, states[first["position"]], drift=True),
            "low_confidence": suppression_case(first, states[first["position"]], low_confidence=True),
        }

    all_reports = replayed + control_reports
    reproduced_count = sum(1 for r in replayed if r["reproduced_exact_margin"])
    active_cases = sum(1 for r in replayed if r["assist_active"])
    top_change_cases = sum(1 for r in replayed if r["top_candidate_changed"])
    route_change_cases = sum(1 for r in replayed if r["route_changed"])
    reconverged_cases = sum(1 for r in replayed if r["route_changed"] and r["final_active_reconverged"])
    controls_clean = all(
        not r["assist_active"] and not r["top_candidate_changed"] and not r["route_changed"]
        for r in control_reports
    )
    no_strong_override = all(not r["strong_decision_overridden"] for r in all_reports)
    caps_ok = all(r["absolute_cap_respected"] and r["relative_cap_respected"] for r in all_reports)
    no_value_threshold_force = all(
        not r["candidate_values_changed"] and not r["threshold_crossing_possible"] and not r["winner_forced_by_shadow"]
        for r in all_reports
    )
    structures_ok = all(r["baseline_structure_unchanged"] and r["assisted_structure_unchanged"] for r in all_reports)
    suppression_ok = bool(suppression) and suppression["drift"]["suppressed"] and suppression["low_confidence"]["suppressed"]
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    replay_pass = bool(
        eligible
        and reproduced_count == len(replayed)
        and controls_clean
        and no_strong_override
        and caps_ok
        and no_value_threshold_force
        and structures_ok
        and suppression_ok
        and brain_file_unchanged
    )

    if replay_pass and route_change_cases > 0:
        verdict = "bounded_shadow_changes_real_routes_only_at_observed_natural_choice_boundaries"
        readiness = "bounded_behavioral_route_effect_observed"
        next_step = "stress_test_observed_route_effect_with_counterfactual_shadow_states_and_repeated_replay"
    elif replay_pass and top_change_cases > 0:
        verdict = "bounded_shadow_changes_observed_tie_winner_but_native_capacity_masks_route_effect"
        readiness = "bounded_behavioral_choice_effect_observed"
        next_step = "audit_when_candidate_rank_changes_become_route_relevant_without_artificial_bottlenecks"
    elif replay_pass:
        verdict = "observed_boundaries_replayed_safely_but_shadow_did_not_change_choice"
        readiness = "bounded_behavioral_safe_no_replay_effect"
        next_step = "inspect_core_local_affinity_at_eligible_boundaries_before_any_cap_change"
    else:
        verdict = "observed_boundary_replay_failed_safety_or_reproduction_contract"
        readiness = "shadow_only"
        next_step = "audit_boundary_reproduction_or_target_step_gating_before_further_behavioral_effect"

    payload = {
        "experiment": "Core Growth Binding v63",
        "purpose": "Replay naturally observed v62 choice-boundary cases under identical Core and input conditions, allowing the validated v61 bounded Behavioral Shadow Assist only at the originally observed step. Compare baseline versus assisted choice and downstream route without relaxing any safety cap.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "max_active_per_step_changed": False,
            "tie_margin_changed": False,
            "assist_only_at_observed_target_step": True,
            "eligible_replay_count": REPLAY_N,
            "strong_control_count": CONTROL_N,
            "absolute_cap": v61.ABS_CAP,
            "relative_cap": v61.REL_CAP,
            "minimum_confidence": 0.90,
            "drift_suppression_required": True,
            "low_confidence_suppression_required": True,
        },
        "eligible_cases": eligible,
        "replays": replayed,
        "strong_controls": control_reports,
        "suppression_controls": suppression,
        "summary": {
            "eligible_boundary_cases": len(eligible),
            "exactly_reproduced_cases": reproduced_count,
            "assist_active_cases": active_cases,
            "top_candidate_changed_cases": top_change_cases,
            "route_changed_cases": route_change_cases,
            "route_changed_then_final_reconverged_cases": reconverged_cases,
            "strong_controls_clean": controls_clean,
            "no_strong_decision_override": no_strong_override,
            "caps_respected": caps_ok,
            "no_candidate_value_threshold_or_forced_winner_change": no_value_threshold_force,
            "drift_and_low_confidence_suppressed": suppression_ok,
            "all_core_structures_unchanged": structures_ok,
            "brain_file_unchanged": brain_file_unchanged,
            "observed_boundary_replay_pass": replay_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v63.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v63</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v63</h1><p class="lead">v62で自然に観測されたeligible choice boundaryを同一条件で再生し、観測されたStepだけにv61 Bounded Behavioral Shadow Assistを許可する。人為的tie・capacity制限・安全Cap緩和は行わない。</p><section class="panel"><div class="controls"><button id="run">Observed Boundaryを再生</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Replay生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Eligible再生<b>${s.eligible_boundary_cases}</b></div><div class="metric">完全再現<b class="${s.exactly_reproduced_cases===s.eligible_boundary_cases?'good':'warn'}">${s.exactly_reproduced_cases}</b></div><div class="metric">Assist作動Case<b>${s.assist_active_cases}</b></div><div class="metric">Top候補変更Case<b class="${s.top_candidate_changed_cases>0?'good':'blue'}">${s.top_candidate_changed_cases}</b></div><div class="metric">Route変更Case<b class="${s.route_changed_cases>0?'good':'blue'}">${s.route_changed_cases}</b></div><div class="metric">変更後再収束<b>${s.route_changed_then_final_reconverged_cases}</b></div><div class="metric">Strong Control<b class="${s.strong_controls_clean?'good':'warn'}">${yn(s.strong_controls_clean)}</b></div><div class="metric">強判断上書きなし<b class="${s.no_strong_decision_override?'good':'warn'}">${yn(s.no_strong_decision_override)}</b></div><div class="metric">Cap遵守<b class="${s.caps_respected?'good':'warn'}">${yn(s.caps_respected)}</b></div><div class="metric">drift/低confidence抑制<b class="${s.drift_and_low_confidence_suppressed?'good':'warn'}">${yn(s.drift_and_low_confidence_suppressed)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">Replay PASS<b class="${s.observed_boundary_replay_pass?'good':'warn'}">${yn(s.observed_boundary_replay_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v63: http://{HOST}:{PORT}")
    print("Observed Boundary Behavioral Replay / natural cases / target-step-only bounded assist")
    serve(app, host=HOST, port=PORT)
