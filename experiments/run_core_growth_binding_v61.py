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
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v54 as v54
from behavioral_shadow_assist import BehavioralShadowAssistConfig, BoundedBehavioralShadowAssist
from core_shadow_state import attach_shadow_state, snapshot_shadow_state

HOST = "127.0.0.1"
START_PORT = 5108
OUT = ROOT / "data" / "core_growth_binding_v61" / "results"
POSITIONS = ["左", "中央"]
CONFIDENCE = 0.96
TIE_MARGIN = 0.0025
ABS_CAP = 2.5e-5
REL_CAP = 0.20


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


def find_full_audit(source: dict) -> dict:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            return row
    raise RuntimeError("v53 full audit not found")


def build_shadow_states(source: dict) -> tuple[str, dict[str, object]]:
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v61 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    full = find_full_audit(source)
    count = len(source.get("conditions", []))
    states = {}
    for position, key in [("左", "left_signature"), ("中央", "center_signature")]:
        state = v54.state_from_signature(motif, full[key][0], count)
        state.kind = "motif_stability_profile_behavioral_candidate"
        state.source = "validated_incremental_stability_shadow"
        state.__dict__["evidence_confidence"] = CONFIDENCE
        state.__dict__["drift_suspected"] = False
        state.__dict__["multi_factor_gate_validated"] = True
        states[position] = state
    return motif, states


def forbidden_label_present(state: dict | None) -> bool:
    if state is None:
        return False
    text = json.dumps(state, ensure_ascii=False)
    return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])


def propagate_custom(brain, sources, *, context=None, steps=10, shadow_assist=None):
    original_initial = brain._initial_activation
    original_assist = brain.structural_assist

    def initial(source_nodes, context_nodes):
        src = list(source_nodes)
        activation = np.zeros(brain.node_count, dtype=float)
        for node in src:
            activation[node] = 1.0
        if context_nodes:
            for node in context_nodes:
                activation[node] = max(activation[node], v3.ECHO_STRENGTH)
        return src, activation

    brain._initial_activation = initial
    if shadow_assist is not None:
        brain.structural_assist = shadow_assist
    try:
        result = brain.propagate(
            sources,
            context_nodes=context or None,
            steps=steps,
            threshold=0.18,
            noise=0.0,
            learn=False,
        )
        traces = copy.deepcopy(brain.last_structural_assist_trace)
    finally:
        brain._initial_activation = original_initial
        brain.structural_assist = original_assist
    return result, traces


def summarize(result) -> dict:
    return {
        "activated_nodes": list(result.activated_nodes),
        "traversed_edges": [list(x) for x in result.traversed_edges],
        "history": [list(x) for x in result.activation_history],
        "final_active": np.flatnonzero(np.asarray(result.final_activation) > 0).tolist(),
    }


def make_binding_custom(brain, position: str, *, assisted: bool, bottleneck: bool = False) -> dict:
    if bottleneck:
        brain.max_active_per_step = 1

    entity_result, _ = propagate_custom(brain, v3.entity_nodes("E"), steps=8)
    echo = np.flatnonzero(np.asarray(entity_result.final_activation) > 0).tolist()[: v3.ECHO_LIMIT]

    assist = None
    if assisted:
        assist = BoundedBehavioralShadowAssist(
            BehavioralShadowAssistConfig(
                enabled=True,
                minimum_confidence=0.90,
                tie_margin=TIE_MARGIN,
                relative_cap_ratio=REL_CAP,
                absolute_cap=ABS_CAP,
            )
        )
    bound_result, traces = propagate_custom(
        brain,
        v3.position_nodes(position),
        context=echo,
        steps=10,
        shadow_assist=assist,
    )
    return {
        "entity": summarize(entity_result),
        "bound": summarize(bound_result),
        "assist_trace": traces,
    }


def route_signature(report: dict) -> dict:
    return {
        "entity_nodes": report["entity"]["activated_nodes"],
        "entity_edges": report["entity"]["traversed_edges"],
        "bound_nodes": report["bound"]["activated_nodes"],
        "bound_edges": report["bound"]["traversed_edges"],
    }


def trace_summary(traces: list[dict]) -> dict:
    eligible = [t for t in traces if t.get("shadow_gate_open")]
    active = [t for t in traces if t.get("tie_gate_active")]
    changed = [t for t in traces if t.get("top_candidate_changed")]
    strong_override = [
        t for t in changed
        if t.get("baseline_margin") is not None and float(t["baseline_margin"]) > TIE_MARGIN
    ]
    max_mod = max((float(t.get("absolute_modulation", 0.0)) for t in traces), default=0.0)
    max_ratio = max((float(t.get("meaningful_relative_ratio", 0.0)) for t in traces), default=0.0)
    values_changed = any(bool(t.get("candidate_values_changed")) for t in traces)
    threshold_cross = any(bool(t.get("threshold_crossing_possible")) for t in traces)
    forced = any(bool(t.get("winner_forced_by_shadow")) for t in traces)
    return {
        "step_count": len(traces),
        "eligible_tie_steps": len(eligible),
        "active_assist_steps": len(active),
        "top_candidate_change_steps": len(changed),
        "strong_decision_override_steps": len(strong_override),
        "max_absolute_modulation": max_mod,
        "max_relative_ratio": max_ratio,
        "candidate_values_changed": values_changed,
        "threshold_crossing_possible": threshold_cross,
        "winner_forced_by_shadow": forced,
        "absolute_cap_respected": max_mod <= ABS_CAP + 1e-15,
        "relative_cap_respected": max_ratio <= REL_CAP + 1e-12,
    }


def one_probe(position: str, state, *, bottleneck: bool) -> dict:
    baseline_brain = copy.deepcopy(v3.base.CORE)
    assisted_brain = copy.deepcopy(v3.base.CORE)
    attach_shadow_state(assisted_brain, state)
    state_snapshot = snapshot_shadow_state(assisted_brain)

    base_hash_before = structural_hashes(baseline_brain)
    assist_hash_before = structural_hashes(assisted_brain)
    baseline = make_binding_custom(baseline_brain, position, assisted=False, bottleneck=bottleneck)
    assisted = make_binding_custom(assisted_brain, position, assisted=True, bottleneck=bottleneck)
    base_hash_after = structural_hashes(baseline_brain)
    assist_hash_after = structural_hashes(assisted_brain)

    traces = trace_summary(assisted["assist_trace"])
    return {
        "position": position,
        "mode": "choice_bottleneck" if bottleneck else "native_capacity",
        "shadow_state": state_snapshot,
        "contains_position_label": forbidden_label_present(state_snapshot),
        "baseline_route": route_signature(baseline),
        "assisted_route": route_signature(assisted),
        "route_changed": route_signature(baseline) != route_signature(assisted),
        "assist": traces,
        "baseline_structure_unchanged": base_hash_before == base_hash_after,
        "assisted_structure_unchanged": assist_hash_before == assist_hash_after,
        "cross_brain_structure_equal": base_hash_after == assist_hash_after,
    }


def suppression_probe(position: str, state, *, drift: bool = False, low_confidence: bool = False) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    altered = copy.deepcopy(state)
    if drift:
        altered.__dict__["drift_suspected"] = True
    if low_confidence:
        altered.__dict__["evidence_confidence"] = 0.50
    attach_shadow_state(brain, altered)
    report = make_binding_custom(brain, position, assisted=True, bottleneck=True)
    ts = trace_summary(report["assist_trace"])
    return {
        "drift": drift,
        "low_confidence": low_confidence,
        "active_assist_steps": ts["active_assist_steps"],
        "top_candidate_change_steps": ts["top_candidate_change_steps"],
        "suppressed": ts["active_assist_steps"] == 0 and ts["top_candidate_change_steps"] == 0,
    }


def observe() -> dict:
    print("v61: reproducing v53 minimal Stability Profile...", flush=True)
    source = v53.observe()
    motif, states = build_shadow_states(source)

    probes = {"native": {}, "bottleneck": {}}
    for position in POSITIONS:
        print(f"v61: {position} native capacity", flush=True)
        probes["native"][position] = one_probe(position, states[position], bottleneck=False)
        print(f"v61: {position} choice bottleneck", flush=True)
        probes["bottleneck"][position] = one_probe(position, states[position], bottleneck=True)

    suppression = {
        "drift": suppression_probe("左", states["左"], drift=True),
        "low_confidence": suppression_probe("左", states["左"], low_confidence=True),
    }

    all_reports = [x for group in probes.values() for x in group.values()]
    no_labels = all(not x["contains_position_label"] for x in all_reports)
    no_strong_override = all(x["assist"]["strong_decision_override_steps"] == 0 for x in all_reports)
    caps_ok = all(x["assist"]["absolute_cap_respected"] and x["assist"]["relative_cap_respected"] for x in all_reports)
    no_value_or_threshold_change = all(
        not x["assist"]["candidate_values_changed"]
        and not x["assist"]["threshold_crossing_possible"]
        and not x["assist"]["winner_forced_by_shadow"]
        for x in all_reports
    )
    structures_unchanged = all(
        x["baseline_structure_unchanged"]
        and x["assisted_structure_unchanged"]
        and x["cross_brain_structure_equal"]
        for x in all_reports
    )
    suppress_ok = suppression["drift"]["suppressed"] and suppression["low_confidence"]["suppressed"]
    eligible_steps = sum(x["assist"]["eligible_tie_steps"] for x in all_reports)
    active_steps = sum(x["assist"]["active_assist_steps"] for x in all_reports)
    changed_steps = sum(x["assist"]["top_candidate_change_steps"] for x in all_reports)
    native_route_changes = sum(1 for x in probes["native"].values() if x["route_changed"])
    bottleneck_route_changes = sum(1 for x in probes["bottleneck"].values() if x["route_changed"])
    behavioral_effect_observed = changed_steps > 0 or native_route_changes > 0 or bottleneck_route_changes > 0
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    bounded_pass = bool(
        no_labels and no_strong_override and caps_ok and no_value_or_threshold_change
        and structures_unchanged and suppress_ok and brain_file_unchanged
    )

    if bounded_pass and behavioral_effect_observed:
        verdict = "bounded_behavioral_shadow_assist_changes_only_eligible_tie_choices_without_core_override"
        next_step = "stress_test_behavioral_shadow_assist_across_unstructured_streams_and_counterfactual_shadow_states"
        readiness = "bounded_behavioral_shadow_active"
    elif bounded_pass:
        verdict = "behavioral_shadow_assist_is_safe_but_no_behavioral_choice_change_was_observed"
        next_step = "seek_natural_choice_boundary_cases_without_relaxing_safety_caps"
        readiness = "bounded_behavioral_shadow_safe_no_effect"
    else:
        verdict = "bounded_behavioral_shadow_assist_violates_non_override_contract"
        next_step = "audit_behavioral_gate_before_any_further_core_effect"
        readiness = "shadow_only"

    payload = {
        "experiment": "Core Growth Binding v61",
        "purpose": "Allow the validated Stability Profile Shadow State to influence Core behavior for the first time, only by permitting a tiny Core-local structural tie-break under strict confidence, drift, tie-margin, absolute-cap, and relative-cap gates.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "candidate_values_changed": False,
            "shadow_names_target_node": False,
            "shadow_forces_winner": False,
            "minimum_shadow_confidence": 0.90,
            "assist_forbidden_during_drift": True,
            "tie_margin": TIE_MARGIN,
            "absolute_modulation_cap": ABS_CAP,
            "relative_modulation_cap": REL_CAP,
            "choice_bottleneck_is_diagnostic_only": True,
            "brain_save_format_changed": False,
        },
        "selected_motif": motif,
        "probes": probes,
        "suppression_controls": suppression,
        "summary": {
            "eligible_tie_steps": eligible_steps,
            "active_assist_steps": active_steps,
            "top_candidate_change_steps": changed_steps,
            "native_route_change_probes": native_route_changes,
            "bottleneck_route_change_probes": bottleneck_route_changes,
            "behavioral_effect_observed": behavioral_effect_observed,
            "no_strong_decision_override": no_strong_override,
            "caps_respected": caps_ok,
            "no_candidate_value_or_threshold_change": no_value_or_threshold_change,
            "drift_and_low_confidence_suppressed": suppress_ok,
            "no_position_labels_in_shadow": no_labels,
            "all_core_structures_unchanged": structures_unchanged,
            "brain_file_unchanged": brain_file_unchanged,
            "bounded_behavioral_assist_pass": bounded_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v61.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v61</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v61</h1><p class="lead">Stability Profile Shadow StateがCoreの行動へ初めて触れる。ただし高confidence・非drift・tie-only・絶対/相対capの範囲で、ShadowはNodeを指定せずCore自身の局所構造による候補順位をわずかに補助する。</p><section class="panel"><div class="controls"><button id="run">Bounded Behavioral Shadow Assistを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Behavioral Assist生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Eligible Tie Step<b>${s.eligible_tie_steps}</b></div><div class="metric">Assist作動Step<b>${s.active_assist_steps}</b></div><div class="metric">Top候補変更Step<b>${s.top_candidate_change_steps}</b></div><div class="metric">Behavior変化観測<b class="${s.behavioral_effect_observed?'good':'blue'}">${yn(s.behavioral_effect_observed)}</b></div><div class="metric">強判断上書きなし<b class="${s.no_strong_decision_override?'good':'warn'}">${yn(s.no_strong_decision_override)}</b></div><div class="metric">Cap遵守<b class="${s.caps_respected?'good':'warn'}">${yn(s.caps_respected)}</b></div><div class="metric">値/threshold変更なし<b class="${s.no_candidate_value_or_threshold_change?'good':'warn'}">${yn(s.no_candidate_value_or_threshold_change)}</b></div><div class="metric">drift/低confidence抑制<b class="${s.drift_and_low_confidence_suppressed?'good':'warn'}">${yn(s.drift_and_low_confidence_suppressed)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">Bounded PASS<b class="${s.bounded_behavioral_assist_pass?'good':'warn'}">${yn(s.bounded_behavioral_assist_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v61: http://{HOST}:{PORT}")
    print("Bounded Behavioral Shadow Assist / tie-only / no learning / strict caps")
    serve(app, host=HOST, port=PORT)
