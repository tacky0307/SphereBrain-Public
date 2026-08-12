from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import threading
import webbrowser
from dataclasses import asdict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v50 as v50
import run_core_growth_binding_v53 as v53
from core_shadow_state import StabilityProfileShadowState, attach_shadow_state, snapshot_shadow_state

HOST = "127.0.0.1"
START_PORT = 5103
OUT = ROOT / "data" / "core_growth_binding_v56" / "results"
POSITIONS = ["左", "中央"]
DECAY = 0.82
PRETRAIN_CYCLES = 3
ADAPT_CYCLES = 5
EPS = 1e-12


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


def condition_names() -> list[str]:
    return [name for name, _, _ in v44.CONDITIONS]


def condition_map() -> dict[str, tuple[float, float]]:
    return {name: (float(e), float(p)) for name, e, p in v44.CONDITIONS}


def full_v53_signature(source: dict, position: str) -> list:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            key = "left_signature" if position == "左" else "center_signature"
            return list(row[key][0])
    raise RuntimeError("v53 full signature not found")


def event_named_row(position: str, condition: str) -> dict[str, float]:
    e, p = condition_map()[condition]
    report = v44.make_scaled_report(position, e, p)
    wrapped = {
        "condition": condition,
        "echo_scale": e,
        "position_scale": p,
        "event_formed": bool(report.get("event_formed")),
        "report": report,
    }
    named = v50.named_rows([wrapped])
    if len(named) != 1:
        raise RuntimeError(f"Expected Contact Event for {position}/{condition}")
    return named[0]


def motif_present(row: dict[str, float], motif: str) -> bool:
    return bool(v50.motif_series([row], motif)[0])


class DecayedEvidence:
    def __init__(self, names: list[str], decay: float = DECAY) -> None:
        self.decay = float(decay)
        self.true_weight = {name: 0.0 for name in names}
        self.false_weight = {name: 0.0 for name in names}
        self.observations = {name: 0 for name in names}
        self.total_experiences = 0

    def update(self, condition: str, present: bool) -> None:
        for name in self.true_weight:
            self.true_weight[name] *= self.decay
            self.false_weight[name] *= self.decay
        if present:
            self.true_weight[condition] += 1.0
        else:
            self.false_weight[condition] += 1.0
        self.observations[condition] += 1
        self.total_experiences += 1

    def condition_probability(self, condition: str) -> float | None:
        t = self.true_weight[condition]
        f = self.false_weight[condition]
        total = t + f
        if total <= EPS:
            return None
        return t / total

    def condition_presence(self, condition: str) -> bool | None:
        p = self.condition_probability(condition)
        if p is None:
            return None
        return p >= 0.5

    def condition_margin(self, condition: str) -> float | None:
        p = self.condition_probability(condition)
        if p is None:
            return None
        return abs(p - 0.5) * 2.0

    def effective_mass(self, condition: str) -> float:
        return self.true_weight[condition] + self.false_weight[condition]

    def snapshot(self) -> dict:
        return {
            "decay": self.decay,
            "true_weight": dict(self.true_weight),
            "false_weight": dict(self.false_weight),
            "observations": dict(self.observations),
            "total_experiences": self.total_experiences,
            "probabilities": {name: self.condition_probability(name) for name in self.true_weight},
            "margins": {name: self.condition_margin(name) for name in self.true_weight},
            "effective_mass": {name: self.effective_mass(name) for name in self.true_weight},
        }


def weighted_profile(evidence: DecayedEvidence, motif: str) -> dict:
    names = condition_names()
    presence = {name: evidence.condition_presence(name) for name in names}
    known = [name for name in names if presence[name] is not None]
    present_count = sum(1 for name in known if presence[name])

    def stability_class() -> str:
        if not known:
            return "unknown"
        fraction = present_count / len(known)
        if len(known) == len(names) and fraction >= 1.0 - EPS:
            return "stable"
        if fraction >= 5 / 7:
            return "mostly"
        if fraction >= 2 / 7:
            return "unstable"
        return "absent"

    baseline = presence.get("baseline")

    groups = {
        "echo": ["echo_0.97", "echo_1.03"],
        "position": ["position_0.97", "position_1.03"],
        "common": ["common_0.97", "common_1.03"],
    }

    def resistant(group: str) -> bool | None:
        if baseline is None or not baseline:
            return None
        values = [presence[name] for name in groups[group]]
        if any(value is None for value in values):
            return None
        return all(bool(value) == bool(baseline) for value in values)

    margins = [evidence.condition_margin(name) for name in known]
    masses = [evidence.effective_mass(name) for name in known]
    coverage = len(known) / len(names)
    certainty = 0.0 if not margins else sum(float(x) for x in margins if x is not None) / len(margins)
    mass_confidence = 0.0 if not masses else min(1.0, sum(masses) / (len(names) * 2.0))
    confidence = 0.50 * coverage + 0.35 * certainty + 0.15 * mass_confidence

    return {
        "motif": motif,
        "stability_class": stability_class(),
        "baseline_present": baseline,
        "echo_resistant": resistant("echo"),
        "position_resistant": resistant("position"),
        "common_resistant": resistant("common"),
        "condition_count": len(known),
        "present_count": present_count,
        "coverage": coverage,
        "certainty": certainty,
        "effective_mass_confidence": mass_confidence,
        "confidence": confidence,
        "condition_presence": presence,
    }


def signature(profile: dict) -> list:
    return list(v53.signature(profile))


def state_from_profile(motif: str, profile: dict, evidence: DecayedEvidence) -> StabilityProfileShadowState:
    state = StabilityProfileShadowState(
        kind="motif_stability_profile_online_adaptive",
        motif=motif,
        stability_class=profile.get("stability_class"),
        baseline_present=profile.get("baseline_present"),
        echo_resistant=profile.get("echo_resistant"),
        position_resistant=profile.get("position_resistant"),
        common_resistant=profile.get("common_resistant"),
        evidence_conditions=int(profile.get("condition_count", 0)),
        source="online_decayed_sequential_core_activity_evidence",
        ttl=2,
    )
    state.__dict__["evidence_experiences"] = int(evidence.total_experiences)
    state.__dict__["evidence_confidence"] = float(profile.get("confidence", 0.0))
    state.__dict__["forgetting_decay"] = float(evidence.decay)
    return state


def forbidden_label_present(state: dict | None) -> bool:
    if state is None:
        return False
    text = json.dumps(state, ensure_ascii=False)
    return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])


def route_signature(binding: dict) -> dict:
    return {
        "entity_nodes": list(binding["entity_stage"]["activated_nodes"]),
        "entity_edges": list(binding["entity_stage"]["traversed_edges"]),
        "bound_nodes": list(binding["bound_stage"]["activated_nodes"]),
        "bound_edges": list(binding["bound_stage"]["traversed_edges"]),
    }


def transition_run(start_position: str, new_position: str, motif: str, targets: dict[str, list]) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    hashes_before = structural_hashes(brain)
    route_before = route_signature(v3.make_binding(copy.deepcopy(brain), "E", start_position, learn=False, assist=False))
    evidence = DecayedEvidence(condition_names(), DECAY)
    timeline = []

    def ingest(position: str, condition: str, phase: str) -> None:
        row = event_named_row(position, condition)
        present = motif_present(row, motif)
        evidence.update(condition, present)
        profile = weighted_profile(evidence, motif)
        state = state_from_profile(motif, profile, evidence)
        attach_shadow_state(brain, state)
        shadow = snapshot_shadow_state(brain)
        timeline.append({
            "index": evidence.total_experiences,
            "phase": phase,
            "condition": condition,
            "motif_present": present,
            "profile": profile,
            "signature": signature(profile),
            "matches_start_target": signature(profile) == targets[start_position],
            "matches_new_target": signature(profile) == targets[new_position],
            "shadow_state": shadow,
            "contains_position_label": forbidden_label_present(shadow),
            "evidence": evidence.snapshot(),
            "structural_hashes": structural_hashes(brain),
        })

    # Establish an old environment strongly enough that one contradiction should not rewrite it.
    for _ in range(PRETRAIN_CYCLES):
        for condition in condition_names():
            ingest(start_position, condition, "old_environment")

    pre_change = timeline[-1]

    # One contradictory observation only.
    shock_condition = "baseline"
    ingest(new_position, shock_condition, "single_contradiction")
    after_single = timeline[-1]

    # Sustained new environment.
    for _ in range(ADAPT_CYCLES):
        for condition in condition_names():
            ingest(new_position, condition, "new_environment")

    final = timeline[-1]

    # Find confidence dip after change and first full adaptation to the new target.
    post_change = [row for row in timeline if row["phase"] != "old_environment"]
    min_conf_row = min(post_change, key=lambda row: float(row["profile"]["confidence"]))
    adapted_rows = [row for row in post_change if row["matches_new_target"]]
    first_adapt = adapted_rows[0] if adapted_rows else None

    hashes_after = structural_hashes(brain)
    route_after = route_signature(v3.make_binding(copy.deepcopy(brain), "E", start_position, learn=False, assist=False))

    single_does_not_flip = not after_single["matches_new_target"]
    sustained_adapts = bool(first_adapt is not None and final["matches_new_target"])
    confidence_dips = float(min_conf_row["profile"]["confidence"]) < float(pre_change["profile"]["confidence"])
    confidence_recovers = float(final["profile"]["confidence"]) > float(min_conf_row["profile"]["confidence"])
    old_evidence_decayed = any(
        final["evidence"]["effective_mass"][name] < pre_change["evidence"]["effective_mass"][name] + ADAPT_CYCLES
        for name in condition_names()
    )

    return {
        "start_environment": start_position,
        "new_environment": new_position,
        "decay": DECAY,
        "pretrain_cycles": PRETRAIN_CYCLES,
        "adapt_cycles": ADAPT_CYCLES,
        "timeline": timeline,
        "pre_change": pre_change,
        "after_single_contradiction": after_single,
        "minimum_confidence_after_change": {
            "index": min_conf_row["index"],
            "confidence": min_conf_row["profile"]["confidence"],
        },
        "first_new_target_match": None if first_adapt is None else first_adapt["index"],
        "final": final,
        "single_contradiction_does_not_flip": single_does_not_flip,
        "sustained_change_adapts": sustained_adapts,
        "confidence_dips_during_transition": confidence_dips,
        "confidence_recovers_after_adaptation": confidence_recovers,
        "old_evidence_decays": old_evidence_decayed,
        "no_position_labels": all(not row["contains_position_label"] for row in timeline),
        "structure_unchanged": hashes_before == hashes_after and all(row["structural_hashes"] == hashes_before for row in timeline),
        "route_unchanged": route_before == route_after,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }


def observe() -> dict:
    print("v56: reproducing v53 one-profile target...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v56 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    targets = {
        "左": full_v53_signature(source, "左"),
        "中央": full_v53_signature(source, "中央"),
    }

    print("v56: old environment -> new environment adaptation...", flush=True)
    transitions = {
        "A_to_B": transition_run("左", "中央", motif, targets),
        "B_to_A": transition_run("中央", "左", motif, targets),
    }

    all_single_resistant = all(x["single_contradiction_does_not_flip"] for x in transitions.values())
    all_adapt = all(x["sustained_change_adapts"] for x in transitions.values())
    all_dip = all(x["confidence_dips_during_transition"] for x in transitions.values())
    all_recover = all(x["confidence_recovers_after_adaptation"] for x in transitions.values())
    all_forget = all(x["old_evidence_decays"] for x in transitions.values())
    all_no_labels = all(x["no_position_labels"] for x in transitions.values())
    all_structure = all(x["structure_unchanged"] for x in transitions.values())
    all_route = all(x["route_unchanged"] for x in transitions.values())
    right_runs = v44.condition_runs("右")
    right_absent = all(not row["event_formed"] for row in right_runs)
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    adaptation_pass = bool(
        all_single_resistant
        and all_adapt
        and all_dip
        and all_recover
        and all_forget
        and all_no_labels
        and all_structure
        and all_route
        and right_absent
        and brain_file_unchanged
    )

    if adaptation_pass:
        verdict = "online_shadow_resists_single_outlier_then_forgets_and_adapts_to_sustained_environment_change"
        next_step = "validate_natural_online_adaptation_without_predefined_condition_cycle_before_behavioral_core_effect"
        readiness = "adaptive_shadow_stable"
    else:
        verdict = "online_forgetting_or_adaptation_contract_not_yet_satisfied"
        next_step = "audit_decay_rate_confidence_or_transition_evidence_before_behavioral_core_effect"
        readiness = "mixed_stream_shadow_only"

    payload = {
        "experiment": "Core Growth Binding v56",
        "purpose": "Test online forgetting and adaptation of the one-motif Stability Profile Shadow State. Old evidence decays continuously; one contradictory experience must not immediately flip the profile, while sustained changed experience must eventually replace the old profile and rebuild confidence.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "activation_changed_by_shadow": False,
            "structural_assist_used": False,
            "decoder_receives_shadow": False,
            "shadow_persisted_to_brain_json": False,
            "position_label_stored_in_shadow": False,
            "online_decay": DECAY,
            "evidence_form": "per-condition decayed present/absent weights",
            "single_outlier_should_not_flip": True,
            "sustained_change_should_adapt": True,
        },
        "selected_motif": motif,
        "target_signatures": targets,
        "transitions": transitions,
        "summary": {
            "single_contradiction_resisted_both_directions": all_single_resistant,
            "sustained_change_adapts_both_directions": all_adapt,
            "confidence_dips_during_transition_both": all_dip,
            "confidence_recovers_both": all_recover,
            "old_evidence_decays_both": all_forget,
            "no_position_labels_in_shadow": all_no_labels,
            "all_routes_unchanged": all_route,
            "all_core_structures_unchanged": all_structure,
            "right_event_absent": right_absent,
            "brain_file_unchanged": brain_file_unchanged,
            "online_adaptation_pass": adaptation_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v56.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v56</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v56</h1><p class="lead">古いEvidenceを毎経験ごとに減衰させ、単発の矛盾には耐えながら、環境変化が継続すればStability Profile Shadow Stateが新しい状態へ適応できるかを検証する。ShadowはRoute・学習・Core構造には作用しない。</p><section class="panel"><div class="controls"><button id="run">Online Forgetting & Adaptationを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Adaptation生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const text=await res.text();if(!res.ok)throw new Error(`HTTP ${res.status}: ${text.slice(0,240)}`);const d=JSON.parse(text),s=d.summary;m.innerHTML=`<div class="metric">単発矛盾に耐性<b class="${s.single_contradiction_resisted_both_directions?'good':'warn'}">${yn(s.single_contradiction_resisted_both_directions)}</b></div><div class="metric">継続変化へ適応<b class="${s.sustained_change_adapts_both_directions?'good':'warn'}">${yn(s.sustained_change_adapts_both_directions)}</b></div><div class="metric">移行中confidence低下<b class="${s.confidence_dips_during_transition_both?'good':'warn'}">${yn(s.confidence_dips_during_transition_both)}</b></div><div class="metric">confidence再上昇<b class="${s.confidence_recovers_both?'good':'warn'}">${yn(s.confidence_recovers_both)}</b></div><div class="metric">古いEvidence減衰<b class="${s.old_evidence_decays_both?'good':'warn'}">${yn(s.old_evidence_decays_both)}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_shadow?'good':'warn'}">${yn(s.no_position_labels_in_shadow)}</b></div><div class="metric">Route不変<b class="${s.all_routes_unchanged?'good':'warn'}">${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">brain.json<b class="${s.brain_file_unchanged?'good':'warn'}">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Online適応PASS<b class="${s.online_adaptation_pass?'good':'warn'}">${yn(s.online_adaptation_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/observe")
def api_observe():
    try:
        return jsonify(observe())
    except Exception as exc:
        return jsonify({"error": type(exc).__name__, "message": str(exc)}), 500


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v56: http://{HOST}:{PORT}")
    print("Online Forgetting & Adaptation / decayed evidence / no route or learning effect")
    serve(app, host=HOST, port=PORT)
