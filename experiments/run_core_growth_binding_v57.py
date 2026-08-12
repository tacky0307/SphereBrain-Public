from __future__ import annotations

import copy
import json
import random
import socket
import sys
import threading
import webbrowser
from collections import deque
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v50 as v50
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v56 as v56
from core_shadow_state import attach_shadow_state, snapshot_shadow_state

HOST = "127.0.0.1"
START_PORT = 5104
OUT = ROOT / "data" / "core_growth_binding_v57" / "results"
BASE_DECAY = 0.92
FAST_DECAY = 0.62
PRETRAIN_CYCLES = 3
NEW_CYCLES = 7
RNG_SEED = 5701


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


def condition_names() -> list[str]:
    return [name for name, _, _ in v44.CONDITIONS]


def full_v53_signature(source: dict, position: str) -> list:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            key = "left_signature" if position == "左" else "center_signature"
            return list(row[key][0])
    raise RuntimeError("v53 full signature not found")


def motif_presence(position: str, condition: str, motif: str) -> bool:
    row = v56.event_named_row(position, condition)
    return v56.motif_present(row, motif)


def distinguishing_conditions(a: str, b: str, motif: str) -> list[str]:
    return [
        name for name in condition_names()
        if motif_presence(a, name, motif) != motif_presence(b, name, motif)
    ]


class NaturalAdaptiveEvidence(v56.DecayedEvidence):
    def __init__(self, names: list[str]) -> None:
        super().__init__(names, BASE_DECAY)
        self.base_decay = BASE_DECAY
        self.fast_decay = FAST_DECAY
        self.surprise_ewma = 0.0
        self.recent_surprise: deque[int] = deque(maxlen=12)
        self.drift_suspected = False
        self.fast_mode = False
        self.stable_steps = 0
        self.trigger_count = 0

    def predict_surprise(self, condition: str, present: bool) -> dict:
        p = self.condition_probability(condition)
        margin = self.condition_margin(condition)
        confident = p is not None and margin is not None and margin >= 0.45
        mismatch = bool(confident and ((p >= 0.5) != bool(present)))
        surprise = 1 if mismatch else 0
        return {
            "probability_before": p,
            "margin_before": margin,
            "confident_prediction": bool(confident),
            "mismatch": mismatch,
            "surprise": surprise,
        }

    def update_natural(self, condition: str, present: bool) -> dict:
        diagnostic = self.predict_surprise(condition, present)
        surprise = int(diagnostic["surprise"])
        self.recent_surprise.append(surprise)
        self.surprise_ewma = 0.72 * self.surprise_ewma + 0.28 * surprise

        # No external change-point enters here. Persistent internal surprise alone
        # changes the forgetting rate.
        recent_hits = sum(self.recent_surprise)
        if not self.fast_mode and recent_hits >= 2 and self.surprise_ewma >= 0.22:
            self.fast_mode = True
            self.drift_suspected = True
            self.stable_steps = 0
            self.trigger_count += 1
        elif self.fast_mode:
            if surprise == 0:
                self.stable_steps += 1
            else:
                self.stable_steps = 0
            if self.stable_steps >= 10 and self.surprise_ewma < 0.10:
                self.fast_mode = False
                self.drift_suspected = False

        self.decay = self.fast_decay if self.fast_mode else self.base_decay
        super().update(condition, present)
        diagnostic.update({
            "surprise_ewma": self.surprise_ewma,
            "recent_surprise_hits": recent_hits,
            "drift_suspected": self.drift_suspected,
            "fast_forgetting": self.fast_mode,
            "active_decay": self.decay,
            "trigger_count": self.trigger_count,
        })
        return diagnostic


def hidden_stream(start_position: str, new_position: str, motif: str, seed_offset: int) -> tuple[list[dict], int, int]:
    rng = random.Random(RNG_SEED + seed_offset)
    names = condition_names()
    diff = distinguishing_conditions(start_position, new_position, motif)
    shock_condition = diff[0] if diff else "baseline"
    stream: list[dict] = []

    # Establish old environment; order is shuffled per cycle.
    for _ in range(PRETRAIN_CYCLES):
        cycle = names[:]
        rng.shuffle(cycle)
        stream.extend({"environment": start_position, "condition": c, "evaluation_phase": "old"} for c in cycle)

    # Isolated outlier, followed by old-environment recovery. The adaptive logic
    # is not told that this item is an outlier.
    isolated_index = len(stream) + 1
    stream.append({"environment": new_position, "condition": shock_condition, "evaluation_phase": "isolated_outlier"})
    recovery = names[:]
    rng.shuffle(recovery)
    stream.extend({"environment": start_position, "condition": c, "evaluation_phase": "recovery"} for c in recovery)

    # Hidden sustained change. This boundary is used only for scoring after the run.
    true_change_index = len(stream) + 1
    for _ in range(NEW_CYCLES):
        cycle = names[:]
        rng.shuffle(cycle)
        stream.extend({"environment": new_position, "condition": c, "evaluation_phase": "new"} for c in cycle)
    return stream, isolated_index, true_change_index


def run_transition(start_position: str, new_position: str, motif: str, targets: dict[str, list], seed_offset: int) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    hashes_before = v56.structural_hashes(brain)
    route_before = v56.route_signature(v3.make_binding(copy.deepcopy(brain), "E", start_position, learn=False, assist=False))
    evidence = NaturalAdaptiveEvidence(condition_names())
    stream, isolated_index, true_change_index = hidden_stream(start_position, new_position, motif, seed_offset)
    timeline = []

    for index, item in enumerate(stream, start=1):
        present = motif_presence(item["environment"], item["condition"], motif)
        detector = evidence.update_natural(item["condition"], present)
        profile = v56.weighted_profile(evidence, motif)
        state = v56.state_from_profile(motif, profile, evidence)
        state.__dict__["drift_suspected"] = bool(detector["drift_suspected"])
        state.__dict__["surprise_ewma"] = float(detector["surprise_ewma"])
        state.__dict__["adaptive_forgetting"] = bool(detector["fast_forgetting"])
        attach_shadow_state(brain, state)
        shadow = snapshot_shadow_state(brain)
        sig = v56.signature(profile)
        timeline.append({
            "index": index,
            "condition": item["condition"],
            "evaluation_phase": item["evaluation_phase"],
            "motif_present": present,
            "detector": detector,
            "profile": profile,
            "signature": sig,
            "matches_start_target": sig == targets[start_position],
            "matches_new_target": sig == targets[new_position],
            "shadow_state": shadow,
            "contains_position_label": v56.forbidden_label_present(shadow),
            "structural_hashes": v56.structural_hashes(brain),
        })

    trigger_rows = [r for r in timeline if r["detector"]["fast_forgetting"]]
    first_trigger = trigger_rows[0]["index"] if trigger_rows else None
    isolated_row = timeline[isolated_index - 1]
    pre_change_rows = timeline[: true_change_index - 1]
    false_trigger_before_change = any(
        r["detector"]["fast_forgetting"] for r in pre_change_rows
        if r["index"] != isolated_index
    )
    isolated_does_not_trigger = not isolated_row["detector"]["fast_forgetting"]

    post_change = timeline[true_change_index - 1 :]
    adapted = [r for r in post_change if r["matches_new_target"]]
    first_adapt = adapted[0]["index"] if adapted else None
    final = timeline[-1]
    pre_change_profile_rows = [r for r in pre_change_rows if r["matches_start_target"]]
    pre_conf = pre_change_profile_rows[-1]["profile"]["confidence"] if pre_change_profile_rows else pre_change_rows[-1]["profile"]["confidence"]
    min_post_conf = min(float(r["profile"]["confidence"]) for r in post_change)
    final_conf = float(final["profile"]["confidence"])

    hashes_after = v56.structural_hashes(brain)
    route_after = v56.route_signature(v3.make_binding(copy.deepcopy(brain), "E", start_position, learn=False, assist=False))

    detected_after_change = first_trigger is not None and first_trigger >= true_change_index
    adapted_without_external_switch = bool(first_adapt is not None and final["matches_new_target"])
    confidence_dip = min_post_conf < float(pre_conf)
    confidence_recovery = final_conf > min_post_conf

    return {
        "evaluation_only_ground_truth": {
            "isolated_outlier_index": isolated_index,
            "true_sustained_change_index": true_change_index,
        },
        "detector_received_change_point": False,
        "detector_received_environment_label": False,
        "first_internal_drift_trigger": first_trigger,
        "detection_delay": None if first_trigger is None else first_trigger - true_change_index,
        "first_new_profile_match": first_adapt,
        "adaptation_delay": None if first_adapt is None else first_adapt - true_change_index,
        "isolated_outlier_does_not_trigger": isolated_does_not_trigger,
        "no_false_trigger_before_sustained_change": not false_trigger_before_change,
        "drift_detected_after_sustained_change": detected_after_change,
        "adapts_without_external_switch": adapted_without_external_switch,
        "confidence_dips": confidence_dip,
        "confidence_recovers": confidence_recovery,
        "final_matches_new_target": final["matches_new_target"],
        "timeline": timeline,
        "no_position_labels": all(not r["contains_position_label"] for r in timeline),
        "structure_unchanged": hashes_before == hashes_after and all(r["structural_hashes"] == hashes_before for r in timeline),
        "route_unchanged": route_before == route_after,
    }


def observe() -> dict:
    print("v57: reproducing v53 one-profile candidate...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v57 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    targets = {"左": full_v53_signature(source, "左"), "中央": full_v53_signature(source, "中央")}

    print("v57: running hidden, change-point-free streams...", flush=True)
    transitions = {
        "A_to_B": run_transition("左", "中央", motif, targets, 0),
        "B_to_A": run_transition("中央", "左", motif, targets, 100),
    }

    right_absent = all(not row["event_formed"] for row in v44.condition_runs("右"))
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)
    all_isolated = all(x["isolated_outlier_does_not_trigger"] for x in transitions.values())
    all_no_false = all(x["no_false_trigger_before_sustained_change"] for x in transitions.values())
    all_detect = all(x["drift_detected_after_sustained_change"] for x in transitions.values())
    all_adapt = all(x["adapts_without_external_switch"] for x in transitions.values())
    all_dip = all(x["confidence_dips"] for x in transitions.values())
    all_recover = all(x["confidence_recovers"] for x in transitions.values())
    all_labels = all(x["no_position_labels"] for x in transitions.values())
    all_structure = all(x["structure_unchanged"] for x in transitions.values())
    all_route = all(x["route_unchanged"] for x in transitions.values())

    natural_pass = bool(all_isolated and all_no_false and all_detect and all_adapt and all_dip and all_recover and all_labels and all_structure and all_route and right_absent and brain_file_unchanged)
    if natural_pass:
        verdict = "shadow_detects_persistent_change_and_adapts_without_external_change_point"
        next_step = "broaden_unstructured_streams_then_consider_bounded_behavioral_shadow_assist"
        readiness = "natural_adaptive_shadow_stable"
    else:
        verdict = "natural_change_point_free_adaptation_not_yet_stable"
        next_step = "audit_surprise_threshold_detection_delay_or_adaptive_decay_before_behavioral_effect"
        readiness = "adaptive_shadow_only"

    payload = {
        "experiment": "Core Growth Binding v57",
        "purpose": "Test natural online adaptation without exposing any change-point or environment label to the Shadow update logic. Persistent prediction surprise must trigger faster forgetting; isolated outliers must not.",
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
            "change_point_given_to_detector": False,
            "environment_label_given_to_detector": False,
            "normal_decay": BASE_DECAY,
            "adaptive_fast_decay": FAST_DECAY,
            "drift_signal": "internal prediction surprise only",
        },
        "selected_motif": motif,
        "target_signatures_for_evaluation_only": targets,
        "transitions": transitions,
        "summary": {
            "isolated_outlier_resisted": all_isolated,
            "no_false_trigger_before_change": all_no_false,
            "persistent_change_detected": all_detect,
            "adapts_without_external_change_point": all_adapt,
            "confidence_dips_during_transition": all_dip,
            "confidence_recovers_after_adaptation": all_recover,
            "no_position_labels_in_shadow": all_labels,
            "all_routes_unchanged": all_route,
            "all_core_structures_unchanged": all_structure,
            "right_event_absent": right_absent,
            "brain_file_unchanged": brain_file_unchanged,
            "natural_online_adaptation_pass": natural_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v57.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v57</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v57</h1><p class="lead">変化点や新環境ラベルをShadowへ教えず、経験ごとの予測外（surprise）の持続だけからdriftを疑い、forgetting速度を自律変更して適応できるかを検証する。</p><section class="panel"><div class="controls"><button id="run">Change-Point Free適応を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Natural Adaptation生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const text=await res.text();if(!res.ok)throw new Error(text.slice(0,500));const d=JSON.parse(text),s=d.summary;m.innerHTML=`<div class="metric">単発外れ値耐性<b class="${s.isolated_outlier_resisted?'good':'warn'}">${yn(s.isolated_outlier_resisted)}</b></div><div class="metric">事前誤検知なし<b class="${s.no_false_trigger_before_change?'good':'warn'}">${yn(s.no_false_trigger_before_change)}</b></div><div class="metric">持続変化検知<b class="${s.persistent_change_detected?'good':'warn'}">${yn(s.persistent_change_detected)}</b></div><div class="metric">外部切替なし適応<b class="${s.adapts_without_external_change_point?'good':'warn'}">${yn(s.adapts_without_external_change_point)}</b></div><div class="metric">confidence低下<b class="${s.confidence_dips_during_transition?'good':'warn'}">${yn(s.confidence_dips_during_transition)}</b></div><div class="metric">confidence再上昇<b class="${s.confidence_recovers_after_adaptation?'good':'warn'}">${yn(s.confidence_recovers_after_adaptation)}</b></div><div class="metric">位置ラベルなし<b>${yn(s.no_position_labels_in_shadow)}</b></div><div class="metric">Route不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b>${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">Natural適応PASS<b class="${s.natural_online_adaptation_pass?'good':'warn'}">${yn(s.natural_online_adaptation_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
</script></body></html>'''

@app.get("/")
def index(): return PAGE

@app.post("/api/observe")
def api_observe(): return jsonify(observe())

def open_browser(): webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v57: http://{HOST}:{PORT}")
    print("Natural Online Adaptation / Change-Point Free / no Core behavior changes")
    serve(app, host=HOST, port=PORT)
