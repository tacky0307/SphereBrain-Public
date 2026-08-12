from __future__ import annotations

import copy
import json
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
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v56 as v56
import run_core_growth_binding_v57 as v57
import run_core_growth_binding_v58 as v58
import run_core_growth_binding_v59 as v59
from core_shadow_state import attach_shadow_state, snapshot_shadow_state

HOST = "127.0.0.1"
START_PORT = 5107
OUT = ROOT / "data" / "core_growth_binding_v60" / "results"
RECENT_WINDOW = 12
MIN_SURPRISE_HITS = 4
MIN_DISTINCT_CONFLICTS = 4
MIN_MARGIN = 0.60
MAX_FRESH_GAP = 9
EWMA_GATE = 0.22


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


def full_v53_signature(source: dict, position: str) -> list:
    return v58.full_v53_signature(source, position)


def names() -> list[str]:
    return v57.condition_names()


class MultiFactorAdaptiveEvidence(v57.NaturalAdaptiveEvidence):
    """v57 evidence with a stricter multi-factor drift-entry gate.

    The underlying surprise definition and base/fast decay values are unchanged.
    Only the transition into fast-forgetting mode is gated by multiple independent
    signals: persistence, condition diversity, prediction margin, and freshness.
    """

    def __init__(self, condition_names: list[str]) -> None:
        super().__init__(condition_names)
        self.recent_conflicts: deque[dict] = deque(maxlen=RECENT_WINDOW)
        self.last_seen_step = {name: None for name in condition_names}
        self.step = 0

    def update_natural(self, condition: str, present: bool) -> dict:
        self.step += 1
        diagnostic = self.predict_surprise(condition, present)
        surprise = int(diagnostic["surprise"])
        margin = diagnostic.get("margin_before")
        previous_seen = self.last_seen_step.get(condition)
        gap = None if previous_seen is None else self.step - int(previous_seen)
        self.last_seen_step[condition] = self.step

        self.recent_surprise.append(surprise)
        self.surprise_ewma = 0.72 * self.surprise_ewma + 0.28 * surprise

        conflict = {
            "condition": condition,
            "surprise": surprise,
            "margin": None if margin is None else float(margin),
            "last_seen_gap": gap,
            "fresh": gap is None or gap <= MAX_FRESH_GAP,
            "high_margin": margin is not None and float(margin) >= MIN_MARGIN,
        }
        self.recent_conflicts.append(conflict)

        surprise_conflicts = [x for x in self.recent_conflicts if x["surprise"]]
        recent_hits = len(surprise_conflicts)
        distinct_conditions = len({x["condition"] for x in surprise_conflicts})
        high_margin_hits = sum(1 for x in surprise_conflicts if x["high_margin"])
        fresh_high_margin_conditions = {
            x["condition"] for x in surprise_conflicts if x["high_margin"] and x["fresh"]
        }

        factors = {
            "ewma": self.surprise_ewma >= EWMA_GATE,
            "persistent_hits": recent_hits >= MIN_SURPRISE_HITS,
            "multi_condition": distinct_conditions >= MIN_DISTINCT_CONFLICTS,
            "high_margin": high_margin_hits >= MIN_SURPRISE_HITS,
            "fresh_evidence": len(fresh_high_margin_conditions) >= 2,
        }
        gate_open = bool(
            factors["ewma"]
            and factors["persistent_hits"]
            and factors["multi_condition"]
            and factors["high_margin"]
            and factors["fresh_evidence"]
        )

        if not self.fast_mode and gate_open:
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
        # Bypass v57.NaturalAdaptiveEvidence.update_natural to avoid its old gate.
        v56.DecayedEvidence.update(self, condition, present)

        diagnostic.update({
            "surprise_ewma": self.surprise_ewma,
            "recent_surprise_hits": recent_hits,
            "distinct_surprise_conditions": distinct_conditions,
            "high_margin_surprise_hits": high_margin_hits,
            "fresh_high_margin_condition_count": len(fresh_high_margin_conditions),
            "last_seen_gap": gap,
            "gate_factors": factors,
            "multi_factor_gate_open": gate_open,
            "drift_suspected": self.drift_suspected,
            "fast_forgetting": self.fast_mode,
            "active_decay": self.decay,
            "trigger_count": self.trigger_count,
        })
        return diagnostic


def run_scenario_multifactor(spec: dict, motif: str, targets: dict[str, list]) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    hashes_before = v56.structural_hashes(brain)
    route_before = v56.route_signature(v3.make_binding(copy.deepcopy(brain), "E", "左", learn=False, assist=False))
    evidence = MultiFactorAdaptiveEvidence(names())
    timeline = []

    for index, item in enumerate(spec["stream"], start=1):
        present = v57.motif_presence(item["environment"], item["condition"], motif)
        detector = evidence.update_natural(item["condition"], present)
        profile = v56.weighted_profile(evidence, motif)
        state = v56.state_from_profile(motif, profile, evidence)
        state.__dict__["drift_suspected"] = bool(detector["drift_suspected"])
        state.__dict__["surprise_ewma"] = float(detector["surprise_ewma"])
        state.__dict__["adaptive_forgetting"] = bool(detector["fast_forgetting"])
        state.__dict__["multi_factor_gate"] = True
        attach_shadow_state(brain, state)
        shadow = snapshot_shadow_state(brain)
        sig = v56.signature(profile)
        timeline.append({
            "index": index,
            "condition": item["condition"],
            "evaluation_segment": item["evaluation_segment"],
            "motif_present": present,
            "detector": detector,
            "profile": profile,
            "signature": sig,
            "matches_left": sig == targets["左"],
            "matches_center": sig == targets["中央"],
            "shadow_state": shadow,
            "contains_position_label": v56.forbidden_label_present(shadow),
            "structural_hashes": v56.structural_hashes(brain),
        })

    persistent_scores = []
    boundaries = spec.get("boundaries", [])
    for boundary in boundaries:
        b = int(boundary["index"])
        target = boundary["target"]
        next_boundaries = [int(x["index"]) for x in boundaries if int(x["index"]) > b]
        end = min(next_boundaries) - 1 if next_boundaries else len(timeline)
        rows = timeline[b - 1:end]
        triggers = [r for r in rows if r["detector"]["fast_forgetting"]]
        matches = [r for r in rows if (r["matches_left"] if target == "左" else r["matches_center"])]
        first_trigger = triggers[0]["index"] if triggers else None
        first_match = matches[0]["index"] if matches else None
        persistent_scores.append({
            "boundary_index": b,
            "target": target,
            "first_trigger": first_trigger,
            "detection_delay": None if first_trigger is None else first_trigger - b,
            "first_target_match": first_match,
            "adaptation_delay": None if first_match is None else first_match - b,
            "detected": first_trigger is not None,
            "adapted_within_segment": first_match is not None,
        })

    false_region = v59.false_region_indices(spec, timeline)
    false_fast_steps = [
        r for r in timeline
        if int(r["index"]) in false_region and bool(r["detector"].get("fast_forgetting"))
    ]
    false_episodes = 0
    active = False
    for row in timeline:
        is_false = int(row["index"]) in false_region and bool(row["detector"].get("fast_forgetting"))
        if is_false and not active:
            false_episodes += 1
            active = True
        elif not is_false:
            active = False

    final = timeline[-1]
    target = spec["final_target"]
    final_matches_target = final["matches_left"] if target == "左" else final["matches_center"]

    outlier_recovery_ok = True
    if spec["name"] == "clustered_outliers_then_recovery":
        outlier_recovery_ok = bool(final["matches_left"])
    temporary_reversion_ok = True
    if spec["name"] == "shift_with_temporary_reversion":
        temporary_reversion_ok = bool(final["matches_center"])

    hashes_after = v56.structural_hashes(brain)
    route_after = v56.route_signature(v3.make_binding(copy.deepcopy(brain), "E", "左", learn=False, assist=False))

    return {
        "name": spec["name"],
        "stream_length": len(timeline),
        "persistent_transition_scores": persistent_scores,
        "false_trigger_steps": len(false_fast_steps),
        "false_trigger_episodes": false_episodes,
        "final_target": target,
        "final_matches_target": final_matches_target,
        "clustered_outlier_recovery_ok": outlier_recovery_ok,
        "temporary_reversion_recovery_ok": temporary_reversion_ok,
        "timeline": timeline,
        "no_position_labels": all(not r["contains_position_label"] for r in timeline),
        "structure_unchanged": hashes_before == hashes_after and all(r["structural_hashes"] == hashes_before for r in timeline),
        "route_unchanged": route_before == route_after,
    }


def summarize(results: list[dict]) -> dict:
    transitions = [x for r in results for x in r["persistent_transition_scores"]]
    detection_rate = 0.0 if not transitions else sum(1 for x in transitions if x["detected"]) / len(transitions)
    adaptation_rate = 0.0 if not transitions else sum(1 for x in transitions if x["adapted_within_segment"]) / len(transitions)
    delays = [x["detection_delay"] for x in transitions if x["detection_delay"] is not None]
    adaptation_delays = [x["adaptation_delay"] for x in transitions if x["adaptation_delay"] is not None]
    return {
        "scenario_count": len(results),
        "persistent_transition_count": len(transitions),
        "detection_rate": detection_rate,
        "adaptation_rate": adaptation_rate,
        "mean_detection_delay": None if not delays else sum(delays) / len(delays),
        "max_detection_delay": None if not delays else max(delays),
        "mean_adaptation_delay": None if not adaptation_delays else sum(adaptation_delays) / len(adaptation_delays),
        "false_trigger_steps": sum(r["false_trigger_steps"] for r in results),
        "false_trigger_episodes": sum(r["false_trigger_episodes"] for r in results),
        "all_final_targets_reached": all(r["final_matches_target"] for r in results),
        "clustered_outlier_recovery": all(r["clustered_outlier_recovery_ok"] for r in results),
        "temporary_reversion_recovery": all(r["temporary_reversion_recovery_ok"] for r in results),
        "all_routes_unchanged": all(r["route_unchanged"] for r in results),
        "all_core_structures_unchanged": all(r["structure_unchanged"] for r in results),
        "no_position_labels": all(r["no_position_labels"] for r in results),
    }


def observe() -> dict:
    print("v60: reproducing v53 one-profile candidate...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v60 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    targets = {"左": full_v53_signature(source, "左"), "中央": full_v53_signature(source, "中央")}
    specs = v58.build_scenarios(motif)

    print("v60: running legacy v57 detector on v58 scenarios...", flush=True)
    legacy_results = [v58.run_scenario(spec, motif, targets) for spec in specs]
    legacy_audits = [v59.audit_scenario(spec, report) for spec, report in zip(specs, legacy_results)]
    legacy_summary = {
        "detection_rate": (
            sum(1 for r in legacy_results for x in r["persistent_transition_scores"] if x["detected"])
            / max(1, sum(len(r["persistent_transition_scores"]) for r in legacy_results))
        ),
        "adaptation_rate": (
            sum(1 for r in legacy_results for x in r["persistent_transition_scores"] if x["adapted_within_segment"])
            / max(1, sum(len(r["persistent_transition_scores"]) for r in legacy_results))
        ),
        "false_trigger_steps": sum(a["false_trigger_step_count"] for a in legacy_audits),
        "false_trigger_episodes": sum(a["false_trigger_episode_count"] for a in legacy_audits),
        "all_final_targets_reached": all(r["final_matches_target"] for r in legacy_results),
    }

    print("v60: running Multi-Factor Drift Gate on same scenarios...", flush=True)
    gated_results = [run_scenario_multifactor(spec, motif, targets) for spec in specs]
    gated_summary = summarize(gated_results)

    right_absent = all(not row["event_formed"] for row in v44.condition_runs("右"))
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    false_episodes_improved = gated_summary["false_trigger_episodes"] < legacy_summary["false_trigger_episodes"]
    zero_false_episodes = gated_summary["false_trigger_episodes"] == 0
    sensitivity_preserved = gated_summary["detection_rate"] >= legacy_summary["detection_rate"] - 1e-12
    adaptation_preserved = gated_summary["adaptation_rate"] >= legacy_summary["adaptation_rate"] - 1e-12
    recovery_preserved = gated_summary["clustered_outlier_recovery"] and gated_summary["temporary_reversion_recovery"]

    pass_gate = bool(
        zero_false_episodes
        and sensitivity_preserved
        and adaptation_preserved
        and gated_summary["all_final_targets_reached"]
        and recovery_preserved
        and gated_summary["all_routes_unchanged"]
        and gated_summary["all_core_structures_unchanged"]
        and gated_summary["no_position_labels"]
        and right_absent
        and brain_file_unchanged
    )

    if pass_gate:
        verdict = "multi_factor_gate_removes_false_trigger_episodes_without_losing_detection_or_adaptation"
        next_step = "consider_bounded_behavioral_shadow_assist_with_strict_non_override_caps"
        readiness = "behavioral_shadow_candidate"
    elif false_episodes_improved and sensitivity_preserved and adaptation_preserved:
        verdict = "multi_factor_gate_improves_specificity_but_false_triggers_remain"
        next_step = "audit_remaining_gate_failures_before_behavioral_shadow_effect"
        readiness = "natural_shadow_only"
    else:
        verdict = "multi_factor_gate_reduces_sensitivity_or_adaptation"
        next_step = "reconsider_gate_factor_balance_before_behavioral_shadow_effect"
        readiness = "natural_shadow_only"

    payload = {
        "experiment": "Core Growth Binding v60",
        "purpose": "Compare the unchanged v57 surprise-only drift detector with a multi-factor drift gate on the same six v58 unstructured scenarios. The gate requires persistent, diverse, high-margin, sufficiently fresh conflicts before fast forgetting begins.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "behavioral_shadow_effect": False,
            "position_label_stored_in_shadow": False,
            "legacy_detector_modified": False,
            "same_v58_scenarios": True,
            "multi_factor_gate": {
                "recent_window": RECENT_WINDOW,
                "min_surprise_hits": MIN_SURPRISE_HITS,
                "min_distinct_conflict_conditions": MIN_DISTINCT_CONFLICTS,
                "min_prediction_margin": MIN_MARGIN,
                "max_fresh_gap": MAX_FRESH_GAP,
                "surprise_ewma_gate": EWMA_GATE,
            },
        },
        "selected_motif": motif,
        "legacy_summary": legacy_summary,
        "gated_summary": gated_summary,
        "gated_results": gated_results,
        "summary": {
            "legacy_false_trigger_episodes": legacy_summary["false_trigger_episodes"],
            "gated_false_trigger_episodes": gated_summary["false_trigger_episodes"],
            "false_trigger_episodes_improved": false_episodes_improved,
            "zero_false_trigger_episodes": zero_false_episodes,
            "legacy_detection_rate": legacy_summary["detection_rate"],
            "gated_detection_rate": gated_summary["detection_rate"],
            "legacy_adaptation_rate": legacy_summary["adaptation_rate"],
            "gated_adaptation_rate": gated_summary["adaptation_rate"],
            "sensitivity_preserved": sensitivity_preserved,
            "adaptation_preserved": adaptation_preserved,
            "recovery_preserved": recovery_preserved,
            "right_event_absent": right_absent,
            "brain_file_unchanged": brain_file_unchanged,
            "multi_factor_gate_pass": pass_gate,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v60.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v60</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v60</h1><p class="lead">v59で分離したfalse trigger Episodeを減らすため、surpriseだけではなく、持続性・複数condition矛盾・prediction margin・evidence freshnessを同時に要求するMulti-Factor Drift Gateを、旧v57 Detectorと同じ6 Scenarioで比較する。</p><section class="panel"><div class="controls"><button id="run">Multi-Factor Drift Gateを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>比較生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return Number(v).toFixed(3)}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">旧False Episode<b>${s.legacy_false_trigger_episodes}</b></div><div class="metric">新False Episode<b class="${s.gated_false_trigger_episodes===0?'good':'warn'}">${s.gated_false_trigger_episodes}</b></div><div class="metric">False改善<b class="${s.false_trigger_episodes_improved?'good':'warn'}">${yn(s.false_trigger_episodes_improved)}</b></div><div class="metric">0 Episode達成<b class="${s.zero_false_trigger_episodes?'good':'warn'}">${yn(s.zero_false_trigger_episodes)}</b></div><div class="metric">旧検知率<b>${f(s.legacy_detection_rate)}</b></div><div class="metric">新検知率<b class="${s.sensitivity_preserved?'good':'warn'}">${f(s.gated_detection_rate)}</b></div><div class="metric">旧適応率<b>${f(s.legacy_adaptation_rate)}</b></div><div class="metric">新適応率<b class="${s.adaptation_preserved?'good':'warn'}">${f(s.gated_adaptation_rate)}</b></div><div class="metric">Recovery維持<b class="${s.recovery_preserved?'good':'warn'}">${yn(s.recovery_preserved)}</b></div><div class="metric">Gate PASS<b class="${s.multi_factor_gate_pass?'good':'warn'}">${yn(s.multi_factor_gate_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v60: http://{HOST}:{PORT}")
    print("Multi-Factor Drift Gate / legacy comparison / no Core behavioral effect")
    serve(app, host=HOST, port=PORT)
