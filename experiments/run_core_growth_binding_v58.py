from __future__ import annotations

import copy
import json
import random
import socket
import sys
import threading
import webbrowser
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
from core_shadow_state import attach_shadow_state, snapshot_shadow_state

HOST = "127.0.0.1"
START_PORT = 5105
OUT = ROOT / "data" / "core_growth_binding_v58" / "results"
RNG_SEED = 5801


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
    return v57.full_v53_signature(source, position)


def names() -> list[str]:
    return v57.condition_names()


def shuffled_cycle(rng: random.Random, allowed: list[str] | None = None) -> list[str]:
    cycle = list(allowed or names())
    rng.shuffle(cycle)
    return cycle


def weighted_conditions(rng: random.Random, count: int, weights: dict[str, float]) -> list[str]:
    pool = list(weights)
    ws = [float(weights[x]) for x in pool]
    return rng.choices(pool, weights=ws, k=count)


def append_events(stream: list[dict], environment: str, conditions: list[str], segment: str) -> None:
    for condition in conditions:
        stream.append({
            "environment": environment,
            "condition": condition,
            "evaluation_segment": segment,
        })


def scenario_random_hard_shift(rng: random.Random) -> dict:
    stream: list[dict] = []
    for _ in range(3):
        append_events(stream, "左", shuffled_cycle(rng), "old")
    boundary = len(stream) + 1
    for _ in range(8):
        append_events(stream, "中央", shuffled_cycle(rng), "new")
    return {
        "name": "random_hard_shift",
        "stream": stream,
        "boundaries": [{"index": boundary, "target": "中央", "kind": "persistent"}],
        "final_target": "中央",
        "expect_no_permanent_switch_before_boundary": True,
    }


def scenario_long_bias_then_shift(rng: random.Random) -> dict:
    stream: list[dict] = []
    old_weights = {
        "baseline": 7,
        "echo_0.97": 5,
        "echo_1.03": 4,
        "position_0.97": 1,
        "position_1.03": 1,
        "common_0.97": 2,
        "common_1.03": 2,
    }
    append_events(stream, "左", weighted_conditions(rng, 42, old_weights), "old_biased")
    # Guarantee broad evidence once before the true change, without exposing that fact to the detector.
    append_events(stream, "左", shuffled_cycle(rng), "old_biased")
    boundary = len(stream) + 1
    new_weights = {
        "baseline": 6,
        "echo_0.97": 2,
        "echo_1.03": 2,
        "position_0.97": 5,
        "position_1.03": 5,
        "common_0.97": 1,
        "common_1.03": 1,
    }
    append_events(stream, "中央", weighted_conditions(rng, 56, new_weights), "new_biased")
    append_events(stream, "中央", shuffled_cycle(rng), "new_biased")
    return {
        "name": "long_bias_then_shift",
        "stream": stream,
        "boundaries": [{"index": boundary, "target": "中央", "kind": "persistent"}],
        "final_target": "中央",
        "expect_no_permanent_switch_before_boundary": True,
    }


def scenario_clustered_outliers_then_recovery(rng: random.Random, motif: str) -> dict:
    stream: list[dict] = []
    for _ in range(3):
        append_events(stream, "左", shuffled_cycle(rng), "stable_old")
    diff = v57.distinguishing_conditions("左", "中央", motif)
    outlier_conditions = (diff[:3] if len(diff) >= 3 else (diff * 3)[:3]) or ["baseline"] * 3
    outlier_start = len(stream) + 1
    append_events(stream, "中央", outlier_conditions, "outlier_cluster")
    recovery_start = len(stream) + 1
    for _ in range(4):
        append_events(stream, "左", shuffled_cycle(rng), "recovery_old")
    return {
        "name": "clustered_outliers_then_recovery",
        "stream": stream,
        "boundaries": [],
        "outlier_window": [outlier_start, recovery_start - 1],
        "final_target": "左",
        "expect_recovery_without_permanent_switch": True,
    }


def scenario_shift_with_temporary_reversion(rng: random.Random) -> dict:
    stream: list[dict] = []
    for _ in range(3):
        append_events(stream, "左", shuffled_cycle(rng), "old")
    b1 = len(stream) + 1
    for _ in range(5):
        append_events(stream, "中央", shuffled_cycle(rng), "new")
    revert_start = len(stream) + 1
    append_events(stream, "左", shuffled_cycle(rng)[:4], "temporary_reversion")
    resume_start = len(stream) + 1
    for _ in range(5):
        append_events(stream, "中央", shuffled_cycle(rng), "new_resumed")
    return {
        "name": "shift_with_temporary_reversion",
        "stream": stream,
        "boundaries": [{"index": b1, "target": "中央", "kind": "persistent"}],
        "temporary_reversion": [revert_start, resume_start - 1],
        "final_target": "中央",
        "expect_recovery_without_permanent_switch": True,
    }


def scenario_change_recovery_change(rng: random.Random) -> dict:
    stream: list[dict] = []
    for _ in range(3):
        append_events(stream, "左", shuffled_cycle(rng), "A1")
    b1 = len(stream) + 1
    for _ in range(6):
        append_events(stream, "中央", shuffled_cycle(rng), "B1")
    b2 = len(stream) + 1
    for _ in range(6):
        append_events(stream, "左", shuffled_cycle(rng), "A2")
    b3 = len(stream) + 1
    for _ in range(6):
        append_events(stream, "中央", shuffled_cycle(rng), "B2")
    return {
        "name": "change_recovery_change",
        "stream": stream,
        "boundaries": [
            {"index": b1, "target": "中央", "kind": "persistent"},
            {"index": b2, "target": "左", "kind": "persistent"},
            {"index": b3, "target": "中央", "kind": "persistent"},
        ],
        "final_target": "中央",
    }


def scenario_missing_condition_family(rng: random.Random) -> dict:
    stream: list[dict] = []
    for _ in range(3):
        append_events(stream, "左", shuffled_cycle(rng), "old")
    boundary = len(stream) + 1
    allowed = [x for x in names() if not x.startswith("position_")]
    append_events(stream, "中央", weighted_conditions(rng, 42, {x: 1.0 for x in allowed}), "new_missing_position")
    # Eventually the missing family returns, but only late in the stream.
    return_start = len(stream) + 1
    for _ in range(4):
        append_events(stream, "中央", shuffled_cycle(rng), "new_full_again")
    return {
        "name": "missing_condition_family",
        "stream": stream,
        "boundaries": [{"index": boundary, "target": "中央", "kind": "persistent"}],
        "missing_family_until": return_start - 1,
        "final_target": "中央",
    }


def build_scenarios(motif: str) -> list[dict]:
    rng = random.Random(RNG_SEED)
    return [
        scenario_random_hard_shift(random.Random(rng.randint(1, 10**9))),
        scenario_long_bias_then_shift(random.Random(rng.randint(1, 10**9))),
        scenario_clustered_outliers_then_recovery(random.Random(rng.randint(1, 10**9)), motif),
        scenario_shift_with_temporary_reversion(random.Random(rng.randint(1, 10**9))),
        scenario_change_recovery_change(random.Random(rng.randint(1, 10**9))),
        scenario_missing_condition_family(random.Random(rng.randint(1, 10**9))),
    ]


def run_scenario(spec: dict, motif: str, targets: dict[str, list]) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    hashes_before = v56.structural_hashes(brain)
    route_before = v56.route_signature(v3.make_binding(copy.deepcopy(brain), "E", "左", learn=False, assist=False))
    evidence = v57.NaturalAdaptiveEvidence(names())
    timeline = []

    for index, item in enumerate(spec["stream"], start=1):
        present = v57.motif_presence(item["environment"], item["condition"], motif)
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
    for boundary in spec.get("boundaries", []):
        b = int(boundary["index"])
        target = boundary["target"]
        next_boundaries = [int(x["index"]) for x in spec.get("boundaries", []) if int(x["index"]) > b]
        end = min(next_boundaries) - 1 if next_boundaries else len(timeline)
        rows = timeline[b - 1 : end]
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

    first_boundary = min((int(x["index"]) for x in spec.get("boundaries", [])), default=len(timeline) + 1)
    pre_boundary_false_triggers = sum(
        1 for r in timeline[: first_boundary - 1]
        if r["detector"]["fast_forgetting"]
    )

    final = timeline[-1]
    final_target = spec["final_target"]
    final_matches_target = final["matches_left"] if final_target == "左" else final["matches_center"]

    outlier_recovery_ok = True
    if spec["name"] == "clustered_outliers_then_recovery":
        out_start, out_end = spec["outlier_window"]
        recovery_rows = timeline[out_end:]
        outlier_recovery_ok = bool(final["matches_left"] and any(r["matches_left"] for r in recovery_rows))

    temporary_reversion_ok = True
    if spec["name"] == "shift_with_temporary_reversion":
        temporary_reversion_ok = bool(final["matches_center"])

    hashes_after = v56.structural_hashes(brain)
    route_after = v56.route_signature(v3.make_binding(copy.deepcopy(brain), "E", "左", learn=False, assist=False))

    confidences = [float(r["profile"]["confidence"]) for r in timeline]
    fast_mode_steps = sum(1 for r in timeline if r["detector"]["fast_forgetting"])
    return {
        "name": spec["name"],
        "stream_length": len(timeline),
        "evaluation_only_spec": {k: v for k, v in spec.items() if k != "stream"},
        "detector_received_change_points": False,
        "detector_received_environment_labels": False,
        "persistent_transition_scores": persistent_scores,
        "pre_boundary_false_trigger_steps": pre_boundary_false_triggers,
        "final_target": final_target,
        "final_matches_target": final_matches_target,
        "clustered_outlier_recovery_ok": outlier_recovery_ok,
        "temporary_reversion_recovery_ok": temporary_reversion_ok,
        "fast_forgetting_steps": fast_mode_steps,
        "confidence_min": min(confidences, default=0.0),
        "confidence_max": max(confidences, default=0.0),
        "confidence_final": confidences[-1] if confidences else 0.0,
        "timeline": timeline,
        "no_position_labels": all(not r["contains_position_label"] for r in timeline),
        "structure_unchanged": hashes_before == hashes_after and all(r["structural_hashes"] == hashes_before for r in timeline),
        "route_unchanged": route_before == route_after,
    }


def observe() -> dict:
    print("v58: reproducing v53 one-profile candidate...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v58 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    targets = {"左": full_v53_signature(source, "左"), "中央": full_v53_signature(source, "中央")}

    scenarios = build_scenarios(motif)
    results = []
    for spec in scenarios:
        print(f"v58: {spec['name']} / {len(spec['stream'])} experiences", flush=True)
        results.append(run_scenario(spec, motif, targets))

    transition_rows = [x for r in results for x in r["persistent_transition_scores"]]
    detection_rate = 0.0 if not transition_rows else sum(1 for x in transition_rows if x["detected"]) / len(transition_rows)
    adaptation_rate = 0.0 if not transition_rows else sum(1 for x in transition_rows if x["adapted_within_segment"]) / len(transition_rows)
    detection_delays = [x["detection_delay"] for x in transition_rows if x["detection_delay"] is not None]
    adaptation_delays = [x["adaptation_delay"] for x in transition_rows if x["adaptation_delay"] is not None]

    all_final_targets = all(r["final_matches_target"] for r in results)
    all_outlier_recovery = all(r["clustered_outlier_recovery_ok"] for r in results)
    all_temp_recovery = all(r["temporary_reversion_recovery_ok"] for r in results)
    all_no_labels = all(r["no_position_labels"] for r in results)
    all_structure = all(r["structure_unchanged"] for r in results)
    all_route = all(r["route_unchanged"] for r in results)
    false_trigger_steps = sum(r["pre_boundary_false_trigger_steps"] for r in results)
    right_absent = all(not row["event_formed"] for row in v44.condition_runs("右"))
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    robust_pass = bool(
        detection_rate >= 0.80
        and adaptation_rate >= 0.80
        and all_final_targets
        and all_outlier_recovery
        and all_temp_recovery
        and false_trigger_steps == 0
        and all_no_labels
        and all_structure
        and all_route
        and right_absent
        and brain_file_unchanged
    )

    if robust_pass:
        verdict = "natural_adaptive_shadow_survives_unstructured_experience_streams"
        next_step = "consider_bounded_behavioral_shadow_assist_with_hard_non_interference_caps"
        readiness = "behavioral_shadow_candidate"
    else:
        verdict = "unstructured_streams_reveal_remaining_shadow_adaptation_failure_modes"
        next_step = "audit_false_triggers_detection_delay_missing_evidence_or_reversion_before_behavioral_effect"
        readiness = "natural_shadow_only"

    payload = {
        "experiment": "Core Growth Binding v58",
        "purpose": "Stress-test the v57 change-point-free Natural Adaptive Shadow on unstructured experience streams: random order, long bias, clustered outliers, temporary reversion, repeated changes, and temporarily missing condition families.",
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
            "v57_detector_parameters_changed": False,
            "evaluation_ground_truth_used_only_after_run": True,
        },
        "selected_motif": motif,
        "target_signatures_for_evaluation_only": targets,
        "scenario_results": results,
        "summary": {
            "scenario_count": len(results),
            "persistent_transition_count": len(transition_rows),
            "persistent_change_detection_rate": detection_rate,
            "persistent_change_adaptation_rate": adaptation_rate,
            "median_detection_delay": None if not detection_delays else sorted(detection_delays)[len(detection_delays)//2],
            "median_adaptation_delay": None if not adaptation_delays else sorted(adaptation_delays)[len(adaptation_delays)//2],
            "false_trigger_steps_before_true_change": false_trigger_steps,
            "all_final_targets_reached": all_final_targets,
            "clustered_outlier_recovery_pass": all_outlier_recovery,
            "temporary_reversion_recovery_pass": all_temp_recovery,
            "no_position_labels_in_shadow": all_no_labels,
            "all_routes_unchanged": all_route,
            "all_core_structures_unchanged": all_structure,
            "right_event_absent": right_absent,
            "brain_file_unchanged": brain_file_unchanged,
            "unstructured_stream_robustness_pass": robust_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v58.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v58</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v58</h1><p class="lead">v57のNatural Adaptive Shadowを、完全ランダム順・長い偏り・外れ値の塊・一時逆戻り・複数回の環境変化・条件群の長期欠落へそのまま投入する。detectorへ変化点や環境ラベルは渡さない。</p><section class="panel"><div class="controls"><button id="run">Unstructured Streamを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Unstructured Stream生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===null||v===undefined?'なし':Number(v).toFixed(3)}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error(`HTTP ${res.status}: ${await res.text()}`)}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Scenario数<b>${s.scenario_count}</b></div><div class="metric">持続変化検知率<b class="${s.persistent_change_detection_rate>=0.8?'good':'warn'}">${f(s.persistent_change_detection_rate)}</b></div><div class="metric">適応率<b class="${s.persistent_change_adaptation_rate>=0.8?'good':'warn'}">${f(s.persistent_change_adaptation_rate)}</b></div><div class="metric">事前誤作動Step<b class="${s.false_trigger_steps_before_true_change===0?'good':'warn'}">${s.false_trigger_steps_before_true_change}</b></div><div class="metric">最終Target到達<b class="${s.all_final_targets_reached?'good':'warn'}">${yn(s.all_final_targets_reached)}</b></div><div class="metric">外れ値塊Recovery<b class="${s.clustered_outlier_recovery_pass?'good':'warn'}">${yn(s.clustered_outlier_recovery_pass)}</b></div><div class="metric">一時逆戻りRecovery<b class="${s.temporary_reversion_recovery_pass?'good':'warn'}">${yn(s.temporary_reversion_recovery_pass)}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_shadow?'good':'warn'}">${yn(s.no_position_labels_in_shadow)}</b></div><div class="metric">Route不変<b class="${s.all_routes_unchanged?'good':'warn'}">${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">Robustness PASS<b class="${s.unstructured_stream_robustness_pass?'good':'warn'}">${yn(s.unstructured_stream_robustness_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v58: http://{HOST}:{PORT}")
    print("Unstructured Experience Stream Robustness / v57 detector unchanged / no Core effects")
    serve(app, host=HOST, port=PORT)
