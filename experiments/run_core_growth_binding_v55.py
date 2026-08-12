from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import sys
import threading
import webbrowser
from collections import Counter
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
START_PORT = 5102
OUT = ROOT / "data" / "core_growth_binding_v55" / "results"
POSITIONS = ["左", "中央"]
RNG_SEED = 5501


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


def full_v53_signature(source: dict, position: str) -> list:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            key = "left_signature" if position == "左" else "center_signature"
            return list(row[key][0])
    raise RuntimeError("v53 full signature not found")


def profile_signature(profile: dict) -> list:
    return list(v53.signature(profile))


def condition_map() -> dict[str, tuple[float, float]]:
    return {name: (float(e), float(p)) for name, e, p in v44.CONDITIONS}


def canonical_names() -> list[str]:
    return [name for name, _, _ in v44.CONDITIONS]


def streams() -> dict[str, list[str]]:
    names = canonical_names()
    rng = random.Random(RNG_SEED)
    shuffled = names * 3
    rng.shuffle(shuffled)
    return {
        "balanced_repeat": names + list(reversed(names)) + names,
        "baseline_heavy": [
            "baseline", "baseline", "baseline", "echo_0.97", "baseline",
            "position_1.03", "baseline", "echo_1.03", "common_0.97",
            "baseline", "position_0.97", "common_1.03", "baseline",
            "position_1.03", "echo_0.97", "baseline", "common_1.03",
        ],
        "echo_heavy": [
            "echo_0.97", "echo_1.03", "echo_0.97", "baseline", "echo_1.03",
            "position_0.97", "echo_0.97", "common_0.97", "position_1.03",
            "echo_1.03", "common_1.03", "echo_0.97", "baseline", "echo_1.03",
        ],
        "position_late": [
            "baseline", "echo_0.97", "echo_1.03", "common_0.97", "baseline",
            "common_1.03", "echo_0.97", "position_0.97", "position_1.03",
            "baseline", "position_1.03", "position_0.97", "common_1.03",
        ],
        "shuffled_mixed": shuffled,
    }


def unique_balanced_profile(
    evidence_by_condition: dict[str, dict[str, float]],
    motif: str,
) -> dict:
    ordered = [name for name in canonical_names() if name in evidence_by_condition]
    rows = [evidence_by_condition[name] for name in ordered]
    return v53.subset_profile(rows, ordered, motif)


def naive_occurrence_profile(
    occurrence_rows: list[dict[str, float]],
    occurrence_names: list[str],
    motif: str,
) -> dict:
    return v53.subset_profile(occurrence_rows, occurrence_names, motif)


def evidence_confidence(seen: Counter[str]) -> dict:
    names = canonical_names()
    unique = {name for name in names if seen[name] > 0}
    group_complete = {
        "baseline": seen["baseline"] > 0,
        "echo": seen["echo_0.97"] > 0 and seen["echo_1.03"] > 0,
        "position": seen["position_0.97"] > 0 and seen["position_1.03"] > 0,
        "common": seen["common_0.97"] > 0 and seen["common_1.03"] > 0,
    }
    group_fraction = sum(1 for x in group_complete.values() if x) / len(group_complete)
    unique_fraction = len(unique) / len(names)
    confidence = 0.65 * unique_fraction + 0.35 * group_fraction
    return {
        "unique_conditions_seen": len(unique),
        "unique_condition_fraction": unique_fraction,
        "group_complete": group_complete,
        "group_completion_fraction": group_fraction,
        "confidence": confidence,
    }


def shadow_from_profile(motif: str, profile: dict, confidence: dict, total_experiences: int) -> StabilityProfileShadowState:
    state = StabilityProfileShadowState(
        kind="motif_stability_profile_repeated_mixed",
        motif=motif,
        stability_class=profile.get("stability_class"),
        baseline_present=profile.get("baseline_present"),
        echo_resistant=profile.get("echo_resistant"),
        position_resistant=profile.get("position_resistant"),
        common_resistant=profile.get("common_resistant"),
        evidence_conditions=int(confidence["unique_conditions_seen"]),
        source="repeated_mixed_sequential_core_activity_evidence",
        ttl=2,
    )
    # v55-specific diagnostic metadata; still transient and not persisted.
    state_dict = state.__dict__
    state_dict["evidence_experiences"] = int(total_experiences)
    state_dict["evidence_confidence"] = float(confidence["confidence"])
    return state


def forbidden_label_present(state: dict | None) -> bool:
    if state is None:
        return False
    text = json.dumps(state, ensure_ascii=False)
    return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])


def event_row(position: str, condition: str) -> dict[str, float]:
    e, p = condition_map()[condition]
    report = v44.make_scaled_report(position, e, p)
    named = v50.named_rows([report])
    if len(named) != 1:
        raise RuntimeError(f"Expected Contact Event for {position}/{condition}")
    return named[0]


def stream_run(position: str, stream_name: str, experience_names: list[str], motif: str, target_signature: list) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    hashes_before = structural_hashes(brain)
    route_before = v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False)

    occurrence_rows: list[dict[str, float]] = []
    occurrence_names: list[str] = []
    latest_by_condition: dict[str, dict[str, float]] = {}
    seen: Counter[str] = Counter()
    timeline = []
    first_full_coverage_index = None
    post_full_mismatch_count = 0

    for index, condition in enumerate(experience_names, start=1):
        row = event_row(position, condition)
        occurrence_rows.append(row)
        occurrence_names.append(condition)
        latest_by_condition[condition] = row
        seen[condition] += 1

        balanced = unique_balanced_profile(latest_by_condition, motif)
        naive = naive_occurrence_profile(occurrence_rows, occurrence_names, motif)
        confidence = evidence_confidence(seen)
        state = shadow_from_profile(motif, balanced, confidence, index)
        attach_shadow_state(brain, state)
        shadow = snapshot_shadow_state(brain)

        balanced_sig = profile_signature(balanced)
        naive_sig = profile_signature(naive)
        full_coverage = confidence["unique_conditions_seen"] == len(canonical_names())
        if full_coverage and first_full_coverage_index is None:
            first_full_coverage_index = index
        if first_full_coverage_index is not None and balanced_sig != target_signature:
            post_full_mismatch_count += 1

        timeline.append({
            "experience_index": index,
            "condition": condition,
            "seen_counts": dict(seen),
            "confidence": confidence,
            "condition_balanced_profile": balanced,
            "condition_balanced_signature": balanced_sig,
            "naive_occurrence_profile": naive,
            "naive_occurrence_signature": naive_sig,
            "naive_differs_from_balanced": naive_sig != balanced_sig,
            "shadow_state": shadow,
            "contains_position_label": forbidden_label_present(shadow),
            "structural_hashes": structural_hashes(brain),
        })

    final_balanced = timeline[-1]["condition_balanced_signature"]
    final_naive = timeline[-1]["naive_occurrence_signature"]
    hashes_after = structural_hashes(brain)
    route_after = v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False)
    route_unchanged = (
        route_before["entity_stage"]["traversed_edges"] == route_after["entity_stage"]["traversed_edges"]
        and route_before["bound_stage"]["traversed_edges"] == route_after["bound_stage"]["traversed_edges"]
        and route_before["entity_stage"]["activated_nodes"] == route_after["entity_stage"]["activated_nodes"]
        and route_before["bound_stage"]["activated_nodes"] == route_after["bound_stage"]["activated_nodes"]
    )

    return {
        "position": position,
        "stream": stream_name,
        "experience_count": len(experience_names),
        "experience_order": experience_names,
        "first_full_coverage_index": first_full_coverage_index,
        "timeline": timeline,
        "final_balanced_signature": final_balanced,
        "final_naive_signature": final_naive,
        "target_batch_signature": target_signature,
        "balanced_matches_batch": final_balanced == target_signature,
        "naive_matches_batch": final_naive == target_signature,
        "post_full_coverage_mismatch_count": post_full_mismatch_count,
        "stable_after_full_coverage": first_full_coverage_index is not None and post_full_mismatch_count == 0,
        "final_confidence": timeline[-1]["confidence"],
        "no_position_labels_all_steps": all(not x["contains_position_label"] for x in timeline),
        "structure_unchanged": hashes_before == hashes_after and all(x["structural_hashes"] == hashes_before for x in timeline),
        "route_unchanged": route_unchanged,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }


def observe() -> dict:
    print("v55: reproducing v53 one-profile target...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v55 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    target = {
        "左": full_v53_signature(source, "左"),
        "中央": full_v53_signature(source, "中央"),
    }

    mixed_streams = streams()
    results: dict[str, dict[str, dict]] = {"左": {}, "中央": {}}
    for position in POSITIONS:
        for stream_name, experience_names in mixed_streams.items():
            print(f"v55: {position} / {stream_name} / {len(experience_names)} experiences", flush=True)
            results[position][stream_name] = stream_run(
                position, stream_name, experience_names, motif, target[position]
            )

    all_balanced_match = all(
        r["balanced_matches_batch"] for side in results.values() for r in side.values()
    )
    all_stable_after_coverage = all(
        r["stable_after_full_coverage"] for side in results.values() for r in side.values()
    )
    all_no_labels = all(
        r["no_position_labels_all_steps"] for side in results.values() for r in side.values()
    )
    all_structures_unchanged = all(
        r["structure_unchanged"] for side in results.values() for r in side.values()
    )
    all_routes_unchanged = all(
        r["route_unchanged"] for side in results.values() for r in side.values()
    )
    confidence_complete = all(
        abs(r["final_confidence"]["confidence"] - 1.0) < 1e-12
        for side in results.values() for r in side.values()
    )
    left_center_distinct = tuple(target["左"]) != tuple(target["中央"])

    naive_bias_cases = sum(
        1 for side in results.values() for r in side.values()
        if r["final_naive_signature"] != r["final_balanced_signature"]
    )

    # Right remains a negative control under the original seven live conditions.
    right_runs = v44.condition_runs("右")
    right_absent = all(not row["event_formed"] for row in right_runs)
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    repeated_mixed_pass = bool(
        all_balanced_match
        and all_stable_after_coverage
        and all_no_labels
        and all_structures_unchanged
        and all_routes_unchanged
        and confidence_complete
        and left_center_distinct
        and right_absent
        and brain_file_unchanged
    )

    if repeated_mixed_pass:
        verdict = "incremental_stability_profile_survives_repeated_biased_mixed_experience_streams"
        next_step = "validate_online_forgetting_and_adaptation_before_any_route_or_learning_effect"
        readiness = "mixed_stream_shadow_stable"
    else:
        verdict = "incremental_stability_profile_is_not_yet_stable_under_repeated_mixed_experience_streams"
        next_step = "audit_frequency_bias_or_recovery_before_behavioral_core_integration"
        readiness = "incremental_shadow_only"

    payload = {
        "experiment": "Core Growth Binding v55",
        "purpose": "Test the one-motif incremental Stability Profile under repeated, biased, and mixed sequential experience streams. Repeated occurrences increase evidence confidence but condition-balanced profile formation prevents frequency from changing the semantic stability definition.",
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
            "live_experience_recomputed_each_occurrence": True,
            "condition_balanced_profile": True,
            "frequency_bias_audited_with_naive_occurrence_profile": True,
        },
        "selected_motif": motif,
        "batch_target_signatures": target,
        "streams": mixed_streams,
        "results": results,
        "summary": {
            "all_balanced_streams_match_v53_batch": all_balanced_match,
            "all_streams_stable_after_full_condition_coverage": all_stable_after_coverage,
            "left_center_final_profiles_distinct": left_center_distinct,
            "confidence_reaches_one_after_full_evidence": confidence_complete,
            "naive_frequency_bias_case_count": naive_bias_cases,
            "no_position_labels_in_shadow": all_no_labels,
            "all_routes_unchanged": all_routes_unchanged,
            "all_core_structures_unchanged": all_structures_unchanged,
            "right_event_absent": right_absent,
            "brain_file_unchanged": brain_file_unchanged,
            "repeated_mixed_experience_pass": repeated_mixed_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v55.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v55</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v55</h1><p class="lead">重複・偏り・順序乱れを含む経験列でも、Stability Profile Shadow Stateが安定して育つかを検証する。重複回数はconfidenceへ反映するが、Profile本体は条件種類を均等に扱い、経験頻度と関係安定性を分離する。</p><section class="panel"><div class="controls"><button id="run">Repeated Mixed Experienceを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mixed Stream生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">全stream v53へ収束<b class="${s.all_balanced_streams_match_v53_batch?'good':'warn'}">${yn(s.all_balanced_streams_match_v53_batch)}</b></div><div class="metric">全被覆後安定<b class="${s.all_streams_stable_after_full_condition_coverage?'good':'warn'}">${yn(s.all_streams_stable_after_full_condition_coverage)}</b></div><div class="metric">左右最終Profile差<b class="${s.left_center_final_profiles_distinct?'good':'warn'}">${yn(s.left_center_final_profiles_distinct)}</b></div><div class="metric">confidence 1.0<b class="${s.confidence_reaches_one_after_full_evidence?'good':'warn'}">${yn(s.confidence_reaches_one_after_full_evidence)}</b></div><div class="metric">Naive頻度bias件数<b>${s.naive_frequency_bias_case_count}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_shadow?'good':'warn'}">${yn(s.no_position_labels_in_shadow)}</b></div><div class="metric">Route不変<b class="${s.all_routes_unchanged?'good':'warn'}">${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">brain.json<b class="${s.brain_file_unchanged?'good':'warn'}">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Mixed経験PASS<b class="${s.repeated_mixed_experience_pass?'good':'warn'}">${yn(s.repeated_mixed_experience_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v55: http://{HOST}:{PORT}")
    print("Repeated Mixed Experience Stability / condition-balanced shadow evidence / no Core behavior effect")
    serve(app, host=HOST, port=PORT)
