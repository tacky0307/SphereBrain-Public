from __future__ import annotations

import copy
import hashlib
import json
import random
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
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v50 as v50
import run_core_growth_binding_v53 as v53
from core_shadow_state import StabilityProfileShadowState, attach_shadow_state, snapshot_shadow_state

HOST = "127.0.0.1"
START_PORT = 5101
OUT = ROOT / "data" / "core_growth_binding_v54b" / "results"
POSITIONS = ["左", "中央"]
SHUFFLE_SEED = 5402


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


def profile_signature(profile: dict) -> tuple:
    return v53.signature(profile)


def shadow_from_profile(motif: str, profile: dict) -> StabilityProfileShadowState:
    return StabilityProfileShadowState(
        kind="motif_stability_profile_incremental",
        motif=motif,
        stability_class=profile.get("stability_class"),
        baseline_present=profile.get("baseline_present"),
        echo_resistant=profile.get("echo_resistant"),
        position_resistant=profile.get("position_resistant"),
        common_resistant=profile.get("common_resistant"),
        evidence_conditions=int(profile.get("condition_count", 0)),
        source="sequential_core_activity_evidence",
        ttl=2,
    )


def forbidden_label_present(state: dict | None) -> bool:
    if state is None:
        return False
    text = json.dumps(state, ensure_ascii=False)
    return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])


def full_v53_signature(source: dict, position: str) -> list:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            key = "left_signature" if position == "左" else "center_signature"
            return list(row[key][0])
    raise RuntimeError("v53 full signature not found")


def ordered_indices(order_name: str, count: int) -> list[int]:
    indices = list(range(count))
    if order_name == "reverse":
        indices.reverse()
    elif order_name == "shuffled":
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(indices)
    return indices


def sequence_run(
    *,
    position: str,
    order_name: str,
    condition_rows: list[dict],
    condition_names: list[str],
    motif: str,
    target_signature: list,
) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    hashes_before = structural_hashes(brain)
    route_before = v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False)

    indices = ordered_indices(order_name, len(condition_rows))
    observed_rows: list[dict[str, float]] = []
    observed_names: list[str] = []
    timeline = []

    for sequence_index, source_index in enumerate(indices, start=1):
        source_row = condition_rows[source_index]
        named = v50.named_rows([source_row])
        if len(named) != 1:
            raise RuntimeError(f"Expected one event-derived context row for {position}/{condition_names[source_index]}")
        observed_rows.append(named[0])
        observed_names.append(condition_names[source_index])

        profile = v53.subset_profile(observed_rows, observed_names, motif)
        state = shadow_from_profile(motif, profile)
        attach_shadow_state(brain, state)
        shadow = snapshot_shadow_state(brain)

        timeline.append({
            "experience_index": sequence_index,
            "condition": condition_names[source_index],
            "profile": profile,
            "signature": list(profile_signature(profile)),
            "shadow_state": shadow,
            "contains_position_label": forbidden_label_present(shadow),
            "structural_hashes": structural_hashes(brain),
        })

    final_profile = timeline[-1]["profile"]
    final_signature = timeline[-1]["signature"]
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
        "order": order_name,
        "condition_order": [condition_names[i] for i in indices],
        "timeline": timeline,
        "final_profile": final_profile,
        "final_signature": final_signature,
        "target_batch_signature": target_signature,
        "matches_batch_profile": final_signature == target_signature,
        "no_position_labels_all_steps": all(not row["contains_position_label"] for row in timeline),
        "structure_unchanged": hashes_before == hashes_after and all(row["structural_hashes"] == hashes_before for row in timeline),
        "route_unchanged": route_unchanged,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }


def observe() -> dict:
    print("v54B: reproducing v53 minimal profile candidate...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v54B requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]

    print("v54B: generating live condition experiences...", flush=True)
    runs = {position: v44.condition_runs(position) for position in ["左", "中央", "右"]}
    condition_names = [name for name, _, _ in v44.CONDITIONS]

    target = {
        "左": full_v53_signature(source, "左"),
        "中央": full_v53_signature(source, "中央"),
    }

    sequences = {}
    for position in POSITIONS:
        sequences[position] = {}
        for order in ["canonical", "reverse", "shuffled"]:
            print(f"v54B: {position} / {order}", flush=True)
            sequences[position][order] = sequence_run(
                position=position,
                order_name=order,
                condition_rows=runs[position],
                condition_names=condition_names,
                motif=motif,
                target_signature=target[position],
            )

    final_signatures = {
        position: {
            order: tuple(report["final_signature"])
            for order, report in orders.items()
        }
        for position, orders in sequences.items()
    }
    order_invariant = all(
        len(set(signatures.values())) == 1
        for signatures in final_signatures.values()
    )
    all_match_batch = all(
        report["matches_batch_profile"]
        for orders in sequences.values() for report in orders.values()
    )
    all_no_labels = all(
        report["no_position_labels_all_steps"]
        for orders in sequences.values() for report in orders.values()
    )
    all_structure_unchanged = all(
        report["structure_unchanged"]
        for orders in sequences.values() for report in orders.values()
    )
    all_routes_unchanged = all(
        report["route_unchanged"]
        for orders in sequences.values() for report in orders.values()
    )
    final_left_center_distinct = tuple(target["左"]) != tuple(target["中央"])
    right_absent = all(not row["event_formed"] for row in runs["右"])
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    formation_pass = bool(
        order_invariant
        and all_match_batch
        and all_no_labels
        and all_structure_unchanged
        and all_routes_unchanged
        and final_left_center_distinct
        and right_absent
        and brain_file_unchanged
    )

    if formation_pass:
        verdict = "incremental_stability_profile_converges_from_sequential_experiences"
        next_step = "validate_incremental_profile_with_repeated_mixed_experience_stream_before_any_route_or_learning_effect"
        readiness = "incremental_shadow_formed"
    else:
        verdict = "incremental_stability_profile_does_not_yet_reproduce_batch_shadow_state"
        next_step = "audit_order_dependence_or_evidence_update_before_core_behavior_integration"
        readiness = "shadow_only"

    payload = {
        "experiment": "Core Growth Binding v54B",
        "purpose": "Form the v53 one-motif Stability Profile incrementally from sequential live-condition experiences, updating only a transient Core Shadow State after each experience, and verify convergence to the batch profile independent of experience order.",
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
            "live_experience_evidence": True,
            "sequential_update": True,
            "unknown_fields_remain_none_until_evidence_exists": True,
        },
        "selected_motif": motif,
        "condition_names": condition_names,
        "batch_target_signatures": target,
        "sequences": sequences,
        "summary": {
            "order_invariant_final_profile": order_invariant,
            "all_sequences_match_v53_batch_profile": all_match_batch,
            "left_center_final_profiles_distinct": final_left_center_distinct,
            "no_position_labels_in_shadow": all_no_labels,
            "all_routes_unchanged": all_routes_unchanged,
            "all_core_structures_unchanged": all_structure_unchanged,
            "right_event_absent": right_absent,
            "brain_file_unchanged": brain_file_unchanged,
            "incremental_formation_pass": formation_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v54b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v54B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v54B</h1><p class="lead">7条件を後から一括計算せず、live経験を1件ずつ受けるたびにMotif evidenceを蓄積し、Coreの非干渉Shadow Stateを逐次更新する。canonical / reverse / shuffled順で、最終Profileがv53の一括結果へ収束するかを検証する。</p><section class="panel"><div class="controls"><button id="run">逐次経験からStabilityを形成</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Incremental生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">順序不変<b class="${s.order_invariant_final_profile?'good':'warn'}">${yn(s.order_invariant_final_profile)}</b></div><div class="metric">v53一括へ収束<b class="${s.all_sequences_match_v53_batch_profile?'good':'warn'}">${yn(s.all_sequences_match_v53_batch_profile)}</b></div><div class="metric">左右最終Profile差<b class="${s.left_center_final_profiles_distinct?'good':'warn'}">${yn(s.left_center_final_profiles_distinct)}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_shadow?'good':'warn'}">${yn(s.no_position_labels_in_shadow)}</b></div><div class="metric">Route不変<b class="${s.all_routes_unchanged?'good':'warn'}">${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">brain.json<b class="${s.brain_file_unchanged?'good':'warn'}">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">逐次形成PASS<b class="${s.incremental_formation_pass?'good':'warn'}">${yn(s.incremental_formation_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v54B: http://{HOST}:{PORT}")
    print("Incremental Stability Formation / sequential live evidence / transient Shadow only")
    serve(app, host=HOST, port=PORT)
