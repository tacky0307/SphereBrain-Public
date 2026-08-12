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
from core_shadow_state import (
    StabilityProfileShadowState,
    attach_shadow_state,
    snapshot_shadow_state,
    tick_shadow_state,
)

HOST = "127.0.0.1"
START_PORT = 5100
OUT = ROOT / "data" / "core_growth_binding_v54" / "results"
POSITIONS = ["左", "中央"]


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


def core_structural_hashes(brain) -> dict[str, str]:
    return {
        "weights": array_hash(brain.weights),
        "adjacency": array_hash(brain.adjacency.astype(np.uint8)),
        "usage": array_hash(brain.usage),
        "node_usage": array_hash(brain.node_usage),
    }


def route_signature(binding: dict) -> dict:
    return {
        "entity_nodes": list(binding["entity_stage"]["activated_nodes"]),
        "entity_edges": list(binding["entity_stage"]["traversed_edges"]),
        "bound_nodes": list(binding["bound_stage"]["activated_nodes"]),
        "bound_edges": list(binding["bound_stage"]["traversed_edges"]),
    }


def route_equal(a: dict, b: dict) -> bool:
    return route_signature(a) == route_signature(b)


def find_full_audit(v53_payload: dict) -> dict:
    for row in v53_payload.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            return row
    raise RuntimeError("v53 full-context audit was not found.")


def state_from_signature(motif: str, signature_values: list, evidence_conditions: int) -> StabilityProfileShadowState:
    stability_class, baseline_present, echo_resistant, position_resistant, common_resistant = signature_values
    return StabilityProfileShadowState(
        kind="motif_stability_profile",
        motif=motif,
        stability_class=stability_class,
        baseline_present=baseline_present,
        echo_resistant=echo_resistant,
        position_resistant=position_resistant,
        common_resistant=common_resistant,
        evidence_conditions=evidence_conditions,
        ttl=2,
    )


def forbidden_label_present(state: dict | None) -> bool:
    if state is None:
        return False
    text = json.dumps(state, ensure_ascii=False)
    return any(label in text for label in ["左", "中央", "右", "left", "center", "right"])


def shadow_probe(position: str, state: StabilityProfileShadowState) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    before_hashes = core_structural_hashes(brain)

    before_binding = v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False)
    before_route = route_signature(before_binding)

    attach_shadow_state(brain, state)
    attached_state = snapshot_shadow_state(brain)
    after_attach_hashes = core_structural_hashes(brain)

    after_binding = v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False)
    after_route = route_signature(after_binding)

    state_after_tick1 = tick_shadow_state(brain)
    tick1_hashes = core_structural_hashes(brain)
    state_after_tick2 = tick_shadow_state(brain)
    tick2_hashes = core_structural_hashes(brain)

    return {
        "shadow_state": attached_state,
        "contains_position_label": forbidden_label_present(attached_state),
        "route_before": before_route,
        "route_after": after_route,
        "route_unchanged": before_route == after_route,
        "structure_hash_before": before_hashes,
        "structure_hash_after_attach": after_attach_hashes,
        "structure_hash_after_tick1": tick1_hashes,
        "structure_hash_after_tick2": tick2_hashes,
        "structure_unchanged_on_attach": before_hashes == after_attach_hashes,
        "structure_unchanged_through_ttl": before_hashes == tick1_hashes == tick2_hashes,
        "state_after_tick1": state_after_tick1,
        "state_after_tick2": state_after_tick2,
        "ttl_retained_after_one_tick": state_after_tick1 is not None,
        "ttl_expired_after_two_ticks": state_after_tick2 is None,
    }


def observe() -> dict:
    print("v54: rebuilding v53 minimal Stability Profile...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v54 requires the v53 one-profile shadow candidate, but it was not reproduced.")

    motif = minimal["profiles"][0]
    full = find_full_audit(source)
    evidence_conditions = len(source.get("conditions", []))

    left_sig = full["left_signature"][0]
    center_sig = full["center_signature"][0]
    left_state = state_from_signature(motif, left_sig, evidence_conditions)
    center_state = state_from_signature(motif, center_sig, evidence_conditions)

    print(f"v54: attaching transient shadow state for motif: {motif}", flush=True)
    probes = {
        "左": shadow_probe("左", left_state),
        "中央": shadow_probe("中央", center_state),
    }

    shadow_states_distinct = probes["左"]["shadow_state"] != probes["中央"]["shadow_state"]
    no_position_labels = all(not row["contains_position_label"] for row in probes.values())
    all_routes_unchanged = all(row["route_unchanged"] for row in probes.values())
    all_structures_unchanged = all(
        row["structure_unchanged_on_attach"] and row["structure_unchanged_through_ttl"]
        for row in probes.values()
    )
    ttl_ok = all(
        row["ttl_retained_after_one_tick"] and row["ttl_expired_after_two_ticks"]
        for row in probes.values()
    )
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    shadow_integration_pass = bool(
        shadow_states_distinct
        and no_position_labels
        and all_routes_unchanged
        and all_structures_unchanged
        and ttl_ok
        and brain_file_unchanged
    )

    if shadow_integration_pass:
        verdict = "stability_profile_shadow_state_integrated_without_core_interference"
        next_step = "v54b_incremental_stability_formation_from_sequential_experiences"
        readiness = "shadow_integrated"
    else:
        verdict = "shadow_state_integration_failed_non_interference_contract"
        next_step = "audit_shadow_attachment_before_any_incremental_learning"
        readiness = "not_yet"

    payload = {
        "experiment": "Core Growth Binding v54",
        "purpose": "Attach the v53 one-profile Stability Profile as a transient Core-instance Shadow State and verify that it does not affect route, weights, topology, usage, learning, persistence, or labels.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "activation_changed_by_shadow": False,
            "structural_assist_used": False,
            "decoder_receives_shadow": False,
            "shadow_persisted_to_brain_json": False,
            "core_file_modified": False,
            "position_label_stored_in_shadow": False,
            "shadow_ttl": 2,
            "integration_form": "transient attribute attached to a SphereBrain instance through core_shadow_state.py; brain.py save/load format remains unchanged",
        },
        "source_v53": {
            "minimal_profile_count": minimal.get("size"),
            "selected_motif": motif,
            "full_plus_loco_separated": source.get("summary", {}).get("full_plus_all_leave_one_out_separated"),
            "core_readiness": source.get("summary", {}).get("core_readiness"),
        },
        "shadow_probes": probes,
        "summary": {
            "shadow_states_distinct": shadow_states_distinct,
            "no_position_labels_in_shadow": no_position_labels,
            "all_routes_unchanged": all_routes_unchanged,
            "all_core_structures_unchanged": all_structures_unchanged,
            "ttl_behavior_pass": ttl_ok,
            "brain_file_unchanged": brain_file_unchanged,
            "shadow_integration_pass": shadow_integration_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v54.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v54</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v54</h1><p class="lead">v53で得た最小1 Stability Profileを、SphereBrain実インスタンスへ非永続Shadow Stateとして装着する。伝播・Edge・weight・threshold・learning・Decoderには一切作用させず、Coreに載せても通常経路が完全不変かを検証する。</p><section class="panel"><div class="controls"><button id="run">Shadow StateをCoreへ装着して検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Shadow State生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Shadow状態差<b class="${s.shadow_states_distinct?'good':'warn'}">${yn(s.shadow_states_distinct)}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_shadow?'good':'warn'}">${yn(s.no_position_labels_in_shadow)}</b></div><div class="metric">Route不変<b class="${s.all_routes_unchanged?'good':'warn'}">${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">TTL挙動<b class="${s.ttl_behavior_pass?'good':'warn'}">${yn(s.ttl_behavior_pass)}</b></div><div class="metric">brain.json<b class="${s.brain_file_unchanged?'good':'warn'}">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Shadow統合PASS<b class="${s.shadow_integration_pass?'good':'warn'}">${yn(s.shadow_integration_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v54: http://{HOST}:{PORT}")
    print("Stability Profile Shadow State Integration / transient / no route or learning effect")
    serve(app, host=HOST, port=PORT)
