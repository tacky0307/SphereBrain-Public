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

from brain import SphereBrain
import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v53 as v53

HOST = "127.0.0.1"
START_PORT = 5111
OUT = ROOT / "data" / "core_growth_binding_v64" / "results"
POSITIONS = ["左", "中央"]
CONFIDENCE = 0.96


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


def structural_hashes(brain: SphereBrain) -> dict[str, str]:
    return {
        "weights": array_hash(brain.weights),
        "adjacency": array_hash(brain.adjacency.astype(np.uint8)),
        "usage": array_hash(brain.usage),
        "node_usage": array_hash(brain.node_usage),
    }


def route_signature(binding: dict) -> dict:
    return {
        "entity_nodes": list(binding["entity_stage"]["activated_nodes"]),
        "entity_edges": [list(x) for x in binding["entity_stage"]["traversed_edges"]],
        "bound_nodes": list(binding["bound_stage"]["activated_nodes"]),
        "bound_edges": [list(x) for x in binding["bound_stage"]["traversed_edges"]],
    }


def find_full_audit(source: dict) -> dict:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            return row
    raise RuntimeError("v53 full audit not found")


def native_state_kwargs(motif: str, signature_values: list, evidence_conditions: int) -> dict:
    stability_class, baseline_present, echo_resistant, position_resistant, common_resistant = signature_values
    return {
        "motif": motif,
        "stability_class": stability_class,
        "baseline_present": baseline_present,
        "echo_resistant": echo_resistant,
        "position_resistant": position_resistant,
        "common_resistant": common_resistant,
        "evidence_conditions": evidence_conditions,
        "evidence_experiences": evidence_conditions,
        "confidence": CONFIDENCE,
        "surprise_ewma": 0.0,
        "drift_suspected": False,
        "adaptive_forgetting": False,
        "forgetting_decay": None,
        "metadata": {
            "source": "validated_stability_profile",
            "behavior_neutral": True,
            "native_core_state": True,
        },
    }


def native_probe(position: str, kwargs: dict) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    before_state = brain.snapshot_experience_state()
    before_hashes = structural_hashes(brain)
    before_route = route_signature(v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False))

    brain.update_experience_state(**kwargs)
    after_state = brain.snapshot_experience_state()
    after_hashes = structural_hashes(brain)
    after_route = route_signature(v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False))

    return {
        "position_for_evaluation_only": position,
        "native_field_exists": hasattr(brain, "experience_state"),
        "state_before": before_state,
        "state_after": after_state,
        "state_changed": before_state != after_state,
        "contains_position_label": brain.experience_state.contains_forbidden_position_label(),
        "route_before": before_route,
        "route_after": after_route,
        "route_unchanged": before_route == after_route,
        "structure_before": before_hashes,
        "structure_after": after_hashes,
        "structure_unchanged": before_hashes == after_hashes,
    }


def persistence_probe(kwargs: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    temp_path = OUT / "native_roundtrip_core.json"
    brain = copy.deepcopy(v3.base.CORE)
    brain.update_experience_state(**kwargs)
    expected = brain.snapshot_experience_state()
    hashes_before = structural_hashes(brain)
    brain.save(temp_path)
    loaded = SphereBrain.load(temp_path)
    restored = loaded.snapshot_experience_state()
    hashes_after = structural_hashes(loaded)
    return {
        "temp_path": str(temp_path.relative_to(ROOT)),
        "state_roundtrip_equal": expected == restored,
        "structure_roundtrip_equal": hashes_before == hashes_after,
        "saved_state": expected,
        "loaded_state": restored,
    }


def backward_compatibility_probe() -> dict:
    original = v3.base.BRAIN_PATH
    data = json.loads(original.read_text(encoding="utf-8"))
    original_contains_state = "experience_state" in data
    loaded = SphereBrain.load(original)
    state = loaded.snapshot_experience_state()
    default_state = SphereBrain(
        node_count=1,
        neighbors_per_node=0,
        seed=1,
    ).snapshot_experience_state()
    # Only state-schema equality matters here; constructing the tiny reference
    # Core is isolated and is never used for propagation.
    return {
        "original_contains_experience_state": original_contains_state,
        "old_format_load_succeeded": True,
        "loaded_has_native_field": hasattr(loaded, "experience_state"),
        "old_format_defaults_native_state": state == default_state if not original_contains_state else True,
        "loaded_state": state,
    }


def observe() -> dict:
    print("v64: reproducing validated v53 one-profile state...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v64 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    full = find_full_audit(source)
    evidence_conditions = len(source.get("conditions", []))

    kwargs = {
        "左": native_state_kwargs(motif, list(full["left_signature"][0]), evidence_conditions),
        "中央": native_state_kwargs(motif, list(full["center_signature"][0]), evidence_conditions),
    }

    print("v64: updating native Core Experience State without behavioral effect...", flush=True)
    probes = {position: native_probe(position, kwargs[position]) for position in POSITIONS}
    persistence = persistence_probe(kwargs["左"])
    backward = backward_compatibility_probe()

    states_distinct = probes["左"]["state_after"] != probes["中央"]["state_after"]
    native_fields = all(row["native_field_exists"] for row in probes.values())
    all_states_changed = all(row["state_changed"] for row in probes.values())
    no_labels = all(not row["contains_position_label"] for row in probes.values())
    routes_unchanged = all(row["route_unchanged"] for row in probes.values())
    structures_unchanged = all(row["structure_unchanged"] for row in probes.values())
    persistence_ok = persistence["state_roundtrip_equal"] and persistence["structure_roundtrip_equal"]
    backward_ok = bool(
        backward["old_format_load_succeeded"]
        and backward["loaded_has_native_field"]
        and backward["old_format_defaults_native_state"]
    )
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    native_pass = bool(
        native_fields
        and all_states_changed
        and states_distinct
        and no_labels
        and routes_unchanged
        and structures_unchanged
        and persistence_ok
        and backward_ok
        and brain_file_unchanged
    )

    if native_pass:
        verdict = "experience_state_is_native_to_core_persistent_backward_compatible_and_behavior_neutral"
        next_step = "move_incremental_evidence_update_from_experiment_runner_into_native_core_experience_state_api"
        readiness = "native_experience_state_integrated"
    else:
        verdict = "native_experience_state_contract_failed"
        next_step = "audit_native_state_persistence_or_non_interference_before_further_integration"
        readiness = "shadow_state_only"

    payload = {
        "experiment": "Core Growth Binding v64",
        "purpose": "Promote the validated experience-derived Stability Profile from an externally attached Shadow attribute to a formal SphereBrain-owned Experience State. Verify behavior neutrality, persistence, backward compatibility and absence of semantic position labels.",
        "contract": {
            "native_field_on_spherebrain": True,
            "propagation_reads_experience_state": False,
            "learning_reads_experience_state": False,
            "structural_assist_reads_experience_state": False,
            "decoder_receives_experience_state": False,
            "weights_changed_by_state": False,
            "topology_changed_by_state": False,
            "threshold_changed_by_state": False,
            "activation_changed_by_state": False,
            "save_load_persistence_supported": True,
            "old_brain_json_backward_compatible": True,
            "production_brain_json_modified_by_experiment": False,
        },
        "selected_motif": motif,
        "native_probes": probes,
        "persistence_probe": persistence,
        "backward_compatibility_probe": backward,
        "summary": {
            "native_field_exists": native_fields,
            "native_state_updates": all_states_changed,
            "left_center_native_states_distinct": states_distinct,
            "no_position_labels_in_native_state": no_labels,
            "routes_unchanged": routes_unchanged,
            "core_structures_unchanged": structures_unchanged,
            "save_load_roundtrip_pass": persistence_ok,
            "old_brain_json_backward_compatible": backward_ok,
            "brain_file_unchanged": brain_file_unchanged,
            "native_integration_pass": native_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v64.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v64</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v64</h1><p class="lead">Experience-derived Stability Profileを外付けShadow属性から、SphereBrainが正式所有するNative Experience Stateへ昇格する。伝播・学習・候補選択にはまだ一切作用させず、永続化・後方互換・非干渉を検証する。</p><section class="panel"><div class="controls"><button id="run">Native Experience Stateを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Native State生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Native field<b class="${s.native_field_exists?'good':'warn'}">${yn(s.native_field_exists)}</b></div><div class="metric">Native State更新<b class="${s.native_state_updates?'good':'warn'}">${yn(s.native_state_updates)}</b></div><div class="metric">左右State差<b class="${s.left_center_native_states_distinct?'good':'warn'}">${yn(s.left_center_native_states_distinct)}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_native_state?'good':'warn'}">${yn(s.no_position_labels_in_native_state)}</b></div><div class="metric">Route不変<b class="${s.routes_unchanged?'good':'warn'}">${yn(s.routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.core_structures_unchanged?'good':'warn'}">${yn(s.core_structures_unchanged)}</b></div><div class="metric">Save/Load復元<b class="${s.save_load_roundtrip_pass?'good':'warn'}">${yn(s.save_load_roundtrip_pass)}</b></div><div class="metric">旧brain.json互換<b class="${s.old_brain_json_backward_compatible?'good':'warn'}">${yn(s.old_brain_json_backward_compatible)}</b></div><div class="metric">brain.json<b class="${s.brain_file_unchanged?'good':'warn'}">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Native統合PASS<b class="${s.native_integration_pass?'good':'warn'}">${yn(s.native_integration_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v64: http://{HOST}:{PORT}")
    print("Core Native Experience State / persistent / backward compatible / behavior neutral")
    serve(app, host=HOST, port=PORT)
