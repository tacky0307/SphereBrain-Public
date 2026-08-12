from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import tempfile
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
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v56 as v56
import run_core_growth_binding_v57 as v57
import run_core_growth_binding_v58 as v58
import run_core_growth_binding_v60 as v60

HOST = "127.0.0.1"
START_PORT = 5112
OUT = ROOT / "data" / "core_growth_binding_v65" / "results"
EPS = 1e-10


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


def route_signature(brain, position: str) -> dict:
    report = v3.make_binding(copy.deepcopy(brain), "E", position, learn=False, assist=False)
    return {
        "entity_nodes": list(report["entity_stage"]["activated_nodes"]),
        "entity_edges": list(report["entity_stage"]["traversed_edges"]),
        "bound_nodes": list(report["bound_stage"]["activated_nodes"]),
        "bound_edges": list(report["bound_stage"]["traversed_edges"]),
    }


def native_signature(state: dict) -> list:
    return list(v53.signature(state))


def close(a, b, eps: float = EPS) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= eps


def compare_step(native_diag: dict, native_state: dict, ref_diag: dict, ref_profile: dict) -> dict:
    checks = {
        "signature": native_signature(native_state) == list(v56.signature(ref_profile)),
        "confidence": close(native_state.get("confidence"), ref_profile.get("confidence")),
        "surprise": int(native_diag.get("surprise", 0)) == int(ref_diag.get("surprise", 0)),
        "surprise_ewma": close(native_diag.get("surprise_ewma"), ref_diag.get("surprise_ewma")),
        "drift": bool(native_diag.get("drift_suspected")) == bool(ref_diag.get("drift_suspected")),
        "adaptive_forgetting": bool(native_diag.get("adaptive_forgetting")) == bool(ref_diag.get("fast_forgetting")),
        "active_decay": close(native_diag.get("active_decay"), ref_diag.get("active_decay")),
        "gate_open": bool(native_diag.get("multi_factor_gate_open")) == bool(ref_diag.get("multi_factor_gate_open")),
    }
    return {"checks": checks, "all_match": all(checks.values())}


def run_stream(spec: dict, motif: str) -> dict:
    names = v57.condition_names()
    brain = copy.deepcopy(v3.base.CORE)
    brain.clear_experience_state()
    brain.experience_state.configure(motif=motif, expected_conditions=names)

    hashes_before = structural_hashes(brain)
    route_before = route_signature(brain, "左")
    reference = v60.MultiFactorAdaptiveEvidence(names)
    timeline = []

    for index, item in enumerate(spec["stream"], start=1):
        present = v57.motif_presence(item["environment"], item["condition"], motif)
        ref_diag = reference.update_natural(item["condition"], present)
        ref_profile = v56.weighted_profile(reference, motif)
        native_diag = brain.experience_state.observe(
            condition=item["condition"],
            present=present,
            motif=motif,
            expected_conditions=names,
        )
        state = brain.snapshot_experience_state()
        parity = compare_step(native_diag, state, ref_diag, ref_profile)
        timeline.append({
            "index": index,
            "condition": item["condition"],
            "evaluation_segment": item.get("evaluation_segment"),
            "motif_present": present,
            "parity": parity,
            "native_signature": native_signature(state),
            "reference_signature": list(v56.signature(ref_profile)),
            "native_confidence": state.get("confidence"),
            "reference_confidence": ref_profile.get("confidence"),
        })

    hashes_after = structural_hashes(brain)
    route_after = route_signature(brain, "左")
    state = brain.snapshot_experience_state()
    return {
        "scenario": spec["name"],
        "experience_count": len(spec["stream"]),
        "all_steps_match_reference": all(row["parity"]["all_match"] for row in timeline),
        "mismatch_steps": [row["index"] for row in timeline if not row["parity"]["all_match"]],
        "native_state_contains_position_label": brain.experience_state.contains_forbidden_position_label(),
        "route_unchanged": route_before == route_after,
        "structure_unchanged": hashes_before == hashes_after,
        "final_state": state,
        "timeline": timeline,
    }


def persistence_round_trip(spec: dict, motif: str) -> dict:
    names = v57.condition_names()
    stream = spec["stream"]
    split = max(1, len(stream) // 2)

    uninterrupted = copy.deepcopy(v3.base.CORE)
    uninterrupted.clear_experience_state()
    uninterrupted.experience_state.configure(motif=motif, expected_conditions=names)

    resumed = copy.deepcopy(v3.base.CORE)
    resumed.clear_experience_state()
    resumed.experience_state.configure(motif=motif, expected_conditions=names)

    def ingest(brain, item):
        present = v57.motif_presence(item["environment"], item["condition"], motif)
        brain.experience_state.observe(
            condition=item["condition"],
            present=present,
            motif=motif,
            expected_conditions=names,
        )

    for item in stream:
        ingest(uninterrupted, item)

    for item in stream[:split]:
        ingest(resumed, item)

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="v65_native_", suffix=".json", dir=OUT, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        resumed.save(temp_path)
        saved_state = resumed.snapshot_experience_state()
        loaded = SphereBrain.load(temp_path)
        loaded_state = loaded.snapshot_experience_state()
        for item in stream[split:]:
            ingest(loaded, item)
        final_loaded = loaded.snapshot_experience_state()
        final_uninterrupted = uninterrupted.snapshot_experience_state()
        return {
            "split_index": split,
            "state_restored_exactly": saved_state == loaded_state,
            "continued_final_state_matches_uninterrupted": final_loaded == final_uninterrupted,
            "saved_state": saved_state,
            "loaded_state": loaded_state,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def observe() -> dict:
    print("v65: reproducing v53 minimal motif...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v65 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]

    print("v65: running v58 streams through native CoreExperienceState.observe()...", flush=True)
    specs = v58.build_scenarios(motif)
    reports = []
    for spec in specs:
        print(f"v65: {spec['name']} / {len(spec['stream'])} experiences", flush=True)
        reports.append(run_stream(spec, motif))

    persistence = persistence_round_trip(specs[0], motif)
    old_brain_compatible = True
    try:
        loaded_old = SphereBrain.load(v3.base.BRAIN_PATH)
        old_brain_compatible = hasattr(loaded_old, "experience_state")
    except Exception:
        old_brain_compatible = False

    native_api_exists = hasattr(v3.base.CORE.experience_state, "observe")
    all_parity = all(r["all_steps_match_reference"] for r in reports)
    all_routes = all(r["route_unchanged"] for r in reports)
    all_structures = all(r["structure_unchanged"] for r in reports)
    no_labels = all(not r["native_state_contains_position_label"] for r in reports)
    persistence_ok = bool(
        persistence["state_restored_exactly"]
        and persistence["continued_final_state_matches_uninterrupted"]
    )
    brain_file_unchanged = v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH)

    native_update_pass = bool(
        native_api_exists
        and all_parity
        and all_routes
        and all_structures
        and no_labels
        and persistence_ok
        and old_brain_compatible
        and brain_file_unchanged
    )

    if native_update_pass:
        verdict = "native_core_experience_state_updates_online_and_matches_validated_reference_without_behavioral_effect"
        next_step = "run_native_core_in_pe_3x3_puzzle_and_compare_behavior_before_after_experience"
        readiness = "native_experience_update_integrated"
    else:
        verdict = "native_experience_update_does_not_yet_match_validated_shadow_reference"
        next_step = "audit_native_reference_parity_or_persistence_before_puzzle_trial"
        readiness = "native_state_only"

    payload = {
        "experiment": "Core Growth Binding v65",
        "purpose": "Move sequential evidence accumulation, confidence, forgetting, surprise and the v60 Multi-Factor Drift Gate into the Core-owned CoreExperienceState API, then verify stepwise parity with the validated experiment-side reference while remaining behavior-neutral.",
        "contract": {
            "native_update_entrypoint": "brain.experience_state.observe(condition, present, motif, expected_conditions)",
            "experiment_runner_computes_production_state": False,
            "reference_updater_used_for_audit_only": True,
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "activation_changed_by_experience_state": False,
            "structural_assist_reads_experience_state": False,
            "decoder_reads_experience_state": False,
            "position_label_stored_in_state": False,
            "state_persisted_by_core_save_load": True,
            "old_brain_json_backward_compatible": True,
        },
        "selected_motif": motif,
        "scenario_reports": reports,
        "persistence_round_trip": persistence,
        "summary": {
            "native_observe_api_exists": native_api_exists,
            "all_steps_match_v60_reference": all_parity,
            "all_routes_unchanged": all_routes,
            "all_core_structures_unchanged": all_structures,
            "no_position_labels": no_labels,
            "save_load_continue_exact": persistence_ok,
            "old_brain_json_compatible": old_brain_compatible,
            "brain_file_unchanged": brain_file_unchanged,
            "native_update_pass": native_update_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v65.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v65</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v65</h1><p class="lead">v54B〜v60で実験側に置いていた逐次Evidence・confidence・forgetting・surprise・Multi-Factor Drift Gateを、Core所有のexperience_state.observe()へ移す。v60 referenceとの全Step一致、非干渉、save/load継続を検証する。</p><section class="panel"><div class="controls"><button id="run">Native Experience Updateを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Native Update生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Native observe API<b class="${s.native_observe_api_exists?'good':'warn'}">${yn(s.native_observe_api_exists)}</b></div><div class="metric">v60全Step一致<b class="${s.all_steps_match_v60_reference?'good':'warn'}">${yn(s.all_steps_match_v60_reference)}</b></div><div class="metric">Route不変<b class="${s.all_routes_unchanged?'good':'warn'}">${yn(s.all_routes_unchanged)}</b></div><div class="metric">Core構造不変<b class="${s.all_core_structures_unchanged?'good':'warn'}">${yn(s.all_core_structures_unchanged)}</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels?'good':'warn'}">${yn(s.no_position_labels)}</b></div><div class="metric">Save/Load継続<b class="${s.save_load_continue_exact?'good':'warn'}">${yn(s.save_load_continue_exact)}</b></div><div class="metric">旧brain.json互換<b class="${s.old_brain_json_compatible?'good':'warn'}">${yn(s.old_brain_json_compatible)}</b></div><div class="metric">brain.json<b class="${s.brain_file_unchanged?'good':'warn'}">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Native更新PASS<b class="${s.native_update_pass?'good':'warn'}">${yn(s.native_update_pass)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v65: http://{HOST}:{PORT}")
    print("Core Native Experience Update / v60 parity / save-load continuity / behavior-neutral")
    serve(app, host=HOST, port=PORT)
