from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import webbrowser
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v89 as v89

HOST = "127.0.0.1"
START_PORT = 5141
OUT = ROOT / "data" / "core_growth_binding_v90" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v89.SEEDS
DOMAINS = v89.DOMAINS


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


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def stage_flags(row: dict) -> dict:
    candidate = row["context_linked_candidate_count"] > 0
    shared_edge = row["experimental"]["direct"]["new_shared_edges"] > 0
    control_beaten = row["effects"]["direct_new_shared_edges"] > 0
    similarity_beaten = (
        row["effects"]["direct_edge_similarity"] > 0
        or row["effects"]["direct_activation_similarity"] > 0
    )
    bridge = bool(candidate and shared_edge and control_beaten and similarity_beaten)
    return {
        "candidate": candidate,
        "shared_edge": shared_edge,
        "control_beaten": control_beaten,
        "similarity_beaten": similarity_beaten,
        "bridge": bridge,
    }


def classify_failure(records: list[dict]) -> str:
    ever = {k: any(stage_flags(r)[k] for r in records) for k in [
        "candidate", "shared_edge", "control_beaten", "similarity_beaten", "bridge"
    ]}
    active = [int(r["cycle"]) for r in records if stage_flags(r)["bridge"]]
    stable = v86.classify(records) == "stable"
    if stable:
        return "stable"
    if active:
        return "ever_bridge_unstable"
    if not ever["candidate"]:
        return "never_candidate"
    if not ever["shared_edge"]:
        return "candidate_only"
    if not ever["control_beaten"]:
        return "shared_edge_only"
    if not ever["similarity_beaten"]:
        return "control_not_beaten"
    return "similarity_not_beaten"


def run_trial(domain, seed: int) -> dict:
    experimental = v89.v84.clean_primary_seed(seed)
    control = v89.v84.clean_primary_seed(seed)
    experimental.set_structural_assist(False)
    control.set_structural_assist(False)

    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)

    records = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            row["stage_flags"] = stage_flags(row)
            records.append(row)
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    cls = classify_failure(records)
    bridge_cycles = [int(r["cycle"]) for r in records if r["stage_flags"]["bridge"]]
    candidate_cycles = [int(r["cycle"]) for r in records if r["stage_flags"]["candidate"]]
    shared_cycles = [int(r["cycle"]) for r in records if r["stage_flags"]["shared_edge"]]
    control_cycles = [int(r["cycle"]) for r in records if r["stage_flags"]["control_beaten"]]
    similarity_cycles = [int(r["cycle"]) for r in records if r["stage_flags"]["similarity_beaten"]]

    return {
        "domain": domain.name,
        "seed": int(seed),
        "classification": cls,
        "stable": cls == "stable",
        "ever_bridge": bool(bridge_cycles),
        "first_candidate_cycle": candidate_cycles[0] if candidate_cycles else None,
        "first_shared_edge_cycle": shared_cycles[0] if shared_cycles else None,
        "first_control_beaten_cycle": control_cycles[0] if control_cycles else None,
        "first_similarity_beaten_cycle": similarity_cycles[0] if similarity_cycles else None,
        "first_bridge_cycle": bridge_cycles[0] if bridge_cycles else None,
        "bridge_cycles": bridge_cycles,
        "records": records,
        "experimental": experimental,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    counts = Counter(x["classification"] for x in trials)

    stage_reach = {
        "candidate": sum(any(r["stage_flags"]["candidate"] for r in x["records"]) for x in trials),
        "shared_edge": sum(any(r["stage_flags"]["shared_edge"] for r in x["records"]) for x in trials),
        "control_beaten": sum(any(r["stage_flags"]["control_beaten"] for r in x["records"]) for x in trials),
        "similarity_beaten": sum(any(r["stage_flags"]["similarity_beaten"] for r in x["records"]) for x in trials),
        "ever_bridge": sum(x["ever_bridge"] for x in trials),
        "stable": sum(x["stable"] for x in trials),
    }

    domain_rows = []
    for domain in DOMAINS:
        subset = [x for x in trials if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "trials": len(subset),
            "ever_bridge": sum(x["ever_bridge"] for x in subset),
            "stable": sum(x["stable"] for x in subset),
            "classification_counts": dict(Counter(x["classification"] for x in subset)),
        })

    representative = next((x for x in trials if x["stable"]), trials[0])
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "formation_failure_microscope_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    ever_bridge = stage_reach["ever_bridge"]
    stable = stage_reach["stable"]
    unstable = counts.get("ever_bridge_unstable", 0)
    formation_limited = ever_bridge <= 20
    stabilization_limited = ever_bridge >= 40 and stable <= max(1, int(0.25 * ever_bridge))

    if formation_limited:
        readiness = "semantic_bridge_formation_probability_is_primary_bottleneck"
        verdict = "most_primary_trials_fail_before_a_full_bridge_is_ever_formed"
        next_step = "attribute_which_pre_bridge_gate_dominates_and_test_formation_assist_without_consolidation"
    elif stabilization_limited:
        readiness = "semantic_bridge_stabilization_is_primary_bottleneck"
        verdict = "bridges_form_in_many_trials_but_rarely_remain_stable_across_late_checkpoints"
        next_step = "focus_on_post_formation_stability_without_changing_bridge_formation"
    else:
        readiness = "semantic_bridge_failure_is_mixed_formation_and_stability"
        verdict = "primary_failures_are_split_between_pre_bridge_formation_gates_and_post_formation_instability"
        next_step = "target_the_largest_failure_class_before_any_primary_core_change"

    payload = {
        "experiment": "Core Growth Binding v90 — Semantic Bridge Formation Failure Microscope",
        "contract": {
            "primary_core_modified": False,
            "consolidation_used": False,
            "assist_used": False,
            "trials": len(trials),
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "checkpoints": CHECKPOINTS,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(trials),
            "stable": stable,
            "ever_bridge": ever_bridge,
            "ever_bridge_unstable": unstable,
            "never_candidate": counts.get("never_candidate", 0),
            "candidate_only": counts.get("candidate_only", 0),
            "shared_edge_only": counts.get("shared_edge_only", 0),
            "control_not_beaten": counts.get("control_not_beaten", 0),
            "similarity_not_beaten": counts.get("similarity_not_beaten", 0),
            "stage_reach": stage_reach,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "classification_counts": dict(counts),
        "domain_rows": domain_rows,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in trials],
    }
    (OUT / "latest_binding_v90.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v90</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v90：Semantic Bridge Formation Failure Microscope</h1><p class="lead">Primary Coreだけで100試行を再生し、Bridge形成のどの関門で失敗しているかを分解する。ConsolidationもAssistも使わない。</p><section class="panel"><div class="controls"><button id="run">形成失敗を分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Domain別</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Stable',s.stable),metric('Ever Bridge',s.ever_bridge),metric('Ever unstable',s.ever_bridge_unstable),metric('Never candidate',s.never_candidate),metric('Candidate only',s.candidate_only),metric('Shared edge only',s.shared_edge_only),metric('Control not beaten',s.control_not_beaten),metric('Similarity not beaten',s.similarity_not_beaten),metric('Save/Load',s.saveload_equal?'YES':'NO',s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変更',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('groups').textContent=JSON.stringify({stage_reach:s.stage_reach,domain_rows:d.domain_rows,classification_counts:d.classification_counts},null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post('/api/run')
def api_run():
    return jsonify(observe())


@app.get('/')
def index():
    return PAGE


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
