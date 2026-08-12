from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import webbrowser
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
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v84b as v84b

HOST = "127.0.0.1"
START_PORT = 5135
OUT = ROOT / "data" / "core_growth_binding_v85" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = [0, 1, 3, 5, 10, 20, 30, 50]
SEEDS = v84.SEEDS
DOMAINS = v84.DOMAINS
STABILITY_WINDOW = [20, 30, 50]
READ_ONLY_PROBES = [10, 50]


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


def cycle_success(row: dict) -> bool:
    return v84b.cycle_success(row)


def make_cycle_row(experimental: SphereBrain, control: SphereBrain, domain, exp_baseline, ctrl_baseline, cycle: int) -> dict:
    exp = v84.measure(experimental, domain, exp_baseline)
    ctrl = v84.measure(control, domain, ctrl_baseline)
    candidates = v84.context_candidates(experimental, control, domain, exp_baseline["direct"])
    row = {
        "cycle": int(cycle),
        "experimental": exp,
        "control": ctrl,
        "effects": {
            "direct_edge_similarity": exp["direct"]["edge_similarity"] - ctrl["direct"]["edge_similarity"],
            "direct_activation_similarity": exp["direct"]["activation_similarity"] - ctrl["direct"]["activation_similarity"],
            "direct_new_shared_edges": exp["direct"]["new_shared_edges"] - ctrl["direct"]["new_shared_edges"],
            "transfer_edge_similarity": exp["transfer"]["mean_edge_similarity"] - ctrl["transfer"]["mean_edge_similarity"],
            "transfer_activation_similarity": exp["transfer"]["mean_activation_similarity"] - ctrl["transfer"]["mean_activation_similarity"],
            "transfer_new_shared_edges": exp["transfer"]["new_shared_edges"] - ctrl["transfer"]["new_shared_edges"],
        },
        "context_linked_candidate_count": len(candidates),
        "context_linked_candidates": candidates,
    }
    row["direct_success_now"] = cycle_success(row)
    return row


def classify(records: list[dict]) -> str:
    active = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    candidates = [int(r["cycle"]) for r in records if r["context_linked_candidate_count"] > 0]
    if not active:
        return "candidate_only" if candidates else "never"
    active_set = set(active)
    if all(c in active_set for c in STABILITY_WINDOW):
        return "stable"
    if active[0] <= 1 and active[-1] < 20:
        return "transient"
    if active[0] >= 3:
        return "late_unstable"
    return "intermittent"


def read_only_probe_survival(brain: SphereBrain, control: SphereBrain, domain, exp_baseline, ctrl_baseline) -> dict:
    initial = make_cycle_row(brain, control, domain, exp_baseline, ctrl_baseline, 50)
    initial_edges = initial["experimental"]["direct"]["new_shared_edge_list"]
    initial_candidates = initial["context_linked_candidate_count"]
    results = []
    for probe_count in READ_ONLY_PROBES:
        last = None
        for _ in range(probe_count):
            # All calls inside measure/context_candidates are learn=False and noise=0.
            last = make_cycle_row(brain, control, domain, exp_baseline, ctrl_baseline, 50)
        assert last is not None
        results.append({
            "probe_count": probe_count,
            "success": bool(last["direct_success_now"]),
            "new_shared_edges_same": last["experimental"]["direct"]["new_shared_edge_list"] == initial_edges,
            "candidate_count_same": last["context_linked_candidate_count"] == initial_candidates,
            "edge_effect": last["effects"]["direct_edge_similarity"],
            "activation_effect": last["effects"]["direct_activation_similarity"],
        })
    return {
        "initial_success": bool(initial["direct_success_now"]),
        "initial_new_shared_edges": initial_edges,
        "initial_candidate_count": initial_candidates,
        "probes": results,
        "read_only_preserved": all(x["new_shared_edges_same"] and x["candidate_count_same"] for x in results),
    }


def run_trial(domain, seed: int) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v84.make_baseline(experimental, domain)
    ctrl_baseline = v84.make_baseline(control, domain)
    exp_episodes = v84.episodes_for(domain, True)
    ctrl_episodes = v84.episodes_for(domain, False)

    records = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            records.append(make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle))
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    cls = classify(records)
    success_cycles = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    candidate_cycles = [int(r["cycle"]) for r in records if r["context_linked_candidate_count"] > 0]
    final = records[-1]
    probe = read_only_probe_survival(experimental, control, domain, exp_baseline, ctrl_baseline)

    return {
        "domain": domain.name,
        "seed": int(seed),
        "classification": cls,
        "first_success_cycle": success_cycles[0] if success_cycles else None,
        "last_success_cycle": success_cycles[-1] if success_cycles else None,
        "success_cycles": success_cycles,
        "candidate_cycles": candidate_cycles,
        "stable_at_20_30_50": cls == "stable",
        "final_success": bool(final["direct_success_now"]),
        "final_candidate_count": int(final["context_linked_candidate_count"]),
        "final_new_shared_edge_effect": int(final["effects"]["direct_new_shared_edges"]),
        "final_edge_effect": float(final["effects"]["direct_edge_similarity"]),
        "final_activation_effect": float(final["effects"]["direct_activation_similarity"]),
        "read_only_probe": probe,
        "records": records,
        "experimental": experimental,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]

    class_counts: dict[str, int] = {}
    for row in trials:
        class_counts[row["classification"]] = class_counts.get(row["classification"], 0) + 1

    ever_success = sum(1 for x in trials if x["success_cycles"])
    stable = class_counts.get("stable", 0)
    transient = class_counts.get("transient", 0)
    late_unstable = class_counts.get("late_unstable", 0)
    candidate_only = class_counts.get("candidate_only", 0)
    never = class_counts.get("never", 0)
    read_only_preserved = sum(1 for x in trials if x["read_only_probe"]["read_only_preserved"])

    domain_rows = []
    for domain in DOMAINS:
        rows = [x for x in trials if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "ever_success": sum(1 for x in rows if x["success_cycles"]),
            "stable": sum(1 for x in rows if x["classification"] == "stable"),
            "late_unstable": sum(1 for x in rows if x["classification"] == "late_unstable"),
            "candidate_only": sum(1 for x in rows if x["classification"] == "candidate_only"),
            "never": sum(1 for x in rows if x["classification"] == "never"),
        })

    seed_rows = []
    for seed in SEEDS:
        rows = [x for x in trials if x["seed"] == seed]
        seed_rows.append({
            "seed": seed,
            "ever_success_domains": sum(1 for x in rows if x["success_cycles"]),
            "stable_domains": sum(1 for x in rows if x["classification"] == "stable"),
        })

    representative = next((x for x in trials if x["classification"] == "stable"), next((x for x in trials if x["success_cycles"]), trials[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_episode_stabilization_roundtrip.json"
    brain = representative["experimental"]
    before_state = brain.snapshot_learning_state()
    before_weights = brain.weights.tolist()
    brain.save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_state == loaded.snapshot_learning_state() and before_weights == loaded.weights.tolist()
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)

    natural_stabilization = stable >= 3
    stabilization_pass = (
        ever_success > 0
        and read_only_preserved == len(trials)
        and saveload_equal
        and native_present
        and production_unchanged
    )

    if natural_stabilization:
        verdict = "longer_semantic_episode_exposure_converts_some_bridge_trials_into_naturally_stable_structures"
        readiness = "semantic_episode_natural_stabilization_observed"
        next_step = "compare_stable_vs_nonstable_trials_before_adding_semantic_specific_consolidation"
    elif ever_success > 0:
        verdict = "semantic_episode_bridges_still_form_but_do_not_reliably_stabilize_even_by_50_episodes"
        readiness = "semantic_bridge_stabilization_missing"
        next_step = "design_semantic_episode_consolidation_candidate_and_validate_against_current_primary_core"
    else:
        verdict = "semantic_episode_bridge_not_observed_in_long_horizon_stabilization_run"
        readiness = "semantic_episode_bridge_not_reproduced"
        next_step = "recheck_v84_trial_equivalence_before_any_core_change"

    payload = {
        "experiment": "Core Growth Binding v85 — Semantic Episode Bridge Stabilization",
        "contract": {
            "primary_core_modified": False,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "trial_count": len(trials),
            "training_checkpoints": CHECKPOINTS,
            "stable_requires_success_at": STABILITY_WINDOW,
            "read_only_probe_counts": READ_ONLY_PROBES,
            "read_only_probe_learning": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(trials),
            "ever_success_trials": ever_success,
            "stable_trials": stable,
            "transient_trials": transient,
            "late_unstable_trials": late_unstable,
            "candidate_only_trials": candidate_only,
            "never_trials": never,
            "read_only_preserved_trials": read_only_preserved,
            "natural_stabilization_observed": natural_stabilization,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "stabilization_observation_pass": stabilization_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "domain_rows": domain_rows,
        "seed_rows": seed_rows,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in trials],
    }
    (OUT / "latest_binding_v85.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v85</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v85：Semantic Episode Bridge Stabilization</h1><p class="lead">3 domain × 5 seed の15試行を50Episodeまで延長し、Bridgeが一時的に出るだけか、20/30/50で自然に安定するかを観察する。学習停止後はread-only Probe ×10 / ×50で構造保存も確認する。</p><section class="panel"><div class="controls"><button id="run">Bridge定着を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Domain / Seed</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('一度でもBridge',s.ever_success_trials),metric('Stable',s.stable_trials,s.stable_trials>0?'good':'warn'),metric('Transient',s.transient_trials),metric('Late unstable',s.late_unstable_trials),metric('Candidate only',s.candidate_only_trials),metric('Never',s.never_trials),metric('Read-only保持',`${s.read_only_preserved_trials}/${s.trial_count}`,s.read_only_preserved_trials===s.trial_count?'good':'warn'),metric('自然安定化',yn(s.natural_stabilization_observed),s.natural_stabilization_observed?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.stabilization_observation_pass),s.stabilization_observation_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('groups').textContent=JSON.stringify({domain_rows:d.domain_rows,seed_rows:d.seed_rows},null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    try:
        return jsonify(observe())
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v85: http://{HOST}:{PORT}")
    print("Semantic Episode long horizon / stabilization / read-only retention / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
