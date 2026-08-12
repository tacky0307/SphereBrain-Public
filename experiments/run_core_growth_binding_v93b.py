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
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v89 as v89
import run_core_growth_binding_v93 as v93

HOST = "127.0.0.1"
START_PORT = 5147
OUT = ROOT / "data" / "core_growth_binding_v93b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v89.SEEDS
DOMAINS = v89.DOMAINS
CONSENSUS_DROP_RATIO = 0.12
SUPPORT_DROP_RATIO = 0.12


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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def run_trial(domain, seed: int) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)
    history: dict[tuple[int, int], int] = {}

    records = []
    consensus = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            records.append(row)
            consensus.append(v93.consensus_snapshot(
                experimental, domain, experimental, control, exp_baseline, cycle, history
            ))
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    active = [i for i, row in enumerate(records) if row["direct_success_now"]]
    if not active:
        return {
            "domain": domain.name,
            "seed": int(seed),
            "ever_bridge": False,
            "stable": False,
            "experimental": experimental,
        }

    cls = v86.classify(records)
    first_i = active[0]
    post = consensus[first_i:]
    first = post[0]
    first_score = float(first["mean_consensus_strength"])
    first_support = float(first["mean_support_sources"])
    first_quorum = int(first["quorum_edge_count"])

    scores = [float(x["mean_consensus_strength"]) for x in post]
    supports = [float(x["mean_support_sources"]) for x in post]
    quorums = [int(x["quorum_edge_count"]) for x in post]
    peak_score = max(scores) if scores else 0.0
    final_score = scores[-1] if scores else 0.0
    min_score = min(scores) if scores else 0.0
    final_support = supports[-1] if supports else 0.0
    final_quorum = quorums[-1] if quorums else 0

    retention = final_score / first_score if first_score > 1e-12 else 0.0
    support_retention = final_support / first_support if first_support > 1e-12 else 0.0
    quorum_retention = final_quorum / first_quorum if first_quorum > 0 else (1.0 if final_quorum == 0 else 0.0)

    first_collapse_i = None
    for i in range(first_i + 1, len(records)):
        if not records[i]["direct_success_now"]:
            first_collapse_i = i
            break

    warning_before_collapse = False
    warning_cycle = None
    collapse_cycle = None
    pre_collapse = None
    if first_collapse_i is not None:
        collapse_cycle = int(records[first_collapse_i]["cycle"])
        for i in range(first_i + 1, first_collapse_i):
            cur_score = float(consensus[i]["mean_consensus_strength"])
            cur_support = float(consensus[i]["mean_support_sources"])
            score_drop = (first_score - cur_score) / first_score if first_score > 1e-12 else 0.0
            support_drop = (first_support - cur_support) / first_support if first_support > 1e-12 else 0.0
            if score_drop >= CONSENSUS_DROP_RATIO or support_drop >= SUPPORT_DROP_RATIO:
                warning_before_collapse = True
                warning_cycle = int(records[i]["cycle"])
                break
        if first_collapse_i > first_i:
            p = consensus[first_collapse_i - 1]
            c = consensus[first_collapse_i]
            pre_collapse = {
                "cycle": int(records[first_collapse_i - 1]["cycle"]),
                "consensus": float(p["mean_consensus_strength"]),
                "support_sources": float(p["mean_support_sources"]),
                "quorum_edges": int(p["quorum_edge_count"]),
                "collapse_cycle": collapse_cycle,
                "collapse_consensus": float(c["mean_consensus_strength"]),
                "collapse_support_sources": float(c["mean_support_sources"]),
                "collapse_quorum_edges": int(c["quorum_edge_count"]),
            }

    return {
        "domain": domain.name,
        "seed": int(seed),
        "ever_bridge": True,
        "stable": cls == "stable",
        "classification": cls,
        "first_bridge_cycle": int(records[first_i]["cycle"]),
        "first_consensus": first_score,
        "peak_consensus": peak_score,
        "minimum_consensus_after_bridge": min_score,
        "final_consensus": final_score,
        "consensus_retention_ratio": retention,
        "first_support_sources": first_support,
        "final_support_sources": final_support,
        "support_retention_ratio": support_retention,
        "first_quorum_edges": first_quorum,
        "final_quorum_edges": final_quorum,
        "quorum_retention_ratio": quorum_retention,
        "first_collapse_cycle": collapse_cycle,
        "warning_before_collapse": warning_before_collapse,
        "warning_cycle": warning_cycle,
        "pre_collapse": pre_collapse,
        "consensus_by_cycle": consensus,
        "experimental": experimental,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    all_trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    ever = [x for x in all_trials if x["ever_bridge"]]
    stable = [x for x in ever if x["stable"]]
    unstable = [x for x in ever if not x["stable"]]
    unstable_observed_collapse = [x for x in unstable if x["first_collapse_cycle"] is not None]

    stable_retention = mean([float(x["consensus_retention_ratio"]) for x in stable])
    unstable_retention = mean([float(x["consensus_retention_ratio"]) for x in unstable])
    retention_delta = (stable_retention - unstable_retention) if stable_retention is not None and unstable_retention is not None else 0.0

    stable_support_retention = mean([float(x["support_retention_ratio"]) for x in stable])
    unstable_support_retention = mean([float(x["support_retention_ratio"]) for x in unstable])
    support_retention_delta = (
        stable_support_retention - unstable_support_retention
        if stable_support_retention is not None and unstable_support_retention is not None else 0.0
    )

    stable_quorum_retention = mean([float(x["quorum_retention_ratio"]) for x in stable])
    unstable_quorum_retention = mean([float(x["quorum_retention_ratio"]) for x in unstable])
    quorum_retention_delta = (
        stable_quorum_retention - unstable_quorum_retention
        if stable_quorum_retention is not None and unstable_quorum_retention is not None else 0.0
    )

    warned = [x for x in unstable_observed_collapse if x["warning_before_collapse"]]
    warning_rate = len(warned) / len(unstable_observed_collapse) if unstable_observed_collapse else 0.0

    domain_rows = []
    for domain in DOMAINS:
        srows = [x for x in stable if x["domain"] == domain.name]
        urows = [x for x in unstable if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "stable": len(srows),
            "unstable": len(urows),
            "stable_consensus_retention": mean([float(x["consensus_retention_ratio"]) for x in srows]),
            "unstable_consensus_retention": mean([float(x["consensus_retention_ratio"]) for x in urows]),
            "warning_before_collapse_rate": (
                sum(1 for x in urows if x.get("first_collapse_cycle") is not None and x["warning_before_collapse"])
                / sum(1 for x in urows if x.get("first_collapse_cycle") is not None)
                if sum(1 for x in urows if x.get("first_collapse_cycle") is not None) else None
            ),
        })

    representative = ever[0] if ever else all_trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "consensus_persistence_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    # Signal requires a meaningful persistence separation OR useful pre-collapse warning rate.
    persistence_signal = retention_delta >= 0.08 or support_retention_delta >= 0.08 or quorum_retention_delta >= 0.15
    precursor_signal = warning_rate >= 0.40
    consensus_persistence_signal = persistence_signal or precursor_signal

    measurement_pass = (
        len(stable) > 0
        and len(unstable) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if consensus_persistence_signal:
        readiness = "consensus_persistence_or_precursor_signal_observed"
        verdict = "consensus_dynamics_separate_or_warn_about_unstable_bridges_better_than_first_bridge_consensus_level"
        next_step = "validate_consensus_persistence_signal_with_higher_checkpoint_resolution_before_using_it_for_core_intervention"
    else:
        readiness = "consensus_persistence_signal_not_observed"
        verdict = "consensus_support_remains_too_similar_between_stable_and_unstable_bridges_even_over_time"
        next_step = "deprioritize_structural_consensus_as_a_stability_signal_and_return_to_post_bridge_relative_dynamics"

    payload = {
        "experiment": "Core Growth Binding v93B — Consensus Persistence Dynamics",
        "contract": {
            "primary_core_modified": False,
            "consolidation_used": False,
            "assist_used": False,
            "trial_count": len(all_trials),
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "consensus_formula_changed_from_v93": False,
            "quorum_definition_changed_from_v93": False,
            "future_stability_used_in_consensus_score": False,
            "warning_drop_ratio": CONSENSUS_DROP_RATIO,
            "support_warning_drop_ratio": SUPPORT_DROP_RATIO,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(all_trials),
            "ever_bridge_trials": len(ever),
            "stable_trials": len(stable),
            "unstable_trials": len(unstable),
            "unstable_with_observed_collapse": len(unstable_observed_collapse),
            "stable_consensus_retention": stable_retention,
            "unstable_consensus_retention": unstable_retention,
            "consensus_retention_delta": retention_delta,
            "stable_support_retention": stable_support_retention,
            "unstable_support_retention": unstable_support_retention,
            "support_retention_delta": support_retention_delta,
            "stable_quorum_retention": stable_quorum_retention,
            "unstable_quorum_retention": unstable_quorum_retention,
            "quorum_retention_delta": quorum_retention_delta,
            "precollapse_warning_trials": len(warned),
            "precollapse_warning_rate": warning_rate,
            "persistence_signal": persistence_signal,
            "precursor_signal": precursor_signal,
            "consensus_persistence_signal": consensus_persistence_signal,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "domain_rows": domain_rows,
        "ever_bridge_trials": [{k: v for k, v in x.items() if k != "experimental"} for x in ever],
    }
    (OUT / "latest_binding_v93b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v93B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v93B：Consensus Persistence Dynamics</h1><p class="lead">v93と同じConsensus定義を使い、初回Bridge後のConsensus維持率・support source維持率・Quorum維持率と、Bridge崩壊前の予兆を観察する。Core・Consolidation・Assistは変更しない。</p><section class="panel"><div class="controls"><button id="run">Consensus持続性を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Domain別</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function pct(v){return v==null?'—':(100*v).toFixed(1)+'%'}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Ever Bridge',s.ever_bridge_trials),metric('Stable',s.stable_trials),metric('Unstable',s.unstable_trials,'warn'),metric('Stable Consensus保持',pct(s.stable_consensus_retention)),metric('Unstable Consensus保持',pct(s.unstable_consensus_retention)),metric('保持率差',pct(s.consensus_retention_delta),s.consensus_retention_delta>=.08?'good':'warn'),metric('Support保持率差',pct(s.support_retention_delta),s.support_retention_delta>=.08?'good':'warn'),metric('Quorum保持率差',pct(s.quorum_retention_delta),s.quorum_retention_delta>=.15?'good':'warn'),metric('崩壊前Warning',`${s.precollapse_warning_trials}/${s.unstable_with_observed_collapse}`),metric('Warning率',pct(s.precollapse_warning_rate),s.precollapse_warning_rate>=.4?'good':'warn'),metric('Persistence signal',yn(s.persistence_signal),s.persistence_signal?'good':'warn'),metric('Precursor signal',yn(s.precursor_signal),s.precursor_signal?'good':'warn'),metric('Consensus signal',yn(s.consensus_persistence_signal),s.consensus_persistence_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('groups').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.ever_bridge_trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post('/api/run')
def api_run():
    return jsonify(observe())


@app.get('/')
def index():
    return PAGE


def main() -> None:
    print(f"Core Growth Binding v93B: http://{HOST}:{PORT}")
    threading.Timer(0.7, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
