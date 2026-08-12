from __future__ import annotations

import hashlib
import json
import socket
import statistics
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
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
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v87 as v87
import run_core_growth_binding_v89 as v89

HOST = "127.0.0.1"
START_PORT = 5146
OUT = ROOT / "data" / "core_growth_binding_v93" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v89.SEEDS
DOMAINS = v89.DOMAINS
QUORUM_SOURCES = 4


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


def edge_tuple(value) -> tuple[int, int]:
    a, b = [int(x) for x in value]
    return tuple(sorted((a, b)))


def min_hops(brain: SphereBrain, starts: set[int], targets: set[int], cap: int = 2) -> int | None:
    if not starts or not targets:
        return None
    if starts & targets:
        return 0
    frontier = set(int(x) for x in starts)
    seen = set(frontier)
    for depth in range(1, cap + 1):
        nxt_frontier = set()
        for node in frontier:
            for nxt in np.flatnonzero(brain.adjacency[node]).tolist():
                nxt = int(nxt)
                if nxt in seen:
                    continue
                if nxt in targets:
                    return depth
                seen.add(nxt)
                nxt_frontier.add(nxt)
        frontier = nxt_frontier
        if not frontier:
            break
    return None


def signature(brain: SphereBrain, item) -> dict:
    s = v83.action_signature(brain, item)
    return {
        "nodes": set(int(x) for x in s["nodes"]),
        "edges": {edge_tuple(x) for x in s["edges"]},
    }


def support_signatures(brain: SphereBrain, domain) -> dict[str, dict]:
    return {
        "left_action": signature(brain, v89.action_item(domain.left_subject, domain.left_action)),
        "right_action": signature(brain, v89.action_item(domain.right_subject, domain.right_action)),
        "left_context": signature(brain, v89.context_item(domain.left_subject, domain.shared_context)),
        "right_context": signature(brain, v89.context_item(domain.right_subject, domain.shared_context)),
        "transfer_left": signature(brain, v89.action_item(domain.transfer_left_subject, domain.left_action)),
        "transfer_right": signature(brain, v89.action_item(domain.transfer_right_subject, domain.right_action)),
    }


def source_support(brain: SphereBrain, edge: tuple[int, int], sig: dict) -> tuple[bool, bool]:
    exact = edge in sig["edges"]
    endpoints = {edge[0], edge[1]}
    proximal = bool(endpoints & sig["nodes"])
    if not proximal:
        hops = min_hops(brain, endpoints, sig["nodes"], cap=1)
        proximal = hops is not None and hops <= 1
    return exact, proximal


def consensus_snapshot(
    brain: SphereBrain,
    domain,
    experimental: SphereBrain,
    control: SphereBrain,
    baseline,
    cycle: int,
    history: dict[tuple[int, int], int],
) -> dict:
    candidates = v89.context_candidates(experimental, control, domain, baseline["direct"])
    candidate_edges = [edge_tuple(row["edge"]) for row in candidates]
    sigs = support_signatures(brain, domain)
    rows = []
    for edge in sorted(set(candidate_edges)):
        exact_sources = []
        proximal_sources = []
        for name, sig in sigs.items():
            exact, proximal = source_support(brain, edge, sig)
            if exact:
                exact_sources.append(name)
            if proximal:
                proximal_sources.append(name)
        prior_seen = int(history.get(edge, 0))
        source_ratio = len(proximal_sources) / max(1, len(sigs))
        exact_ratio = len(exact_sources) / max(1, len(sigs))
        temporal_ratio = min(1.0, prior_seen / 3.0)
        consensus_strength = 0.55 * source_ratio + 0.20 * exact_ratio + 0.25 * temporal_ratio
        rows.append({
            "edge": list(edge),
            "exact_support_count": len(exact_sources),
            "proximal_support_count": len(proximal_sources),
            "exact_sources": exact_sources,
            "proximal_sources": proximal_sources,
            "prior_checkpoint_recurrence": prior_seen,
            "quorum": len(proximal_sources) >= QUORUM_SOURCES,
            "consensus_strength": float(consensus_strength),
            "hub_score": float(v87.hub_score(brain, edge)),
        })

    for edge in set(candidate_edges):
        history[edge] = int(history.get(edge, 0)) + 1

    if not rows:
        return {
            "cycle": int(cycle),
            "candidate_count": 0,
            "quorum_edge_count": 0,
            "mean_support_sources": 0.0,
            "max_support_sources": 0,
            "mean_consensus_strength": 0.0,
            "max_consensus_strength": 0.0,
            "mean_hub_score": 0.0,
            "edges": [],
        }
    return {
        "cycle": int(cycle),
        "candidate_count": len(rows),
        "quorum_edge_count": sum(1 for x in rows if x["quorum"]),
        "mean_support_sources": sum(x["proximal_support_count"] for x in rows) / len(rows),
        "max_support_sources": max(x["proximal_support_count"] for x in rows),
        "mean_consensus_strength": sum(x["consensus_strength"] for x in rows) / len(rows),
        "max_consensus_strength": max(x["consensus_strength"] for x in rows),
        "mean_hub_score": sum(x["hub_score"] for x in rows) / len(rows),
        "edges": rows,
    }


def run_trial(domain, seed: int) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)
    history: dict[tuple[int, int], int] = {}
    records = []
    consensus_by_cycle = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            records.append(row)
            consensus_by_cycle.append(consensus_snapshot(
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
    first_consensus = consensus_by_cycle[first_i]
    return {
        "domain": domain.name,
        "seed": int(seed),
        "ever_bridge": True,
        "stable": cls == "stable",
        "classification": cls,
        "first_bridge_cycle": int(records[first_i]["cycle"]),
        "first_bridge_consensus": first_consensus,
        "consensus_by_cycle": consensus_by_cycle,
        "experimental": experimental,
    }


def mean(rows: list[dict], path: str) -> float | None:
    vals = []
    for row in rows:
        cur = row
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            vals.append(float(cur))
    return sum(vals) / len(vals) if vals else None


def compare_groups(stable: list[dict], unstable: list[dict], metric: str) -> dict:
    s = mean(stable, f"first_bridge_consensus.{metric}")
    u = mean(unstable, f"first_bridge_consensus.{metric}")
    return {
        "metric": metric,
        "stable_mean": s,
        "unstable_mean": u,
        "difference_stable_minus_unstable": None if s is None or u is None else s - u,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    all_trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    ever = [x for x in all_trials if x["ever_bridge"]]
    stable = [x for x in ever if x["stable"]]
    unstable = [x for x in ever if not x["stable"]]

    metrics = [
        "candidate_count",
        "quorum_edge_count",
        "mean_support_sources",
        "max_support_sources",
        "mean_consensus_strength",
        "max_consensus_strength",
        "mean_hub_score",
    ]
    comparisons = [compare_groups(stable, unstable, metric) for metric in metrics]
    ranked = sorted(
        [x for x in comparisons if x["difference_stable_minus_unstable"] is not None],
        key=lambda x: abs(float(x["difference_stable_minus_unstable"])),
        reverse=True,
    )

    stable_quorum_trials = sum(1 for x in stable if x["first_bridge_consensus"]["quorum_edge_count"] > 0)
    unstable_quorum_trials = sum(1 for x in unstable if x["first_bridge_consensus"]["quorum_edge_count"] > 0)
    stable_quorum_rate = stable_quorum_trials / len(stable) if stable else 0.0
    unstable_quorum_rate = unstable_quorum_trials / len(unstable) if unstable else 0.0
    consensus_delta = next((x["difference_stable_minus_unstable"] for x in comparisons if x["metric"] == "mean_consensus_strength"), 0.0) or 0.0
    quorum_rate_delta = stable_quorum_rate - unstable_quorum_rate

    domain_rows = []
    for domain in DOMAINS:
        srows = [x for x in stable if x["domain"] == domain.name]
        urows = [x for x in unstable if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "stable": len(srows),
            "unstable": len(urows),
            "stable_mean_consensus": mean(srows, "first_bridge_consensus.mean_consensus_strength"),
            "unstable_mean_consensus": mean(urows, "first_bridge_consensus.mean_consensus_strength"),
            "stable_quorum_rate": sum(1 for x in srows if x["first_bridge_consensus"]["quorum_edge_count"] > 0) / len(srows) if srows else None,
            "unstable_quorum_rate": sum(1 for x in urows if x["first_bridge_consensus"]["quorum_edge_count"] > 0) / len(urows) if urows else None,
        })

    representative = ever[0] if ever else all_trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "structural_consensus_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    # This is a microscope, not a promotion gate. PASS means the measurement was valid.
    measurement_pass = (
        len(stable) > 0
        and len(unstable) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )
    consensus_signal = consensus_delta > 0.03 or quorum_rate_delta > 0.15

    if consensus_signal:
        readiness = "structural_consensus_signal_observed"
        verdict = "stable_bridges_show_stronger_multi_path_structural_support_than_unstable_bridges_at_first_bridge_formation"
        next_step = "validate_structural_consensus_signal_with_more_seeds_and_ablation_before_using_consensus_for_core_stabilization"
    else:
        readiness = "structural_consensus_signal_not_observed"
        verdict = "stable_bridges_do_not_show_a_clear_structural_consensus_advantage_under_this_measurement"
        next_step = "do_not_add_consensus_to_core_until_support_definition_is_reconsidered_or_independent_evidence_appears"

    payload = {
        "experiment": "Core Growth Binding v93 — Structural Consensus Microscope",
        "contract": {
            "primary_core_modified": False,
            "consolidation_used": False,
            "assist_used": False,
            "trial_count": len(all_trials),
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "support_sources": ["left_action", "right_action", "left_context", "right_context", "transfer_left", "transfer_right"],
            "quorum_sources": QUORUM_SOURCES,
            "stable_label_used_in_consensus_score": False,
            "future_stability_used_in_consensus_score": False,
            "hub_penalty_used_in_consensus_score": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(all_trials),
            "ever_bridge_trials": len(ever),
            "stable_trials": len(stable),
            "unstable_trials": len(unstable),
            "stable_mean_consensus": mean(stable, "first_bridge_consensus.mean_consensus_strength"),
            "unstable_mean_consensus": mean(unstable, "first_bridge_consensus.mean_consensus_strength"),
            "consensus_delta": consensus_delta,
            "stable_quorum_rate": stable_quorum_rate,
            "unstable_quorum_rate": unstable_quorum_rate,
            "quorum_rate_delta": quorum_rate_delta,
            "top_separating_metric": ranked[0]["metric"] if ranked else None,
            "top_separating_difference": ranked[0]["difference_stable_minus_unstable"] if ranked else None,
            "consensus_signal_observed": consensus_signal,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "stable_vs_unstable": ranked,
        "domain_rows": domain_rows,
        "ever_bridge_trials": [{k: v for k, v in x.items() if k != "experimental"} for x in ever],
    }
    (OUT / "latest_binding_v93.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v93</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v93：Structural Consensus Microscope</h1><p class="lead">StableなSemantic Bridgeは、初回形成時点ですでに複数の独立した内部経路から強く支持されているかを測る。Consensusは観測のみで、Core・学習・Assistは変更しない。</p><section class="panel"><div class="controls"><button id="run">構造合意を観測</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Stable vs Unstable</h2><pre id="compare" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function pct(v){return `${(100*v).toFixed(1)}%`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Ever Bridge',s.ever_bridge_trials),metric('Stable',s.stable_trials),metric('Unstable',s.unstable_trials,'warn'),metric('Stable Consensus',s.stable_mean_consensus?.toFixed(4)??'—'),metric('Unstable Consensus',s.unstable_mean_consensus?.toFixed(4)??'—'),metric('Consensus差',s.consensus_delta?.toFixed(4)??'—',s.consensus_delta>0?'good':'warn'),metric('Stable Quorum率',pct(s.stable_quorum_rate)),metric('Unstable Quorum率',pct(s.unstable_quorum_rate)),metric('Quorum差',pct(s.quorum_rate_delta),s.quorum_rate_delta>0?'good':'warn'),metric('Consensus signal',yn(s.consensus_signal_observed),s.consensus_signal_observed?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('compare').textContent=JSON.stringify(d.stable_vs_unstable,null,2);document.getElementById('groups').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.ever_bridge_trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def main() -> None:
    print(f"v93 running: http://{HOST}:{PORT}")
    threading.Timer(0.8, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
