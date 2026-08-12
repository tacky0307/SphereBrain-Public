from __future__ import annotations

import hashlib
import json
import socket
import statistics
import sys
import threading
import webbrowser
from collections import deque
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
import run_core_growth_binding_v82 as v82
import run_core_growth_binding_v82b as v82b
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84

HOST = "127.0.0.1"
START_PORT = 5134
OUT = ROOT / "data" / "core_growth_binding_v84b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v84.CHECKPOINTS
SEEDS = v84.SEEDS
DOMAINS = v84.DOMAINS


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


def min_hops(brain: SphereBrain, starts: set[int], targets: set[int], cap: int = 12) -> int | None:
    if not starts or not targets:
        return None
    if starts & targets:
        return 0
    q = deque((int(v), 0) for v in starts)
    seen = set(int(v) for v in starts)
    while q:
        node, depth = q.popleft()
        if depth >= cap:
            continue
        for nxt in np.flatnonzero(brain.adjacency[node]).tolist():
            nxt = int(nxt)
            if nxt in seen:
                continue
            if nxt in targets:
                return depth + 1
            seen.add(nxt)
            q.append((nxt, depth + 1))
    return None


def usage_hub_score(brain: SphereBrain, nodes: set[int]) -> float:
    if not nodes:
        return 0.0
    usage = np.asarray(brain.node_usage, dtype=float)
    if usage.size == 0:
        return 0.0
    scores = []
    for node in nodes:
        value = float(usage[int(node)])
        scores.append(float(np.mean(usage <= value)))
    return sum(scores) / len(scores)


def structural_snapshot(brain: SphereBrain, domain) -> dict:
    left = v84.action_item(domain.left_subject, domain.left_action)
    right = v84.action_item(domain.right_subject, domain.right_action)
    left_sig = v83.action_signature(brain, left)
    right_sig = v83.action_signature(brain, right)
    ctx = v84.domain_context_signature(brain, domain)
    action_nodes = set(left_sig["nodes"]) | set(right_sig["nodes"])
    action_shared_nodes = set(left_sig["nodes"]) & set(right_sig["nodes"])
    action_shared_edges = set(left_sig["edges"]) & set(right_sig["edges"])
    context_nodes = set(ctx["nodes"])
    context_shared_nodes = set(ctx["shared_nodes"])
    return {
        "action_node_hops": min_hops(brain, set(left_sig["nodes"]), set(right_sig["nodes"])),
        "context_to_action_hops": min_hops(brain, context_nodes, action_nodes),
        "shared_context_to_action_hops": min_hops(brain, context_shared_nodes, action_nodes),
        "action_shared_nodes": len(action_shared_nodes),
        "action_shared_edges": len(action_shared_edges),
        "action_edge_similarity": v82.jaccard(left_sig["edges"], right_sig["edges"]),
        "action_activation_similarity": v82b.weighted_similarity(left_sig["activation"], right_sig["activation"]),
        "context_shared_nodes": len(context_shared_nodes),
        "context_shared_edges": len(ctx["shared_edges"]),
        "action_hub_score": usage_hub_score(brain, action_nodes),
        "context_hub_score": usage_hub_score(brain, context_nodes),
    }


def cycle_success(row: dict) -> bool:
    return bool(
        row["experimental"]["direct"]["new_shared_edges"] > 0
        and row["effects"]["direct_new_shared_edges"] > 0
        and row["context_linked_candidate_count"] > 0
        and (
            row["effects"]["direct_edge_similarity"] > 0
            or row["effects"]["direct_activation_similarity"] > 0
        )
    )


def classify_timeline(records: list[dict]) -> str:
    active = [int(r["cycle"]) for r in records if cycle_success(r)]
    candidates = [int(r["cycle"]) for r in records if r["context_linked_candidate_count"] > 0]
    if not active:
        return "candidate_only" if candidates else "never_success"
    first = active[0]
    last_checkpoint = int(records[-1]["cycle"])
    if first <= 1 and last_checkpoint in active and len(active) >= 3:
        return "stable_success"
    if first <= 1 and last_checkpoint not in active:
        return "transient_success"
    if first >= 3:
        return "late_success"
    return "intermittent_success"


def run_attribution_trial(domain, seed: int) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v84.make_baseline(experimental, domain)
    ctrl_baseline = v84.make_baseline(control, domain)
    exp_episodes = v84.episodes_for(domain, True)
    ctrl_episodes = v84.episodes_for(domain, False)

    initial_exp = structural_snapshot(experimental, domain)
    initial_ctrl = structural_snapshot(control, domain)
    records = []

    previous_weights = experimental.weights.copy()
    previous_usage = experimental.usage.copy()

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            exp = v84.measure(experimental, domain, exp_baseline)
            ctrl = v84.measure(control, domain, ctrl_baseline)
            candidates = v84.context_candidates(experimental, control, domain, exp_baseline["direct"])
            dw = np.abs(np.asarray(experimental.weights, dtype=float) - np.asarray(previous_weights, dtype=float))
            du = np.abs(np.asarray(experimental.usage, dtype=float) - np.asarray(previous_usage, dtype=float))
            changed_weight_edges = int(np.count_nonzero(np.triu(dw > 1e-12, 1)))
            changed_usage_edges = int(np.count_nonzero(np.triu(du > 1e-12, 1)))
            snap = structural_snapshot(experimental, domain)
            row = {
                "cycle": cycle,
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
                "structure": snap,
                "changed_weight_edges_since_previous_checkpoint": changed_weight_edges,
                "changed_usage_edges_since_previous_checkpoint": changed_usage_edges,
            }
            row["direct_success_now"] = cycle_success(row)
            records.append(row)
            previous_weights = experimental.weights.copy()
            previous_usage = experimental.usage.copy()
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    success_cycles = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    final = records[-1]
    return {
        "domain": domain.name,
        "seed": seed,
        "v84_success": bool(success_cycles),
        "first_success_cycle": success_cycles[0] if success_cycles else None,
        "timeline_class": classify_timeline(records),
        "initial_experimental": initial_exp,
        "initial_control": initial_ctrl,
        "final_context_candidates": final["context_linked_candidate_count"],
        "final_edge_effect": final["effects"]["direct_edge_similarity"],
        "final_activation_effect": final["effects"]["direct_activation_similarity"],
        "final_new_shared_edge_effect": final["effects"]["direct_new_shared_edges"],
        "records": records,
        "experimental": experimental,
    }


def mean_metric(rows: list[dict], key: str) -> float | None:
    vals = [r["initial_experimental"].get(key) for r in rows]
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def separation(successes: list[dict], failures: list[dict], key: str) -> dict:
    s = mean_metric(successes, key)
    f = mean_metric(failures, key)
    return {
        "metric": key,
        "success_mean": s,
        "failure_mean": f,
        "difference_success_minus_failure": None if s is None or f is None else s - f,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_attribution_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    successes = [x for x in trials if x["v84_success"]]
    failures = [x for x in trials if not x["v84_success"]]

    metrics = [
        "action_node_hops",
        "context_to_action_hops",
        "shared_context_to_action_hops",
        "action_shared_nodes",
        "action_shared_edges",
        "action_edge_similarity",
        "action_activation_similarity",
        "context_shared_nodes",
        "context_shared_edges",
        "action_hub_score",
        "context_hub_score",
    ]
    comparisons = [separation(successes, failures, key) for key in metrics]
    numeric = [x for x in comparisons if x["difference_success_minus_failure"] is not None]
    ranked = sorted(numeric, key=lambda x: abs(float(x["difference_success_minus_failure"])), reverse=True)

    class_counts: dict[str, int] = {}
    for row in trials:
        class_counts[row["timeline_class"]] = class_counts.get(row["timeline_class"], 0) + 1

    domain_rows = []
    for domain in DOMAINS:
        rows = [x for x in trials if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "successes": sum(1 for x in rows if x["v84_success"]),
            "trials": len(rows),
            "timeline_classes": {k: sum(1 for x in rows if x["timeline_class"] == k) for k in class_counts},
        })

    seed_rows = []
    for seed in SEEDS:
        rows = [x for x in trials if x["seed"] == seed]
        seed_rows.append({
            "seed": seed,
            "successes": sum(1 for x in rows if x["v84_success"]),
            "domains": [x["domain"] for x in rows if x["v84_success"]],
        })

    representative = successes[0] if successes else trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_episode_attribution_roundtrip.json"
    before_state = representative["experimental"].snapshot_learning_state()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_state == loaded.snapshot_learning_state()
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)

    seed_variation = len({x["successes"] for x in seed_rows}) > 1
    domain_variation = len({x["successes"] for x in domain_rows}) > 1
    transient_count = class_counts.get("transient_success", 0)
    never_count = class_counts.get("never_success", 0)
    attribution_pass = (
        len(successes) > 0
        and len(failures) > 0
        and len(ranked) > 0
        and saveload_equal
        and native_present
        and production_unchanged
    )

    if attribution_pass:
        readiness = "semantic_episode_failure_modes_attributed"
        verdict = "success_and_failure_trials_show_measurable_initial_structure_and_timeline_differences"
        next_step = "validate_top_attribution_factors_with_controlled_topology_or_targeted_episode_intervention"
    else:
        readiness = "semantic_episode_attribution_inconclusive"
        verdict = "success_failure_difference_not_yet_explained_by_measured_structural_factors"
        next_step = "expand_structural_observation_before_modifying_primary_core"

    payload = {
        "experiment": "Core Growth Binding v84B — Semantic Episode Success / Failure Attribution",
        "contract": {
            "primary_core_modified": False,
            "reuses_v84_domains": [x.name for x in DOMAINS],
            "reuses_v84_seeds": SEEDS,
            "trial_count": len(trials),
            "checkpoints": CHECKPOINTS,
            "production_brain_json_saved": False,
        },
        "summary": {
            "success_trials": len(successes),
            "failure_trials": len(failures),
            "timeline_classes": class_counts,
            "transient_success_count": transient_count,
            "never_success_count": never_count,
            "seed_variation_observed": seed_variation,
            "domain_variation_observed": domain_variation,
            "top_attribution_metric": ranked[0]["metric"] if ranked else None,
            "top_attribution_difference": ranked[0]["difference_success_minus_failure"] if ranked else None,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "comparisons_ranked": ranked,
        "domain_rows": domain_rows,
        "seed_rows": seed_rows,
        "trials": [
            {k: v for k, v in x.items() if k != "experimental"}
            for x in trials
        ],
    }
    (OUT / "latest_binding_v84b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v84B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v84B：Semantic Episode Success / Failure Attribution</h1><p class="lead">v84の15試行を再現し、成功と失敗を初期Core構造・Context距離・Hub依存・時間推移から比較する。Coreは変更しない。</p><section class="panel"><div class="controls"><button id="run">成功/失敗を帰属分析</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>差が大きい指標</h2><pre id="compare" class="raw">未実行</pre></section><section class="panel"><h2>Trial別</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;const tc=s.timeline_classes||{};document.getElementById('metrics').innerHTML=[metric('成功Trial',s.success_trials),metric('失敗Trial',s.failure_trials),metric('Stable',tc.stable_success||0),metric('Transient',tc.transient_success||0),metric('Late',tc.late_success||0),metric('Never',tc.never_success||0),metric('Candidate only',tc.candidate_only||0),metric('Seed差',yn(s.seed_variation_observed),s.seed_variation_observed?'good':'warn'),metric('Domain差',yn(s.domain_variation_observed),s.domain_variation_observed?'good':'warn'),metric('最大差指標',s.top_attribution_metric||'—'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('compare').textContent=JSON.stringify(d.comparisons_ranked,null,2);document.getElementById('rows').textContent=JSON.stringify({domain_rows:d.domain_rows,seed_rows:d.seed_rows,trials:d.trials},null,2)}document.getElementById('run').onclick=run;</script></body></html>'''


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
    print(f"Core Growth Binding v84B: http://{HOST}:{PORT}")
    print("Semantic Episode success/failure attribution / Primary Core変更なし / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
