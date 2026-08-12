from __future__ import annotations

import copy
import hashlib
import json
import socket
import statistics
import sys
import threading
import webbrowser
from collections import Counter, defaultdict
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
import run_core_growth_binding_v99 as v99
import run_core_growth_binding_v99b as v99b
import run_core_growth_binding_v100 as v100

HOST = "127.0.0.1"
START_PORT = 5159
OUT = ROOT / "data" / "core_growth_binding_v100b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS = v100.SEEDS
TRIGGER_SETS = v100.TRIGGER_SETS
FORMATION_CYCLES = v100.FORMATION_CYCLES
AFTERMATH_CYCLES = v100.AFTERMATH_CYCLES
MATCHED_NOVEL_EPISODES = v100.MATCHED_NOVEL_EPISODES


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


def prepare_seed(seed: int) -> SphereBrain:
    return v100.prepare_seed(seed)


def stage_edges(result: dict) -> dict[str, set[tuple[int, int]]]:
    out = {}
    for stage in ("subject", "place_relation", "context", "action_relation", "action"):
        out[stage] = {
            tuple(sorted((int(a), int(b))))
            for a, b in result[stage].traversed_edges
        }
    return out


def stage_nodes(result: dict) -> dict[str, set[int]]:
    return {
        stage: set(int(x) for x in result[stage].activated_nodes)
        for stage in ("subject", "place_relation", "context", "action_relation", "action")
    }


def edge_stats(brain: SphereBrain, edges: set[tuple[int, int]]) -> dict:
    if not edges:
        return {
            "mean_weight": 0.0,
            "mean_usage": 0.0,
            "mean_endpoint_node_usage": 0.0,
        }
    weights = [float(brain.weights[a, b]) for a, b in edges]
    usages = [float(brain.usage[a, b]) for a, b in edges]
    node_usages = []
    for a, b in edges:
        node_usages.extend([float(brain.node_usage[a]), float(brain.node_usage[b])])
    return {
        "mean_weight": statistics.mean(weights),
        "mean_usage": statistics.mean(usages),
        "mean_endpoint_node_usage": statistics.mean(node_usages),
    }


def trigger_trace_features(base_brain: SphereBrain, episodes) -> tuple[SphereBrain, dict]:
    brain = copy.deepcopy(base_brain)
    history: dict[tuple[int, int], int] = {}
    before = v99.structural_snapshot(brain, history)
    before_quorum = v99b.quorum_edges(before)
    reused_before = set(v99.reused_edges(before["signatures"]))

    all_edges: set[tuple[int, int]] = set()
    all_nodes: set[int] = set()
    stage_edge_sets: dict[str, set[tuple[int, int]]] = defaultdict(set)
    stage_node_sets: dict[str, set[int]] = defaultdict(set)

    for spec in episodes:
        result = v99.v83.episode_experience(brain, spec, learn=True)
        se = stage_edges(result)
        sn = stage_nodes(result)
        for stage in se:
            stage_edge_sets[stage].update(se[stage])
            stage_node_sets[stage].update(sn[stage])
            all_edges.update(se[stage])
            all_nodes.update(sn[stage])

    immediate = v99.structural_snapshot(brain, history)
    immediate_quorum = v99b.quorum_edges(immediate)
    immediate_new = immediate_quorum - before_quorum

    overlap = v99b.module_overlap_stats(all_edges, before["modules"])
    overlap_reused = len(all_edges & reused_before) / max(1, len(all_edges))

    pre_consensus_edges = {
        tuple(x["edge"]): x for x in before["consensus"]["edges"]
    }
    proximal_hits = 0
    consensus_strengths = []
    for edge in all_edges:
        if edge in pre_consensus_edges:
            proximal_hits += 1
            consensus_strengths.append(float(pre_consensus_edges[edge]["consensus_strength"]))
            continue
        a, b = edge
        for qedge, row in pre_consensus_edges.items():
            if a in qedge or b in qedge:
                proximal_hits += 1
                consensus_strengths.append(float(row["consensus_strength"]))
                break
    pre_quorum_proximity = proximal_hits / max(1, len(all_edges))

    stage_pair_overlaps = []
    stages = list(stage_edge_sets)
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            left = stage_edge_sets[stages[i]]
            right = stage_edge_sets[stages[j]]
            union = left | right
            stage_pair_overlaps.append(len(left & right) / len(union) if union else 0.0)
    stage_convergence = statistics.mean(stage_pair_overlaps) if stage_pair_overlaps else 0.0
    active_stage_count = sum(1 for s in stages if stage_edge_sets[s])
    stage_diversity = active_stage_count / 5.0

    stats = edge_stats(brain, all_edges)
    reused_topology = v99.v98b.topology_from_signatures(before["signatures"])
    local_density = (
        len(reused_before) / max(1, sum(len(x["edges"]) for x in before["signatures"]))
    )

    immediate_supports = []
    immediate_strengths = []
    for row in immediate["consensus"]["edges"]:
        e = tuple(row["edge"])
        if e in immediate_new:
            immediate_supports.append(float(row["proximal_support"]))
            immediate_strengths.append(float(row["consensus_strength"]))

    features = {
        **overlap,
        "trigger_edge_count": len(all_edges),
        "reuse_overlap_fraction": overlap_reused,
        "pre_quorum_proximity": pre_quorum_proximity,
        "pre_consensus_strength_near_trigger": statistics.mean(consensus_strengths) if consensus_strengths else 0.0,
        "stage_diversity": stage_diversity,
        "stage_convergence": stage_convergence,
        "mean_traversed_weight": stats["mean_weight"],
        "mean_traversed_usage": stats["mean_usage"],
        "mean_traversed_node_usage": stats["mean_endpoint_node_usage"],
        "pre_reused_edge_density": local_density,
        "pre_module_count": len(before["modules"]),
        "pre_quorum_count": len(before_quorum),
        "pre_mean_consensus_strength": before["consensus"]["mean_consensus_strength"],
        "immediate_new_quorum": len(immediate_new),
        "immediate_new_quorum_mean_support": statistics.mean(immediate_supports) if immediate_supports else 0.0,
        "immediate_new_quorum_mean_strength": statistics.mean(immediate_strengths) if immediate_strengths else 0.0,
        "pre_giant_ratio": reused_topology["giant_component_edge_ratio"],
        "pre_module_topology_count": reused_topology["local_module_count"],
    }
    return brain, features


def persistence_label(brain_after_trigger: SphereBrain, before_snapshot: dict) -> tuple[int, float]:
    history: dict[tuple[int, int], int] = {}
    # Recreate a comparable pre-quorum set from the saved pre-trigger snapshot.
    pre_quorum = v99b.quorum_edges(before_snapshot)
    immediate = v99.structural_snapshot(brain_after_trigger, history)
    immediate_new = v99b.quorum_edges(immediate) - pre_quorum

    for _ in range(AFTERMATH_CYCLES):
        v99.v98.train_one_cycle(brain_after_trigger)
        v99.structural_snapshot(brain_after_trigger, history)
    final = v99.structural_snapshot(brain_after_trigger, history)
    final_quorum = v99b.quorum_edges(final)
    persistent = immediate_new & final_quorum
    return len(persistent), len(persistent) / max(1, len(immediate_new))


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(x[key]) for x in rows) if rows else 0.0


def standardized_gap(success: list[dict], failure: list[dict], key: str) -> float:
    if not success or not failure:
        return 0.0
    svals = [float(x[key]) for x in success]
    fvals = [float(x[key]) for x in failure]
    pooled = svals + fvals
    sd = statistics.pstdev(pooled) if len(pooled) > 1 else 0.0
    raw = statistics.mean(svals) - statistics.mean(fvals)
    return raw / sd if sd > 1e-12 else raw


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    rows = []
    representatives = []

    for si, seed in enumerate(SEEDS, start=1):
        print(f"[v100B] seed {si}/{len(SEEDS)}: {seed}", flush=True)
        base = prepare_seed(seed)
        matched = v100.run_branch(base, MATCHED_NOVEL_EPISODES)
        matched_persistent = int(matched["persistent_new_quorum"])

        for ti, (trigger_id, episodes) in enumerate(TRIGGER_SETS.items(), start=1):
            print(f"[v100B]   trigger {ti}/{len(TRIGGER_SETS)}: {trigger_id}", flush=True)
            pre_history: dict[tuple[int, int], int] = {}
            before_snapshot = v99.structural_snapshot(copy.deepcopy(base), pre_history)
            brain_after_trigger, features = trigger_trace_features(base, episodes)
            persistent_count, persistent_fraction = persistence_label(brain_after_trigger, before_snapshot)
            success = persistent_count > matched_persistent
            rows.append({
                "seed": seed,
                "trigger_id": trigger_id,
                "matched_persistent_new_quorum": matched_persistent,
                "persistent_new_quorum": persistent_count,
                "persistent_fraction_label": persistent_fraction,
                "success": bool(success),
                **features,
            })
            if not representatives:
                representatives.append(brain_after_trigger)

    success = [x for x in rows if x["success"]]
    failure = [x for x in rows if not x["success"]]

    feature_keys = [
        "touched_module_count",
        "cross_module_edge_count",
        "cross_module_fraction",
        "existing_module_overlap_fraction",
        "reuse_overlap_fraction",
        "pre_quorum_proximity",
        "pre_consensus_strength_near_trigger",
        "stage_diversity",
        "stage_convergence",
        "mean_traversed_weight",
        "mean_traversed_usage",
        "mean_traversed_node_usage",
        "pre_reused_edge_density",
        "pre_module_count",
        "pre_quorum_count",
        "pre_mean_consensus_strength",
        "immediate_new_quorum",
        "immediate_new_quorum_mean_support",
        "immediate_new_quorum_mean_strength",
        "pre_giant_ratio",
        "pre_module_topology_count",
    ]

    feature_rows = []
    for key in feature_keys:
        gap = mean(success, key) - mean(failure, key)
        effect = standardized_gap(success, failure, key)
        per_trigger_positive = 0
        comparable = 0
        for trigger_id in TRIGGER_SETS:
            s = [x for x in success if x["trigger_id"] == trigger_id]
            f = [x for x in failure if x["trigger_id"] == trigger_id]
            if s and f:
                comparable += 1
                if mean(s, key) - mean(f, key) > 0:
                    per_trigger_positive += 1
        feature_rows.append({
            "feature": key,
            "success_mean": mean(success, key),
            "failure_mean": mean(failure, key),
            "gap": gap,
            "standardized_gap": effect,
            "positive_triggers": per_trigger_positive,
            "comparable_triggers": comparable,
        })

    ranked = sorted(feature_rows, key=lambda x: (-abs(x["standardized_gap"]), -x["positive_triggers"], x["feature"]))
    top = ranked[0] if ranked else None
    robust_candidates = [x for x in ranked if abs(x["standardized_gap"]) >= 0.35 and x["positive_triggers"] >= 3]
    formation_signal = bool(robust_candidates) and len(success) >= 20 and len(failure) >= 80

    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    native_present = False
    if representatives:
        temp = OUT / "persistence_formation_attribution_roundtrip.json"
        before_weights = representatives[0].weights.tolist()
        representatives[0].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
        native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    attribution_pass = (
        len(rows) == len(SEEDS) * len(TRIGGER_SETS)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if formation_signal:
        readiness = "persistence_formation_precursor_candidate_identified"
        verdict = "pre_or_immediate_trigger_structure_contains_replicated_features_associated_with_later_persistent_quorum_formation"
        next_step = "hold_out_the_candidate_feature_thresholds_and_test_prediction_on_new_seeds_without_using_aftermath_information"
    else:
        readiness = "persistence_formation_precursor_not_yet_isolated"
        verdict = "persistent_quorum_formation_is_not_yet_explained_by_a_robust_pre_or_immediate_trigger_feature_across_multiple_triggers"
        next_step = "increase_temporal_resolution_inside_the_trigger_and_trace_first_quorum_assembly_before_modifying_core_learning"

    payload = {
        "experiment": "Core Growth Binding v100B — Persistence Formation Attribution",
        "contract": {
            "primary_core_modified": False,
            "seed_count": len(SEEDS),
            "trigger_count": len(TRIGGER_SETS),
            "trial_count": len(rows),
            "success_definition": "persistent_new_quorum_greater_than_same_seed_matched_novel",
            "aftermath_persistence_not_used_as_feature": True,
            "semantic_labels_used_for_attribution": False,
            "v100_trigger_sets_unchanged": True,
            "v100_seeds_unchanged": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(rows),
            "success_count": len(success),
            "failure_count": len(failure),
            "success_rate": len(success) / len(rows) if rows else 0.0,
            "top_attribution": top["feature"] if top else None,
            "top_standardized_gap": top["standardized_gap"] if top else 0.0,
            "top_positive_triggers": top["positive_triggers"] if top else 0,
            "robust_candidate_count": len(robust_candidates),
            "formation_signal": formation_signal,
            "saveload_equal": saveload_equal,
            "brain_file_unchanged": production_unchanged,
            "primary_native_learning_present": native_present,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "feature_ranking": ranked,
        "robust_candidates": robust_candidates,
        "trials": rows,
    }
    (OUT / "latest_binding_v100b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v100B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v100B：Persistence Formation Attribution</h1><p class="lead">v100で強く再現したPersistent quorumについて、成立する前までのCore状態とTrigger内部経路だけから、36成功Trialと124非成功Trialの違いを探す。最終Persistenceそのものは説明変数に使わない。</p><section class="panel"><div class="controls"><button id="run">Formationを分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Feature ranking</h2><pre id="features" class="raw">未実行</pre></section><section class="panel"><h2>Robust candidates</h2><pre id="candidates" class="raw">未実行</pre></section><section class="panel"><h2>Raw JSON</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Success',s.success_count),metric('Failure',s.failure_count),metric('Success rate',`${(s.success_rate*100).toFixed(1)}%`),metric('Top attribution',s.top_attribution||'—'),metric('Top effect',s.top_standardized_gap.toFixed(3),Math.abs(s.top_standardized_gap)>=0.35?'good':'blue'),metric('Top + triggers',`${s.top_positive_triggers}/4`),metric('Robust candidates',s.robust_candidate_count,s.robust_candidate_count?'good':'warn'),metric('Formation signal',yn(s.formation_signal),s.formation_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('features').textContent=JSON.stringify(d.feature_ranking,null,2);document.getElementById('candidates').textContent=JSON.stringify(d.robust_candidates,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v100B: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
