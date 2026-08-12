from __future__ import annotations

import copy
import hashlib
import json
import socket
import statistics
import sys
import threading
import webbrowser
from collections import defaultdict
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
START_PORT = 5160
OUT = ROOT / "data" / "core_growth_binding_v100c" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS = v100.SEEDS
TRIGGER_SETS = v100.TRIGGER_SETS
FORMATION_CYCLES = v100.FORMATION_CYCLES
AFTERMATH_CYCLES = v100.AFTERMATH_CYCLES
MATCHED_NOVEL_EPISODES = v100.MATCHED_NOVEL_EPISODES
STAGES = ("subject", "place_relation", "context", "action_relation", "action")


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


def qedges(snapshot: dict) -> dict[tuple[int, int], dict]:
    return {
        tuple(sorted((int(x["edge"][0]), int(x["edge"][1])))): x
        for x in snapshot["consensus"]["edges"] if bool(x["quorum"])
    }


def edge_near(brain: SphereBrain, edge: tuple[int, int], trace_edges: set[tuple[int, int]], trace_nodes: set[int]) -> bool:
    if edge in trace_edges:
        return True
    a, b = edge
    if a in trace_nodes or b in trace_nodes:
        return True
    for endpoint in edge:
        for nxt in np.flatnonzero(brain.adjacency[endpoint]).tolist():
            if int(nxt) in trace_nodes:
                return True
    return False


def trigger_with_trace(base: SphereBrain, episodes) -> tuple[SphereBrain, dict, dict, list[dict]]:
    brain = copy.deepcopy(base)
    history: dict[tuple[int, int], int] = {}
    before = v99.structural_snapshot(brain, history)
    pre_q = qedges(before)
    reused_before = set(v99.reused_edges(before["signatures"]))
    modules = before["modules"]
    membership = v99.node_to_module(modules)

    traces: list[dict] = []
    all_trigger_edges: set[tuple[int, int]] = set()
    edge_visit_count: defaultdict[tuple[int, int], int] = defaultdict(int)

    for episode_index, spec in enumerate(episodes):
        result = v99.v83.episode_experience(brain, spec, learn=True)
        for stage in STAGES:
            stage_edges = {
                tuple(sorted((int(a), int(b))))
                for a, b in result[stage].traversed_edges
            }
            stage_nodes = set(int(x) for x in result[stage].activated_nodes)
            all_trigger_edges.update(stage_edges)
            for edge in stage_edges:
                edge_visit_count[edge] += 1
            traces.append({
                "episode": episode_index,
                "stage": stage,
                "edges": stage_edges,
                "nodes": stage_nodes,
            })

    immediate = v99.structural_snapshot(brain, history)
    immediate_q = qedges(immediate)
    immediate_new = set(immediate_q) - set(pre_q)

    rows = []
    for edge in sorted(immediate_new):
        exact_episode = set()
        proximal_episode = set()
        exact_stage = set()
        proximal_stage = set()
        exact_trace_count = 0
        proximal_trace_count = 0
        for tr in traces:
            exact = edge in tr["edges"]
            proximal = edge_near(brain, edge, tr["edges"], tr["nodes"])
            if exact:
                exact_episode.add(tr["episode"])
                exact_stage.add(tr["stage"])
                exact_trace_count += 1
            if proximal:
                proximal_episode.add(tr["episode"])
                proximal_stage.add(tr["stage"])
                proximal_trace_count += 1

        a, b = edge
        ma = membership.get(a)
        mb = membership.get(b)
        pre_near = 0
        if edge in pre_q:
            pre_near = 1
        else:
            for pe in pre_q:
                if a in pe or b in pe:
                    pre_near = 1
                    break
        local_reused = sum(1 for e in reused_before if a in e or b in e)
        immediate_row = immediate_q[edge]
        rows.append({
            "edge": list(edge),
            "exact_episode_count": len(exact_episode),
            "proximal_episode_count": len(proximal_episode),
            "exact_stage_count": len(exact_stage),
            "proximal_stage_count": len(proximal_stage),
            "exact_trace_count": exact_trace_count,
            "proximal_trace_count": proximal_trace_count,
            "stage_diversity": len(proximal_stage) / len(STAGES),
            "episode_diversity": len(proximal_episode) / max(1, len(episodes)),
            "revisit_count": edge_visit_count.get(edge, 0),
            "pre_reuse_exact": 1 if edge in reused_before else 0,
            "pre_quorum_near": pre_near,
            "local_reused_degree": local_reused,
            "same_pre_module": 1 if ma is not None and ma == mb else 0,
            "cross_pre_module": 1 if ma is not None and mb is not None and ma != mb else 0,
            "weight_at_formation": float(brain.weights[a, b]),
            "usage_at_formation": float(brain.usage[a, b]),
            "node_usage_at_formation": (float(brain.node_usage[a]) + float(brain.node_usage[b])) / 2.0,
            "support_sources_at_formation": float(immediate_row["proximal_support"]),
            "consensus_strength_at_formation": float(immediate_row["consensus_strength"]),
        })

    return brain, before, immediate, rows


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(x[key]) for x in rows) if rows else 0.0


def standardized_gap(persistent: list[dict], vanished: list[dict], key: str) -> float:
    if not persistent or not vanished:
        return 0.0
    vals = [float(x[key]) for x in persistent + vanished]
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    raw = mean(persistent, key) - mean(vanished, key)
    return raw / sd if sd > 1e-12 else raw


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    quorum_rows: list[dict] = []
    representative = None

    for si, seed in enumerate(SEEDS, start=1):
        print(f"[v100C] seed {si}/{len(SEEDS)}: {seed}", flush=True)
        base = v100.prepare_seed(seed)
        matched = v100.run_branch(base, MATCHED_NOVEL_EPISODES)
        matched_persistent = int(matched["persistent_new_quorum"])

        for ti, (trigger_id, episodes) in enumerate(TRIGGER_SETS.items(), start=1):
            print(f"[v100C]   trigger {ti}/{len(TRIGGER_SETS)}: {trigger_id}", flush=True)
            brain, before, immediate, assemblies = trigger_with_trace(base, episodes)
            immediate_set = set(qedges(immediate)) - set(qedges(before))

            history: dict[tuple[int, int], int] = {}
            for _ in range(AFTERMATH_CYCLES):
                v99.v98.train_one_cycle(brain)
                v99.structural_snapshot(brain, history)
            final = v99.structural_snapshot(brain, history)
            final_set = set(qedges(final))
            persistent_set = immediate_set & final_set
            trial_persistent_count = len(persistent_set)
            trial_success = trial_persistent_count > matched_persistent

            for row in assemblies:
                edge = tuple(row["edge"])
                quorum_rows.append({
                    "seed": seed,
                    "trigger_id": trigger_id,
                    "trial_success": bool(trial_success),
                    "persistent": edge in persistent_set,
                    **row,
                })
            if representative is None:
                representative = brain

    persistent = [x for x in quorum_rows if x["persistent"]]
    vanished = [x for x in quorum_rows if not x["persistent"]]
    features = [
        "exact_episode_count", "proximal_episode_count", "exact_stage_count",
        "proximal_stage_count", "exact_trace_count", "proximal_trace_count",
        "stage_diversity", "episode_diversity", "revisit_count", "pre_reuse_exact",
        "pre_quorum_near", "local_reused_degree", "same_pre_module", "cross_pre_module",
        "weight_at_formation", "usage_at_formation", "node_usage_at_formation",
        "support_sources_at_formation", "consensus_strength_at_formation",
    ]

    ranking = []
    for key in features:
        effect = standardized_gap(persistent, vanished, key)
        positive_triggers = 0
        comparable = 0
        for trigger_id in TRIGGER_SETS:
            p = [x for x in persistent if x["trigger_id"] == trigger_id]
            v = [x for x in vanished if x["trigger_id"] == trigger_id]
            if p and v:
                comparable += 1
                if mean(p, key) > mean(v, key):
                    positive_triggers += 1
        ranking.append({
            "feature": key,
            "persistent_mean": mean(persistent, key),
            "vanished_mean": mean(vanished, key),
            "gap": mean(persistent, key) - mean(vanished, key),
            "standardized_gap": effect,
            "positive_triggers": positive_triggers,
            "comparable_triggers": comparable,
        })
    ranking.sort(key=lambda x: (-abs(x["standardized_gap"]), -x["positive_triggers"], x["feature"]))
    top = ranking[0] if ranking else None
    robust = [x for x in ranking if abs(x["standardized_gap"]) >= 0.35 and x["positive_triggers"] >= 3]
    assembly_signal = bool(robust) and len(persistent) >= 20 and len(vanished) >= 20

    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    native_present = False
    if representative is not None:
        temp = OUT / "immediate_quorum_assembly_roundtrip.json"
        before_weights = representative.weights.tolist()
        representative.save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
        native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    measurement_pass = saveload_equal and production_unchanged and native_present and len(quorum_rows) > 0

    readiness = (
        "immediate_quorum_assembly_pattern_identified"
        if assembly_signal else
        "immediate_quorum_assembly_pattern_not_yet_isolated"
    )
    verdict = (
        "persistent_and_vanishing_immediate_quorum_edges_differ_in_replicated_assembly_features"
        if assembly_signal else
        "immediate_quorum_persistence_is_not_yet_explained_by_a_robust_assembly_feature"
    )
    next_step = (
        "freeze_the_top_assembly_feature_and_test_prediction_on_new_seeds_and_new_trigger_streams"
        if assembly_signal else
        "increase_within_trigger_temporal_resolution_and_observe_quorum_formation_step_by_step"
    )

    payload = {
        "experiment": "Core Growth Binding v100C — Immediate Quorum Assembly Decomposition",
        "contract": {
            "primary_core_modified": False,
            "seed_count": len(SEEDS),
            "trigger_count": len(TRIGGER_SETS),
            "semantic_labels_used_for_assembly_scoring": False,
            "persistence_used_only_as_posthoc_edge_label": True,
            "v100_trigger_sets_unchanged": True,
            "v100_seeds_unchanged": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "quorum_edge_count": len(quorum_rows),
            "persistent_quorum_edges": len(persistent),
            "vanished_quorum_edges": len(vanished),
            "persistence_rate": len(persistent) / len(quorum_rows) if quorum_rows else 0.0,
            "top_attribution": top["feature"] if top else None,
            "top_standardized_gap": top["standardized_gap"] if top else 0.0,
            "top_positive_triggers": top["positive_triggers"] if top else 0,
            "robust_candidate_count": len(robust),
            "assembly_signal": assembly_signal,
            "saveload_equal": saveload_equal,
            "brain_file_unchanged": production_unchanged,
            "primary_native_learning_present": native_present,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "feature_ranking": ranking,
        "robust_candidates": robust,
        "quorum_edges": quorum_rows,
    }
    (OUT / "latest_binding_v100c.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v100C</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v100C：Immediate Quorum Assembly Decomposition</h1><p class="lead">Trigger直後に新しく形成されたQuorum Edgeを1本ずつ分解し、後にPersistenceしたものと消えたものの組み上がり方を比較する。Persistenceは事後ラベルとしてのみ使用する。</p><section class="panel"><div class="controls"><button id="run">Assemblyを分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Feature ranking</h2><pre id="ranking" class="raw">未実行</pre></section><section class="panel"><h2>Robust candidates</h2><pre id="robust" class="raw">未実行</pre></section><section class="panel"><h2>Raw JSON</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Quorum Edge',s.quorum_edge_count),metric('Persistent',s.persistent_quorum_edges,'good'),metric('Vanished',s.vanished_quorum_edges),metric('Persistence rate',`${(s.persistence_rate*100).toFixed(1)}%`),metric('Top attribution',s.top_attribution||'—'),metric('Top effect',s.top_standardized_gap.toFixed(3),Math.abs(s.top_standardized_gap)>=0.35?'good':'warn'),metric('Top + triggers',`${s.top_positive_triggers}/4`),metric('Robust candidates',s.robust_candidate_count),metric('Assembly signal',yn(s.assembly_signal),s.assembly_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('ranking').textContent=JSON.stringify(d.feature_ranking,null,2);document.getElementById('robust').textContent=JSON.stringify(d.robust_candidates,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v100C: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
