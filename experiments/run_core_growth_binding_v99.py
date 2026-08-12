from __future__ import annotations

import hashlib
import json
import socket
import statistics
import sys
import threading
import webbrowser
from collections import Counter
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
import run_core_growth_binding_v98_hotfix as hotfix
import run_core_growth_binding_v98b as v98b

v98 = hotfix.v98

HOST = "127.0.0.1"
START_PORT = 5156
OUT = ROOT / "data" / "core_growth_binding_v99" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS = v98.SEEDS
MODES = ["sham", "matched_novel", "trigger", "trigger_consensus_guided"]
FORMATION_CYCLES = 6
AFTERMATH_CYCLES = 6
QUORUM_SOURCES = 4

# A new subject crosses several already experienced contexts. These are ordinary
# episode inputs; no semantic class label is used by the observer.
TRIGGER_EPISODES = [
    v83.EpisodeSpec("旅人", "空港", "歩く"),
    v83.EpisodeSpec("旅人", "港", "歩く"),
    v83.EpisodeSpec("旅人", "道", "歩く"),
    v83.EpisodeSpec("旅人", "工房", "歩く"),
]

# Same number of novel episodes, but each episode has a separate subject/context/action
# so it does not deliberately create one repeated cross-context trajectory.
MATCHED_NOVEL_EPISODES = [
    v83.EpisodeSpec("時計", "棚", "時を刻む"),
    v83.EpisodeSpec("ランプ", "机", "光る"),
    v83.EpisodeSpec("カップ", "食卓", "置かれる"),
    v83.EpisodeSpec("本", "本棚", "閉じる"),
]


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


def entity_signatures(brain: SphereBrain) -> list[dict]:
    return [v98.entity_signature(brain, world) for world in v98.WORLDS]


def reused_edges(signatures: list[dict], minimum_support: int = 3) -> dict[tuple[int, int], int]:
    counts: Counter = Counter()
    for sig in signatures:
        for a, b in sig["edges"]:
            counts[tuple(sorted((int(a), int(b))))] += 1
    return {edge: int(count) for edge, count in counts.items() if count >= minimum_support}


def modules_from_signatures(signatures: list[dict]) -> list[dict]:
    edges = set(reused_edges(signatures))
    return v98b.graph_components(edges)


def node_to_module(modules: list[dict]) -> dict[int, int]:
    out = {}
    for index, module in enumerate(modules):
        for node in module["nodes"]:
            out[int(node)] = int(index)
    return out


def proximal_to_signature(brain: SphereBrain, edge: tuple[int, int], sig: dict) -> bool:
    a, b = edge
    nodes = set(int(x) for x in sig["nodes"])
    if a in nodes or b in nodes:
        return True
    for endpoint in (a, b):
        for nxt in np.flatnonzero(brain.adjacency[endpoint]).tolist():
            if int(nxt) in nodes:
                return True
    return False


def generic_consensus(brain: SphereBrain, signatures: list[dict], history: dict[tuple[int, int], int]) -> dict:
    reused = reused_edges(signatures)
    rows = []
    for edge, exact_support in sorted(reused.items()):
        proximal_support = sum(1 for sig in signatures if proximal_to_signature(brain, edge, sig))
        prior = int(history.get(edge, 0))
        source_ratio = proximal_support / max(1, len(signatures))
        exact_ratio = exact_support / max(1, len(signatures))
        temporal_ratio = min(1.0, prior / 3.0)
        strength = 0.55 * source_ratio + 0.20 * exact_ratio + 0.25 * temporal_ratio
        rows.append({
            "edge": [int(edge[0]), int(edge[1])],
            "exact_support": int(exact_support),
            "proximal_support": int(proximal_support),
            "quorum": bool(proximal_support >= QUORUM_SOURCES),
            "consensus_strength": float(strength),
        })
        history[edge] = prior + 1
    quorum = [row for row in rows if row["quorum"]]
    return {
        "edge_count": len(rows),
        "quorum_edge_count": len(quorum),
        "mean_support_sources": statistics.mean([r["proximal_support"] for r in rows]) if rows else 0.0,
        "mean_consensus_strength": statistics.mean([r["consensus_strength"] for r in rows]) if rows else 0.0,
        "max_consensus_strength": max([r["consensus_strength"] for r in rows], default=0.0),
        "edges": rows,
    }


def structural_snapshot(brain: SphereBrain, history: dict[tuple[int, int], int]) -> dict:
    signatures = entity_signatures(brain)
    modules = modules_from_signatures(signatures)
    topo = v98b.topology_from_signatures(signatures)
    consensus = generic_consensus(brain, signatures, history)
    return {
        "signatures": signatures,
        "modules": modules,
        "topology": topo,
        "consensus": consensus,
    }


def trace_edges(result: dict) -> set[tuple[int, int]]:
    edges = set()
    for stage in ("subject", "place_relation", "context", "action_relation", "action"):
        for a, b in result[stage].traversed_edges:
            edges.add(tuple(sorted((int(a), int(b)))))
    return edges


def train_normal(brain: SphereBrain, episodes: list[v83.EpisodeSpec]) -> set[tuple[int, int]]:
    traversed = set()
    for spec in episodes:
        traversed.update(trace_edges(v83.episode_experience(brain, spec, learn=True)))
    return traversed


def guided_nodes_from_consensus(snapshot: dict, limit: int = 10) -> list[int]:
    rows = [x for x in snapshot["consensus"]["edges"] if x["quorum"]]
    rows.sort(key=lambda x: (-float(x["consensus_strength"]), -int(x["proximal_support"]), x["edge"]))
    nodes = []
    for row in rows:
        for node in row["edge"]:
            node = int(node)
            if node not in nodes:
                nodes.append(node)
            if len(nodes) >= limit:
                return nodes
    return nodes


def guided_episode_experience(brain: SphereBrain, spec: v83.EpisodeSpec, guide_nodes: list[int]) -> dict:
    # Same staged episode as v83. The only extra signal is a weak, label-blind
    # context made from high-consensus nodes observed before the trigger.
    noise = 0.004
    guide = list(guide_nodes[:10])

    subject_sources = v98.v82.component_nodes(brain, "role:subject", "subject", 2) + v98.v82.component_nodes(brain, "entity", spec.subject, 3)
    subject = brain.propagate(subject_sources, steps=8, threshold=0.18, noise=noise, learn=True, context_nodes=guide)

    place_relation_sources = v98.v82.component_nodes(brain, "role:relation", "relation", 2) + v98.v82.component_nodes(brain, "relation", "場所", 3)
    place_context = list(dict.fromkeys(v98.v82.context_tail(subject) + guide))
    place_relation = brain.propagate(place_relation_sources, steps=8, threshold=0.18, noise=noise, learn=True, context_nodes=place_context)

    context_sources = v98.v82.component_nodes(brain, "role:content", "content", 2) + v98.v82.component_nodes(brain, "content", spec.context, 3)
    context_ctx = list(dict.fromkeys(v98.v82.context_tail(place_relation) + guide))
    context = brain.propagate(context_sources, steps=10, threshold=0.18, noise=noise, learn=True, context_nodes=context_ctx)

    action_relation_sources = v98.v82.component_nodes(brain, "role:relation", "relation", 2) + v98.v82.component_nodes(brain, "relation", "動作", 3)
    action_relation_ctx = list(dict.fromkeys(v98.v82.context_tail(context) + guide))
    action_relation = brain.propagate(action_relation_sources, steps=8, threshold=0.18, noise=noise, learn=True, context_nodes=action_relation_ctx)

    action_sources = v98.v82.component_nodes(brain, "role:content", "content", 2) + v98.v82.component_nodes(brain, "content", spec.action, 3)
    action_ctx = list(dict.fromkeys(v98.v82.context_tail(action_relation) + guide))
    action = brain.propagate(action_sources, steps=10, threshold=0.18, noise=noise, learn=True, context_nodes=action_ctx)

    return {
        "subject": subject,
        "place_relation": place_relation,
        "context": context,
        "action_relation": action_relation,
        "action": action,
    }


def train_guided(brain: SphereBrain, episodes: list[v83.EpisodeSpec], guide_nodes: list[int]) -> set[tuple[int, int]]:
    traversed = set()
    for spec in episodes:
        traversed.update(trace_edges(guided_episode_experience(brain, spec, guide_nodes)))
    return traversed


def crossing_edges(edges: set[tuple[int, int]], modules_before: list[dict]) -> set[tuple[int, int]]:
    membership = node_to_module(modules_before)
    out = set()
    for a, b in edges:
        ma = membership.get(int(a))
        mb = membership.get(int(b))
        if ma is not None and mb is not None and ma != mb:
            out.add(tuple(sorted((int(a), int(b)))))
    return out


def module_pair_count(edges: set[tuple[int, int]], modules_before: list[dict]) -> int:
    membership = node_to_module(modules_before)
    pairs = set()
    for a, b in edges:
        ma = membership.get(int(a))
        mb = membership.get(int(b))
        if ma is not None and mb is not None and ma != mb:
            pairs.add(tuple(sorted((ma, mb))))
    return len(pairs)


def run_mode(seed: int, mode: str) -> dict:
    brain = v84.clean_primary_seed(seed)
    history: dict[tuple[int, int], int] = {}

    # Phase 1: form the existing world.
    for _ in range(FORMATION_CYCLES):
        v98.train_one_cycle(brain)
    before = structural_snapshot(brain, history)
    pre_quorum_edges = {tuple(x["edge"]) for x in before["consensus"]["edges"] if x["quorum"]}
    guide_nodes = guided_nodes_from_consensus(before)

    # Phase 2: one matched intervention block.
    if mode == "sham":
        trigger_edges = set()
        v98.train_one_cycle(brain)
    elif mode == "matched_novel":
        trigger_edges = train_normal(brain, MATCHED_NOVEL_EPISODES)
    elif mode == "trigger":
        trigger_edges = train_normal(brain, TRIGGER_EPISODES)
    elif mode == "trigger_consensus_guided":
        trigger_edges = train_guided(brain, TRIGGER_EPISODES, guide_nodes)
    else:
        raise ValueError(mode)

    immediate = structural_snapshot(brain, history)
    immediate_crossing = crossing_edges(trigger_edges, before["modules"])
    immediate_pairs = module_pair_count(trigger_edges, before["modules"])

    # Phase 3: aftermath uses ordinary world experience only for all modes.
    aftermath_rows = []
    for cycle in range(1, AFTERMATH_CYCLES + 1):
        v98.train_one_cycle(brain)
        snap = structural_snapshot(brain, history)
        quorum_edges = {tuple(x["edge"]) for x in snap["consensus"]["edges"] if x["quorum"]}
        aftermath_rows.append({
            "cycle": cycle,
            "module_count": len(snap["modules"]),
            "reused_edge_count": snap["topology"]["reused_edge_count"],
            "quorum_edge_count": snap["consensus"]["quorum_edge_count"],
            "mean_support_sources": snap["consensus"]["mean_support_sources"],
            "mean_consensus_strength": snap["consensus"]["mean_consensus_strength"],
            "pre_quorum_retained": len(pre_quorum_edges & quorum_edges),
            "new_quorum_edges": len(quorum_edges - pre_quorum_edges),
        })

    final = structural_snapshot(brain, history)
    final_quorum_edges = {tuple(x["edge"]) for x in final["consensus"]["edges"] if x["quorum"]}
    immediate_quorum_edges = {tuple(x["edge"]) for x in immediate["consensus"]["edges"] if x["quorum"]}

    persistent_new_quorum = len((immediate_quorum_edges - pre_quorum_edges) & final_quorum_edges)
    new_quorum_final = len(final_quorum_edges - pre_quorum_edges)
    module_delta = len(final["modules"]) - len(before["modules"])
    reuse_delta = final["topology"]["reused_edge_count"] - before["topology"]["reused_edge_count"]
    support_delta = final["consensus"]["mean_support_sources"] - before["consensus"]["mean_support_sources"]
    consensus_delta = final["consensus"]["mean_consensus_strength"] - before["consensus"]["mean_consensus_strength"]
    quorum_delta = final["consensus"]["quorum_edge_count"] - before["consensus"]["quorum_edge_count"]

    return {
        "seed": int(seed),
        "mode": mode,
        "before": {
            "module_count": len(before["modules"]),
            "reused_edge_count": before["topology"]["reused_edge_count"],
            "quorum_edge_count": before["consensus"]["quorum_edge_count"],
            "mean_support_sources": before["consensus"]["mean_support_sources"],
            "mean_consensus_strength": before["consensus"]["mean_consensus_strength"],
        },
        "immediate": {
            "module_count": len(immediate["modules"]),
            "reused_edge_count": immediate["topology"]["reused_edge_count"],
            "quorum_edge_count": immediate["consensus"]["quorum_edge_count"],
            "mean_support_sources": immediate["consensus"]["mean_support_sources"],
            "mean_consensus_strength": immediate["consensus"]["mean_consensus_strength"],
            "cross_module_trigger_edges": len(immediate_crossing),
            "crossed_module_pairs": immediate_pairs,
        },
        "final": {
            "module_count": len(final["modules"]),
            "reused_edge_count": final["topology"]["reused_edge_count"],
            "quorum_edge_count": final["consensus"]["quorum_edge_count"],
            "mean_support_sources": final["consensus"]["mean_support_sources"],
            "mean_consensus_strength": final["consensus"]["mean_consensus_strength"],
        },
        "module_delta": int(module_delta),
        "reuse_delta": int(reuse_delta),
        "support_delta": float(support_delta),
        "consensus_delta": float(consensus_delta),
        "quorum_delta": int(quorum_delta),
        "persistent_new_quorum": int(persistent_new_quorum),
        "new_quorum_final": int(new_quorum_final),
        "aftermath": aftermath_rows,
        "guide_node_count": len(guide_nodes),
        "brain": brain,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    def mean(key):
        vals = [float(x[key]) for x in subset]
        return statistics.mean(vals) if vals else 0.0
    return {
        "mode": mode,
        "trials": len(subset),
        "mean_module_delta": mean("module_delta"),
        "mean_reuse_delta": mean("reuse_delta"),
        "mean_support_delta": mean("support_delta"),
        "mean_consensus_delta": mean("consensus_delta"),
        "mean_quorum_delta": mean("quorum_delta"),
        "mean_persistent_new_quorum": mean("persistent_new_quorum"),
        "mean_new_quorum_final": mean("new_quorum_final"),
        "mean_cross_module_trigger_edges": statistics.mean([x["immediate"]["cross_module_trigger_edges"] for x in subset]) if subset else 0.0,
        "mean_crossed_module_pairs": statistics.mean([x["immediate"]["crossed_module_pairs"] for x in subset]) if subset else 0.0,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    rows = [run_mode(seed, mode) for seed in SEEDS for mode in MODES]
    summaries = {mode: summarize(rows, mode) for mode in MODES}

    sham = summaries["sham"]
    novel = summaries["matched_novel"]
    trigger = summaries["trigger"]
    guided = summaries["trigger_consensus_guided"]

    trigger_specific_quorum = trigger["mean_new_quorum_final"] - novel["mean_new_quorum_final"]
    trigger_specific_support = trigger["mean_support_delta"] - novel["mean_support_delta"]
    trigger_specific_consensus = trigger["mean_consensus_delta"] - novel["mean_consensus_delta"]
    persistence_margin = trigger["mean_persistent_new_quorum"] - novel["mean_persistent_new_quorum"]
    guided_gain = guided["mean_new_quorum_final"] - trigger["mean_new_quorum_final"]

    positive_trigger_seeds = 0
    positive_guided_seeds = 0
    for seed in SEEDS:
        n = next(x for x in rows if x["seed"] == seed and x["mode"] == "matched_novel")
        t = next(x for x in rows if x["seed"] == seed and x["mode"] == "trigger")
        g = next(x for x in rows if x["seed"] == seed and x["mode"] == "trigger_consensus_guided")
        if t["new_quorum_final"] > n["new_quorum_final"]:
            positive_trigger_seeds += 1
        if g["new_quorum_final"] > t["new_quorum_final"]:
            positive_guided_seeds += 1

    reorganization_signal = (
        trigger["mean_crossed_module_pairs"] > novel["mean_crossed_module_pairs"]
        and trigger_specific_quorum > 0.0
        and persistence_margin >= 0.0
        and positive_trigger_seeds >= 3
    )
    guided_signal = guided_gain > 0.0 and positive_guided_seeds >= 3

    representative = next(x for x in rows if x["mode"] == "trigger")
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "structural_reorganization_roundtrip.json"
    before_weights = representative["brain"].weights.tolist()
    representative["brain"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    measurement_pass = saveload_equal and production_unchanged and native_present

    if reorganization_signal:
        readiness = "experience_triggered_structural_reorganization_candidate"
        verdict = "a_matched_cross_context_experience_block_creates_more_persistent_new_distributed_support_than_equal_novel_input"
        next_step = "increase_trigger_variety_and_checkpoint_resolution_then_test_whether_reorganization_changes_future_core_outputs"
    else:
        readiness = "experience_triggered_structural_reorganization_not_yet_established"
        verdict = "cross_context_trigger_experience_does_not_yet_produce_specific_persistent_reorganization_over_matched_novel_input"
        next_step = "inspect_trigger_edge_entry_points_module_pair_coverage_and_consensus_timing_before_modifying_core"

    payload = {
        "experiment": "Core Growth Binding v99 — Structural Reorganization Trigger Microscope",
        "contract": {
            "primary_core_modified": False,
            "semantic_labels_used_for_reorganization_judgment": False,
            "v98_world_used_as_preexisting_experience_substrate": True,
            "formation_cycles": FORMATION_CYCLES,
            "aftermath_cycles": AFTERMATH_CYCLES,
            "modes": MODES,
            "matched_novel_episode_count": len(MATCHED_NOVEL_EPISODES),
            "trigger_episode_count": len(TRIGGER_EPISODES),
            "distributed_consensus_definition": "reused internal edge supported proximally by independent entity signatures with temporal recurrence",
            "consensus_guidance_uses_pretrigger_structure_only": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "seed_count": len(SEEDS),
            "sham": sham,
            "matched_novel": novel,
            "trigger": trigger,
            "trigger_consensus_guided": guided,
            "trigger_specific_quorum_margin": float(trigger_specific_quorum),
            "trigger_specific_support_margin": float(trigger_specific_support),
            "trigger_specific_consensus_margin": float(trigger_specific_consensus),
            "persistence_margin": float(persistence_margin),
            "guided_gain": float(guided_gain),
            "positive_trigger_seeds": int(positive_trigger_seeds),
            "positive_guided_seeds": int(positive_guided_seeds),
            "reorganization_signal": bool(reorganization_signal),
            "guided_signal": bool(guided_signal),
            "measurement_pass": bool(measurement_pass),
            "saveload_equal": bool(saveload_equal),
            "brain_file_unchanged": bool(production_unchanged),
            "primary_native_learning_present": bool(native_present),
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "trials": [{k: v for k, v in row.items() if k != "brain"} for row in rows],
    }
    (OUT / "latest_binding_v99.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v99</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v99：Structural Reorganization Trigger Microscope</h1><p class="lead">既存の局所構造が形成されたCoreへ、新しい具体的経験を与えた瞬間とその後を観察する。Matched novel入力と比較し、Module間接続・再利用・分散合意・Quorum・Persistenceを意味ラベルなしで追跡する。</p><section class="panel"><div class="controls"><button id="run">Reorganizationを観察</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>条件別</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="trials" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Seed',s.seed_count),metric('Trigger Quorum margin',s.trigger_specific_quorum_margin.toFixed(2),s.trigger_specific_quorum_margin>0?'good':'warn'),metric('Support margin',s.trigger_specific_support_margin.toFixed(3),s.trigger_specific_support_margin>0?'good':'warn'),metric('Consensus margin',s.trigger_specific_consensus_margin.toFixed(3),s.trigger_specific_consensus_margin>0?'good':'warn'),metric('Persistence margin',s.persistence_margin.toFixed(2),s.persistence_margin>=0?'good':'warn'),metric('Trigger positive',`${s.positive_trigger_seeds}/${s.seed_count}`),metric('Guided gain',s.guided_gain.toFixed(2),s.guided_gain>0?'good':'blue'),metric('Guided positive',`${s.positive_guided_seeds}/${s.seed_count}`),metric('Reorganization signal',yn(s.reorganization_signal),s.reorganization_signal?'good':'warn'),metric('Guided signal',yn(s.guided_signal),s.guided_signal?'good':'blue'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify({sham:s.sham,matched_novel:s.matched_novel,trigger:s.trigger,trigger_consensus_guided:s.trigger_consensus_guided},null,2);document.getElementById('trials').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v99: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
