from __future__ import annotations

import hashlib
import json
import socket
import statistics
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
import run_core_growth_binding_v99 as v99

HOST = "127.0.0.1"
START_PORT = 5157
OUT = ROOT / "data" / "core_growth_binding_v99b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS = v99.SEEDS


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


def edge_tuple(edge) -> tuple[int, int]:
    a, b = [int(x) for x in edge]
    return tuple(sorted((a, b)))


def quorum_edges(snapshot: dict) -> set[tuple[int, int]]:
    return {
        edge_tuple(x["edge"])
        for x in snapshot["consensus"]["edges"]
        if bool(x["quorum"])
    }


def module_overlap_stats(edges: set[tuple[int, int]], modules: list[dict]) -> dict:
    membership = v99.node_to_module(modules)
    touched_modules: set[int] = set()
    cross_edges: set[tuple[int, int]] = set()
    inside_edges = 0
    outside_edges = 0
    for a, b in edges:
        ma = membership.get(int(a))
        mb = membership.get(int(b))
        if ma is not None:
            touched_modules.add(ma)
        if mb is not None:
            touched_modules.add(mb)
        if ma is not None and mb is not None:
            if ma != mb:
                cross_edges.add((a, b))
            else:
                inside_edges += 1
        elif ma is None and mb is None:
            outside_edges += 1
    known_edges = max(1, inside_edges + len(cross_edges))
    return {
        "touched_module_count": len(touched_modules),
        "cross_module_edge_count": len(cross_edges),
        "within_existing_module_edge_count": inside_edges,
        "outside_existing_module_edge_count": outside_edges,
        "cross_module_fraction": len(cross_edges) / known_edges,
        "existing_module_overlap_fraction": (inside_edges + len(cross_edges)) / max(1, len(edges)),
    }


def run_detailed(seed: int, *, guided: bool) -> dict:
    brain = v99.v84.clean_primary_seed(seed)
    history: dict[tuple[int, int], int] = {}

    for _ in range(v99.FORMATION_CYCLES):
        v99.v98.train_one_cycle(brain)

    before = v99.structural_snapshot(brain, history)
    pre_quorum = quorum_edges(before)
    guide_nodes = v99.guided_nodes_from_consensus(before)

    if guided:
        trigger_edges = v99.train_guided(brain, v99.TRIGGER_EPISODES, guide_nodes)
        mode = "trigger_consensus_guided"
    else:
        trigger_edges = v99.train_normal(brain, v99.TRIGGER_EPISODES)
        mode = "trigger"

    immediate = v99.structural_snapshot(brain, history)
    immediate_quorum = quorum_edges(immediate)
    immediate_new = immediate_quorum - pre_quorum
    overlap = module_overlap_stats(trigger_edges, before["modules"])

    aftermath = []
    for cycle in range(1, v99.AFTERMATH_CYCLES + 1):
        v99.v98.train_one_cycle(brain)
        snap = v99.structural_snapshot(brain, history)
        q = quorum_edges(snap)
        aftermath.append({
            "cycle": cycle,
            "new_quorum_edges": len(q - pre_quorum),
            "immediate_new_retained": len(immediate_new & q),
            "mean_support_sources": snap["consensus"]["mean_support_sources"],
            "mean_consensus_strength": snap["consensus"]["mean_consensus_strength"],
        })

    final = v99.structural_snapshot(brain, history)
    final_quorum = quorum_edges(final)
    persistent_edges = immediate_new & final_quorum

    return {
        "seed": int(seed),
        "mode": mode,
        **overlap,
        "trigger_edge_count": len(trigger_edges),
        "before_module_count": len(before["modules"]),
        "before_quorum_edge_count": len(pre_quorum),
        "immediate_new_quorum": len(immediate_new),
        "persistent_new_quorum": len(persistent_edges),
        "persistent_fraction": len(persistent_edges) / max(1, len(immediate_new)),
        "final_new_quorum": len(final_quorum - pre_quorum),
        "support_delta": final["consensus"]["mean_support_sources"] - before["consensus"]["mean_support_sources"],
        "consensus_delta": final["consensus"]["mean_consensus_strength"] - before["consensus"]["mean_consensus_strength"],
        "module_delta": len(final["modules"]) - len(before["modules"]),
        "aftermath": aftermath,
        "brain": brain,
    }


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(x[key]) for x in rows) if rows else 0.0


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)

    matched = {seed: v99.run_mode(seed, "matched_novel") for seed in SEEDS}
    native = {seed: run_detailed(seed, guided=False) for seed in SEEDS}
    guided = {seed: run_detailed(seed, guided=True) for seed in SEEDS}

    for seed in SEEDS:
        n = native[seed]
        m = matched[seed]
        g = guided[seed]
        matched_persistent = int(m.get("persistent_new_quorum", 0))
        n["matched_persistent_new_quorum"] = matched_persistent
        n["native_success"] = int(n["persistent_new_quorum"]) > matched_persistent
        g["native_persistent_new_quorum"] = int(n["persistent_new_quorum"])
        g["guided_improvement"] = int(g["persistent_new_quorum"]) > int(n["persistent_new_quorum"])

    native_rows = list(native.values())
    success = [x for x in native_rows if x["native_success"]]
    failure = [x for x in native_rows if not x["native_success"]]
    guided_rows = list(guided.values())
    guided_improved = [x for x in guided_rows if x["guided_improvement"]]

    crossing_gap = mean(success, "touched_module_count") - mean(failure, "touched_module_count")
    cross_edge_gap = mean(success, "cross_module_edge_count") - mean(failure, "cross_module_edge_count")
    persistence_gap = mean(success, "persistent_fraction") - mean(failure, "persistent_fraction")
    support_gap = mean(success, "support_delta") - mean(failure, "support_delta")
    consensus_gap = mean(success, "consensus_delta") - mean(failure, "consensus_delta")

    if crossing_gap > 0 and cross_edge_gap > 0:
        top_attribution = "trigger_crossed_more_existing_modules"
    elif persistence_gap > 0:
        top_attribution = "post_trigger_quorum_persistence"
    elif support_gap > 0:
        top_attribution = "distributed_support_growth"
    else:
        top_attribution = "no_single_structural_attribution"

    structural_attribution_signal = (
        len(success) >= 1
        and len(failure) >= 1
        and (crossing_gap >= 1.0 or cross_edge_gap >= 1.0 or persistence_gap >= 0.20)
    )

    OUT.mkdir(parents=True, exist_ok=True)
    representative = native_rows[0]
    temp = OUT / "reorganization_attribution_roundtrip.json"
    before_weights = representative["brain"].weights.tolist()
    representative["brain"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    attribution_pass = saveload_equal and production_unchanged and native_present and len(native_rows) == len(SEEDS)

    if structural_attribution_signal:
        readiness = "reorganization_success_has_structural_precursors"
        verdict = "successful_triggered_reorganization_differs_from_failure_in_observed_core_path_or_persistence_structure"
        next_step = "replicate_the_identified_structural_precursor_without_using_human_semantic_labels_to_select_trigger_content"
    else:
        readiness = "reorganization_success_precursor_not_yet_isolated"
        verdict = "the_single_native_success_is_not_yet_explained_by_a_consistent_measured_module_crossing_or_persistence_difference"
        next_step = "increase_seed_count_and_capture_trigger_stage_level_paths_before_modifying_core_or_trigger_design"

    payload = {
        "experiment": "Core Growth Binding v99B — Reorganization Success/Failure Attribution",
        "contract": {
            "primary_core_modified": False,
            "v99_trigger_episodes_unchanged": True,
            "v99_matched_novel_unchanged": True,
            "v99_formation_cycles_unchanged": True,
            "v99_aftermath_cycles_unchanged": True,
            "semantic_labels_used_for_success_attribution": False,
            "success_definition": "native_trigger_persistent_new_quorum_greater_than_same_seed_matched_novel",
            "production_brain_json_saved": False,
        },
        "summary": {
            "seed_count": len(SEEDS),
            "native_success_count": len(success),
            "native_failure_count": len(failure),
            "guided_improvement_count": len(guided_improved),
            "success_touched_modules": mean(success, "touched_module_count"),
            "failure_touched_modules": mean(failure, "touched_module_count"),
            "module_crossing_gap": crossing_gap,
            "success_cross_module_edges": mean(success, "cross_module_edge_count"),
            "failure_cross_module_edges": mean(failure, "cross_module_edge_count"),
            "cross_edge_gap": cross_edge_gap,
            "success_persistent_fraction": mean(success, "persistent_fraction"),
            "failure_persistent_fraction": mean(failure, "persistent_fraction"),
            "persistence_gap": persistence_gap,
            "support_delta_gap": support_gap,
            "consensus_delta_gap": consensus_gap,
            "top_attribution": top_attribution,
            "structural_attribution_signal": structural_attribution_signal,
            "saveload_equal": saveload_equal,
            "brain_file_unchanged": production_unchanged,
            "primary_native_learning_present": native_present,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "native": [{k: v for k, v in x.items() if k != "brain"} for x in native_rows],
        "guided": [{k: v for k, v in x.items() if k != "brain"} for x in guided_rows],
        "matched": [matched[seed] for seed in SEEDS],
    }
    (OUT / "latest_binding_v99b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v99B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v99B：Reorganization Success/Failure Attribution</h1><p class="lead">v99でNative Triggerが成功したSeedと失敗したSeedを、Triggerが実際に横断した既存Module・cross-module Edge・新規Quorumの持続で比較する。意味ラベルは成功判定にもAttributionにも使わない。</p><section class="panel"><div class="controls"><button id="run">成功/失敗を分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Native Seed別</h2><pre id="native" class="raw">未実行</pre></section><section class="panel"><h2>Guided Seed別</h2><pre id="guided" class="raw">未実行</pre></section><section class="panel"><h2>Raw JSON</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Seed',s.seed_count),metric('Native success',`${s.native_success_count}/${s.seed_count}`,s.native_success_count?'good':'warn'),metric('Guided improvement',`${s.guided_improvement_count}/${s.seed_count}`),metric('Success touched modules',s.success_touched_modules.toFixed(2)),metric('Failure touched modules',s.failure_touched_modules.toFixed(2)),metric('Module crossing gap',s.module_crossing_gap.toFixed(2),s.module_crossing_gap>0?'good':'warn'),metric('Cross-edge gap',s.cross_edge_gap.toFixed(2),s.cross_edge_gap>0?'good':'warn'),metric('Persistence gap',s.persistence_gap.toFixed(3),s.persistence_gap>0?'good':'warn'),metric('Support delta gap',s.support_delta_gap.toFixed(3)),metric('Consensus delta gap',s.consensus_delta_gap.toFixed(3)),metric('Top attribution',s.top_attribution),metric('Structural signal',yn(s.structural_attribution_signal),s.structural_attribution_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('native').textContent=JSON.stringify(d.native,null,2);document.getElementById('guided').textContent=JSON.stringify(d.guided,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v99B: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
