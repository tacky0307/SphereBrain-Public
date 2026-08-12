from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
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
import run_core_growth_binding_v98_hotfix as hotfix

v98 = hotfix.v98
HOST = "127.0.0.1"
START_PORT = 5155
OUT = ROOT / "data" / "core_growth_binding_v98b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS = v98.SEEDS
CHECKPOINTS = [0, 12]


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


def reuse_edge_counts(signatures: list[dict]) -> Counter:
    counts: Counter = Counter()
    for sig in signatures:
        for edge in sig["edges"]:
            counts[tuple(sorted((int(edge[0]), int(edge[1]))))] += 1
    return counts


def graph_components(edges: set[tuple[int, int]]) -> list[dict]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen: set[int] = set()
    rows = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        nodes: set[int] = set()
        while stack:
            cur = stack.pop()
            nodes.add(cur)
            for nxt in adjacency.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        component_edges = {e for e in edges if e[0] in nodes and e[1] in nodes}
        rows.append({
            "node_count": len(nodes),
            "edge_count": len(component_edges),
            "nodes": sorted(nodes),
        })
    return sorted(rows, key=lambda x: (-x["edge_count"], -x["node_count"], x["nodes"][0] if x["nodes"] else -1))


def normalized_entropy(values: list[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0]
    total = sum(vals)
    if total <= 0 or len(vals) <= 1:
        return 0.0
    probs = [v / total for v in vals]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(probs))


def topology_from_signatures(signatures: list[dict]) -> dict:
    counts = reuse_edge_counts(signatures)
    reused = {e: c for e, c in counts.items() if c >= 3}
    broad = {e: c for e, c in counts.items() if c >= 5}
    edges = set(reused)
    components = graph_components(edges)

    node_degree: Counter = Counter()
    node_support: Counter = Counter()
    for (a, b), support in reused.items():
        node_degree[a] += 1
        node_degree[b] += 1
        node_support[a] += int(support)
        node_support[b] += int(support)

    total_endpoints = max(1, 2 * len(edges))
    top_endpoint_hits = sum(v for _, v in node_degree.most_common(5))
    top5_node_concentration = top_endpoint_hits / total_endpoints
    giant_edges = components[0]["edge_count"] if components else 0
    giant_ratio = giant_edges / len(edges) if edges else 0.0
    local_modules = sum(1 for c in components if c["edge_count"] >= 3)
    mean_degree = statistics.mean(node_degree.values()) if node_degree else 0.0
    component_entropy = normalized_entropy([c["edge_count"] for c in components])
    broad_ratio = len(broad) / len(reused) if reused else 0.0

    return {
        "reused_edge_count": len(reused),
        "broad_reuse_edge_count": len(broad),
        "broad_reuse_ratio": float(broad_ratio),
        "component_count": len(components),
        "local_module_count": int(local_modules),
        "giant_component_edge_ratio": float(giant_ratio),
        "top5_node_concentration": float(top5_node_concentration),
        "mean_reused_node_degree": float(mean_degree),
        "component_entropy": float(component_entropy),
        "top_nodes": [
            {"node": int(node), "reused_degree": int(degree), "support_mass": int(node_support[node])}
            for node, degree in node_degree.most_common(12)
        ],
        "components": components[:20],
    }


def topology_probe(brain: SphereBrain) -> dict:
    signatures = [v98.entity_signature(brain, world) for world in v98.WORLDS]
    return topology_from_signatures(signatures)


def classify_change(first: dict, final: dict) -> str:
    reuse_gain = final["reused_edge_count"] - first["reused_edge_count"]
    giant_gain = final["giant_component_edge_ratio"] - first["giant_component_edge_ratio"]
    concentration_gain = final["top5_node_concentration"] - first["top5_node_concentration"]
    modules_gain = final["local_module_count"] - first["local_module_count"]
    entropy_gain = final["component_entropy"] - first["component_entropy"]

    if reuse_gain <= 0:
        return "no_reuse_growth"
    if giant_gain >= 0.18 and concentration_gain >= 0.08:
        return "hub_collapse"
    if modules_gain > 0 and giant_gain <= 0.10 and concentration_gain <= 0.06:
        return "localized_modular_growth"
    if entropy_gain > 0.08 and concentration_gain <= 0.04:
        return "diffuse_reuse"
    return "mixed_topology"


def run_seed(seed: int) -> dict:
    brain = v98.v84.clean_primary_seed(seed)
    first = topology_probe(brain)
    for _ in range(12):
        v98.train_one_cycle(brain)
    final = topology_probe(brain)
    return {
        "seed": int(seed),
        "first": first,
        "final": final,
        "reused_edge_gain": final["reused_edge_count"] - first["reused_edge_count"],
        "giant_ratio_gain": final["giant_component_edge_ratio"] - first["giant_component_edge_ratio"],
        "concentration_gain": final["top5_node_concentration"] - first["top5_node_concentration"],
        "module_gain": final["local_module_count"] - first["local_module_count"],
        "entropy_gain": final["component_entropy"] - first["component_entropy"],
        "broad_ratio_gain": final["broad_reuse_ratio"] - first["broad_reuse_ratio"],
        "classification": classify_change(first, final),
        "brain": brain,
    }


def mean(rows, key: str) -> float:
    vals = [float(x[key]) for x in rows]
    return statistics.mean(vals) if vals else 0.0


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    rows = [run_seed(seed) for seed in SEEDS]
    classes = Counter(x["classification"] for x in rows)
    dominant = classes.most_common(1)[0][0] if classes else None

    modular_seeds = sum(1 for x in rows if x["classification"] == "localized_modular_growth")
    hub_seeds = sum(1 for x in rows if x["classification"] == "hub_collapse")
    mixed_seeds = sum(1 for x in rows if x["classification"] == "mixed_topology")
    diffuse_seeds = sum(1 for x in rows if x["classification"] == "diffuse_reuse")

    topology_signal = len(rows) == len(SEEDS) and max(classes.values(), default=0) >= 3
    hub_dominance = hub_seeds >= 3
    modular_growth = modular_seeds >= 3

    representative = rows[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "reuse_topology_roundtrip.json"
    before_topology = topology_probe(representative["brain"])
    representative["brain"].save(temp)
    loaded = SphereBrain.load(temp)
    after_topology = topology_probe(loaded)
    saveload_equal = before_topology == after_topology
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    measurement_pass = saveload_equal and production_unchanged and native_present

    if modular_growth:
        readiness = "reuse_growth_is_predominantly_localized_and_modular"
        verdict = "experience_driven_reuse_growth_forms_multiple_local_topological_modules_more_often_than_a_single_hub_collapse"
        next_step = "test_whether_local_modules_are_recalled_across_novel_stimuli_without_using_human_semantic_labels"
    elif hub_dominance:
        readiness = "reuse_growth_is_predominantly_hub_concentrated"
        verdict = "experience_driven_reuse_growth_is_dominated_by_a_small_number_of_nodes_and_a_giant_reused_edge_component"
        next_step = "design_label_blind_anti_hub_competition_or_capacity_pressure_and_retest_emergent_structure"
    else:
        readiness = "reuse_growth_topology_is_mixed_or_diffuse"
        verdict = "reuse_edges_grow_consistently_but_their_topology_is_not_explained_by_one_dominant_modular_or_hub_pattern"
        next_step = "attribute_seed_level_topology_transitions_and_compare_early_vs_late_reuse_growth_before_modifying_core_learning"

    payload = {
        "experiment": "Core Growth Binding v98B — Reuse Structure Topology Attribution",
        "contract": {
            "primary_core_modified": False,
            "v98_worlds_unchanged": True,
            "v98_seeds_unchanged": True,
            "v98_training_cycles_unchanged": True,
            "semantic_labels_used_for_topology": False,
            "hidden_family_used_for_topology": False,
            "reused_edge_definition": "edge_present_in_at_least_3_entity_signatures",
            "production_brain_json_saved": False,
        },
        "summary": {
            "seed_count": len(rows),
            "mean_reused_edge_gain": mean(rows, "reused_edge_gain"),
            "mean_giant_ratio_gain": mean(rows, "giant_ratio_gain"),
            "mean_concentration_gain": mean(rows, "concentration_gain"),
            "mean_module_gain": mean(rows, "module_gain"),
            "mean_entropy_gain": mean(rows, "entropy_gain"),
            "mean_broad_ratio_gain": mean(rows, "broad_ratio_gain"),
            "localized_modular_seeds": modular_seeds,
            "hub_collapse_seeds": hub_seeds,
            "mixed_topology_seeds": mixed_seeds,
            "diffuse_reuse_seeds": diffuse_seeds,
            "classification_counts": dict(classes),
            "dominant_topology": dominant,
            "topology_signal": topology_signal,
            "saveload_equal": saveload_equal,
            "brain_file_unchanged": production_unchanged,
            "primary_native_learning_present": native_present,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "seeds": [
            {k: v for k, v in x.items() if k != "brain"}
            for x in rows
        ],
    }
    (OUT / "latest_binding_v98b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v98B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v98B：Reuse Structure Topology Attribution</h1><p class="lead">v98で全Seedに増えた再利用Edgeが、局所モジュール化・Hub集中・拡散のどれとして成長したかを、意味ラベルを使わずTopologyだけで観察する。Coreとv98の経験条件は変更しない。</p><section class="panel"><div class="controls"><button id="run">Topologyを観察</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Seed別</h2><pre id="seeds" class="raw">未実行</pre></section><section class="panel"><h2>Raw JSON</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Seed',s.seed_count),metric('Reuse Edge gain',s.mean_reused_edge_gain.toFixed(1),s.mean_reused_edge_gain>0?'good':'warn'),metric('Giant ratio gain',s.mean_giant_ratio_gain.toFixed(3),s.mean_giant_ratio_gain>0.18?'warn':'blue'),metric('Top5集中 gain',s.mean_concentration_gain.toFixed(3),s.mean_concentration_gain>0.08?'warn':'blue'),metric('Module gain',s.mean_module_gain.toFixed(2),s.mean_module_gain>0?'good':'blue'),metric('Component entropy gain',s.mean_entropy_gain.toFixed(3)),metric('Broad reuse gain',s.mean_broad_ratio_gain.toFixed(3)),metric('Modular',`${s.localized_modular_seeds}/${s.seed_count}`,s.localized_modular_seeds>=3?'good':'blue'),metric('Hub collapse',`${s.hub_collapse_seeds}/${s.seed_count}`,s.hub_collapse_seeds>=3?'warn':'blue'),metric('Mixed',`${s.mixed_topology_seeds}/${s.seed_count}`),metric('Diffuse',`${s.diffuse_reuse_seeds}/${s.seed_count}`),metric('Dominant topology',s.dominant_topology||'—'),metric('Topology signal',yn(s.topology_signal),s.topology_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('seeds').textContent=JSON.stringify(d.seeds.map(x=>({seed:x.seed,reused_edge_gain:x.reused_edge_gain,giant_ratio_gain:x.giant_ratio_gain,concentration_gain:x.concentration_gain,module_gain:x.module_gain,entropy_gain:x.entropy_gain,classification:x.classification})),null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v98B: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
