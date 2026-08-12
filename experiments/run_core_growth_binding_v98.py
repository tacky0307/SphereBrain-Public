from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
import sys
import threading
import webbrowser
from dataclasses import dataclass
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
START_PORT = 5154
OUT = ROOT / "data" / "core_growth_binding_v98" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = [0, 1, 3, 6, 12]
SEEDS = [42, 314, 2718, 8088, 12021]


@dataclass(frozen=True)
class EntityWorld:
    entity: str
    hidden_family: str
    episodes: tuple[v83.EpisodeSpec, ...]


# hidden_family is NEVER used by clustering or threshold selection.
# It exists only for the final human-side post-hoc interpretation.
WORLDS = [
    EntityWorld("鳥", "F1", (
        v83.EpisodeSpec("鳥", "空", "羽ばたく"),
        v83.EpisodeSpec("鳥", "森", "飛ぶ"),
    )),
    EntityWorld("飛行機", "F1", (
        v83.EpisodeSpec("飛行機", "空", "飛行する"),
        v83.EpisodeSpec("飛行機", "空港", "進む"),
    )),
    EntityWorld("蝶", "F1", (
        v83.EpisodeSpec("蝶", "花畑", "羽ばたく"),
        v83.EpisodeSpec("蝶", "空", "飛ぶ"),
    )),
    EntityWorld("ドローン", "F1", (
        v83.EpisodeSpec("ドローン", "空", "飛行する"),
        v83.EpisodeSpec("ドローン", "工場", "進む"),
    )),

    EntityWorld("魚", "F2", (
        v83.EpisodeSpec("魚", "海", "泳ぐ"),
        v83.EpisodeSpec("魚", "川", "移動する"),
    )),
    EntityWorld("船", "F2", (
        v83.EpisodeSpec("船", "海", "進む"),
        v83.EpisodeSpec("船", "港", "移動する"),
    )),
    EntityWorld("イルカ", "F2", (
        v83.EpisodeSpec("イルカ", "海", "泳ぐ"),
        v83.EpisodeSpec("イルカ", "湾", "跳ぶ"),
    )),
    EntityWorld("潜水艦", "F2", (
        v83.EpisodeSpec("潜水艦", "海", "進む"),
        v83.EpisodeSpec("潜水艦", "深海", "潜る"),
    )),

    EntityWorld("馬", "F3", (
        v83.EpisodeSpec("馬", "道", "走る"),
        v83.EpisodeSpec("馬", "草原", "移動する"),
    )),
    EntityWorld("車", "F3", (
        v83.EpisodeSpec("車", "道", "進む"),
        v83.EpisodeSpec("車", "駐車場", "移動する"),
    )),
    EntityWorld("鹿", "F3", (
        v83.EpisodeSpec("鹿", "森", "走る"),
        v83.EpisodeSpec("鹿", "道", "移動する"),
    )),
    EntityWorld("バス", "F3", (
        v83.EpisodeSpec("バス", "道", "進む"),
        v83.EpisodeSpec("バス", "駅", "移動する"),
    )),

    EntityWorld("ハンマー", "F4", (
        v83.EpisodeSpec("ハンマー", "工房", "叩く"),
        v83.EpisodeSpec("ハンマー", "工具箱", "置かれる"),
    )),
    EntityWorld("ドリル", "F4", (
        v83.EpisodeSpec("ドリル", "工房", "穴を開ける"),
        v83.EpisodeSpec("ドリル", "工具箱", "回る"),
    )),
    EntityWorld("のこぎり", "F4", (
        v83.EpisodeSpec("のこぎり", "工房", "切る"),
        v83.EpisodeSpec("のこぎり", "木材", "動く"),
    )),
    EntityWorld("ミキサー", "F4", (
        v83.EpisodeSpec("ミキサー", "台所", "混ぜる"),
        v83.EpisodeSpec("ミキサー", "収納庫", "回る"),
    )),
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


def episode_direct_nodes(brain: SphereBrain, spec: v83.EpisodeSpec) -> set[int]:
    nodes = set()
    parts = [
        ("role:subject", "subject", 2),
        ("entity", spec.subject, 3),
        ("role:relation", "relation", 2),
        ("relation", "場所", 3),
        ("role:content", "content", 2),
        ("content", spec.context, 3),
        ("relation", "動作", 3),
        ("content", spec.action, 3),
    ]
    for namespace, value, count in parts:
        nodes.update(int(x) for x in v82.component_nodes(brain, namespace, value, count))
    return nodes


def trace_signature(brain: SphereBrain, spec: v83.EpisodeSpec) -> dict:
    result = v83.episode_experience(brain, spec, learn=False)
    direct = episode_direct_nodes(brain, spec)
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    activation = np.zeros(brain.n_nodes, dtype=float)

    for stage in ("subject", "place_relation", "context", "action_relation", "action"):
        trace = result[stage]
        nodes.update(int(x) for x in trace.activated_nodes if int(x) not in direct)
        edges.update(
            tuple(sorted((int(a), int(b))))
            for a, b in trace.traversed_edges
            if int(a) not in direct and int(b) not in direct
        )
        arr = np.asarray(trace.final_activation, dtype=float)
        activation = np.maximum(activation, arr)

    for node in direct:
        activation[int(node)] = 0.0
    return {"nodes": nodes, "edges": edges, "activation": activation}


def entity_signature(brain: SphereBrain, world: EntityWorld) -> dict:
    parts = [trace_signature(brain, spec) for spec in world.episodes]
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    activation = np.zeros(brain.n_nodes, dtype=float)
    for part in parts:
        nodes.update(part["nodes"])
        edges.update(part["edges"])
        activation = np.maximum(activation, part["activation"])
    return {"nodes": nodes, "edges": edges, "activation": activation}


def similarity(left: dict, right: dict) -> float:
    node_sim = v82.jaccard(left["nodes"], right["nodes"])
    edge_sim = v82.jaccard(left["edges"], right["edges"])
    act_sim = v82b.weighted_similarity(left["activation"], right["activation"])
    return float(0.20 * node_sim + 0.45 * edge_sim + 0.35 * act_sim)


def similarity_matrix(signatures: list[dict]) -> np.ndarray:
    n = len(signatures)
    matrix = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = similarity(signatures[i], signatures[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def connected_components(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    n = matrix.shape[0]
    seen = set()
    groups = []
    for start in range(n):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in range(n):
                if nxt in seen or nxt == cur:
                    continue
                if float(matrix[cur, nxt]) >= threshold:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def cluster_objective(matrix: np.ndarray, groups: list[list[int]]) -> dict:
    within = []
    between = []
    n = matrix.shape[0]
    membership = {}
    for gi, group in enumerate(groups):
        for idx in group:
            membership[idx] = gi
    for i in range(n):
        for j in range(i + 1, n):
            if membership[i] == membership[j]:
                within.append(float(matrix[i, j]))
            else:
                between.append(float(matrix[i, j]))
    within_mean = statistics.mean(within) if within else 0.0
    between_mean = statistics.mean(between) if between else 0.0
    singleton_rate = sum(1 for g in groups if len(g) == 1) / max(1, len(groups))
    giant_fraction = max((len(g) for g in groups), default=0) / max(1, n)
    count_bonus = min(len(groups), max(2, int(round(math.sqrt(n))))) / max(1, n)
    objective = (
        within_mean - between_mean
        + 0.05 * count_bonus
        - 0.08 * singleton_rate
        - 0.06 * max(0.0, giant_fraction - 0.70)
    )
    return {
        "objective": float(objective),
        "within_similarity": float(within_mean),
        "between_similarity": float(between_mean),
        "separation": float(within_mean - between_mean),
        "singleton_rate": float(singleton_rate),
        "giant_fraction": float(giant_fraction),
    }


def discover_structure(matrix: np.ndarray) -> dict:
    offdiag = [float(matrix[i, j]) for i in range(matrix.shape[0]) for j in range(i + 1, matrix.shape[0])]
    if not offdiag:
        return {"threshold": 1.0, "groups": [[0]], **cluster_objective(matrix, [[0]])}
    candidates = sorted(set(float(np.quantile(offdiag, q)) for q in (0.50, 0.60, 0.70, 0.78, 0.84, 0.90)))
    rows = []
    for threshold in candidates:
        groups = connected_components(matrix, threshold)
        score = cluster_objective(matrix, groups)
        rows.append({"threshold": threshold, "groups": groups, **score})
    valid = [x for x in rows if 2 <= len(x["groups"]) <= max(8, matrix.shape[0] - 1)]
    chosen = max(valid or rows, key=lambda x: x["objective"])
    return {**chosen, "candidates": rows}


def structural_reuse(signatures: list[dict]) -> dict:
    edge_counts: dict[tuple[int, int], int] = {}
    node_counts: dict[int, int] = {}
    for sig in signatures:
        for edge in sig["edges"]:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        for node in sig["nodes"]:
            node_counts[node] = node_counts.get(node, 0) + 1
    reused_edges = {e: c for e, c in edge_counts.items() if c >= 3}
    broad_edges = {e: c for e, c in edge_counts.items() if c >= 5}
    reused_nodes = {n: c for n, c in node_counts.items() if c >= 4}
    return {
        "reused_edge_count": len(reused_edges),
        "broad_reuse_edge_count": len(broad_edges),
        "reused_node_count": len(reused_nodes),
        "top_reused_edges": [
            {"edge": list(e), "support": c}
            for e, c in sorted(reused_edges.items(), key=lambda x: (-x[1], x[0]))[:20]
        ],
    }


def posthoc_interpretation(groups: list[list[int]], matrix: np.ndarray) -> dict:
    # Human-side labels are used only AFTER label-blind structure discovery.
    labels = [w.hidden_family for w in WORLDS]
    purity_hits = 0
    group_rows = []
    for gi, group in enumerate(groups):
        counts: dict[str, int] = {}
        for idx in group:
            counts[labels[idx]] = counts.get(labels[idx], 0) + 1
        majority = max(counts.values()) if counts else 0
        purity_hits += majority
        group_rows.append({
            "cluster": gi,
            "entities": [WORLDS[idx].entity for idx in group],
            "posthoc_family_counts": counts,
        })
    same = []
    different = []
    for i in range(len(WORLDS)):
        for j in range(i + 1, len(WORLDS)):
            target = same if labels[i] == labels[j] else different
            target.append(float(matrix[i, j]))
    return {
        "posthoc_purity": purity_hits / len(WORLDS),
        "same_family_similarity": statistics.mean(same) if same else 0.0,
        "different_family_similarity": statistics.mean(different) if different else 0.0,
        "posthoc_family_separation": (statistics.mean(same) - statistics.mean(different)) if same and different else 0.0,
        "clusters": group_rows,
    }


def probe(brain: SphereBrain, cycle: int) -> dict:
    signatures = [entity_signature(brain, world) for world in WORLDS]
    matrix = similarity_matrix(signatures)
    discovered = discover_structure(matrix)
    reuse = structural_reuse(signatures)
    posthoc = posthoc_interpretation(discovered["groups"], matrix)
    return {
        "cycle": int(cycle),
        "cluster_count": len(discovered["groups"]),
        "threshold": float(discovered["threshold"]),
        "objective": float(discovered["objective"]),
        "within_similarity": float(discovered["within_similarity"]),
        "between_similarity": float(discovered["between_similarity"]),
        "separation": float(discovered["separation"]),
        "groups": discovered["groups"],
        "reuse": reuse,
        "posthoc": posthoc,
        "similarity_matrix": matrix.tolist(),
    }


def train_one_cycle(brain: SphereBrain) -> None:
    # Interleave the worlds so no family is trained as one contiguous block.
    max_len = max(len(w.episodes) for w in WORLDS)
    for episode_index in range(max_len):
        for world in WORLDS:
            if episode_index < len(world.episodes):
                v83.episode_experience(brain, world.episodes[episode_index], learn=True)


def run_seed(seed: int) -> dict:
    brain = v84.clean_primary_seed(seed)
    records = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            records.append(probe(brain, cycle))
        if cycle < max(CHECKPOINTS):
            train_one_cycle(brain)
    first = records[0]
    final = records[-1]
    return {
        "seed": int(seed),
        "records": records,
        "objective_gain": float(final["objective"] - first["objective"]),
        "separation_gain": float(final["separation"] - first["separation"]),
        "reused_edge_gain": int(final["reuse"]["reused_edge_count"] - first["reuse"]["reused_edge_count"]),
        "posthoc_purity_gain": float(final["posthoc"]["posthoc_purity"] - first["posthoc"]["posthoc_purity"]),
        "posthoc_family_separation_gain": float(final["posthoc"]["posthoc_family_separation"] - first["posthoc"]["posthoc_family_separation"]),
        "brain": brain,
    }


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    seed_results = [run_seed(seed) for seed in SEEDS]
    positive_objective = sum(1 for x in seed_results if x["objective_gain"] > 0)
    positive_reuse = sum(1 for x in seed_results if x["reused_edge_gain"] > 0)
    positive_separation = sum(1 for x in seed_results if x["separation_gain"] > 0)

    final_cluster_counts = [x["records"][-1]["cluster_count"] for x in seed_results]
    label_blind_signal = (
        positive_objective >= 3
        and positive_reuse >= 3
        and positive_separation >= 3
        and mean(x["objective_gain"] for x in seed_results) > 0.01
    )

    # Post-hoc interpretation is supportive only; it is not required for the label-blind PASS.
    posthoc_alignment = mean(x["posthoc_family_separation_gain"] for x in seed_results) > 0.0

    representative = seed_results[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "emergent_structure_roundtrip.json"
    before_probe = probe(representative["brain"], max(CHECKPOINTS))
    representative["brain"].save(temp)
    loaded = SphereBrain.load(temp)
    after_probe = probe(loaded, max(CHECKPOINTS))
    saveload_equal = (
        before_probe["groups"] == after_probe["groups"]
        and abs(before_probe["objective"] - after_probe["objective"]) < 1e-12
        and before_probe["reuse"]["reused_edge_count"] == after_probe["reuse"]["reused_edge_count"]
    )
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    measurement_pass = saveload_equal and production_unchanged and native_present
    if label_blind_signal:
        readiness = "label_blind_emergent_structural_organization_observed"
        verdict = "experience_increases_reusable_internal_structure_and_unsupervised_separation_before_human_semantic_labels_are_consulted"
        next_step = "repeat_with_nonlinguistic_numeric_stimulus_families_and_larger_mixed_experience_streams"
    else:
        readiness = "label_blind_emergent_structural_organization_not_yet_observed"
        verdict = "current_small_world_does_not_yet_show_consistent_experience_driven_unsupervised_structural_organization"
        next_step = "inspect_encoder_baseline_dominance_and_signature_definition_before_scaling_experience"

    payload = {
        "experiment": "Core Growth Binding v98 — Emergent Structural Meaning Microscope",
        "contract": {
            "primary_core_modified": False,
            "language_used_as_input_stimulus": True,
            "semantic_labels_used_for_clustering": False,
            "semantic_labels_used_for_threshold_selection": False,
            "hidden_family_used_only_after_structure_discovery": True,
            "direct_input_nodes_excluded_from_signatures": True,
            "seed_count": len(SEEDS),
            "entity_count": len(WORLDS),
            "episode_count_per_cycle": sum(len(w.episodes) for w in WORLDS),
            "checkpoints": CHECKPOINTS,
            "production_brain_json_saved": False,
        },
        "summary": {
            "seed_count": len(SEEDS),
            "mean_objective_gain": mean(x["objective_gain"] for x in seed_results),
            "mean_separation_gain": mean(x["separation_gain"] for x in seed_results),
            "mean_reused_edge_gain": mean(x["reused_edge_gain"] for x in seed_results),
            "positive_objective_seeds": positive_objective,
            "positive_separation_seeds": positive_separation,
            "positive_reuse_seeds": positive_reuse,
            "mean_posthoc_purity_gain": mean(x["posthoc_purity_gain"] for x in seed_results),
            "mean_posthoc_family_separation_gain": mean(x["posthoc_family_separation_gain"] for x in seed_results),
            "posthoc_alignment": posthoc_alignment,
            "final_cluster_counts": final_cluster_counts,
            "label_blind_structural_signal": label_blind_signal,
            "measurement_pass": measurement_pass,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "seeds": [
            {
                "seed": x["seed"],
                "objective_gain": x["objective_gain"],
                "separation_gain": x["separation_gain"],
                "reused_edge_gain": x["reused_edge_gain"],
                "posthoc_purity_gain": x["posthoc_purity_gain"],
                "posthoc_family_separation_gain": x["posthoc_family_separation_gain"],
                "records": x["records"],
            }
            for x in seed_results
        ],
    }
    (OUT / "latest_binding_v98.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v98</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v98：Emergent Structural Meaning Microscope</h1><p class="lead">言語は刺激として使うが、人間の意味ラベルを構造抽出に使わない。Core内部のNode・Edge・Activationだけから自然なまとまりを先に抽出し、その後にだけ人間側ラベルと照合する。</p><section class="panel"><div class="controls"><button id="run">構造を観察</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Seed別</h2><pre id="seeds" class="raw">未実行</pre></section><section class="panel"><h2>代表Seedの形成過程</h2><pre id="timeline" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Seed',s.seed_count),metric('Objective gain',s.mean_objective_gain.toFixed(4),s.mean_objective_gain>0?'good':'warn'),metric('Separation gain',s.mean_separation_gain.toFixed(4),s.mean_separation_gain>0?'good':'warn'),metric('Reuse Edge gain',s.mean_reused_edge_gain.toFixed(1),s.mean_reused_edge_gain>0?'good':'warn'),metric('Objective正方向',`${s.positive_objective_seeds}/${s.seed_count}`),metric('Separation正方向',`${s.positive_separation_seeds}/${s.seed_count}`),metric('Reuse正方向',`${s.positive_reuse_seeds}/${s.seed_count}`),metric('Label-blind signal',yn(s.label_blind_structural_signal),s.label_blind_structural_signal?'good':'warn'),metric('Post-hoc alignment',yn(s.posthoc_alignment),s.posthoc_alignment?'good':'blue'),metric('Purity gain',s.mean_posthoc_purity_gain.toFixed(3)),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('seeds').textContent=JSON.stringify(d.seeds.map(x=>({seed:x.seed,objective_gain:x.objective_gain,separation_gain:x.separation_gain,reused_edge_gain:x.reused_edge_gain,posthoc_purity_gain:x.posthoc_purity_gain,posthoc_family_separation_gain:x.posthoc_family_separation_gain})),null,2);document.getElementById('timeline').textContent=JSON.stringify(d.seeds[0]?.records||[],null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v98: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
