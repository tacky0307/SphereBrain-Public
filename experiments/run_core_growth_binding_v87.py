from __future__ import annotations

import hashlib
import json
import socket
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
from semantic_bridge_homeostasis import SemanticBridgeHomeostasis
from semantic_bridge_delayed_selective import DelayedSelectiveConsolidation
import run_core_growth_binding_v82c as v82c
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86

HOST = "127.0.0.1"
START_PORT = 5138
OUT = ROOT / "data" / "core_growth_binding_v87" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v85.CHECKPOINTS
SEEDS = v84.SEEDS
DOMAINS = v84.DOMAINS
STABILITY_WINDOW = v85.STABILITY_WINDOW
MODES = ["primary", "homeostatic", "delayed_selective", "delayed_selective_assist"]


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


def action_shared_edges(brain: SphereBrain, domain) -> set[tuple[int, int]]:
    left = v84.action_item(domain.left_subject, domain.left_action)
    right = v84.action_item(domain.right_subject, domain.right_action)
    return v83.action_signature(brain, left)["edges"] & v83.action_signature(brain, right)["edges"]


def usage_percentile(brain: SphereBrain, node: int) -> float:
    values = np.asarray(brain.node_usage, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= float(values[int(node)])))


def hub_score(brain: SphereBrain, edge: tuple[int, int]) -> float:
    a, b = edge
    usage = max(usage_percentile(brain, a), usage_percentile(brain, b))
    degree = max(int(np.count_nonzero(brain.adjacency[a])), int(np.count_nonzero(brain.adjacency[b])))
    degree_ref = max(1, int(brain.neighbors_per_node) * 2)
    degree_score = min(1.0, degree / degree_ref)
    return min(1.0, 0.7 * usage + 0.3 * degree_score)


def selective_candidates(brain: SphereBrain, domain, baseline: set[tuple[int, int]], shared_context: bool) -> list[dict]:
    new_shared = action_shared_edges(brain, domain) - baseline
    ctx = v84.domain_context_signature(brain, domain) if shared_context else v84.control_context_signature(brain, domain)
    context_nodes = set(int(x) for x in ctx["nodes"])
    rows = []
    for edge in sorted(new_shared):
        edge = edge_tuple(edge)
        endpoints = {edge[0], edge[1]}
        hops = v82c.min_hops(brain, endpoints, context_nodes, cap=4)
        if endpoints & context_nodes:
            context_score = 1.0
        elif hops == 1:
            context_score = 0.72
        elif hops == 2:
            context_score = 0.35
        else:
            context_score = 0.0
        if context_score <= 0:
            continue
        rows.append({
            "edge": list(edge),
            "context_score": context_score,
            "hub_score": hub_score(brain, edge),
        })
    return rows


def classify(records: list[dict]) -> str:
    return v86.classify(records)


def run_trial(domain, seed: int, mode: str) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "delayed_selective_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = v84.make_baseline(experimental, domain)
    ctrl_baseline = v84.make_baseline(control, domain)
    exp_episodes = v84.episodes_for(domain, True)
    ctrl_episodes = v84.episodes_for(domain, False)

    old_exp = SemanticBridgeHomeostasis()
    old_ctrl = SemanticBridgeHomeostasis()
    delayed_exp = DelayedSelectiveConsolidation()
    delayed_ctrl = DelayedSelectiveConsolidation()

    records = []
    protection_events = []
    observed_candidate_edges: set[tuple[int, int]] = set()
    protected_edges: set[tuple[int, int]] = set()
    hub_protected_edges: set[tuple[int, int]] = set()

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v85.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            row["mode"] = mode
            records.append(row)

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

            if mode == "homeostatic":
                exp_edges = v86.intrinsic_context_candidate_edges(experimental, domain, exp_baseline["direct"], shared_context=True)
                ctrl_edges = v86.intrinsic_context_candidate_edges(control, domain, ctrl_baseline["direct"], shared_context=False)
                result = old_exp.observe(experimental, exp_edges)
                old_ctrl.observe(control, ctrl_edges)
                for item in result.get("protected", []):
                    edge = edge_tuple(item["edge"])
                    protected_edges.add(edge)
                    if hub_score(experimental, edge) >= 0.88:
                        hub_protected_edges.add(edge)
                    protection_events.append({"cycle": cycle + 1, "edge": list(edge), "hub_score": hub_score(experimental, edge)})

            elif mode in {"delayed_selective", "delayed_selective_assist"}:
                exp_rows = selective_candidates(experimental, domain, exp_baseline["direct"], True)
                ctrl_rows = selective_candidates(control, domain, ctrl_baseline["direct"], False)
                observed_candidate_edges.update(edge_tuple(x["edge"]) for x in exp_rows)
                result = delayed_exp.observe(experimental, exp_rows)
                delayed_ctrl.observe(control, ctrl_rows)
                for item in result.get("protected", []):
                    edge = edge_tuple(item["edge"])
                    protected_edges.add(edge)
                    h = hub_score(experimental, edge)
                    if h >= 0.88:
                        hub_protected_edges.add(edge)
                    protection_events.append({
                        "cycle": cycle + 1,
                        "edge": list(edge),
                        "score": float(item.get("score", 0.0)),
                        "hub_score": h,
                    })

    cls = classify(records)
    active = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    final = records[-1]
    return {
        "domain": domain.name,
        "seed": int(seed),
        "mode": mode,
        "classification": cls,
        "ever_bridge": bool(active),
        "stable": cls == "stable",
        "never": cls == "never",
        "first_bridge_cycle": active[0] if active else None,
        "final_direct_similarity": float(final["experimental"]["direct"]["edge_similarity"]),
        "final_candidate_count": int(final["context_linked_candidate_count"]),
        "observed_candidate_edge_count": len(observed_candidate_edges),
        "protected_edge_count": len(protected_edges),
        "hub_protected_count": len(hub_protected_edges),
        "protection_events": protection_events,
        "records": records,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    classes: dict[str, int] = {}
    for row in subset:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    protected = sum(x["protected_edge_count"] for x in subset)
    hub_protected = sum(x["hub_protected_count"] for x in subset)
    return {
        "mode": mode,
        "trials": len(subset),
        "ever_bridge": sum(1 for x in subset if x["ever_bridge"]),
        "stable": sum(1 for x in subset if x["stable"]),
        "never": sum(1 for x in subset if x["never"]),
        "class_counts": classes,
        "protected_edges": protected,
        "hub_protected_edges": hub_protected,
        "hub_protection_rate": hub_protected / protected if protected else 0.0,
        "mean_final_direct_similarity": sum(x["final_direct_similarity"] for x in subset) / len(subset),
    }


def paired(rows: list[dict], mode: str) -> dict:
    p = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == "primary"}
    t = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == mode}
    deltas = [t[k]["final_direct_similarity"] - p[k]["final_direct_similarity"] for k in p]
    return {
        "stable_gain": sum(1 for k in p if t[k]["stable"] and not p[k]["stable"]),
        "stable_loss": sum(1 for k in p if p[k]["stable"] and not t[k]["stable"]),
        "never_gain": sum(1 for k in p if t[k]["never"] and not p[k]["never"]),
        "never_loss": sum(1 for k in p if p[k]["never"] and not t[k]["never"]),
        "max_similarity_delta": max(deltas),
        "mean_similarity_delta": sum(deltas) / len(deltas),
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed, mode) for mode in MODES for domain in DOMAINS for seed in SEEDS]
    summaries = {m: summarize(trials, m) for m in MODES}
    delayed_pair = paired(trials, "delayed_selective")
    assist_pair = paired(trials, "delayed_selective_assist")

    primary = summaries["primary"]
    old = summaries["homeostatic"]
    delayed = summaries["delayed_selective"]
    assist = summaries["delayed_selective_assist"]

    delayed_distinctness = delayed_pair["max_similarity_delta"] <= 0.12
    assist_distinctness = assist_pair["max_similarity_delta"] <= 0.12
    delayed_safe = delayed["never"] <= primary["never"] and delayed_distinctness
    assist_safe = assist["never"] <= primary["never"] and assist_distinctness
    delayed_improves = delayed["stable"] > primary["stable"] and delayed_safe
    assist_improves = assist["stable"] > primary["stable"] and assist_safe

    old_hub = old["hub_protection_rate"]
    delayed_hub_reduced = delayed["hub_protection_rate"] < old_hub if old["protected_edges"] else True
    assist_hub_reduced = assist["hub_protection_rate"] < old_hub if old["protected_edges"] else True

    representative = next((x for x in trials if x["mode"] == "delayed_selective" and x["stable"]), next(x for x in trials if x["mode"] == "delayed_selective"))
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "delayed_selective_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    overall_pass = (
        (delayed_improves or assist_improves)
        and (delayed_hub_reduced or assist_hub_reduced)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if delayed_improves and assist_improves:
        winner = "delayed_selective_assist" if assist["stable"] > delayed["stable"] else "delayed_selective"
    elif delayed_improves:
        winner = "delayed_selective"
    elif assist_improves:
        winner = "delayed_selective_assist"
    else:
        winner = "primary"

    if overall_pass:
        verdict = "delayed_selective_consolidation_improves_semantic_bridge_stability_while_reducing_early_hub_protection"
        readiness = "semantic_delayed_selective_consolidation_candidate"
        next_step = "stress_test_winner_under_context_switching_and_contradictory_semantic_episodes"
    else:
        verdict = "delayed_selective_consolidation_does_not_yet_outperform_primary_core_safely"
        readiness = "semantic_delayed_selective_consolidation_not_yet_validated"
        next_step = "attribute_selection_threshold_failures_before_primary_core_change"

    payload = {
        "experiment": "Core Growth Binding v87 — Delayed Selective Consolidation",
        "contract": {
            "primary_core_modified": False,
            "modes": MODES,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "checkpoints": CHECKPOINTS,
            "free_formation_until": 10,
            "observation_only_until": 20,
            "semantic_answer_labels_used": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "primary_stable": primary["stable"],
            "old_homeostatic_stable": old["stable"],
            "delayed_stable": delayed["stable"],
            "delayed_assist_stable": assist["stable"],
            "primary_never": primary["never"],
            "old_homeostatic_never": old["never"],
            "delayed_never": delayed["never"],
            "delayed_assist_never": assist["never"],
            "old_hub_protection_rate": old_hub,
            "delayed_hub_protection_rate": delayed["hub_protection_rate"],
            "delayed_assist_hub_protection_rate": assist["hub_protection_rate"],
            "delayed_hub_reduced": delayed_hub_reduced,
            "assist_hub_reduced": assist_hub_reduced,
            "delayed_distinctness_safe": delayed_distinctness,
            "assist_distinctness_safe": assist_distinctness,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "delayed_selective_pass": overall_pass,
            "winner": winner,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "paired": {"delayed_selective": delayed_pair, "delayed_selective_assist": assist_pair},
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in trials],
    }
    (OUT / "latest_binding_v87.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v87</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v87：Delayed Selective Consolidation</h1><p class="lead">0〜10 Episodeは自由形成、10〜20は観察のみ、20以降に再出現・持続性・Context近接・低Hub性を満たすEdgeだけを保護する。Primary / v86 Homeostasis / v87 / v87+Assistを比較する。</p><section class="panel"><div class="controls"><button id="run">遅延選別定着を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode比較</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Primary Stable',s.primary_stable),metric('v86 Stable',s.old_homeostatic_stable,'warn'),metric('Delayed Stable',s.delayed_stable,s.delayed_stable>s.primary_stable?'good':'warn'),metric('Delayed+Assist Stable',s.delayed_assist_stable,s.delayed_assist_stable>s.primary_stable?'good':'warn'),metric('Primary Never',s.primary_never),metric('Delayed Never',s.delayed_never,s.delayed_never<=s.primary_never?'good':'warn'),metric('Delayed+Assist Never',s.delayed_assist_never,s.delayed_assist_never<=s.primary_never?'good':'warn'),metric('v86 Hub保護率',(s.old_hub_protection_rate*100).toFixed(1)+'%','warn'),metric('Delayed Hub保護率',(s.delayed_hub_protection_rate*100).toFixed(1)+'%',s.delayed_hub_reduced?'good':'warn'),metric('Assist Hub保護率',(s.delayed_assist_hub_protection_rate*100).toFixed(1)+'%',s.assist_hub_reduced?'good':'warn'),metric('Distinctness',yn(s.delayed_distinctness_safe),s.delayed_distinctness_safe?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変更',s.brain_file_unchanged?'good':'warn'),metric('v87 PASS',yn(s.delayed_selective_pass),s.delayed_selective_pass?'good':'warn'),metric('Winner',s.winner),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify({mode_summaries:d.mode_summaries,paired:d.paired},null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"v87: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
