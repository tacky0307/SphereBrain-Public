from __future__ import annotations

import hashlib
import json
import math
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
import run_core_growth_binding_v89 as v89
import run_core_growth_binding_v93 as v93
import run_core_growth_binding_v94 as v94
import run_core_growth_binding_v95 as v95

HOST = "127.0.0.1"
START_PORT = 5150
OUT = ROOT / "data" / "core_growth_binding_v95b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v95.CHECKPOINTS
SEEDS = v94.SEEDS
DOMAINS = v94.DOMAINS
STABILITY_WINDOW = v95.STABILITY_WINDOW
MODES = ["sham", "consensus", "matched_random", "hub_matched"]
ABLATION_FACTOR = 0.50


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


def domain_by_name(name: str):
    return next(x for x in DOMAINS if x.name == name)


def stable_from_records(records: list[dict]) -> bool:
    by_cycle = {int(r["cycle"]): bool(r["direct_success_now"]) for r in records}
    return all(by_cycle.get(c, False) for c in STABILITY_WINDOW)


def screen_stable_trials() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        for seed in SEEDS:
            trial = v94.v93b.run_trial(domain, seed)
            if trial.get("ever_bridge") and trial.get("stable"):
                rows.append({
                    "domain": domain.name,
                    "seed": int(seed),
                    "first_bridge_cycle": int(trial["first_bridge_cycle"]),
                })
    return rows


def usage_percentile(brain: SphereBrain, node: int) -> float:
    vals = np.asarray(brain.node_usage, dtype=float)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals <= float(vals[int(node)])))


def edge_features(brain: SphereBrain, edge: tuple[int, int]) -> dict:
    a, b = edge
    degree = max(int(np.count_nonzero(brain.adjacency[a])), int(np.count_nonzero(brain.adjacency[b])))
    usage = max(usage_percentile(brain, a), usage_percentile(brain, b))
    weight = float(brain.weights[a, b])
    hub = float(v95.v93.v87.hub_score(brain, edge)) if hasattr(v95.v93, "v87") else float(0.7 * usage + 0.3 * min(1.0, degree / max(1, brain.neighbors_per_node * 2)))
    return {"weight": weight, "degree": degree, "usage": usage, "hub": hub}


def local_signature_edges(brain: SphereBrain, domain) -> set[tuple[int, int]]:
    items = [
        v89.action_item(domain.left_subject, domain.left_action),
        v89.action_item(domain.right_subject, domain.right_action),
        v89.context_item(domain.left_subject, domain.shared_context),
        v89.context_item(domain.right_subject, domain.shared_context),
        v89.action_item(domain.transfer_left_subject, domain.left_action),
        v89.action_item(domain.transfer_right_subject, domain.right_action),
    ]
    out: set[tuple[int, int]] = set()
    for item in items:
        sig = v83.action_signature(brain, item)
        out.update(edge_tuple(e) for e in sig["edges"])
    return out


def consensus_targets(snapshot: dict) -> list[tuple[int, int]]:
    ranked = v95.rank_consensus_edges(snapshot)
    quorum = [x for x in ranked if bool(x.get("quorum"))]
    pool = quorum if quorum else ranked
    return [x["edge_tuple"] for x in pool]


def normalized_distance(a: dict, b: dict, *, hub_only: bool = False) -> float:
    if hub_only:
        return abs(float(a["hub"]) - float(b["hub"]))
    return (
        2.0 * abs(float(a["weight"]) - float(b["weight"]))
        + 0.7 * abs(float(a["usage"]) - float(b["usage"]))
        + 0.04 * abs(float(a["degree"]) - float(b["degree"]))
        + 0.5 * abs(float(a["hub"]) - float(b["hub"]))
    )


def match_edges(
    brain: SphereBrain,
    target_edges: list[tuple[int, int]],
    local_pool: set[tuple[int, int]],
    *,
    hub_only: bool,
) -> list[tuple[int, int]]:
    target_set = set(target_edges)
    candidates = sorted(e for e in local_pool if e not in target_set)
    chosen: list[tuple[int, int]] = []
    available = set(candidates)

    for target in target_edges:
        if not available:
            break
        tf = edge_features(brain, target)
        best = min(
            available,
            key=lambda e: (
                normalized_distance(tf, edge_features(brain, e), hub_only=hub_only),
                e,
            ),
        )
        chosen.append(best)
        available.remove(best)
    return chosen


def apply_ablation(brain: SphereBrain, edges: list[tuple[int, int]]) -> list[dict]:
    events = []
    for a, b in edges:
        before = float(brain.weights[a, b])
        after = max(0.0, before * ABLATION_FACTOR)
        brain.weights[a, b] = after
        brain.weights[b, a] = after
        f = edge_features(brain, (a, b))
        events.append({
            "edge": [a, b],
            "before": before,
            "after": after,
            "removed_weight": before - after,
            "degree": f["degree"],
            "usage": f["usage"],
            "hub": f["hub"],
        })
    return events


def run_trial(spec: dict, mode: str) -> dict:
    domain = domain_by_name(spec["domain"])
    seed = int(spec["seed"])
    target_cycle = int(spec["first_bridge_cycle"])

    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)

    history: dict[tuple[int, int], int] = {}
    records = []
    consensus_rows = []
    selected_edges: list[tuple[int, int]] = []
    target_edges: list[tuple[int, int]] = []
    events = []
    match_coverage = 1.0
    ablated = False

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            records.append(row)
            snap = v93.consensus_snapshot(experimental, domain, experimental, control, exp_baseline, cycle, history)
            consensus_rows.append(snap)

            if not ablated and cycle == target_cycle:
                target_edges = consensus_targets(snap)
                local_pool = local_signature_edges(experimental, domain)
                if mode == "consensus":
                    selected_edges = list(target_edges)
                elif mode == "matched_random":
                    selected_edges = match_edges(experimental, target_edges, local_pool, hub_only=False)
                elif mode == "hub_matched":
                    selected_edges = match_edges(experimental, target_edges, local_pool, hub_only=True)
                else:
                    selected_edges = []
                match_coverage = (
                    len(selected_edges) / len(target_edges)
                    if target_edges and mode not in {"sham", "consensus"}
                    else 1.0
                )
                events = apply_ablation(experimental, selected_edges) if mode != "sham" else []
                ablated = True

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    stable_after = stable_from_records(records)
    first_idx = next((i for i, r in enumerate(records) if int(r["cycle"]) == target_cycle), None)
    first_consensus = float(consensus_rows[first_idx]["mean_consensus_strength"]) if first_idx is not None else 0.0
    final_consensus = float(consensus_rows[-1]["mean_consensus_strength"]) if consensus_rows else 0.0
    first_quorum = int(consensus_rows[first_idx]["quorum_edge_count"]) if first_idx is not None else 0
    final_quorum = int(consensus_rows[-1]["quorum_edge_count"]) if consensus_rows else 0

    return {
        "domain": domain.name,
        "seed": seed,
        "mode": mode,
        "target_cycle": target_cycle,
        "stable_after": stable_after,
        "target_edge_count": len(target_edges),
        "selected_edge_count": len(selected_edges),
        "match_coverage": float(match_coverage),
        "removed_weight_total": sum(float(x["removed_weight"]) for x in events),
        "mean_selected_weight": statistics.mean([float(x["before"]) for x in events]) if events else 0.0,
        "mean_selected_degree": statistics.mean([float(x["degree"]) for x in events]) if events else 0.0,
        "mean_selected_usage": statistics.mean([float(x["usage"]) for x in events]) if events else 0.0,
        "mean_selected_hub": statistics.mean([float(x["hub"]) for x in events]) if events else 0.0,
        "events": events,
        "consensus_retention": final_consensus / first_consensus if first_consensus > 1e-12 else 0.0,
        "quorum_retention": final_quorum / first_quorum if first_quorum > 0 else (1.0 if final_quorum == 0 else 0.0),
        "records": records,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    stable = sum(1 for x in subset if x["stable_after"])
    return {
        "mode": mode,
        "trials": len(subset),
        "stable": stable,
        "stable_rate": stable / len(subset) if subset else 0.0,
        "collapse_rate": 1.0 - (stable / len(subset) if subset else 0.0),
        "mean_target_edges": statistics.mean([x["target_edge_count"] for x in subset]) if subset else 0.0,
        "mean_selected_edges": statistics.mean([x["selected_edge_count"] for x in subset]) if subset else 0.0,
        "mean_match_coverage": statistics.mean([x["match_coverage"] for x in subset]) if subset else 0.0,
        "mean_removed_weight_total": statistics.mean([x["removed_weight_total"] for x in subset]) if subset else 0.0,
        "mean_selected_weight": statistics.mean([x["mean_selected_weight"] for x in subset]) if subset else 0.0,
        "mean_selected_degree": statistics.mean([x["mean_selected_degree"] for x in subset]) if subset else 0.0,
        "mean_selected_usage": statistics.mean([x["mean_selected_usage"] for x in subset]) if subset else 0.0,
        "mean_selected_hub": statistics.mean([x["mean_selected_hub"] for x in subset]) if subset else 0.0,
        "mean_consensus_retention": statistics.mean([x["consensus_retention"] for x in subset]) if subset else 0.0,
        "mean_quorum_retention": statistics.mean([x["quorum_retention"] for x in subset]) if subset else 0.0,
    }


def domain_summary(rows: list[dict]) -> list[dict]:
    out = []
    for domain in DOMAINS:
        drows = [x for x in rows if x["domain"] == domain.name]
        if not drows:
            continue
        item = {"domain": domain.name}
        for mode in MODES:
            m = [x for x in drows if x["mode"] == mode]
            item[f"{mode}_trials"] = len(m)
            item[f"{mode}_stable"] = sum(1 for x in m if x["stable_after"])
        out.append(item)
    return out


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    stable_specs = screen_stable_trials()
    rows = [run_trial(spec, mode) for spec in stable_specs for mode in MODES]
    summaries = {m: summarize(rows, m) for m in MODES}

    sham = summaries["sham"]
    consensus = summaries["consensus"]
    random = summaries["matched_random"]
    hub = summaries["hub_matched"]

    consensus_effect = sham["stable_rate"] - consensus["stable_rate"]
    random_effect = sham["stable_rate"] - random["stable_rate"]
    hub_effect = sham["stable_rate"] - hub["stable_rate"]
    specificity_margin_random = consensus_effect - random_effect
    specificity_margin_hub = consensus_effect - hub_effect

    weight_ratio_random = (
        random["mean_removed_weight_total"] / consensus["mean_removed_weight_total"]
        if consensus["mean_removed_weight_total"] > 1e-12 else 0.0
    )
    weight_ratio_hub = (
        hub["mean_removed_weight_total"] / consensus["mean_removed_weight_total"]
        if consensus["mean_removed_weight_total"] > 1e-12 else 0.0
    )
    matching_quality = (
        random["mean_match_coverage"] >= 0.80
        and hub["mean_match_coverage"] >= 0.80
        and 0.70 <= weight_ratio_random <= 1.30
        and 0.70 <= weight_ratio_hub <= 1.30
    )

    domains = domain_summary(rows)
    comparable_domains = [x for x in domains if x.get("consensus_trials", 0) > 0]
    specificity_domains = [
        x for x in comparable_domains
        if x.get("consensus_stable", 0) < x.get("matched_random_stable", 0)
        or x.get("consensus_stable", 0) < x.get("hub_matched_stable", 0)
    ]
    domain_specificity_rate = len(specificity_domains) / len(comparable_domains) if comparable_domains else 0.0

    representative = next((x for x in rows if x["mode"] == "consensus"), rows[0] if rows else None)
    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    if representative is not None:
        temp = OUT / "consensus_specificity_roundtrip.json"
        before_weights = representative["experimental"].weights.tolist()
        representative["experimental"].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = representative is not None and hasattr(representative["experimental"], "learning_state")

    specificity_signal = (
        len(stable_specs) >= 10
        and sham["stable_rate"] >= 0.80
        and consensus_effect >= 0.20
        and specificity_margin_random >= 0.15
        and specificity_margin_hub >= 0.15
        and matching_quality
        and domain_specificity_rate >= 0.50
    )
    measurement_pass = (
        len(stable_specs) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if specificity_signal:
        readiness = "consensus_ablation_specificity_candidate"
        verdict = "consensus_support_edge_ablation_harms_stable_bridge_survival_more_than_weight_degree_usage_or_hub_matched_local_controls"
        next_step = "replicate_specificity_control_with_independent_seed_holdout_before_consensus_stability_mechanism_integration"
    else:
        readiness = "consensus_ablation_specificity_not_established"
        verdict = "consensus_edge_ablation_is_not_yet_more_harmful_than_matched_local_edge_controls"
        next_step = "inspect_matching_quality_and_edge_role_overlap_before_interpreting_consensus_as_causal"

    payload = {
        "experiment": "Core Growth Binding v95B — Consensus Ablation Specificity Control",
        "contract": {
            "primary_core_modified": False,
            "stable_trial_screening_uses_v94_definition": True,
            "consensus_definition_changed_from_v93": False,
            "semantic_answer_labels_used_for_targeting": False,
            "ablation_factor": ABLATION_FACTOR,
            "control_pool": "same_local_action_context_transfer_signature_edges_excluding_consensus_targets",
            "modes": MODES,
            "production_brain_json_saved": False,
        },
        "summary": {
            "stable_cohort": len(stable_specs),
            "sham_stable_rate": sham["stable_rate"],
            "consensus_stable_rate": consensus["stable_rate"],
            "matched_random_stable_rate": random["stable_rate"],
            "hub_matched_stable_rate": hub["stable_rate"],
            "consensus_effect": consensus_effect,
            "matched_random_effect": random_effect,
            "hub_matched_effect": hub_effect,
            "specificity_margin_vs_random": specificity_margin_random,
            "specificity_margin_vs_hub": specificity_margin_hub,
            "random_match_coverage": random["mean_match_coverage"],
            "hub_match_coverage": hub["mean_match_coverage"],
            "random_removed_weight_ratio": weight_ratio_random,
            "hub_removed_weight_ratio": weight_ratio_hub,
            "matching_quality": matching_quality,
            "specificity_domains": len(specificity_domains),
            "comparable_domains": len(comparable_domains),
            "domain_specificity_rate": domain_specificity_rate,
            "consensus_specificity_signal": specificity_signal,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "domain_rows": domains,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in rows],
    }
    (OUT / "latest_binding_v95b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v95B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v95B：Consensus Ablation Specificity Control</h1><p class="lead">v94 Stable cohortに対し、Consensus支持Edgeと、同じ局所構造からweight/degree/usageまたはHub度を合わせた非Consensus Edgeを同量Ablationし、Consensus特異的な因果効果かを検証する。</p><section class="panel"><div class="controls"><button id="run">Specificityを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode別</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function pct(v){return `${(100*Number(v||0)).toFixed(1)}%`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Stable cohort',s.stable_cohort),metric('Sham Stable',pct(s.sham_stable_rate)),metric('Consensus Stable',pct(s.consensus_stable_rate),'warn'),metric('Matched random Stable',pct(s.matched_random_stable_rate)),metric('Hub matched Stable',pct(s.hub_matched_stable_rate)),metric('Consensus effect',pct(s.consensus_effect)),metric('vs Random margin',pct(s.specificity_margin_vs_random),s.specificity_margin_vs_random>=.15?'good':'warn'),metric('vs Hub margin',pct(s.specificity_margin_vs_hub),s.specificity_margin_vs_hub>=.15?'good':'warn'),metric('Random match coverage',pct(s.random_match_coverage),s.random_match_coverage>=.8?'good':'warn'),metric('Hub match coverage',pct(s.hub_match_coverage),s.hub_match_coverage>=.8?'good':'warn'),metric('Weight match R/H',`${Number(s.random_removed_weight_ratio).toFixed(2)} / ${Number(s.hub_removed_weight_ratio).toFixed(2)}`,s.matching_quality?'good':'warn'),metric('Domain specificity',`${s.specificity_domains}/${s.comparable_domains}`),metric('Specificity signal',yn(s.consensus_specificity_signal),s.consensus_specificity_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"v95B running: {url}")
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
