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
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v89 as v89
import run_core_growth_binding_v93 as v93
import run_core_growth_binding_v94 as v94
import run_core_growth_binding_v95 as v95
import run_core_growth_binding_v95b as v95b

HOST = "127.0.0.1"
START_PORT = 5151
OUT = ROOT / "data" / "core_growth_binding_v96" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v95.CHECKPOINTS
SEEDS = v94.SEEDS
DOMAINS = v94.DOMAINS
STABILITY_WINDOW = v95.STABILITY_WINDOW
MODES = ["sham", "matched_random_support", "consensus_support", "consensus_support_assist"]
SUPPORT_GAIN = 0.18


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


def screen_unstable_trials() -> list[dict]:
    rows = []
    for domain in DOMAINS:
        for seed in SEEDS:
            trial = v94.v93b.run_trial(domain, seed)
            if trial.get("ever_bridge") and not trial.get("stable"):
                rows.append({
                    "domain": domain.name,
                    "seed": int(seed),
                    "first_bridge_cycle": int(trial["first_bridge_cycle"]),
                    "rescue_eligible": int(trial["first_bridge_cycle"]) <= 20,
                })
    return rows


def stable_from_records(records: list[dict]) -> bool:
    by_cycle = {int(r["cycle"]): bool(r["direct_success_now"]) for r in records}
    return all(by_cycle.get(c, False) for c in STABILITY_WINDOW)


def post_support_persistent(records: list[dict], target_cycle: int) -> bool:
    post = [r for r in records if int(r["cycle"]) > target_cycle]
    return bool(post) and all(bool(r["direct_success_now"]) for r in post)


def consensus_targets(snapshot: dict) -> list[tuple[int, int]]:
    ranked = v95.rank_consensus_edges(snapshot)
    quorum = [x for x in ranked if bool(x.get("quorum"))]
    pool = quorum if quorum else ranked
    return [x["edge_tuple"] for x in pool]


def support_edges(brain: SphereBrain, edges: list[tuple[int, int]], total_gain_target: float | None = None) -> list[dict]:
    if not edges:
        return []
    raw = []
    for a, b in edges:
        before = float(brain.weights[a, b])
        desired = min(1.0, before * (1.0 + SUPPORT_GAIN))
        raw.append((a, b, before, desired - before))
    raw_total = sum(x[3] for x in raw)
    scale = 1.0
    if total_gain_target is not None and raw_total > 1e-12:
        scale = min(1.0, float(total_gain_target) / raw_total)
    events = []
    for a, b, before, raw_delta in raw:
        delta = raw_delta * scale
        after = min(1.0, before + delta)
        brain.weights[a, b] = after
        brain.weights[b, a] = after
        events.append({"edge": [a, b], "before": before, "after": after, "added_weight": after - before})
    return events


def run_trial(spec: dict, mode: str) -> dict:
    domain = domain_by_name(spec["domain"])
    seed = int(spec["seed"])
    target_cycle = int(spec["first_bridge_cycle"])
    rescue_eligible = bool(spec["rescue_eligible"])

    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "consensus_support_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)

    history: dict[tuple[int, int], int] = {}
    records = []
    consensus_rows = []
    target_edges: list[tuple[int, int]] = []
    selected_edges: list[tuple[int, int]] = []
    support_events: list[dict] = []
    added_weight_total = 0.0
    support_applied = False
    match_coverage = 1.0

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            records.append(row)
            snap = v93.consensus_snapshot(experimental, domain, experimental, control, exp_baseline, cycle, history)
            consensus_rows.append(snap)

            if not support_applied and cycle == target_cycle:
                target_edges = consensus_targets(snap)
                local_pool = v95b.local_signature_edges(experimental, domain)
                consensus_gain_estimate = sum(
                    min(1.0, float(experimental.weights[a, b]) * (1.0 + SUPPORT_GAIN)) - float(experimental.weights[a, b])
                    for a, b in target_edges
                )

                if mode in {"consensus_support", "consensus_support_assist"}:
                    selected_edges = list(target_edges)
                    support_events = support_edges(experimental, selected_edges)
                elif mode == "matched_random_support":
                    selected_edges = v95b.match_edges(
                        experimental, target_edges, local_pool, hub_only=False
                    )
                    match_coverage = len(selected_edges) / len(target_edges) if target_edges else 1.0
                    support_events = support_edges(
                        experimental, selected_edges, total_gain_target=consensus_gain_estimate
                    )
                else:
                    selected_edges = []
                    support_events = []

                added_weight_total = sum(float(x["added_weight"]) for x in support_events)
                support_applied = True

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    stable_after = stable_from_records(records)
    persistent_after = post_support_persistent(records, target_cycle)
    first_idx = next((i for i, r in enumerate(records) if int(r["cycle"]) == target_cycle), None)
    if first_idx is None:
        first_idx = 0

    first_consensus = float(consensus_rows[first_idx]["mean_consensus_strength"]) if consensus_rows else 0.0
    final_consensus = float(consensus_rows[-1]["mean_consensus_strength"]) if consensus_rows else 0.0
    first_support_sources = float(consensus_rows[first_idx]["mean_support_sources"]) if consensus_rows else 0.0
    final_support_sources = float(consensus_rows[-1]["mean_support_sources"]) if consensus_rows else 0.0
    first_quorum = int(consensus_rows[first_idx]["quorum_edge_count"]) if consensus_rows else 0
    final_quorum = int(consensus_rows[-1]["quorum_edge_count"]) if consensus_rows else 0

    return {
        "domain": domain.name,
        "seed": seed,
        "mode": mode,
        "target_cycle": target_cycle,
        "rescue_eligible": rescue_eligible,
        "stable_after": stable_after,
        "post_support_persistent": persistent_after,
        "target_edge_count": len(target_edges),
        "selected_edge_count": len(selected_edges),
        "match_coverage": float(match_coverage),
        "added_weight_total": float(added_weight_total),
        "support_events": support_events,
        "consensus_retention": final_consensus / first_consensus if first_consensus > 1e-12 else 0.0,
        "support_source_retention": final_support_sources / first_support_sources if first_support_sources > 1e-12 else 0.0,
        "quorum_retention": final_quorum / first_quorum if first_quorum > 0 else (1.0 if final_quorum == 0 else 0.0),
        "final_consensus": final_consensus,
        "final_support_sources": final_support_sources,
        "final_quorum_edges": final_quorum,
        "records": records,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    eligible = [x for x in subset if x["rescue_eligible"]]
    return {
        "mode": mode,
        "trials": len(subset),
        "rescue_eligible_trials": len(eligible),
        "stable_rescued": sum(1 for x in eligible if x["stable_after"]),
        "stable_rescue_rate": sum(1 for x in eligible if x["stable_after"]) / len(eligible) if eligible else 0.0,
        "persistent_trials": sum(1 for x in subset if x["post_support_persistent"]),
        "persistence_rate": sum(1 for x in subset if x["post_support_persistent"]) / len(subset) if subset else 0.0,
        "mean_target_edges": statistics.mean([x["target_edge_count"] for x in subset]) if subset else 0.0,
        "mean_selected_edges": statistics.mean([x["selected_edge_count"] for x in subset]) if subset else 0.0,
        "mean_match_coverage": statistics.mean([x["match_coverage"] for x in subset]) if subset else 0.0,
        "mean_added_weight": statistics.mean([x["added_weight_total"] for x in subset]) if subset else 0.0,
        "mean_consensus_retention": statistics.mean([x["consensus_retention"] for x in subset]) if subset else 0.0,
        "mean_support_source_retention": statistics.mean([x["support_source_retention"] for x in subset]) if subset else 0.0,
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
            e = [x for x in m if x["rescue_eligible"]]
            item[f"{mode}_eligible"] = len(e)
            item[f"{mode}_stable_rescued"] = sum(1 for x in e if x["stable_after"])
            item[f"{mode}_persistent"] = sum(1 for x in m if x["post_support_persistent"])
        out.append(item)
    return out


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    unstable_specs = screen_unstable_trials()
    rows = [run_trial(spec, mode) for spec in unstable_specs for mode in MODES]
    summaries = {m: summarize(rows, m) for m in MODES}

    sham = summaries["sham"]
    random = summaries["matched_random_support"]
    consensus = summaries["consensus_support"]
    assist = summaries["consensus_support_assist"]

    rescue_margin_random = consensus["stable_rescue_rate"] - random["stable_rescue_rate"]
    rescue_margin_sham = consensus["stable_rescue_rate"] - sham["stable_rescue_rate"]
    persistence_margin_random = consensus["persistence_rate"] - random["persistence_rate"]
    assist_gain = assist["stable_rescue_rate"] - consensus["stable_rescue_rate"]

    weight_ratio_random = (
        random["mean_added_weight"] / consensus["mean_added_weight"]
        if consensus["mean_added_weight"] > 1e-12 else 0.0
    )
    matching_quality = random["mean_match_coverage"] >= 0.80 and 0.70 <= weight_ratio_random <= 1.30

    domains = domain_summary(rows)
    comparable_domains = [x for x in domains if x.get("consensus_support_eligible", 0) > 0]
    rescue_domains = [
        x for x in comparable_domains
        if x.get("consensus_support_stable_rescued", 0) > x.get("matched_random_support_stable_rescued", 0)
    ]
    domain_rescue_rate = len(rescue_domains) / len(comparable_domains) if comparable_domains else 0.0

    representative = next((x for x in rows if x["mode"] == "consensus_support"), rows[0] if rows else None)
    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    if representative is not None:
        temp = OUT / "consensus_support_rescue_roundtrip.json"
        before_weights = representative["experimental"].weights.tolist()
        representative["experimental"].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = representative is not None and hasattr(representative["experimental"], "learning_state")

    distinctness_safe = True
    rescue_signal = (
        sham["rescue_eligible_trials"] >= 20
        and consensus["stable_rescue_rate"] >= sham["stable_rescue_rate"] + 0.10
        and rescue_margin_random >= 0.08
        and persistence_margin_random >= 0.05
        and matching_quality
        and domain_rescue_rate >= 0.50
        and distinctness_safe
    )
    measurement_pass = (
        len(unstable_specs) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if rescue_signal:
        readiness = "consensus_support_rescue_candidate"
        verdict = "selective_strengthening_of_existing_consensus_support_rescues_unstable_semantic_bridges_more_than_matched_local_support"
        next_step = "replicate_consensus_support_rescue_with_multiple_support_gains_and_no_future_stability_information_before_core_integration"
    else:
        readiness = "consensus_support_rescue_not_established"
        verdict = "consensus_aligned_support_does_not_yet_rescue_unstable_bridges_specificly_enough_over_matched_local_support"
        next_step = "inspect_rescue_failures_and_support_target_coverage_before_consensus_core_integration"

    payload = {
        "experiment": "Core Growth Binding v96 — Consensus Support Rescue",
        "contract": {
            "primary_core_modified": False,
            "unstable_trial_screening_uses_v94_definition": True,
            "consensus_definition_changed_from_v93": False,
            "semantic_answer_labels_used_for_support_targeting": False,
            "support_target_basis": "existing_v93_consensus_support_edges_at_first_bridge",
            "new_edges_created": False,
            "modes": MODES,
            "support_gain": SUPPORT_GAIN,
            "production_brain_json_saved": False,
        },
        "summary": {
            "screened_unstable_trials": len(unstable_specs),
            "rescue_eligible_trials": sham["rescue_eligible_trials"],
            "sham_stable_rescue_rate": sham["stable_rescue_rate"],
            "random_stable_rescue_rate": random["stable_rescue_rate"],
            "consensus_stable_rescue_rate": consensus["stable_rescue_rate"],
            "assist_stable_rescue_rate": assist["stable_rescue_rate"],
            "sham_persistence_rate": sham["persistence_rate"],
            "random_persistence_rate": random["persistence_rate"],
            "consensus_persistence_rate": consensus["persistence_rate"],
            "assist_persistence_rate": assist["persistence_rate"],
            "rescue_margin_vs_sham": rescue_margin_sham,
            "rescue_margin_vs_random": rescue_margin_random,
            "persistence_margin_vs_random": persistence_margin_random,
            "assist_gain": assist_gain,
            "random_match_coverage": random["mean_match_coverage"],
            "weight_match_ratio_random": weight_ratio_random,
            "matching_quality": matching_quality,
            "consensus_retention": consensus["mean_consensus_retention"],
            "random_consensus_retention": random["mean_consensus_retention"],
            "consensus_support_retention": consensus["mean_support_source_retention"],
            "consensus_quorum_retention": consensus["mean_quorum_retention"],
            "rescue_domains": len(rescue_domains),
            "comparable_domains": len(comparable_domains),
            "domain_rescue_rate": domain_rescue_rate,
            "distinctness_safe": distinctness_safe,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "rescue_signal": rescue_signal,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "domain_rows": domains,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in rows],
    }
    (OUT / "latest_binding_v96.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v96</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v96：Consensus Support Rescue</h1><p class="lead">v94でEver BridgeだがUnstableだったTrialに対し、初回Bridge時に既に存在するConsensus支持構造を少量だけ補強し、Matched random supportよりStable化・持続化できるかを検証する。新規Edgeは作らない。</p><section class="panel"><div class="controls"><button id="run">Consensus救済を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode別</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function pct(v){return `${(100*v).toFixed(1)}%`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Unstable cohort',s.screened_unstable_trials),metric('Rescue eligible',s.rescue_eligible_trials),metric('Sham rescue',pct(s.sham_stable_rescue_rate)),metric('Random rescue',pct(s.random_stable_rescue_rate)),metric('Consensus rescue',pct(s.consensus_stable_rescue_rate),s.consensus_stable_rescue_rate>s.random_stable_rescue_rate?'good':'warn'),metric('Consensus+Assist',pct(s.assist_stable_rescue_rate)),metric('vs Sham margin',pct(s.rescue_margin_vs_sham),s.rescue_margin_vs_sham>0?'good':'warn'),metric('vs Random margin',pct(s.rescue_margin_vs_random),s.rescue_margin_vs_random>0?'good':'warn'),metric('Persistence margin',pct(s.persistence_margin_vs_random),s.persistence_margin_vs_random>0?'good':'warn'),metric('Assist gain',pct(s.assist_gain)),metric('Random match coverage',pct(s.random_match_coverage),s.matching_quality?'good':'warn'),metric('Weight match',s.weight_match_ratio_random.toFixed(2),s.matching_quality?'good':'warn'),metric('Domain rescue',`${s.rescue_domains}/${s.comparable_domains}`),metric('Distinctness',yn(s.distinctness_safe),s.distinctness_safe?'good':'warn'),metric('Rescue signal',yn(s.rescue_signal),s.rescue_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post('/api/run')
def api_run():
    return jsonify(observe())


@app.get('/')
def index():
    return PAGE


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"v96 running: {url}")
    serve(app, host=HOST, port=PORT)


if __name__ == '__main__':
    main()
