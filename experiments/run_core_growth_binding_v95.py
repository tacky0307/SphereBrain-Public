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
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v89 as v89
import run_core_growth_binding_v93 as v93
import run_core_growth_binding_v94 as v94

HOST = "127.0.0.1"
START_PORT = 5149
OUT = ROOT / "data" / "core_growth_binding_v95" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v94.SEEDS
DOMAINS = v94.DOMAINS
STABILITY_WINDOW = [20, 30, 50]
MODES = ["sham", "low", "partial", "strong"]
ABLATION_FACTORS = {"sham": 1.0, "low": 0.82, "partial": 0.68, "strong": 0.50}


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


def domain_by_name(name: str):
    return next(x for x in DOMAINS if x.name == name)


def rank_consensus_edges(snapshot: dict) -> list[dict]:
    rows = []
    for row in snapshot.get("edges", []):
        rows.append({
            **row,
            "edge_tuple": edge_tuple(row["edge"]),
        })
    return sorted(
        rows,
        key=lambda x: (
            int(x.get("proximal_support_count", 0)),
            int(x.get("exact_support_count", 0)),
            float(x.get("consensus_strength", 0.0)),
        ),
        reverse=True,
    )


def select_ablation_edges(snapshot: dict, mode: str) -> list[tuple[int, int]]:
    ranked = rank_consensus_edges(snapshot)
    quorum = [x for x in ranked if bool(x.get("quorum"))]
    pool = quorum if quorum else ranked
    if mode == "sham" or not pool:
        return []
    if mode == "low":
        return [pool[0]["edge_tuple"]]
    if mode == "partial":
        n = max(1, len(pool) // 2)
        return [x["edge_tuple"] for x in pool[:n]]
    return [x["edge_tuple"] for x in pool]


def apply_ablation(brain: SphereBrain, edges: list[tuple[int, int]], mode: str) -> list[dict]:
    factor = float(ABLATION_FACTORS[mode])
    events = []
    for a, b in edges:
        before = float(brain.weights[a, b])
        after = max(0.0, before * factor)
        brain.weights[a, b] = after
        brain.weights[b, a] = after
        events.append({"edge": [a, b], "before": before, "after": after, "delta": after - before})
    return events


def consensus_at(brain, control, domain, baseline, cycle: int, history: dict[tuple[int, int], int]) -> dict:
    return v93.consensus_snapshot(brain, domain, brain, control, baseline, cycle, history)


def run_ablation_trial(spec: dict, mode: str) -> dict:
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
    ablation_events = []
    selected_edges: list[tuple[int, int]] = []
    ablated = False
    pre_ablation_consensus = None

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            records.append(row)
            snap = consensus_at(experimental, control, domain, exp_baseline, cycle, history)
            consensus_rows.append(snap)

            if not ablated and cycle == target_cycle:
                pre_ablation_consensus = snap
                selected_edges = select_ablation_edges(snap, mode)
                ablation_events = apply_ablation(experimental, selected_edges, mode)
                ablated = True

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    stable_after = stable_from_records(records)
    by_cycle = {int(r["cycle"]): bool(r["direct_success_now"]) for r in records}
    bridge_cycles = [int(r["cycle"]) for r in records if r["direct_success_now"]]

    first_idx = next((i for i, r in enumerate(records) if int(r["cycle"]) == target_cycle), None)
    first_consensus = float(consensus_rows[first_idx]["mean_consensus_strength"]) if first_idx is not None else 0.0
    final_consensus = float(consensus_rows[-1]["mean_consensus_strength"]) if consensus_rows else 0.0
    first_quorum = int(consensus_rows[first_idx]["quorum_edge_count"]) if first_idx is not None else 0
    final_quorum = int(consensus_rows[-1]["quorum_edge_count"]) if consensus_rows else 0
    consensus_retention = final_consensus / first_consensus if first_consensus > 1e-12 else 0.0
    quorum_retention = final_quorum / first_quorum if first_quorum > 0 else (1.0 if final_quorum == 0 else 0.0)

    return {
        "domain": domain.name,
        "seed": seed,
        "mode": mode,
        "target_cycle": target_cycle,
        "selected_edge_count": len(selected_edges),
        "ablation_factor": float(ABLATION_FACTORS[mode]),
        "ablation_events": ablation_events,
        "pre_ablation_consensus": pre_ablation_consensus,
        "stable_after": stable_after,
        "bridge_cycles": bridge_cycles,
        "success_20": by_cycle.get(20, False),
        "success_30": by_cycle.get(30, False),
        "success_50": by_cycle.get(50, False),
        "first_consensus": first_consensus,
        "final_consensus": final_consensus,
        "consensus_retention": consensus_retention,
        "first_quorum_edges": first_quorum,
        "final_quorum_edges": final_quorum,
        "quorum_retention": quorum_retention,
        "records": records,
        "consensus_by_cycle": consensus_rows,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    stable = sum(1 for x in subset if x["stable_after"])
    collapsed = len(subset) - stable
    return {
        "mode": mode,
        "trials": len(subset),
        "stable": stable,
        "stable_rate": stable / len(subset) if subset else 0.0,
        "collapsed": collapsed,
        "collapse_rate": collapsed / len(subset) if subset else 0.0,
        "mean_selected_edges": statistics.mean([x["selected_edge_count"] for x in subset]) if subset else 0.0,
        "mean_consensus_retention": statistics.mean([x["consensus_retention"] for x in subset]) if subset else 0.0,
        "mean_quorum_retention": statistics.mean([x["quorum_retention"] for x in subset]) if subset else 0.0,
        "success_20": sum(1 for x in subset if x["success_20"]),
        "success_30": sum(1 for x in subset if x["success_30"]),
        "success_50": sum(1 for x in subset if x["success_50"]),
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


def dose_response(summaries: dict[str, dict]) -> bool:
    rates = [summaries[m]["stable_rate"] for m in MODES]
    return all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1)) and rates[0] > rates[-1]


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    stable_specs = screen_stable_trials()
    rows = [run_ablation_trial(spec, mode) for spec in stable_specs for mode in MODES]
    summaries = {m: summarize(rows, m) for m in MODES}

    sham = summaries["sham"]
    low = summaries["low"]
    partial = summaries["partial"]
    strong = summaries["strong"]

    dose = dose_response(summaries)
    strong_effect = sham["stable_rate"] - strong["stable_rate"]
    partial_effect = sham["stable_rate"] - partial["stable_rate"]
    consensus_dose = (
        sham["mean_consensus_retention"] >= low["mean_consensus_retention"]
        >= partial["mean_consensus_retention"] >= strong["mean_consensus_retention"]
    )
    quorum_dose = (
        sham["mean_quorum_retention"] >= low["mean_quorum_retention"]
        >= partial["mean_quorum_retention"] >= strong["mean_quorum_retention"]
    )

    domains = domain_summary(rows)
    comparable_domains = [x for x in domains if x.get("sham_trials", 0) > 0]
    strong_harm_domains = [x for x in comparable_domains if x.get("strong_stable", 0) < x.get("sham_stable", 0)]
    domain_effect_rate = len(strong_harm_domains) / len(comparable_domains) if comparable_domains else 0.0

    representative = next((x for x in rows if x["mode"] == "strong"), rows[0] if rows else None)
    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    if representative is not None:
        temp = OUT / "consensus_causality_ablation_roundtrip.json"
        before_weights = representative["experimental"].weights.tolist()
        representative["experimental"].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = representative is not None and hasattr(representative["experimental"], "learning_state")

    causality_signal = (
        len(stable_specs) >= 10
        and sham["stable_rate"] >= 0.80
        and strong_effect >= 0.20
        and dose
        and (consensus_dose or quorum_dose)
        and domain_effect_rate >= 0.50
    )
    measurement_pass = (
        len(stable_specs) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if causality_signal:
        readiness = "consensus_persistence_causality_candidate"
        verdict = "graded_ablation_of_consensus_support_reduces_stable_bridge_survival_with_a_dose_response_pattern"
        next_step = "replicate_ablation_with_random_edge_controls_and_source-matched_sham_before_any_consensus mechanism_integration"
    else:
        readiness = "consensus_persistence_causality_not_established"
        verdict = "consensus_support_ablation_does_not_show_a_strong_enough_dose-dependent_effect_on_stable_bridge_survival"
        next_step = "add_random_edge_and_degree-matched_ablation_controls_before interpreting consensus_persistence_as_causal"

    payload = {
        "experiment": "Core Growth Binding v95 — Consensus Causality Ablation",
        "contract": {
            "primary_core_modified": False,
            "stable_trial_screening_uses_v94_definition": True,
            "consensus_definition_changed_from_v93": False,
            "semantic_answer_labels_used_for_ablation_targeting": False,
            "ablation_target_basis": "v93_structural_support_edges_at_first_bridge",
            "modes": MODES,
            "ablation_factors": ABLATION_FACTORS,
            "stability_window": STABILITY_WINDOW,
            "production_brain_json_saved": False,
        },
        "summary": {
            "screened_stable_trials": len(stable_specs),
            "sham_stable": sham["stable"],
            "low_stable": low["stable"],
            "partial_stable": partial["stable"],
            "strong_stable": strong["stable"],
            "sham_stable_rate": sham["stable_rate"],
            "low_stable_rate": low["stable_rate"],
            "partial_stable_rate": partial["stable_rate"],
            "strong_stable_rate": strong["stable_rate"],
            "partial_effect": partial_effect,
            "strong_effect": strong_effect,
            "dose_response": dose,
            "consensus_retention_dose_response": consensus_dose,
            "quorum_retention_dose_response": quorum_dose,
            "strong_harm_domains": len(strong_harm_domains),
            "comparable_domains": len(comparable_domains),
            "domain_effect_rate": domain_effect_rate,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "causality_signal": causality_signal,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "domain_rows": domains,
        "stable_screen": stable_specs,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in rows],
    }
    (OUT / "latest_binding_v95.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v95</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v95：Consensus Causality Ablation</h1><p class="lead">v94でStableだったTrialを同条件で再生し、初回Bridge直後にConsensus支持Edgeを段階的に弱める。Sham / Low / Partial / StrongでStable維持率・Consensus保持率・Quorum保持率の用量反応を検証する。</p><section class="panel"><div class="controls"><button id="run">Consensus因果性を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode別</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function pct(v){return `${(100*v).toFixed(1)}%`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Stable cohort',s.screened_stable_trials),metric('Sham Stable',pct(s.sham_stable_rate)),metric('Low Stable',pct(s.low_stable_rate)),metric('Partial Stable',pct(s.partial_stable_rate)),metric('Strong Stable',pct(s.strong_stable_rate),s.strong_stable_rate<s.sham_stable_rate?'warn':'blue'),metric('Strong effect',pct(s.strong_effect),s.strong_effect>=.2?'good':'warn'),metric('Dose response',yn(s.dose_response),s.dose_response?'good':'warn'),metric('Consensus dose',yn(s.consensus_retention_dose_response),s.consensus_retention_dose_response?'good':'warn'),metric('Quorum dose',yn(s.quorum_retention_dose_response),s.quorum_retention_dose_response?'good':'warn'),metric('Domain effect',`${s.strong_harm_domains}/${s.comparable_domains}`,s.domain_effect_rate>=.5?'good':'warn'),metric('Causality signal',yn(s.causality_signal),s.causality_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"v95 running: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
