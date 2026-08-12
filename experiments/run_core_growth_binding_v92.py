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
from semantic_relative_selectivity import RelativeSelectivityPreserver
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v87 as v87
import run_core_growth_binding_v89 as v89

HOST = "127.0.0.1"
START_PORT = 5144
OUT = ROOT / "data" / "core_growth_binding_v92" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v89.SEEDS
DOMAINS = v89.DOMAINS
MODES = ["primary", "relative_selectivity", "relative_selectivity_assist"]


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


def run_trial(domain, seed: int, mode: str) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "relative_selectivity_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)

    exp_preserver = RelativeSelectivityPreserver()
    ctrl_preserver = RelativeSelectivityPreserver()
    use_preserver = mode != "primary"

    records = []
    exp_intervention_cycles: list[int] = []
    ctrl_intervention_cycles: list[int] = []
    exp_restored_total = 0.0
    ctrl_restored_total = 0.0
    exp_peak_drop = 0.0
    mature_seen = False

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            row["mode"] = mode
            records.append(row)

        if cycle < max(CHECKPOINTS):
            exp_prev = experimental.weights.copy()
            ctrl_prev = control.weights.copy()

            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

            if use_preserver:
                exp_rows = v87.selective_candidates(experimental, domain, exp_baseline["direct"], True)
                ctrl_rows = v87.selective_candidates(control, domain, ctrl_baseline["direct"], False)
                exp_result = exp_preserver.observe(experimental, exp_rows, exp_prev)
                ctrl_result = ctrl_preserver.observe(control, ctrl_rows, ctrl_prev)
                mature_seen = mature_seen or exp_result["mature_edge_count"] > 0
                exp_peak_drop = max(exp_peak_drop, float(exp_result["drop_ratio"]))
                if exp_result["triggered"]:
                    exp_intervention_cycles.append(cycle + 1)
                    exp_restored_total += sum(float(x["restored"]) for x in exp_result["restored"])
                if ctrl_result["triggered"]:
                    ctrl_intervention_cycles.append(cycle + 1)
                    ctrl_restored_total += sum(float(x["restored"]) for x in ctrl_result["restored"])

    cls = v86.classify(records)
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
        "intervention_count": len(exp_intervention_cycles),
        "intervention_cycles": sorted(set(exp_intervention_cycles)),
        "control_intervention_count": len(ctrl_intervention_cycles),
        "restored_total": float(exp_restored_total),
        "control_restored_total": float(ctrl_restored_total),
        "max_selectivity_drop": float(exp_peak_drop),
        "mature_ensemble_seen": bool(mature_seen),
        "preserver_state": exp_preserver.snapshot(),
        "records": records,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    intervention_cycles = [c for x in subset for c in x["intervention_cycles"]]
    return {
        "mode": mode,
        "trials": len(subset),
        "ever_bridge": sum(1 for x in subset if x["ever_bridge"]),
        "ever_bridge_rate": sum(1 for x in subset if x["ever_bridge"]) / len(subset),
        "stable": sum(1 for x in subset if x["stable"]),
        "stable_rate": sum(1 for x in subset if x["stable"]) / len(subset),
        "never": sum(1 for x in subset if x["never"]),
        "never_rate": sum(1 for x in subset if x["never"]) / len(subset),
        "intervention_trials": sum(1 for x in subset if x["intervention_count"] > 0),
        "intervention_rate": sum(1 for x in subset if x["intervention_count"] > 0) / len(subset),
        "intervention_events": sum(x["intervention_count"] for x in subset),
        "intervention_cycle_median": statistics.median(intervention_cycles) if intervention_cycles else None,
        "restored_total": sum(x["restored_total"] for x in subset),
        "mature_ensemble_trials": sum(1 for x in subset if x["mature_ensemble_seen"]),
        "mean_final_direct_similarity": sum(x["final_direct_similarity"] for x in subset) / len(subset),
    }


def paired(rows: list[dict], mode: str) -> dict:
    primary = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == "primary"}
    target = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == mode}
    gains = [k for k in primary if target[k]["stable"] and not primary[k]["stable"]]
    losses = [k for k in primary if primary[k]["stable"] and not target[k]["stable"]]
    never_gains = [k for k in primary if target[k]["never"] and not primary[k]["never"]]
    never_losses = [k for k in primary if primary[k]["never"] and not target[k]["never"]]
    similarity_deltas = [target[k]["final_direct_similarity"] - primary[k]["final_direct_similarity"] for k in primary]
    ordered = sorted(similarity_deltas)
    p95 = ordered[min(len(ordered) - 1, int(round(.95 * (len(ordered) - 1))))]
    return {
        "stable_gain": len(gains),
        "stable_loss": len(losses),
        "net_stable_gain": len(gains) - len(losses),
        "never_gain": len(never_gains),
        "never_loss": len(never_losses),
        "net_never_change": len(never_gains) - len(never_losses),
        "max_similarity_delta": max(similarity_deltas),
        "p95_similarity_delta": p95,
        "mean_similarity_delta": sum(similarity_deltas) / len(similarity_deltas),
        "improved_trials": [{"domain": k[0], "seed": k[1]} for k in gains],
        "worsened_trials": [{"domain": k[0], "seed": k[1]} for k in losses],
    }


def domain_summary(rows: list[dict], mode: str) -> list[dict]:
    result = []
    for domain in DOMAINS:
        subset = [x for x in rows if x["mode"] == mode and x["domain"] == domain.name]
        result.append({
            "domain": domain.name,
            "stable": sum(1 for x in subset if x["stable"]),
            "never": sum(1 for x in subset if x["never"]),
            "ever_bridge": sum(1 for x in subset if x["ever_bridge"]),
            "intervention_trials": sum(1 for x in subset if x["intervention_count"] > 0),
        })
    return result


def seed_summary(rows: list[dict], mode: str) -> list[dict]:
    result = []
    for seed in SEEDS:
        subset = [x for x in rows if x["mode"] == mode and x["seed"] == seed]
        result.append({
            "seed": seed,
            "stable_domains": sum(1 for x in subset if x["stable"]),
            "never_domains": sum(1 for x in subset if x["never"]),
            "intervention_domains": sum(1 for x in subset if x["intervention_count"] > 0),
        })
    return result


def count_domain_wins(primary_rows: list[dict], target_rows: list[dict]) -> int:
    p = {x["domain"]: x for x in primary_rows}
    t = {x["domain"]: x for x in target_rows}
    return sum(1 for name in p if t[name]["stable"] > p[name]["stable"] and t[name]["never"] <= p[name]["never"])


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed, mode) for mode in MODES for domain in DOMAINS for seed in SEEDS]

    summaries = {m: summarize(trials, m) for m in MODES}
    preserve_pair = paired(trials, "relative_selectivity")
    assist_pair = paired(trials, "relative_selectivity_assist")

    primary = summaries["primary"]
    preserve = summaries["relative_selectivity"]
    assist = summaries["relative_selectivity_assist"]

    domains = {m: domain_summary(trials, m) for m in MODES}
    seeds = {m: seed_summary(trials, m) for m in MODES}
    preserve_domain_wins = count_domain_wins(domains["primary"], domains["relative_selectivity"])
    assist_domain_wins = count_domain_wins(domains["primary"], domains["relative_selectivity_assist"])

    preserve_distinctness = preserve_pair["p95_similarity_delta"] <= 0.12 and preserve_pair["max_similarity_delta"] <= 0.20
    assist_distinctness = assist_pair["p95_similarity_delta"] <= 0.12 and assist_pair["max_similarity_delta"] <= 0.20

    preserve_safe = preserve["never"] <= primary["never"] and preserve_distinctness
    assist_safe = assist["never"] <= primary["never"] and assist_distinctness
    preserve_improves = preserve_pair["net_stable_gain"] >= 5 and preserve_safe and preserve_domain_wins >= 2
    assist_improves = assist_pair["net_stable_gain"] >= 5 and assist_safe and assist_domain_wins >= 2

    representative = next(
        (x for x in trials if x["mode"] == "relative_selectivity" and x["intervention_count"] > 0),
        next(x for x in trials if x["mode"] == "relative_selectivity"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "relative_selectivity_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    intervention_observed = preserve["intervention_trials"] > 0 or assist["intervention_trials"] > 0
    overall_pass = (
        (preserve_improves or assist_improves)
        and intervention_observed
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if preserve_improves and assist_improves:
        winner = "relative_selectivity_assist" if assist["stable"] > preserve["stable"] else "relative_selectivity"
    elif preserve_improves:
        winner = "relative_selectivity"
    elif assist_improves:
        winner = "relative_selectivity_assist"
    else:
        winner = "primary"

    if overall_pass:
        readiness = "semantic_relative_selectivity_preservation_candidate"
        verdict = "reversible_local_selectivity_preservation_improves_post_bridge_stability_without_increasing_never_or_collapsing_distinctness"
        next_step = "validate_winner_under_context_switching_and_contradictory_semantic_episode_streams"
    else:
        readiness = "semantic_relative_selectivity_preservation_not_yet_validated"
        verdict = "relative_selectivity_preservation_does_not_yet_outperform_primary_core_robustly"
        next_step = "attribute_intervention_true_positive_false_positive_and_missed_collapse_cases_before_primary_core_change"

    payload = {
        "experiment": "Core Growth Binding v92 — Relative Selectivity Preservation",
        "contract": {
            "primary_core_modified": False,
            "modes": MODES,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "trials_per_mode": len(DOMAINS) * len(SEEDS),
            "total_trials": len(trials),
            "semantic_answer_labels_used": False,
            "control_results_used_by_preserver": False,
            "new_edges_created_by_preserver": False,
            "preservation_reversible": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "primary_stable": primary["stable"],
            "preserve_stable": preserve["stable"],
            "assist_stable": assist["stable"],
            "primary_never": primary["never"],
            "preserve_never": preserve["never"],
            "assist_never": assist["never"],
            "preserve_net_stable_gain": preserve_pair["net_stable_gain"],
            "assist_net_stable_gain": assist_pair["net_stable_gain"],
            "preserve_intervention_trials": preserve["intervention_trials"],
            "assist_intervention_trials": assist["intervention_trials"],
            "preserve_intervention_rate": preserve["intervention_rate"],
            "assist_intervention_rate": assist["intervention_rate"],
            "preserve_domain_wins": preserve_domain_wins,
            "assist_domain_wins": assist_domain_wins,
            "preserve_distinctness_safe": preserve_distinctness,
            "assist_distinctness_safe": assist_distinctness,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "relative_selectivity_pass": overall_pass,
            "winner": winner,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "paired": {
            "relative_selectivity": preserve_pair,
            "relative_selectivity_assist": assist_pair,
        },
        "domain_rows": domains,
        "seed_rows": seeds,
        "trials": [{k: v for k, v in x.items() if k not in {"experimental", "records"}} for x in trials],
    }
    (OUT / "latest_binding_v92.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v92</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v92：Relative Selectivity Preservation</h1><p class="lead">Bridge Edgeを固定せず、成熟した局所Bridge集合の相対選択性が急落した時だけ可逆的に減衰を緩和する。Primary / Preservation / Preservation+Assistを100試行ずつ比較する。</p><section class="panel"><div class="controls"><button id="run">相対選択性維持を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode比較</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Paired変化</h2><pre id="paired" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','300試行を実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial / mode',100),metric('Primary Stable',s.primary_stable),metric('Preserve Stable',s.preserve_stable,s.preserve_stable>s.primary_stable?'good':'blue'),metric('Preserve+Assist Stable',s.assist_stable,s.assist_stable>s.primary_stable?'good':'blue'),metric('Primary Never',s.primary_never),metric('Preserve Never',s.preserve_never,s.preserve_never<=s.primary_never?'good':'warn'),metric('Assist Never',s.assist_never,s.assist_never<=s.primary_never?'good':'warn'),metric('Preserve net gain',s.preserve_net_stable_gain,s.preserve_net_stable_gain>=5?'good':'warn'),metric('Assist net gain',s.assist_net_stable_gain,s.assist_net_stable_gain>=5?'good':'warn'),metric('介入Trial',`${s.preserve_intervention_trials} / ${s.assist_intervention_trials}`),metric('Domain勝ち',`${s.preserve_domain_wins} / ${s.assist_domain_wins}`),metric('Distinctness',s.preserve_distinctness_safe&&s.assist_distinctness_safe?'YES':'NO',s.preserve_distinctness_safe&&s.assist_distinctness_safe?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('v92 PASS',yn(s.relative_selectivity_pass),s.relative_selectivity_pass?'good':'warn'),metric('Winner',s.winner),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('paired').textContent=JSON.stringify(d.paired,null,2)}document.getElementById('run').onclick=run;
</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"v92: {url}")
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
