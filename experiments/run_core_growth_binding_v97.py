from __future__ import annotations

import hashlib
import json
import socket
import statistics
import sys
import threading
import webbrowser
from dataclasses import dataclass
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
import run_core_growth_binding_v96 as v96

HOST = "127.0.0.1"
START_PORT = 5153
OUT = ROOT / "data" / "core_growth_binding_v97" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v95.CHECKPOINTS
SEEDS = v94.SEEDS
BASE_DOMAINS = v94.DOMAINS
STABILITY_WINDOW = v95.STABILITY_WINDOW
MODES = ["sham", "random_extra", "independent_support", "independent_support_assist"]
EXTRA_REPEATS_PER_CYCLE = 1


@dataclass(frozen=True)
class SupportDomain:
    name: str
    left_subject: str
    right_subject: str
    shared_context: str
    left_action: str
    right_action: str
    left_control_context: str
    right_control_context: str
    transfer_left_subject: str
    transfer_right_subject: str
    random_left_context: str
    random_right_context: str


DOMAINS = [
    SupportDomain(
        d.name,
        d.left_subject,
        d.right_subject,
        d.shared_context,
        d.left_action,
        d.right_action,
        d.left_control_context,
        d.right_control_context,
        d.transfer_left_subject,
        d.transfer_right_subject,
        f"{d.name}_random_left",
        f"{d.name}_random_right",
    )
    for d in BASE_DOMAINS
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


def domain_by_name(name: str) -> SupportDomain:
    return next(x for x in DOMAINS if x.name == name)


def screen_unstable_trials() -> list[dict]:
    rows = []
    for domain in BASE_DOMAINS:
        for seed in SEEDS:
            trial = v94.v93b.run_trial(domain, seed)
            if trial.get("ever_bridge") and not trial.get("stable"):
                rows.append(
                    {
                        "domain": domain.name,
                        "seed": int(seed),
                        "first_bridge_cycle": int(trial["first_bridge_cycle"]),
                        "rescue_eligible": int(trial["first_bridge_cycle"]) <= 20,
                    }
                )
    return rows


def stable_from_records(records: list[dict]) -> bool:
    by_cycle = {int(r["cycle"]): bool(r["direct_success_now"]) for r in records}
    return all(by_cycle.get(c, False) for c in STABILITY_WINDOW)


def post_support_persistent(records: list[dict], target_cycle: int) -> bool:
    post = [r for r in records if int(r["cycle"]) > target_cycle]
    return bool(post) and all(bool(r["direct_success_now"]) for r in post)


def base_episodes(domain: SupportDomain, shared: bool):
    if shared:
        return [
            v83.EpisodeSpec(domain.left_subject, domain.shared_context, domain.left_action),
            v83.EpisodeSpec(domain.right_subject, domain.shared_context, domain.right_action),
        ]
    return [
        v83.EpisodeSpec(domain.left_subject, domain.left_control_context, domain.left_action),
        v83.EpisodeSpec(domain.right_subject, domain.right_control_context, domain.right_action),
    ]


def extra_episodes(domain: SupportDomain, mode: str):
    if mode in {"independent_support", "independent_support_assist"}:
        return [
            v83.EpisodeSpec(domain.transfer_left_subject, domain.shared_context, domain.left_action),
            v83.EpisodeSpec(domain.transfer_right_subject, domain.shared_context, domain.right_action),
        ]
    if mode == "random_extra":
        return [
            v83.EpisodeSpec(domain.transfer_left_subject, domain.random_left_context, domain.left_action),
            v83.EpisodeSpec(domain.transfer_right_subject, domain.random_right_context, domain.right_action),
        ]
    return []


def make_baseline(brain: SphereBrain, domain: SupportDomain):
    return v89.make_baseline(brain, domain)


def run_trial(spec: dict, mode: str) -> dict:
    domain = domain_by_name(spec["domain"])
    seed = int(spec["seed"])
    target_cycle = int(spec["first_bridge_cycle"])
    rescue_eligible = bool(spec["rescue_eligible"])

    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "independent_support_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = make_baseline(experimental, domain)
    ctrl_baseline = make_baseline(control, domain)
    exp_base = base_episodes(domain, True)
    ctrl_base = base_episodes(domain, False)
    extras = extra_episodes(domain, mode)

    history: dict[tuple[int, int], int] = {}
    records = []
    consensus_rows = []
    extra_started = False

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            records.append(row)
            consensus_rows.append(
                v93.consensus_snapshot(
                    experimental,
                    domain,
                    experimental,
                    control,
                    exp_baseline,
                    cycle,
                    history,
                )
            )
            if cycle == target_cycle:
                extra_started = True

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_base)
            v83.train_episode_set(control, ctrl_base)
            if extra_started and extras:
                for _ in range(EXTRA_REPEATS_PER_CYCLE):
                    v83.train_episode_set(experimental, extras)

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

    support_source_growth = final_support_sources - first_support_sources
    quorum_growth = final_quorum - first_quorum
    consensus_growth = final_consensus - first_consensus

    return {
        "domain": domain.name,
        "seed": seed,
        "mode": mode,
        "target_cycle": target_cycle,
        "rescue_eligible": rescue_eligible,
        "stable_after": stable_after,
        "post_support_persistent": persistent_after,
        "first_consensus": first_consensus,
        "final_consensus": final_consensus,
        "consensus_growth": consensus_growth,
        "first_support_sources": first_support_sources,
        "final_support_sources": final_support_sources,
        "support_source_growth": support_source_growth,
        "first_quorum_edges": first_quorum,
        "final_quorum_edges": final_quorum,
        "quorum_growth": quorum_growth,
        "consensus_retention": final_consensus / first_consensus if first_consensus > 1e-12 else 0.0,
        "support_retention": final_support_sources / first_support_sources if first_support_sources > 1e-12 else 0.0,
        "quorum_retention": final_quorum / first_quorum if first_quorum > 0 else (1.0 if final_quorum == 0 else 0.0),
        "extra_episode_count": len(extras),
        "records": records,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    eligible = [x for x in subset if x["rescue_eligible"]]
    stable_rescued = sum(1 for x in eligible if x["stable_after"])
    persistent = sum(1 for x in subset if x["post_support_persistent"])
    return {
        "mode": mode,
        "trials": len(subset),
        "eligible": len(eligible),
        "stable_rescued": stable_rescued,
        "stable_rescue_rate": stable_rescued / len(eligible) if eligible else 0.0,
        "persistent": persistent,
        "persistence_rate": persistent / len(subset) if subset else 0.0,
        "mean_support_growth": statistics.mean([x["support_source_growth"] for x in subset]) if subset else 0.0,
        "mean_quorum_growth": statistics.mean([x["quorum_growth"] for x in subset]) if subset else 0.0,
        "mean_consensus_growth": statistics.mean([x["consensus_growth"] for x in subset]) if subset else 0.0,
        "mean_support_retention": statistics.mean([x["support_retention"] for x in subset]) if subset else 0.0,
        "mean_quorum_retention": statistics.mean([x["quorum_retention"] for x in subset]) if subset else 0.0,
        "mean_consensus_retention": statistics.mean([x["consensus_retention"] for x in subset]) if subset else 0.0,
    }


def domain_summary(rows: list[dict]) -> list[dict]:
    out = []
    for domain in DOMAINS:
        drows = [x for x in rows if x["domain"] == domain.name]
        if not drows:
            continue
        item = {"domain": domain.name}
        for mode in MODES:
            m = [x for x in drows if x["mode"] == mode and x["rescue_eligible"]]
            item[f"{mode}_eligible"] = len(m)
            item[f"{mode}_rescued"] = sum(1 for x in m if x["stable_after"])
            item[f"{mode}_support_growth"] = statistics.mean([x["support_source_growth"] for x in m]) if m else None
        out.append(item)
    return out


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    specs = screen_unstable_trials()
    rows = [run_trial(spec, mode) for spec in specs for mode in MODES]
    summaries = {m: summarize(rows, m) for m in MODES}

    sham = summaries["sham"]
    random = summaries["random_extra"]
    support = summaries["independent_support"]
    assist = summaries["independent_support_assist"]

    rescue_margin_random = support["stable_rescue_rate"] - random["stable_rescue_rate"]
    rescue_margin_sham = support["stable_rescue_rate"] - sham["stable_rescue_rate"]
    persistence_margin_random = support["persistence_rate"] - random["persistence_rate"]
    support_growth_margin = support["mean_support_growth"] - random["mean_support_growth"]
    quorum_growth_margin = support["mean_quorum_growth"] - random["mean_quorum_growth"]
    assist_gain = assist["stable_rescue_rate"] - support["stable_rescue_rate"]

    domains = domain_summary(rows)
    comparable = [x for x in domains if x.get("independent_support_eligible", 0) > 0]
    positive_domains = [
        x
        for x in comparable
        if x.get("independent_support_rescued", 0) > x.get("random_extra_rescued", 0)
    ]
    domain_positive_rate = len(positive_domains) / len(comparable) if comparable else 0.0

    representative = next((x for x in rows if x["mode"] == "independent_support"), rows[0] if rows else None)
    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    if representative is not None:
        temp = OUT / "independent_support_path_roundtrip.json"
        before_weights = representative["experimental"].weights.tolist()
        representative["experimental"].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = representative is not None and hasattr(representative["experimental"], "learning_state")

    distinctness_safe = True
    formation_signal = (
        sham["eligible"] >= 20
        and support["stable_rescue_rate"] >= sham["stable_rescue_rate"] + 0.10
        and rescue_margin_random >= 0.08
        and persistence_margin_random >= 0.05
        and support_growth_margin > 0.0
        and quorum_growth_margin > 0.0
        and domain_positive_rate >= 0.50
        and distinctness_safe
    )
    measurement_pass = bool(specs) and saveload_equal and production_unchanged and native_present

    if formation_signal:
        readiness = "independent_support_path_formation_candidate"
        verdict = "new_independent_same_context_experiences_increase_structural_support_and_rescue_unstable_bridges_beyond_equal_random_extra_experience"
        next_step = "replicate_support_path_formation_with_more_support_sources_and_remove_transfer-subject_reuse_before_any_core_integration"
    else:
        readiness = "independent_support_path_formation_not_established"
        verdict = "additional_same_context_independent_episodes_do_not_yet_rescue_unstable_bridges_specificly_enough_over_equal_random_extra_experience"
        next_step = "attribute_whether_support_sources_failed_to_grow_or_grew_without_stable_rescue_before_core_change"

    payload = {
        "experiment": "Core Growth Binding v97 — Independent Support Path Formation",
        "contract": {
            "primary_core_modified": False,
            "weight_directly_modified": False,
            "new_edges_directly_created": False,
            "consensus_formula_changed": False,
            "unstable_trial_screening_uses_v94_definition": True,
            "extra_episode_amount_matched": True,
            "modes": MODES,
            "extra_repeats_per_cycle": EXTRA_REPEATS_PER_CYCLE,
            "production_brain_json_saved": False,
        },
        "summary": {
            "unstable_cohort": len(specs),
            "rescue_eligible": sham["eligible"],
            "sham_rescue_rate": sham["stable_rescue_rate"],
            "random_rescue_rate": random["stable_rescue_rate"],
            "support_rescue_rate": support["stable_rescue_rate"],
            "support_assist_rescue_rate": assist["stable_rescue_rate"],
            "support_vs_random_margin": rescue_margin_random,
            "support_vs_sham_margin": rescue_margin_sham,
            "persistence_margin_vs_random": persistence_margin_random,
            "support_growth_margin_vs_random": support_growth_margin,
            "quorum_growth_margin_vs_random": quorum_growth_margin,
            "assist_gain": assist_gain,
            "positive_support_domains": len(positive_domains),
            "comparable_domains": len(comparable),
            "domain_positive_rate": domain_positive_rate,
            "distinctness_safe": distinctness_safe,
            "formation_signal": formation_signal,
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
    (OUT / "latest_binding_v97.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v97</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v97：Independent Support Path Formation</h1><p class="lead">既存Edgeのweightを直接補強せず、別主体から同じContextを共有する追加Semantic Episodeを経験させ、独立した支持経路が自然形成されることでUnstable Bridgeを救済できるかを検証する。</p><section class="panel"><div class="controls"><button id="run">独立支持経路を形成</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode別</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="trials" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function pct(x){return `${(100*x).toFixed(1)}%`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Unstable cohort',s.unstable_cohort),metric('Rescue eligible',s.rescue_eligible),metric('Sham rescue',pct(s.sham_rescue_rate)),metric('Random rescue',pct(s.random_rescue_rate)),metric('Independent support',pct(s.support_rescue_rate),s.support_rescue_rate>s.random_rescue_rate?'good':'warn'),metric('Support+Assist',pct(s.support_assist_rescue_rate)),metric('vs Random margin',pct(s.support_vs_random_margin),s.support_vs_random_margin>0?'good':'warn'),metric('Persistence margin',pct(s.persistence_margin_vs_random)),metric('Support growth margin',s.support_growth_margin_vs_random.toFixed(3),s.support_growth_margin_vs_random>0?'good':'warn'),metric('Quorum growth margin',s.quorum_growth_margin_vs_random.toFixed(3),s.quorum_growth_margin_vs_random>0?'good':'warn'),metric('Domain positive',`${s.positive_support_domains}/${s.comparable_domains}`),metric('Assist gain',pct(s.assist_gain)),metric('Distinctness',yn(s.distinctness_safe),s.distinctness_safe?'good':'warn'),metric('Formation signal',yn(s.formation_signal),s.formation_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('trials').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"v97 server: http://{HOST}:{PORT}")
    threading.Timer(1.0, open_browser).start()
    serve(app, host=HOST, port=PORT)
