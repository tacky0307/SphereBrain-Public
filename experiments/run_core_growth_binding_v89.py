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
from semantic_bridge_event_triggered import EventTriggeredConsolidation
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v87 as v87

HOST = "127.0.0.1"
START_PORT = 5140
OUT = ROOT / "data" / "core_growth_binding_v89" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v85.CHECKPOINTS
MODES = ["primary", "event_triggered", "event_triggered_assist"]
SEEDS = [
    42, 314, 2718, 8088, 12021,
    7, 73, 101, 509, 997,
    2027, 4099, 6029, 9011, 15013,
    22003, 30011, 41017, 52021, 65029,
]


@dataclass(frozen=True)
class DomainSpec:
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


DOMAINS = [
    DomainSpec("sky", "鳥", "飛行機", "空", "羽ばたく", "飛行する", "森", "空港", "蝶", "ドローン"),
    DomainSpec("sea", "魚", "船", "海", "泳ぐ", "進む", "川", "港", "イルカ", "潜水艦"),
    DomainSpec("road", "馬", "車", "道", "走る", "進む", "草原", "駐車場", "鹿", "バス"),
    DomainSpec("forest", "鹿", "キツツキ", "森", "走る", "つつく", "草原", "木立", "ウサギ", "クマゲラ"),
    DomainSpec("workshop", "ハンマー", "ドリル", "工房", "叩く", "穴を開ける", "倉庫", "工具箱", "木槌", "キリ"),
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


def edge_tuple(value) -> tuple[int, int]:
    a, b = [int(x) for x in value]
    return tuple(sorted((a, b)))


def action_item(subject: str, action: str):
    return v84.action_item(subject, action)


def context_item(subject: str, context: str):
    return v84.context_item(subject, context)


def episodes_for(domain: DomainSpec, shared: bool):
    if shared:
        return [
            v83.EpisodeSpec(domain.left_subject, domain.shared_context, domain.left_action),
            v83.EpisodeSpec(domain.right_subject, domain.shared_context, domain.right_action),
        ]
    return [
        v83.EpisodeSpec(domain.left_subject, domain.left_control_context, domain.left_action),
        v83.EpisodeSpec(domain.right_subject, domain.right_control_context, domain.right_action),
    ]


def make_baseline(brain: SphereBrain, domain: DomainSpec):
    left = action_item(domain.left_subject, domain.left_action)
    right = action_item(domain.right_subject, domain.right_action)
    transfer_left = action_item(domain.transfer_left_subject, domain.left_action)
    transfer_right = action_item(domain.transfer_right_subject, domain.right_action)
    return {
        "direct": v83.action_signature(brain, left)["edges"] & v83.action_signature(brain, right)["edges"],
        "transfer_left": v83.action_signature(brain, transfer_left)["edges"] & v83.action_signature(brain, right)["edges"],
        "transfer_right": v83.action_signature(brain, transfer_right)["edges"] & v83.action_signature(brain, left)["edges"],
    }


def domain_context_signature(brain: SphereBrain, domain: DomainSpec):
    left = v83.action_signature(brain, context_item(domain.left_subject, domain.shared_context))
    right = v83.action_signature(brain, context_item(domain.right_subject, domain.shared_context))
    return {
        "nodes": left["nodes"] | right["nodes"],
        "edges": left["edges"] | right["edges"],
        "shared_nodes": left["nodes"] & right["nodes"],
        "shared_edges": left["edges"] & right["edges"],
    }


def control_context_signature(brain: SphereBrain, domain: DomainSpec):
    left = v83.action_signature(brain, context_item(domain.left_subject, domain.left_control_context))
    right = v83.action_signature(brain, context_item(domain.right_subject, domain.right_control_context))
    return {"nodes": left["nodes"] | right["nodes"], "edges": left["edges"] | right["edges"]}


def pair_metrics(brain: SphereBrain, left, right, baseline: set[tuple[int, int]]):
    return v84.pair_metrics(brain, left, right, baseline)


def measure(brain: SphereBrain, domain: DomainSpec, baseline):
    left = action_item(domain.left_subject, domain.left_action)
    right = action_item(domain.right_subject, domain.right_action)
    transfer_left = action_item(domain.transfer_left_subject, domain.left_action)
    transfer_right = action_item(domain.transfer_right_subject, domain.right_action)
    direct = pair_metrics(brain, left, right, baseline["direct"])
    tl = pair_metrics(brain, transfer_left, right, baseline["transfer_left"])
    tr = pair_metrics(brain, transfer_right, left, baseline["transfer_right"])
    return {
        "direct": direct,
        "transfer": {
            "left_to_right": tl,
            "right_to_left": tr,
            "mean_edge_similarity": (tl["edge_similarity"] + tr["edge_similarity"]) / 2.0,
            "mean_activation_similarity": (tl["activation_similarity"] + tr["activation_similarity"]) / 2.0,
            "new_shared_edges": tl["new_shared_edges"] + tr["new_shared_edges"],
        },
    }


def context_candidates(experimental, control, domain: DomainSpec, exp_baseline):
    # v84 attribution is generic once the domain fields are supplied.
    return v84.context_candidates(experimental, control, domain, exp_baseline)


def make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle: int):
    exp = measure(experimental, domain, exp_baseline)
    ctrl = measure(control, domain, ctrl_baseline)
    candidates = context_candidates(experimental, control, domain, exp_baseline["direct"])
    row = {
        "cycle": int(cycle),
        "experimental": exp,
        "control": ctrl,
        "effects": {
            "direct_edge_similarity": exp["direct"]["edge_similarity"] - ctrl["direct"]["edge_similarity"],
            "direct_activation_similarity": exp["direct"]["activation_similarity"] - ctrl["direct"]["activation_similarity"],
            "direct_new_shared_edges": exp["direct"]["new_shared_edges"] - ctrl["direct"]["new_shared_edges"],
        },
        "context_linked_candidate_count": len(candidates),
    }
    row["direct_success_now"] = v84b_cycle_success(row)
    return row


def v84b_cycle_success(row: dict) -> bool:
    return bool(
        row["experimental"]["direct"]["new_shared_edges"] > 0
        and row["effects"]["direct_new_shared_edges"] > 0
        and row["context_linked_candidate_count"] > 0
        and (
            row["effects"]["direct_edge_similarity"] > 0
            or row["effects"]["direct_activation_similarity"] > 0
        )
    )


def selective_candidates(brain, domain: DomainSpec, baseline, shared_context: bool):
    # Same candidate/Hub scoring as v87, but works with the two added DomainSpec rows.
    return v87.selective_candidates(brain, domain, baseline, shared_context)


def run_trial(domain: DomainSpec, seed: int, mode: str) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "event_triggered_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = make_baseline(experimental, domain)
    ctrl_baseline = make_baseline(control, domain)
    exp_episodes = episodes_for(domain, True)
    ctrl_episodes = episodes_for(domain, False)
    event_exp = EventTriggeredConsolidation()
    event_ctrl = EventTriggeredConsolidation()

    records = []
    protected_edges: set[tuple[int, int]] = set()
    hub_protected_edges: set[tuple[int, int]] = set()
    trigger_cycles: list[int] = []
    trigger_events = 0

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            records.append(make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle))

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

            if mode in {"event_triggered", "event_triggered_assist"}:
                exp_rows = selective_candidates(experimental, domain, exp_baseline["direct"], True)
                ctrl_rows = selective_candidates(control, domain, ctrl_baseline["direct"], False)
                result = event_exp.observe(experimental, exp_rows)
                event_ctrl.observe(control, ctrl_rows)
                for item in result.get("triggered", []):
                    trigger_cycles.append(int(item["cycle"]))
                    trigger_events += 1
                for item in result.get("protected", []):
                    edge = edge_tuple(item["edge"])
                    protected_edges.add(edge)
                    if v87.hub_score(experimental, edge) >= 0.88:
                        hub_protected_edges.add(edge)

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
        "protected_edge_count": len(protected_edges),
        "hub_protected_count": len(hub_protected_edges),
        "trigger_count": trigger_events,
        "first_trigger_cycle": min(trigger_cycles) if trigger_cycles else None,
        "trigger_cycles": sorted(set(trigger_cycles)),
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    protected = sum(x["protected_edge_count"] for x in subset)
    hub_protected = sum(x["hub_protected_count"] for x in subset)
    trigger_cycles = [x["first_trigger_cycle"] for x in subset if x["first_trigger_cycle"] is not None]
    return {
        "mode": mode,
        "trials": len(subset),
        "ever_bridge": sum(1 for x in subset if x["ever_bridge"]),
        "stable": sum(1 for x in subset if x["stable"]),
        "stable_rate": sum(1 for x in subset if x["stable"]) / len(subset),
        "never": sum(1 for x in subset if x["never"]),
        "never_rate": sum(1 for x in subset if x["never"]) / len(subset),
        "protected_edges": protected,
        "hub_protected_edges": hub_protected,
        "hub_protection_rate": hub_protected / protected if protected else 0.0,
        "triggered_trials": sum(1 for x in subset if x["first_trigger_cycle"] is not None),
        "trigger_rate": sum(1 for x in subset if x["first_trigger_cycle"] is not None) / len(subset),
        "first_trigger_cycle_median": statistics.median(trigger_cycles) if trigger_cycles else None,
        "unique_first_trigger_cycles": sorted(set(int(x) for x in trigger_cycles)),
        "mean_final_direct_similarity": sum(x["final_direct_similarity"] for x in subset) / len(subset),
    }


def paired(rows: list[dict], mode: str) -> dict:
    p = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == "primary"}
    t = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == mode}
    deltas = [t[k]["final_direct_similarity"] - p[k]["final_direct_similarity"] for k in p]
    ordered = sorted(deltas)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    improved = [k for k in p if t[k]["stable"] and not p[k]["stable"]]
    worsened = [k for k in p if p[k]["stable"] and not t[k]["stable"]]
    return {
        "stable_gain": len(improved),
        "stable_loss": len(worsened),
        "net_stable_gain": len(improved) - len(worsened),
        "never_gain": sum(1 for k in p if t[k]["never"] and not p[k]["never"]),
        "never_loss": sum(1 for k in p if p[k]["never"] and not t[k]["never"]),
        "max_similarity_delta": max(deltas),
        "p95_similarity_delta": p95,
        "mean_similarity_delta": sum(deltas) / len(deltas),
        "improved_trials": [{"domain": k[0], "seed": k[1]} for k in improved],
        "worsened_trials": [{"domain": k[0], "seed": k[1]} for k in worsened],
    }


def domain_summary(rows, mode: str):
    result = []
    for domain in DOMAINS:
        subset = [x for x in rows if x["mode"] == mode and x["domain"] == domain.name]
        result.append({
            "domain": domain.name,
            "stable": sum(1 for x in subset if x["stable"]),
            "stable_rate": sum(1 for x in subset if x["stable"]) / len(subset),
            "never": sum(1 for x in subset if x["never"]),
            "never_rate": sum(1 for x in subset if x["never"]) / len(subset),
            "triggered": sum(1 for x in subset if x["first_trigger_cycle"] is not None),
        })
    return result


def seed_summary(rows):
    result = []
    for seed in SEEDS:
        p = [x for x in rows if x["mode"] == "primary" and x["seed"] == seed]
        e = [x for x in rows if x["mode"] == "event_triggered" and x["seed"] == seed]
        a = [x for x in rows if x["mode"] == "event_triggered_assist" and x["seed"] == seed]
        result.append({
            "seed": seed,
            "primary_stable_domains": sum(1 for x in p if x["stable"]),
            "event_stable_domains": sum(1 for x in e if x["stable"]),
            "assist_stable_domains": sum(1 for x in a if x["stable"]),
            "assist_minus_primary": sum(1 for x in a if x["stable"]) - sum(1 for x in p if x["stable"]),
        })
    return result


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed, mode) for mode in MODES for domain in DOMAINS for seed in SEEDS]
    summaries = {m: summarize(trials, m) for m in MODES}
    event_pair = paired(trials, "event_triggered")
    assist_pair = paired(trials, "event_triggered_assist")

    primary = summaries["primary"]
    event = summaries["event_triggered"]
    assist = summaries["event_triggered_assist"]

    primary_domains = {x["domain"]: x for x in domain_summary(trials, "primary")}
    event_domains = domain_summary(trials, "event_triggered")
    assist_domains = domain_summary(trials, "event_triggered_assist")

    event_domain_wins = sum(1 for x in event_domains if x["stable"] > primary_domains[x["domain"]]["stable"])
    assist_domain_wins = sum(1 for x in assist_domains if x["stable"] > primary_domains[x["domain"]]["stable"])
    event_domain_losses = sum(1 for x in event_domains if x["stable"] < primary_domains[x["domain"]]["stable"])
    assist_domain_losses = sum(1 for x in assist_domains if x["stable"] < primary_domains[x["domain"]]["stable"])

    seeds = seed_summary(trials)
    assist_seed_wins = sum(1 for x in seeds if x["assist_minus_primary"] > 0)
    assist_seed_losses = sum(1 for x in seeds if x["assist_minus_primary"] < 0)

    event_distinctness = event_pair["max_similarity_delta"] <= 0.12
    assist_distinctness = assist_pair["max_similarity_delta"] <= 0.12
    adaptive = len(event["unique_first_trigger_cycles"]) >= 3 or len(assist["unique_first_trigger_cycles"]) >= 3

    # Robust validation requires more than a one-trial win.
    event_robust = (
        event_pair["net_stable_gain"] >= 5
        and event["never"] <= primary["never"]
        and event_domain_wins >= 3
        and event_domain_losses <= 1
        and event_distinctness
    )
    assist_robust = (
        assist_pair["net_stable_gain"] >= 5
        and assist["never"] <= primary["never"]
        and assist_domain_wins >= 3
        and assist_domain_losses <= 1
        and assist_seed_wins >= 5
        and assist_seed_losses <= 3
        and assist_distinctness
    )

    representative = next((x for x in trials if x["mode"] == "event_triggered_assist" and x["stable"]), next(x for x in trials if x["mode"] == "event_triggered_assist"))
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "event_triggered_robustness_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    overall_pass = (
        (event_robust or assist_robust)
        and adaptive
        and event["hub_protection_rate"] == 0.0
        and assist["hub_protection_rate"] == 0.0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if assist_robust:
        winner = "event_triggered_assist"
    elif event_robust:
        winner = "event_triggered"
    else:
        winner = "primary"

    if overall_pass:
        readiness = "semantic_event_triggered_consolidation_robustness_validated"
        verdict = "event_triggered_semantic_consolidation_reproduces_across_expanded_seeds_and_domains_without_hub_collapse"
        next_step = "stress_test_winner_under_context_switching_contradiction_and_forgetting_before_primary_core_integration"
    else:
        readiness = "semantic_event_triggered_consolidation_robustness_not_yet_validated"
        verdict = "v88_gain_does_not_yet_reproduce_strongly_enough_across_expanded_seeds_and_domains"
        next_step = "attribute_seed_domain_and_assist_failure_modes_without_retuning_event_thresholds"

    payload = {
        "experiment": "Core Growth Binding v89 — Event-Triggered Consolidation Robustness Validation",
        "contract": {
            "primary_core_modified": False,
            "event_algorithm_modified": False,
            "modes": MODES,
            "seed_count": len(SEEDS),
            "domain_count": len(DOMAINS),
            "trials_per_mode": len(SEEDS) * len(DOMAINS),
            "total_trials": len(trials),
            "checkpoints": CHECKPOINTS,
            "semantic_answer_labels_used": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "primary_stable": primary["stable"],
            "primary_stable_rate": primary["stable_rate"],
            "event_stable": event["stable"],
            "event_stable_rate": event["stable_rate"],
            "assist_stable": assist["stable"],
            "assist_stable_rate": assist["stable_rate"],
            "primary_never": primary["never"],
            "event_never": event["never"],
            "assist_never": assist["never"],
            "event_trigger_rate": event["trigger_rate"],
            "assist_trigger_rate": assist["trigger_rate"],
            "event_trigger_cycles": event["unique_first_trigger_cycles"],
            "assist_trigger_cycles": assist["unique_first_trigger_cycles"],
            "adaptive_timing": adaptive,
            "event_hub_protection_rate": event["hub_protection_rate"],
            "assist_hub_protection_rate": assist["hub_protection_rate"],
            "event_domain_wins": event_domain_wins,
            "assist_domain_wins": assist_domain_wins,
            "event_domain_losses": event_domain_losses,
            "assist_domain_losses": assist_domain_losses,
            "assist_seed_wins": assist_seed_wins,
            "assist_seed_losses": assist_seed_losses,
            "event_net_stable_gain": event_pair["net_stable_gain"],
            "assist_net_stable_gain": assist_pair["net_stable_gain"],
            "event_distinctness_safe": event_distinctness,
            "assist_distinctness_safe": assist_distinctness,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "robustness_pass": overall_pass,
            "winner": winner,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "paired": {"event_triggered": event_pair, "event_triggered_assist": assist_pair},
        "domain_rows": {
            "primary": list(primary_domains.values()),
            "event_triggered": event_domains,
            "event_triggered_assist": assist_domains,
        },
        "seed_rows": seeds,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in trials],
    }
    (OUT / "latest_binding_v89.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v89</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v89：Event-Triggered Consolidation Robustness Validation</h1><p class="lead">v88の閾値を変更せず、20 seed × 5 domain × 3 mode = 300試行へ拡大。Stable率、Never率、発火分布、Hub保護、Distinctness、seed/domain依存、Assistの改善/悪化を検証する。</p><section class="panel"><div class="controls"><button id="run">100試行/条件で再現性を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Seed別</h2><pre id="seeds" class="raw">未実行</pre></section><section class="panel"><h2>Paired / Raw</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function pct(x){return (100*x).toFixed(1)+'%'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','300 Trial 実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial / mode','100'),metric('Primary Stable',`${s.primary_stable} (${pct(s.primary_stable_rate)})`),metric('Event Stable',`${s.event_stable} (${pct(s.event_stable_rate)})`,s.event_stable>s.primary_stable?'good':'blue'),metric('Event+Assist Stable',`${s.assist_stable} (${pct(s.assist_stable_rate)})`,s.assist_stable>s.primary_stable?'good':'blue'),metric('Primary Never',s.primary_never),metric('Event Never',s.event_never,s.event_never<=s.primary_never?'good':'warn'),metric('Event+Assist Never',s.assist_never,s.assist_never<=s.primary_never?'good':'warn'),metric('Assist net Stable gain',s.assist_net_stable_gain,s.assist_net_stable_gain>=5?'good':'warn'),metric('Event発火率',pct(s.event_trigger_rate)),metric('Assist発火率',pct(s.assist_trigger_rate)),metric('Adaptive timing',s.adaptive_timing?'YES':'NO',s.adaptive_timing?'good':'warn'),metric('Hub保護率',`${pct(s.event_hub_protection_rate)} / ${pct(s.assist_hub_protection_rate)}`,(s.event_hub_protection_rate===0&&s.assist_hub_protection_rate===0)?'good':'warn'),metric('Domain勝ち Event/Assist',`${s.event_domain_wins} / ${s.assist_domain_wins}`),metric('Assist Seed勝ち/負け',`${s.assist_seed_wins} / ${s.assist_seed_losses}`),metric('Distinctness',s.event_distinctness_safe&&s.assist_distinctness_safe?'YES':'NO',s.event_distinctness_safe&&s.assist_distinctness_safe?'good':'warn'),metric('Save/Load',s.saveload_equal?'YES':'NO',s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変更',s.brain_file_unchanged?'good':'warn'),metric('v89 PASS',s.robustness_pass?'YES':'NO',s.robustness_pass?'good':'warn'),metric('Winner',s.winner),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('seeds').textContent=JSON.stringify(d.seed_rows,null,2);document.getElementById('raw').textContent=JSON.stringify({paired:d.paired,mode_summaries:d.mode_summaries,summary:d.summary},null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"v89 UI: {url}")
    threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
