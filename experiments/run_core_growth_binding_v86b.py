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
import run_core_growth_binding_v82c as v82c
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86

HOST = "127.0.0.1"
START_PORT = 5137
OUT = ROOT / "data" / "core_growth_binding_v86b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v85.CHECKPOINTS
SEEDS = v84.SEEDS
DOMAINS = v84.DOMAINS
MODES = v86.MODES


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


def node_usage_percentile(brain: SphereBrain, node: int) -> float:
    values = np.asarray(brain.node_usage, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= float(values[int(node)])))


def edge_hub_like(brain: SphereBrain, edge: tuple[int, int]) -> bool:
    a, b = edge
    degree_a = int(np.count_nonzero(brain.adjacency[a]))
    degree_b = int(np.count_nonzero(brain.adjacency[b]))
    usage_pct = max(node_usage_percentile(brain, a), node_usage_percentile(brain, b))
    return usage_pct >= 0.90 or max(degree_a, degree_b) >= max(12, brain.neighbors_per_node * 2)


def success_candidate_edges(experimental: SphereBrain, control: SphereBrain, domain, baseline) -> set[tuple[int, int]]:
    rows = v84.context_candidates(experimental, control, domain, baseline["direct"])
    return {edge_tuple(row["edge"]) for row in rows}


def run_detailed_mode(domain, seed: int, mode: str) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "homeostatic_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = v84.make_baseline(experimental, domain)
    ctrl_baseline = v84.make_baseline(control, domain)
    exp_episodes = v84.episodes_for(domain, True)
    ctrl_episodes = v84.episodes_for(domain, False)
    exp_homeostasis = SemanticBridgeHomeostasis()
    ctrl_homeostasis = SemanticBridgeHomeostasis()
    use_homeostasis = mode != "primary"

    records: list[dict] = []
    protection_events: list[dict] = []
    unique_candidates: set[tuple[int, int]] = set()
    success_bridge_edges: set[tuple[int, int]] = set()
    first_candidate_cycle = None
    first_protection_cycle = None
    assist_steps = 0

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v85.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            current_candidates = {
                edge_tuple(x["edge"])
                for x in row.get("context_linked_candidates", [])
            }
            unique_candidates.update(current_candidates)
            if current_candidates and first_candidate_cycle is None:
                first_candidate_cycle = cycle
            if row["direct_success_now"]:
                success_bridge_edges.update(current_candidates)
            row["candidate_edge_list"] = [list(x) for x in sorted(current_candidates)]
            row["semantic_stability"] = exp_homeostasis.snapshot()
            records.append(row)

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            assist_steps += sum(
                1 for trace in getattr(experimental, "last_structural_assist_trace", [])
                if trace.get("eligible") or trace.get("applied")
            )
            v83.train_episode_set(control, ctrl_episodes)

            if use_homeostasis:
                exp_candidates = v86.intrinsic_context_candidate_edges(
                    experimental, domain, exp_baseline["direct"], shared_context=True
                )
                ctrl_candidates = v86.intrinsic_context_candidate_edges(
                    control, domain, ctrl_baseline["direct"], shared_context=False
                )
                unique_candidates.update(tuple(x) for x in exp_candidates)
                if exp_candidates and first_candidate_cycle is None:
                    first_candidate_cycle = cycle + 1
                exp_result = exp_homeostasis.observe(experimental, exp_candidates)
                ctrl_homeostasis.observe(control, ctrl_candidates)
                protected = exp_result.get("protected", [])
                if protected and first_protection_cycle is None:
                    first_protection_cycle = cycle + 1
                for item in protected:
                    edge = edge_tuple(item["edge"])
                    protection_events.append({
                        "cycle": cycle + 1,
                        "edge": list(edge),
                        "count": int(item.get("count", 0)),
                        "stability_before": float(item.get("stability_before", 0.0)),
                        "stability_after": float(item.get("stability_after", 0.0)),
                        "weight_delta": float(item.get("weight_delta", 0.0)),
                        "hub_like_at_protection": edge_hub_like(experimental, edge),
                    })

    classification = v86.classify(records)
    active_cycles = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    promoted_edges = {
        edge_tuple(item["edge"])
        for item in protection_events
        if item["stability_before"] < 0.35 <= item["stability_after"]
    }
    protected_edges = {edge_tuple(item["edge"]) for item in protection_events}
    hub_protected = {edge_tuple(item["edge"]) for item in protection_events if item["hub_like_at_protection"]}

    return {
        "domain": domain.name,
        "seed": int(seed),
        "mode": mode,
        "classification": classification,
        "ever_bridge": bool(active_cycles),
        "stable": classification == "stable",
        "never": classification == "never",
        "first_bridge_cycle": active_cycles[0] if active_cycles else None,
        "first_candidate_cycle": first_candidate_cycle,
        "first_protection_cycle": first_protection_cycle,
        "unique_candidate_count": len(unique_candidates),
        "unique_candidate_edges": [list(x) for x in sorted(unique_candidates)],
        "success_bridge_edges": [list(x) for x in sorted(success_bridge_edges)],
        "protected_edge_count": len(protected_edges),
        "protected_edges": [list(x) for x in sorted(protected_edges)],
        "promoted_edges": [list(x) for x in sorted(promoted_edges)],
        "hub_protected_count": len(hub_protected),
        "hub_protected_edges": [list(x) for x in sorted(hub_protected)],
        "protection_events": protection_events,
        "assist_eligible_or_applied_steps": assist_steps,
        "final_semantic_stability": exp_homeostasis.snapshot(),
        "records": records,
        "experimental": experimental,
    }


def pair_index(rows: list[dict], mode: str) -> dict[tuple[str, int], dict]:
    return {(x["domain"], int(x["seed"])): x for x in rows if x["mode"] == mode}


def set_edges(rows) -> set[tuple[int, int]]:
    return {edge_tuple(x) for x in rows}


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [
        run_detailed_mode(domain, seed, mode)
        for mode in MODES
        for domain in DOMAINS
        for seed in SEEDS
    ]

    primary = pair_index(trials, "primary")
    homeo = pair_index(trials, "homeostatic")
    assist = pair_index(trials, "homeostatic_assist")

    degraded = []
    rescued = []
    all_keys = sorted(primary)

    total_protected = 0
    total_hub_protected = 0
    total_overlap_edges = 0
    total_primary_success_edges = 0
    early_protection_cases = 0
    diversity_losses = 0

    for key in all_keys:
        p = primary[key]
        h = homeo[key]
        a = assist[key]

        if p["ever_bridge"] and h["never"]:
            p_success = set_edges(p["success_bridge_edges"])
            h_protected = set_edges(h["protected_edges"])
            overlap = p_success & h_protected
            wrong = h_protected - p_success
            early = (
                h["first_protection_cycle"] is not None
                and p["first_bridge_cycle"] is not None
                and int(h["first_protection_cycle"]) < int(p["first_bridge_cycle"])
            )
            early_protection_cases += int(early)
            diversity_delta = h["unique_candidate_count"] - p["unique_candidate_count"]
            diversity_losses += int(diversity_delta < 0)
            total_protected += len(h_protected)
            total_hub_protected += int(h["hub_protected_count"])
            total_overlap_edges += len(overlap)
            total_primary_success_edges += len(p_success)
            degraded.append({
                "domain": key[0],
                "seed": key[1],
                "primary_first_bridge_cycle": p["first_bridge_cycle"],
                "homeo_first_candidate_cycle": h["first_candidate_cycle"],
                "homeo_first_protection_cycle": h["first_protection_cycle"],
                "protection_before_primary_bridge": early,
                "primary_success_bridge_edges": [list(x) for x in sorted(p_success)],
                "homeo_protected_edges": [list(x) for x in sorted(h_protected)],
                "protected_overlap_with_primary_success": [list(x) for x in sorted(overlap)],
                "protected_not_in_primary_success": [list(x) for x in sorted(wrong)],
                "homeo_hub_protected_count": h["hub_protected_count"],
                "primary_candidate_diversity": p["unique_candidate_count"],
                "homeo_candidate_diversity": h["unique_candidate_count"],
                "candidate_diversity_delta_homeo_minus_primary": diversity_delta,
            })

        if h["never"] and a["ever_bridge"]:
            rescued.append({
                "domain": key[0],
                "seed": key[1],
                "assist_first_bridge_cycle": a["first_bridge_cycle"],
                "homeo_candidate_diversity": h["unique_candidate_count"],
                "assist_candidate_diversity": a["unique_candidate_count"],
                "candidate_diversity_gain": a["unique_candidate_count"] - h["unique_candidate_count"],
                "homeo_first_candidate_cycle": h["first_candidate_cycle"],
                "assist_first_candidate_cycle": a["first_candidate_cycle"],
                "assist_eligible_or_applied_steps": a["assist_eligible_or_applied_steps"],
                "homeo_hub_protected_count": h["hub_protected_count"],
                "assist_hub_protected_count": a["hub_protected_count"],
            })

    overlap_ratio = (
        total_overlap_edges / total_primary_success_edges
        if total_primary_success_edges else 0.0
    )
    wrong_protection_ratio = (
        max(0, total_protected - total_overlap_edges) / total_protected
        if total_protected else 0.0
    )
    hub_protection_ratio = (
        total_hub_protected / total_protected if total_protected else 0.0
    )

    representative = homeo[all_keys[0]]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_consolidation_attribution_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    early_signal = bool(degraded) and early_protection_cases >= max(1, len(degraded) // 2)
    wrong_target_signal = bool(degraded) and wrong_protection_ratio >= 0.50
    diversity_signal = bool(degraded) and diversity_losses >= max(1, len(degraded) // 2)
    assist_signal = len(rescued) > 0

    attribution_pass = (
        bool(degraded)
        and (early_signal or wrong_target_signal or diversity_signal)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if early_signal and wrong_target_signal:
        readiness = "semantic_consolidation_timing_and_target_failure_attributed"
        verdict = "homeostasis_often_protects_edges_before_primary_bridge_selection_and_many_protected_edges_do_not_match_later_primary_success_bridges"
        next_step = "test_delayed_competition_window_before_semantic_consolidation_without_changing_primary_core"
    elif early_signal:
        readiness = "semantic_consolidation_timing_failure_attributed"
        verdict = "homeostasis_protection_begins_too_early_relative_to_primary_bridge_selection"
        next_step = "test_delayed_semantic_consolidation_gate_after_candidate_competition_window"
    elif wrong_target_signal:
        readiness = "semantic_consolidation_target_failure_attributed"
        verdict = "homeostasis_protects_many_edges_that_do_not_match_later_primary_success_bridge_edges"
        next_step = "test_reobservation_consensus_gate_before_semantic_edge_protection"
    elif diversity_signal:
        readiness = "semantic_candidate_diversity_suppression_observed"
        verdict = "homeostasis_reduces_candidate_diversity_in_primary_success_homeostasis_failure_trials"
        next_step = "test_candidate_competition_period_before_any_semantic_protection"
    else:
        readiness = "semantic_consolidation_failure_partially_attributed"
        verdict = "homeostasis_failure_is_reproduced_but_no_single_measured_failure_mechanism_dominates"
        next_step = "inspect_per_edge_protection_timeline_before_new_consolidation_design"

    payload = {
        "experiment": "Core Growth Binding v86B — Semantic Consolidation Failure Attribution",
        "contract": {
            "primary_core_modified": False,
            "semantic_homeostasis_modified": False,
            "replays_v86_modes": MODES,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "checkpoints": CHECKPOINTS,
            "focus": "primary_success_homeostasis_never_and_homeostasis_never_assist_rescue",
            "production_brain_json_saved": False,
        },
        "summary": {
            "degraded_trial_count": len(degraded),
            "assist_rescue_count": len(rescued),
            "early_protection_degraded_trials": early_protection_cases,
            "candidate_diversity_loss_degraded_trials": diversity_losses,
            "protected_edge_count_in_degraded_trials": total_protected,
            "protected_overlap_with_primary_success_count": total_overlap_edges,
            "primary_success_bridge_edge_count": total_primary_success_edges,
            "protected_overlap_ratio": overlap_ratio,
            "wrong_protection_ratio": wrong_protection_ratio,
            "hub_protected_count": total_hub_protected,
            "hub_protection_ratio": hub_protection_ratio,
            "early_protection_signal": early_signal,
            "wrong_target_signal": wrong_target_signal,
            "candidate_diversity_suppression_signal": diversity_signal,
            "assist_rescue_signal": assist_signal,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "degraded_trials": degraded,
        "assist_rescues": rescued,
        "mode_trials": [
            {k: v for k, v in x.items() if k not in {"experimental", "records", "protection_events"}}
            for x in trials
        ],
    }
    (OUT / "latest_binding_v86b.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v86B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v86B：Semantic Consolidation Failure Attribution</h1><p class="lead">PrimaryでBridgeが形成されたのにHomeostasisでNeverになった試行を中心に、保護開始cycle・保護Edge・Primary成功Bridgeとの重なり・Hub誤保護・候補多様性を追う。Homeo+Assistが救済した試行も同時に比較する。</p><section class="panel"><div class="controls"><button id="run">失敗原因を追跡</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Primary成功 → Homeo失敗</h2><pre id="degraded" class="raw">未実行</pre></section><section class="panel"><h2>Assist救済</h2><pre id="rescued" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function pct(v){return `${(100*v).toFixed(1)}%`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('劣化Trial',s.degraded_trial_count,s.degraded_trial_count?'warn':'good'),metric('早期保護',s.early_protection_degraded_trials,s.early_protection_signal?'warn':'blue'),metric('誤保護率',pct(s.wrong_protection_ratio),s.wrong_target_signal?'warn':'blue'),metric('Primary成功Edge重複率',pct(s.protected_overlap_ratio)),metric('Hub保護率',pct(s.hub_protection_ratio)),metric('候補多様性低下',s.candidate_diversity_loss_degraded_trials,s.candidate_diversity_suppression_signal?'warn':'blue'),metric('Assist救済',s.assist_rescue_count,s.assist_rescue_signal?'good':'blue'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('degraded').textContent=JSON.stringify(d.degraded_trials,null,2);document.getElementById('rescued').textContent=JSON.stringify(d.assist_rescues,null,2)}document.getElementById('run').onclick=run;</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    try:
        return jsonify(observe())
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v86B: http://{HOST}:{PORT}")
    print("Semantic consolidation failure attribution / Primary vs Homeo vs Assist / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
