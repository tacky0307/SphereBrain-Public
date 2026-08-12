from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import sys
import threading
import webbrowser
from collections import defaultdict
from pathlib import Path
from statistics import median

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v70 as v70
import run_core_growth_binding_v75 as v75
import run_core_growth_binding_v77 as v77

HOST = "127.0.0.1"
START_PORT = 5124
OUT = ROOT / "data" / "core_growth_binding_v78" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEED_COUNT = 12
BASE_SEED = 780100
EPISODES_PER_PHASE = 24
MODES = ["temporal_credit", "consolidated_temporal"]

A = {"name": "base_A", "start": 0, "goal": 8, "blocked": {3, 4}}
B = {"name": "route_blocked_B", "start": 0, "goal": 8, "blocked": {2, 4}}
C = {"name": "alternate_block_C", "start": 0, "goal": 8, "blocked": {1, 4}}
PHASES = [A, B, A, C, B, A]


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


def shortest_steps(case: dict) -> int | None:
    return v75.shortest_steps(case)


def eval_episode(brain: SphereBrain, case: dict) -> dict:
    ep = v70.run_episode(brain, case, relative_mode=True, assist=False)
    optimal = shortest_steps(case)
    return {
        **ep,
        "optimal": bool(ep["success"] and optimal is not None and int(ep["steps"]) == optimal),
        "optimal_steps": optimal,
    }


def run_long_horizon(pretrained: SphereBrain, mode: str, seed: int) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    success_counts: dict[str, int] = defaultdict(int)
    stability: dict[str, float] = defaultdict(float)

    phase_rows = []
    total_credited = 0
    total_protected = 0
    total_promotions = 0
    total_extra_replays = 0
    total_loops = 0
    stability_ceiling_hits = 0
    stability_observations = 0
    prior_first_optimal: dict[str, int | None] = {}

    global_episode = 0
    for phase_index, case in enumerate(PHASES, start=1):
        start_eval = eval_episode(brain, case)
        first_success = None
        first_optimal = None
        success_episodes = 0
        optimal_episodes = 0
        phase_loops = 0
        phase_credited = 0
        phase_protected = 0
        phase_promotions = 0

        for local_episode in range(1, EPISODES_PER_PHASE + 1):
            global_episode += 1
            row = v77.training_episode(
                brain,
                case,
                mode,
                rng,
                motif_counts,
                success_counts,
                stability,
                global_episode,
            )
            ep = row["episode"]
            events = row["events"]
            consolidation = row["consolidation"]
            optimal_steps = shortest_steps(case)
            is_optimal = bool(ep["success"] and optimal_steps is not None and int(ep["steps"]) == optimal_steps)

            phase_loops += int(ep.get("loop_steps", 0))
            total_loops += int(ep.get("loop_steps", 0))
            credited = sum(1 for e in events if e.get("credited"))
            protected = sum(1 for e in events if e.get("credited") and e.get("protected"))
            phase_credited += credited
            phase_protected += protected
            total_credited += credited
            total_protected += protected
            promoted = int(consolidation.get("promoted", 0))
            phase_promotions += promoted
            total_promotions += promoted
            total_extra_replays += int(consolidation.get("extra_replayed", 0))

            if ep["success"]:
                success_episodes += 1
                if first_success is None:
                    first_success = local_episode
            if is_optimal:
                optimal_episodes += 1
                if first_optimal is None:
                    first_optimal = local_episode

            if mode == "consolidated_temporal":
                values = list(stability.values())
                stability_observations += max(1, len(values))
                if values:
                    stability_ceiling_hits += sum(1 for v in values if float(v) >= v77.MAX_STABILITY - 1e-12)

        end_eval = eval_episode(brain, case)
        name = str(case["name"])
        previous = prior_first_optimal.get(name)
        revisit_faster = bool(previous is not None and first_optimal is not None and first_optimal < previous)
        if first_optimal is not None and name not in prior_first_optimal:
            prior_first_optimal[name] = first_optimal

        phase_rows.append({
            "phase": phase_index,
            "environment": name,
            "start_success": bool(start_eval["success"]),
            "start_optimal": bool(start_eval["optimal"]),
            "end_success": bool(end_eval["success"]),
            "end_optimal": bool(end_eval["optimal"]),
            "first_success_episode": first_success,
            "first_optimal_episode": first_optimal,
            "success_episode_count": success_episodes,
            "optimal_episode_count": optimal_episodes,
            "mean_loop_steps": phase_loops / EPISODES_PER_PHASE,
            "credited_loop_events": phase_credited,
            "protected_loop_events": phase_protected,
            "consolidation_promotions": phase_promotions,
            "consolidated_transition_count": sum(1 for v in stability.values() if float(v) >= v77.PROTECTION_STABILITY_GATE),
            "max_stability": max([float(v) for v in stability.values()] or [0.0]),
            "revisit_faster_than_first": revisit_faster,
            "end_path": end_eval["path"],
            "end_steps": int(end_eval["steps"]),
        })

    final_a = eval_episode(brain, A)
    max_stability = max([float(v) for v in stability.values()] or [0.0])
    ceiling_rate = 0.0 if stability_observations == 0 else stability_ceiling_hits / stability_observations
    return {
        "phase_rows": phase_rows,
        "final_A_success": bool(final_a["success"]),
        "final_A_optimal": bool(final_a["optimal"]),
        "final_A_path": final_a["path"],
        "total_credited_loop_events": total_credited,
        "total_protected_loop_events": total_protected,
        "total_consolidation_promotions": total_promotions,
        "total_extra_replays": total_extra_replays,
        "mean_loop_steps": total_loops / (len(PHASES) * EPISODES_PER_PHASE),
        "consolidated_transition_count": sum(1 for v in stability.values() if float(v) >= v77.PROTECTION_STABILITY_GATE),
        "max_stability": max_stability,
        "stability_ceiling_rate": ceiling_rate,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    all_phases = [p for r in rows for p in r["phase_rows"]]
    first_opts = [int(p["first_optimal_episode"]) for p in all_phases if p["first_optimal_episode"] is not None]
    returns = [p for p in all_phases if p["environment"] == "base_A" and int(p["phase"]) > 1]
    switch_phases = [p for p in all_phases if int(p["phase"]) > 1]
    rigid_failures = sum(1 for p in switch_phases if not p["end_success"])
    return {
        "seed_count": n,
        "phase_count": len(all_phases),
        "phase_end_success_rate": sum(1 for p in all_phases if p["end_success"]) / len(all_phases),
        "phase_end_optimal_rate": sum(1 for p in all_phases if p["end_optimal"]) / len(all_phases),
        "switch_phase_failure_rate": rigid_failures / len(switch_phases),
        "median_first_optimal_episode": None if not first_opts else float(median(first_opts)),
        "mean_loop_steps": sum(float(r["mean_loop_steps"]) for r in rows) / n,
        "return_A_end_success_rate": sum(1 for p in returns if p["end_success"]) / len(returns),
        "return_A_end_optimal_rate": sum(1 for p in returns if p["end_optimal"]) / len(returns),
        "revisit_faster_count": sum(1 for p in all_phases if p["revisit_faster_than_first"]),
        "final_A_success_rate": sum(1 for r in rows if r["final_A_success"]) / n,
        "final_A_optimal_rate": sum(1 for r in rows if r["final_A_optimal"]) / n,
        "mean_consolidated_transitions": sum(int(r["consolidated_transition_count"]) for r in rows) / n,
        "mean_max_stability": sum(float(r["max_stability"]) for r in rows) / n,
        "mean_stability_ceiling_rate": sum(float(r["stability_ceiling_rate"]) for r in rows) / n,
        "mean_protected_loop_events": sum(int(r["total_protected_loop_events"]) for r in rows) / n,
    }


def metric(row: dict) -> tuple[int, int, float, float]:
    return (
        1 if row["final_A_success"] else 0,
        1 if row["final_A_optimal"] else 0,
        -float(row["mean_loop_steps"]),
        -float(row["stability_ceiling_rate"]),
    )


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = v70.pretrain(base)

    by_mode = {m: [] for m in MODES}
    seed_rows = []
    consolidated_better = consolidated_worse = 0

    for i in range(SEED_COUNT):
        seed = BASE_SEED + i
        results = {m: run_long_horizon(pretrained, m, seed) for m in MODES}
        for m in MODES:
            by_mode[m].append(results[m])
        if metric(results["consolidated_temporal"]) > metric(results["temporal_credit"]):
            consolidated_better += 1
        elif metric(results["consolidated_temporal"]) < metric(results["temporal_credit"]):
            consolidated_worse += 1
        seed_rows.append({"seed": seed, **results})

    aggregates = {m: aggregate(by_mode[m]) for m in MODES}
    t = aggregates["temporal_credit"]
    c = aggregates["consolidated_temporal"]

    rigidity_not_worse = c["switch_phase_failure_rate"] <= t["switch_phase_failure_rate"] + 1e-12
    return_memory_not_worse = c["return_A_end_optimal_rate"] >= t["return_A_end_optimal_rate"] - 1e-12
    final_not_worse = c["final_A_optimal_rate"] >= t["final_A_optimal_rate"] - 1e-12
    ceiling_safe = c["mean_stability_ceiling_rate"] < 0.35
    brain_unchanged = before_hash == file_hash(BRAIN_PATH)

    pass_all = rigidity_not_worse and return_memory_not_worse and final_not_worse and ceiling_safe and brain_unchanged
    if pass_all and c["return_A_end_optimal_rate"] > t["return_A_end_optimal_rate"]:
        verdict = "consolidation_preserves_plasticity_and_improves_long_horizon_return_memory"
        readiness = "temporal_consolidation_long_horizon_ready"
        next_step = "promote_temporal_credit_and_success_consolidation_to_native_core_learning_candidate"
    elif pass_all:
        verdict = "consolidation_remains_stable_long_horizon_without_rigidity_but_additional_memory_gain_is_small"
        readiness = "temporal_consolidation_long_horizon_stable"
        next_step = "native_integrate_temporal_credit_first_and_keep_consolidation_optional_until_broader_memory_tests"
    elif not rigidity_not_worse or not ceiling_safe:
        verdict = "consolidation_accumulates_rigidity_or_excessive_stability_over_long_horizon"
        readiness = "consolidation_not_ready_for_native_core"
        next_step = "reduce_protection_or_add_stability_homeostasis_before_native_integration"
    else:
        verdict = "long_horizon_memory_or_final_recovery_regresses_under_consolidation"
        readiness = "consolidation_needs_reanalysis"
        next_step = "audit_phase_specific_interference_before_native_integration"

    payload = {
        "experiment": "Core Growth Binding v78 — Temporal + Consolidation Long-Horizon Stability",
        "contract": {
            "seed_count": SEED_COUNT,
            "phase_sequence": [p["name"] for p in PHASES],
            "episodes_per_phase": EPISODES_PER_PHASE,
            "total_episodes_per_seed": len(PHASES) * EPISODES_PER_PHASE,
            "paired_rng_seed_per_mode": True,
            "same_pretrained_core_per_mode": True,
            "assist_used": False,
            "consolidation_permanent_lock": False,
            "production_brain_json_saved": False,
        },
        "aggregates": aggregates,
        "paired_comparison": {
            "consolidated_better_seeds": consolidated_better,
            "consolidated_worse_seeds": consolidated_worse,
            "rigidity_not_worse": rigidity_not_worse,
            "return_memory_not_worse": return_memory_not_worse,
            "final_A_not_worse": final_not_worse,
            "stability_ceiling_safe": ceiling_safe,
        },
        "summary": {
            "long_horizon_pass": pass_all,
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "seed_rows": seed_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v78.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v78</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:950px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v78：Temporal + Consolidation Long-Horizon Stability</h1><p class="lead">A→B→A→C→B→A を各24 Episode、計144 Episode流し、Temporal Credit単体とSuccess Consolidation併用を長期比較する。適応・復帰・硬直・stability飽和を同時監査する。</p><section class="panel"><div class="controls"><button id="run">Long-Horizonを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>集計</h2><pre id="agg" class="raw">未実行</pre></section><section class="panel"><h2>Seed別</h2><pre id="rows" class="raw">未実行</pre></section><script>
const pct=x=>(100*x).toFixed(1)+'%';function m(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=m('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=m('エラー',d.error||'失敗','warn');return}const t=d.aggregates.temporal_credit,c=d.aggregates.consolidated_temporal,p=d.paired_comparison,s=d.summary;document.getElementById('metrics').innerHTML=[m('Temporal phase成功',pct(t.phase_end_success_rate)),m('Consolidated phase成功',pct(c.phase_end_success_rate)),m('Temporal phase最短',pct(t.phase_end_optimal_rate)),m('Consolidated phase最短',pct(c.phase_end_optimal_rate)),m('Temporal A復帰最短',pct(t.return_A_end_optimal_rate)),m('Consolidated A復帰最短',pct(c.return_A_end_optimal_rate)),m('Temporal 最終A',pct(t.final_A_optimal_rate)),m('Consolidated 最終A',pct(c.final_A_optimal_rate)),m('Consolidated改善seed',p.consolidated_better_seeds),m('Consolidated悪化seed',p.consolidated_worse_seeds),m('硬直悪化なし',p.rigidity_not_worse?'YES':'NO',p.rigidity_not_worse?'good':'warn'),m('Stability飽和安全',p.stability_ceiling_safe?'YES':'NO',p.stability_ceiling_safe?'good':'warn'),m('Long-Horizon PASS',s.long_horizon_pass?'YES':'NO',s.long_horizon_pass?'good':'warn'),m('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),m('Core readiness',s.core_readiness),m('総合判定',s.overall_verdict)].join('');document.getElementById('agg').textContent=JSON.stringify({aggregates:d.aggregates,paired_comparison:d.paired_comparison,contract:d.contract},null,2);document.getElementById('rows').textContent=JSON.stringify(d.seed_rows,null,2)}document.getElementById('run').onclick=run;
</script></body></html>'''


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
    print(f"Core Growth Binding v78: http://{HOST}:{PORT}")
    print("A→B→A→C→B→A / 24 episodes per phase / 12 paired seeds / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
