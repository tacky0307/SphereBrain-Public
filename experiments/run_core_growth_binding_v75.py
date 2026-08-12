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
import run_core_growth_binding_v73 as v73

HOST = "127.0.0.1"
START_PORT = 5121
OUT = ROOT / "data" / "core_growth_binding_v75" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS_PER_ENV = 20
BASE_SEED = 750100
ADAPT_EPISODES = 20
RESTORE_CHANGE_EPISODES = 12
RESTORE_BASE_EPISODES = 8
MODES = ["legacy_relative", "temporal_credit"]

ENVIRONMENTS = [
    {"name": "route_blocked", "start": 0, "goal": 8, "blocked": {2, 4}},
    {"name": "alternate_block", "start": 0, "goal": 8, "blocked": {1, 4}},
    {"name": "new_goal", "start": 0, "goal": 5, "blocked": {3, 4}},
    {"name": "new_start", "start": 1, "goal": 8, "blocked": {3, 4}},
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


def shortest_steps(case: dict) -> int | None:
    start, goal, blocked = int(case["start"]), int(case["goal"]), set(case["blocked"])
    q = [(start, 0)]
    seen = {start}
    for node, dist in q:
        if node == goal:
            return dist
        for _, nxt in v70.legal_moves(node, blocked):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return None


def metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def run_training_episode(brain: SphereBrain, case: dict, mode: str, rng: random.Random, motif_counts: dict[str, int], episode_index: int) -> dict:
    temporal = mode == "temporal_credit"
    ep = v70.run_episode(
        brain,
        case,
        relative_mode=True,
        assist=False,
        rng=rng,
        explore=v70.EXPLORATION,
        max_steps=v70.TRAIN_MAX_STEPS,
    )
    v70.observe_episode(brain, ep, include_relative=True)
    credited = 0
    if ep["success"]:
        v70.reinforce_success(brain, ep, include_relative=True)
    elif temporal:
        events = v73.temporal_attribution(brain, ep, motif_counts, episode_index=episode_index)
        credited = sum(1 for x in events if x.get("credited"))
    return {"episode": ep, "credited": credited}


def paired_branch(pretrained: SphereBrain, case: dict, mode: str, seed: int, episodes: int = ADAPT_EPISODES) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    first_success = None
    success_episodes = 0
    total_loops = 0
    credited_events = 0

    for i in range(1, episodes + 1):
        row = run_training_episode(brain, case, mode, rng, motif_counts, i)
        ep = row["episode"]
        credited_events += int(row["credited"])
        total_loops += int(ep.get("loop_steps", 0))
        if ep["success"]:
            success_episodes += 1
            if first_success is None:
                first_success = i

    final_ep = v70.run_episode(brain, case, relative_mode=True, assist=False)
    optimal_steps = shortest_steps(case)
    return {
        "final": final_ep,
        "final_success": bool(final_ep["success"]),
        "final_optimal": bool(final_ep["success"] and optimal_steps is not None and int(final_ep["steps"]) == optimal_steps),
        "optimal_steps": optimal_steps,
        "first_success_episode": first_success,
        "success_episodes": success_episodes,
        "mean_loop_steps": total_loops / episodes,
        "credited_loop_events": credited_events,
        "brain": brain,
    }


def restore_branch(pretrained: SphereBrain, mode: str, seed: int) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    changed = ENVIRONMENTS[0]

    before_base = v70.run_episode(brain, v70.BASE, relative_mode=True, assist=False)

    for i in range(1, RESTORE_CHANGE_EPISODES + 1):
        run_training_episode(brain, changed, mode, rng, motif_counts, i)

    after_change_base = v70.run_episode(brain, v70.BASE, relative_mode=True, assist=False)
    changed_final = v70.run_episode(brain, changed, relative_mode=True, assist=False)

    for j in range(1, RESTORE_BASE_EPISODES + 1):
        run_training_episode(brain, v70.BASE, mode, rng, motif_counts, RESTORE_CHANGE_EPISODES + j)

    restored_base = v70.run_episode(brain, v70.BASE, relative_mode=True, assist=False)
    base_optimal = shortest_steps(v70.BASE)
    return {
        "before_base": before_base,
        "after_change_base": after_change_base,
        "changed_final": changed_final,
        "restored_base": restored_base,
        "preserved_immediately": bool(after_change_base["success"]),
        "restored_success": bool(restored_base["success"]),
        "restored_optimal": bool(restored_base["success"] and base_optimal is not None and int(restored_base["steps"]) == base_optimal),
    }


def aggregate(rows: list[dict]) -> dict:
    firsts = [int(r["first_success_episode"]) for r in rows if r["first_success_episode"] is not None]
    n = len(rows)
    return {
        "seed_count": n,
        "final_success_count": sum(1 for r in rows if r["final_success"]),
        "final_success_rate": sum(1 for r in rows if r["final_success"]) / n,
        "optimal_count": sum(1 for r in rows if r["final_optimal"]),
        "optimal_rate": sum(1 for r in rows if r["final_optimal"]) / n,
        "ever_success_count": len(firsts),
        "ever_success_rate": len(firsts) / n,
        "median_first_success_episode": None if not firsts else float(median(firsts)),
        "mean_success_episodes": sum(int(r["success_episodes"]) for r in rows) / n,
        "mean_loop_steps": sum(float(r["mean_loop_steps"]) for r in rows) / n,
        "mean_credited_loop_events": sum(int(r["credited_loop_events"]) for r in rows) / n,
    }


def restore_aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "seed_count": n,
        "preserved_immediately_count": sum(1 for r in rows if r["preserved_immediately"]),
        "preserved_immediately_rate": sum(1 for r in rows if r["preserved_immediately"]) / n,
        "restored_success_count": sum(1 for r in rows if r["restored_success"]),
        "restored_success_rate": sum(1 for r in rows if r["restored_success"]) / n,
        "restored_optimal_count": sum(1 for r in rows if r["restored_optimal"]),
        "restored_optimal_rate": sum(1 for r in rows if r["restored_optimal"]) / n,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = v70.pretrain(base)

    environment_rows = []
    total_improved = 0
    total_worsened = 0
    temporal_wins = 0
    legacy_wins = 0
    ties = 0

    for env_index, case in enumerate(ENVIRONMENTS):
        paired_rows = []
        by_mode = {m: [] for m in MODES}
        improved = worsened = 0
        for i in range(SEEDS_PER_ENV):
            seed = BASE_SEED + env_index * 1000 + i
            results = {m: paired_branch(pretrained, case, m, seed) for m in MODES}
            compact = {}
            for m in MODES:
                brain_obj = results[m].pop("brain")
                del brain_obj
                by_mode[m].append(results[m])
                compact[m] = results[m]
            a, b = compact["legacy_relative"]["final"], compact["temporal_credit"]["final"]
            if metric(b) > metric(a):
                improved += 1
            elif metric(b) < metric(a):
                worsened += 1
            paired_rows.append({"seed": seed, **compact})

        aggs = {m: aggregate(by_mode[m]) for m in MODES}
        if aggs["temporal_credit"]["final_success_rate"] > aggs["legacy_relative"]["final_success_rate"]:
            winner = "temporal_credit"
            temporal_wins += 1
        elif aggs["temporal_credit"]["final_success_rate"] < aggs["legacy_relative"]["final_success_rate"]:
            winner = "legacy_relative"
            legacy_wins += 1
        else:
            # tie-break only for reporting; primary judgment remains success rate
            if aggs["temporal_credit"]["optimal_rate"] > aggs["legacy_relative"]["optimal_rate"]:
                winner = "temporal_credit"
                temporal_wins += 1
            elif aggs["temporal_credit"]["optimal_rate"] < aggs["legacy_relative"]["optimal_rate"]:
                winner = "legacy_relative"
                legacy_wins += 1
            else:
                winner = "tie"
                ties += 1
        total_improved += improved
        total_worsened += worsened
        environment_rows.append({
            "environment": case["name"],
            "case": {"start": case["start"], "goal": case["goal"], "blocked": sorted(case["blocked"])},
            "winner": winner,
            "legacy_to_temporal_improved_seeds": improved,
            "legacy_to_temporal_worsened_seeds": worsened,
            "aggregates": aggs,
            "seed_rows": paired_rows,
        })

    restore_rows_by_mode = {m: [] for m in MODES}
    restore_seed_rows = []
    for i in range(SEEDS_PER_ENV):
        seed = BASE_SEED + 9000 + i
        results = {m: restore_branch(pretrained, m, seed) for m in MODES}
        for m in MODES:
            restore_rows_by_mode[m].append(results[m])
        restore_seed_rows.append({"seed": seed, **results})
    restore_aggs = {m: restore_aggregate(restore_rows_by_mode[m]) for m in MODES}

    all_legacy_successes = sum(row["aggregates"]["legacy_relative"]["final_success_count"] for row in environment_rows)
    all_temporal_successes = sum(row["aggregates"]["temporal_credit"]["final_success_count"] for row in environment_rows)
    all_trials = len(ENVIRONMENTS) * SEEDS_PER_ENV
    legacy_rate = all_legacy_successes / all_trials
    temporal_rate = all_temporal_successes / all_trials

    brain_unchanged = before_hash == file_hash(BRAIN_PATH)
    if temporal_wins >= 3 and total_worsened == 0 and temporal_rate > legacy_rate:
        verdict = "temporal_credit_generalizes_recovery_value_across_multiple_environment_changes"
        readiness = "temporal_credit_multi_environment_value_observed"
        next_step = "promote_temporal_credit_to_native_core_candidate_then_validate_native_equivalence"
    elif temporal_rate > legacy_rate:
        verdict = "temporal_credit_improves_overall_recovery_but_has_environment_specific_limits"
        readiness = "temporal_credit_promising_but_not_uniform"
        next_step = "inspect_environment_specific_failures_before_native_core_integration"
    elif temporal_rate < legacy_rate:
        verdict = "temporal_credit_does_not_generalize_and_can_reduce_recovery"
        readiness = "temporal_credit_not_ready_for_native_core"
        next_step = "revisit_loop_credit_scope_and_preservation_rules"
    else:
        verdict = "temporal_credit_and_legacy_relative_are_tied_across_current_environment_suite"
        readiness = "temporal_credit_role_still_mixed"
        next_step = "expand_environment_suite_or_episode_budget_before_core_integration"

    payload = {
        "experiment": "Core Growth Binding v75 — Temporal Credit Multi-Environment Robustness",
        "contract": {
            "independent_environment_count": len(ENVIRONMENTS),
            "restore_scenario_included": True,
            "seeds_per_environment": SEEDS_PER_ENV,
            "paired_rng_seed_per_mode": True,
            "same_pretrained_core_per_seed_and_mode": True,
            "same_exploration_rate": True,
            "same_episode_budget": True,
            "assist_excluded_from_primary_test": True,
            "production_brain_json_saved": False,
        },
        "environment_rows": environment_rows,
        "restore": {
            "description": "adapt route_blocked for 12 episodes then return to original environment for 8 episodes",
            "aggregates": restore_aggs,
            "seed_rows": restore_seed_rows,
        },
        "summary": {
            "legacy_overall_success_rate": legacy_rate,
            "temporal_overall_success_rate": temporal_rate,
            "temporal_environment_wins": temporal_wins,
            "legacy_environment_wins": legacy_wins,
            "tied_environments": ties,
            "legacy_to_temporal_improved_seed_trials": total_improved,
            "legacy_to_temporal_worsened_seed_trials": total_worsened,
            "restore_legacy_success_rate": restore_aggs["legacy_relative"]["restored_success_rate"],
            "restore_temporal_success_rate": restore_aggs["temporal_credit"]["restored_success_rate"],
            "restore_legacy_optimal_rate": restore_aggs["legacy_relative"]["restored_optimal_rate"],
            "restore_temporal_optimal_rate": restore_aggs["temporal_credit"]["restored_optimal_rate"],
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v75.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v75</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v75：Temporal Credit Multi-Environment Robustness</h1><p class="lead">4種類の環境変更＋元環境への復帰を、paired seedsでv70 RelativeとTemporal Creditに同条件で経験させる。</p><section class="panel"><div class="controls"><button id="run">Multi-Environment Benchmarkを実行</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>環境別集計</h2><pre id="env" class="raw">未実行</pre></section><section class="panel"><h2>復帰シナリオ</h2><pre id="restore" class="raw">未実行</pre></section><script>
const pct=x=>(100*x).toFixed(1)+'%';function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('v70 全体到達率',pct(s.legacy_overall_success_rate)),metric('Temporal 全体到達率',pct(s.temporal_overall_success_rate),s.temporal_overall_success_rate>s.legacy_overall_success_rate?'good':'blue'),metric('Temporal勝利環境',s.temporal_environment_wins),metric('Legacy勝利環境',s.legacy_environment_wins),metric('改善seed試行',s.legacy_to_temporal_improved_seed_trials,'good'),metric('悪化seed試行',s.legacy_to_temporal_worsened_seed_trials,s.legacy_to_temporal_worsened_seed_trials===0?'good':'warn'),metric('復帰 v70',pct(s.restore_legacy_success_rate)),metric('復帰 Temporal',pct(s.restore_temporal_success_rate)),metric('復帰最短 v70',pct(s.restore_legacy_optimal_rate)),metric('復帰最短 Temporal',pct(s.restore_temporal_optimal_rate)),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('env').textContent=JSON.stringify(d.environment_rows.map(x=>({environment:x.environment,winner:x.winner,improved:x.legacy_to_temporal_improved_seeds,worsened:x.legacy_to_temporal_worsened_seeds,aggregates:x.aggregates})),null,2);document.getElementById('restore').textContent=JSON.stringify(d.restore,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v75: http://{HOST}:{PORT}")
    print("4 environment changes + restoration / paired seeds / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
