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
import run_core_growth_binding_v75 as v75

HOST = "127.0.0.1"
START_PORT = 5123
OUT = ROOT / "data" / "core_growth_binding_v77" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEEDS_PER_ENV = v75.SEEDS_PER_ENV
BASE_SEED = 770100
MODES = ["temporal_credit", "consolidated_temporal"]

# Consolidation contract: repeated success increases protection, but sustained
# contradiction erodes it again. Nothing is permanently locked.
CONSOLIDATE_AFTER = 2
MAX_STABILITY = 1.0
SUCCESS_STABILITY_GAIN = 0.22
FAIL_STABILITY_LOSS = 0.16
EXTRA_REPLAY_STABILITY_GATE = 0.45
PROTECTION_STABILITY_GATE = 0.35
MAX_EXTRA_REPLAY = 1
PROTECTED_RETURN_WEIGHT_DECAY = 0.982
PROTECTED_RETURN_USAGE_DECAY = 0.88
NORMAL_RETURN_WEIGHT_DECAY = v73.RETURN_WEIGHT_DECAY
NORMAL_RETURN_USAGE_DECAY = v73.RETURN_USAGE_DECAY
ENTRY_WEIGHT_DECAY = v73.ENTRY_WEIGHT_DECAY
ENTRY_USAGE_DECAY = v73.ENTRY_USAGE_DECAY
LOOP_GATE = v73.LOOP_GATE
MIN_FAILURE_EVIDENCE = v73.MIN_FAILURE_EVIDENCE


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


def metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def pair_key(source: int, target: int) -> str:
    return f"{int(source)}>{int(target)}"


def successful_pairs(ep: dict) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for source, target in ep.get("transitions", []):
        pair = (int(source), int(target))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def update_success_stability(
    brain: SphereBrain,
    ep: dict,
    success_counts: dict[str, int],
    stability: dict[str, float],
) -> dict:
    promoted = 0
    reinforced = 0
    for source, target in successful_pairs(ep):
        key = pair_key(source, target)
        success_counts[key] += 1
        before = float(stability.get(key, 0.0))
        gain = SUCCESS_STABILITY_GAIN if success_counts[key] >= CONSOLIDATE_AFTER else SUCCESS_STABILITY_GAIN * 0.45
        after = min(MAX_STABILITY, before + gain)
        stability[key] = after
        if before < PROTECTION_STABILITY_GATE <= after:
            promoted += 1
        if after >= EXTRA_REPLAY_STABILITY_GATE:
            # Small additional replay only after repeated successful confirmation.
            for _ in range(MAX_EXTRA_REPLAY):
                v70.transition_route(brain, source, target, learn=True, relative=False)
            reinforced += 1
    return {"promoted": promoted, "extra_replayed": reinforced}


def erode_stability_for_failed_episode(ep: dict, stability: dict[str, float]) -> int:
    changed = 0
    for source, target in successful_pairs(ep):
        key = pair_key(source, target)
        before = float(stability.get(key, 0.0))
        if before <= 0:
            continue
        after = max(0.0, before - FAIL_STABILITY_LOSS)
        stability[key] = after
        if after != before:
            changed += 1
    return changed


def consolidation_temporal_attribution(
    brain: SphereBrain,
    episode: dict,
    motif_counts: dict[str, int],
    stability: dict[str, float],
    *,
    episode_index: int,
) -> list[dict]:
    events = []
    for motif in v73.immediate_return_motifs(episode):
        motif_counts[motif["key"]] += 1
        count = motif_counts[motif["key"]]
        source, target = motif["return"]
        key = pair_key(source, target)
        stable = float(stability.get(key, 0.0))
        failure_evidence = 1.0 - v70.evidence_probability(brain, v70.specific_token(source, target))
        event = {
            "episode": episode_index,
            "motif": motif["nodes"],
            "motif_count": count,
            "entry": motif["entry"],
            "return": motif["return"],
            "failure_evidence": failure_evidence,
            "return_stability_before": stable,
            "protected": False,
            "credited": False,
        }
        if count >= LOOP_GATE and failure_evidence >= MIN_FAILURE_EVIDENCE:
            protected = stable >= PROTECTION_STABILITY_GATE
            event["protected"] = protected
            wf = PROTECTED_RETURN_WEIGHT_DECAY if protected else NORMAL_RETURN_WEIGHT_DECAY
            uf = PROTECTED_RETURN_USAGE_DECAY if protected else NORMAL_RETURN_USAGE_DECAY
            event["return_decay"] = v73.decay_specific_credit(
                brain,
                source,
                target,
                weight_factor=wf,
                usage_factor=uf,
            )
            es, et = motif["entry"]
            event["entry_decay"] = v73.decay_specific_credit(
                brain,
                es,
                et,
                weight_factor=ENTRY_WEIGHT_DECAY,
                usage_factor=ENTRY_USAGE_DECAY,
            )
            # Contradiction weakens consolidation; repeated contradictions eventually
            # remove protection and restore full plasticity.
            if stable > 0:
                stability[key] = max(0.0, stable - FAIL_STABILITY_LOSS)
            event["return_stability_after"] = float(stability.get(key, 0.0))
            event["credited"] = True
        events.append(event)
    return events


def training_episode(
    brain: SphereBrain,
    case: dict,
    mode: str,
    rng: random.Random,
    motif_counts: dict[str, int],
    success_counts: dict[str, int],
    stability: dict[str, float],
    episode_index: int,
) -> dict:
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
    events = []
    consolidation = {"promoted": 0, "extra_replayed": 0, "eroded": 0}
    if ep["success"]:
        v70.reinforce_success(brain, ep, include_relative=True)
        if mode == "consolidated_temporal":
            consolidation.update(update_success_stability(brain, ep, success_counts, stability))
    else:
        if mode == "consolidated_temporal":
            consolidation["eroded"] = erode_stability_for_failed_episode(ep, stability)
            events = consolidation_temporal_attribution(
                brain, ep, motif_counts, stability, episode_index=episode_index
            )
        else:
            events = v73.temporal_attribution(
                brain, ep, motif_counts, episode_index=episode_index
            )
    return {"episode": ep, "events": events, "consolidation": consolidation}


def branch(
    pretrained: SphereBrain,
    case: dict,
    mode: str,
    seed: int,
    episodes: int,
) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    success_counts: dict[str, int] = defaultdict(int)
    stability: dict[str, float] = defaultdict(float)
    first_success = None
    success_episodes = 0
    credited_events = 0
    protected_events = 0
    promotions = 0
    extra_replays = 0
    total_loops = 0

    for i in range(1, episodes + 1):
        row = training_episode(
            brain, case, mode, rng, motif_counts, success_counts, stability, i
        )
        ep = row["episode"]
        events = row["events"]
        c = row["consolidation"]
        total_loops += int(ep.get("loop_steps", 0))
        credited_events += sum(1 for e in events if e.get("credited"))
        protected_events += sum(1 for e in events if e.get("credited") and e.get("protected"))
        promotions += int(c.get("promoted", 0))
        extra_replays += int(c.get("extra_replayed", 0))
        if ep["success"]:
            success_episodes += 1
            if first_success is None:
                first_success = i

    final_ep = v70.run_episode(brain, case, relative_mode=True, assist=False)
    optimal = shortest_steps(case)
    consolidated_count = sum(1 for v in stability.values() if float(v) >= PROTECTION_STABILITY_GATE)
    return {
        "brain": brain,
        "final": final_ep,
        "final_success": bool(final_ep["success"]),
        "final_optimal": bool(final_ep["success"] and optimal is not None and int(final_ep["steps"]) == optimal),
        "first_success_episode": first_success,
        "success_episodes": success_episodes,
        "mean_loop_steps": total_loops / episodes,
        "credited_loop_events": credited_events,
        "protected_loop_events": protected_events,
        "consolidation_promotions": promotions,
        "extra_replays": extra_replays,
        "consolidated_transition_count": consolidated_count,
        "stability": dict(stability),
    }


def restore_branch(pretrained: SphereBrain, mode: str, seed: int) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    success_counts: dict[str, int] = defaultdict(int)
    stability: dict[str, float] = defaultdict(float)
    changed = v75.ENVIRONMENTS[0]

    # Seed consolidation with successful original-environment behavior without
    # altering the production brain file.
    for i in range(1, 5):
        ep = v70.run_episode(
            brain, v70.BASE, relative_mode=True, assist=False,
            rng=rng, explore=v70.EXPLORATION, max_steps=v70.TRAIN_MAX_STEPS,
        )
        v70.observe_episode(brain, ep, include_relative=True)
        if ep["success"]:
            v70.reinforce_success(brain, ep, include_relative=True)
            if mode == "consolidated_temporal":
                update_success_stability(brain, ep, success_counts, stability)

    for i in range(1, v75.RESTORE_CHANGE_EPISODES + 1):
        training_episode(
            brain, changed, mode, rng, motif_counts, success_counts, stability, i
        )

    after_change_base = v70.run_episode(brain, v70.BASE, relative_mode=True, assist=False)

    restore_successes = 0
    for j in range(1, v75.RESTORE_BASE_EPISODES + 1):
        row = training_episode(
            brain, v70.BASE, mode, rng, motif_counts, success_counts, stability,
            v75.RESTORE_CHANGE_EPISODES + j,
        )
        if row["episode"]["success"]:
            restore_successes += 1

    restored = v70.run_episode(brain, v70.BASE, relative_mode=True, assist=False)
    optimal = shortest_steps(v70.BASE)
    return {
        "after_change_base": after_change_base,
        "preserved_immediately": bool(after_change_base["success"]),
        "restore_success_episodes": restore_successes,
        "restored": restored,
        "restored_success": bool(restored["success"]),
        "restored_optimal": bool(restored["success"] and optimal is not None and int(restored["steps"]) == optimal),
        "consolidated_transition_count": sum(1 for v in stability.values() if float(v) >= PROTECTION_STABILITY_GATE),
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    firsts = [int(r["first_success_episode"]) for r in rows if r["first_success_episode"] is not None]
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
        "mean_protected_loop_events": sum(int(r["protected_loop_events"]) for r in rows) / n,
        "mean_consolidation_promotions": sum(int(r["consolidation_promotions"]) for r in rows) / n,
        "mean_consolidated_transitions": sum(int(r["consolidated_transition_count"]) for r in rows) / n,
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
    consolidated_wins = 0
    temporal_wins = 0

    for env_index, case in enumerate(v75.ENVIRONMENTS):
        by_mode = {m: [] for m in MODES}
        seed_rows = []
        improved = worsened = 0
        for i in range(SEEDS_PER_ENV):
            seed = BASE_SEED + env_index * 1000 + i
            results = {
                m: branch(pretrained, case, m, seed, v75.ADAPT_EPISODES)
                for m in MODES
            }
            compact = {}
            for m in MODES:
                brain_obj = results[m].pop("brain")
                del brain_obj
                by_mode[m].append(results[m])
                compact[m] = results[m]
            a = compact["temporal_credit"]["final"]
            b = compact["consolidated_temporal"]["final"]
            if metric(b) > metric(a):
                improved += 1
            elif metric(b) < metric(a):
                worsened += 1
            seed_rows.append({"seed": seed, **compact})

        aggs = {m: aggregate(by_mode[m]) for m in MODES}
        ta = aggs["temporal_credit"]
        ca = aggs["consolidated_temporal"]
        if (ca["final_success_rate"], ca["optimal_rate"]) > (ta["final_success_rate"], ta["optimal_rate"]):
            winner = "consolidated_temporal"
            consolidated_wins += 1
        elif (ca["final_success_rate"], ca["optimal_rate"]) < (ta["final_success_rate"], ta["optimal_rate"]):
            winner = "temporal_credit"
            temporal_wins += 1
        else:
            winner = "tie"
        total_improved += improved
        total_worsened += worsened
        environment_rows.append({
            "environment": case["name"],
            "winner": winner,
            "temporal_to_consolidated_improved_seeds": improved,
            "temporal_to_consolidated_worsened_seeds": worsened,
            "aggregates": aggs,
            "seed_rows": seed_rows,
        })

    restore_by_mode = {m: [] for m in MODES}
    restore_seed_rows = []
    for i in range(SEEDS_PER_ENV):
        seed = BASE_SEED + 9000 + i
        results = {m: restore_branch(pretrained, m, seed) for m in MODES}
        for m in MODES:
            restore_by_mode[m].append(results[m])
        restore_seed_rows.append({"seed": seed, **results})
    restore_aggs = {m: restore_aggregate(restore_by_mode[m]) for m in MODES}

    total_trials = len(v75.ENVIRONMENTS) * SEEDS_PER_ENV
    temporal_successes = sum(r["aggregates"]["temporal_credit"]["final_success_count"] for r in environment_rows)
    consolidated_successes = sum(r["aggregates"]["consolidated_temporal"]["final_success_count"] for r in environment_rows)
    temporal_rate = temporal_successes / total_trials
    consolidated_rate = consolidated_successes / total_trials

    brain_unchanged = before_hash == file_hash(BRAIN_PATH)
    restore_better = restore_aggs["consolidated_temporal"]["restored_success_rate"] > restore_aggs["temporal_credit"]["restored_success_rate"]

    if consolidated_rate > temporal_rate and total_worsened == 0 and restore_better:
        verdict = "success_consolidation_improves_policy_stability_and_restore_without_harming_temporal_recovery"
        readiness = "temporal_plus_consolidation_native_candidate"
        next_step = "promote_temporal_credit_and_success_consolidation_to_native_core_candidate_then_validate_equivalence"
    elif consolidated_rate > temporal_rate and total_worsened == 0:
        verdict = "success_consolidation_improves_final_policy_stability_without_observed_seed_regression"
        readiness = "success_consolidation_population_value_observed"
        next_step = "audit_restore_preservation_then_prepare_native_core_integration"
    elif consolidated_rate >= temporal_rate and restore_better and total_worsened <= 1:
        verdict = "success_consolidation_mainly_improves_restore_memory_while_preserving_temporal_recovery"
        readiness = "success_consolidation_restore_value_observed"
        next_step = "stress_test_consolidation_under_repeated_environment_switching"
    elif total_worsened > 0:
        verdict = "success_consolidation_can_overstabilize_and_reduce_adaptation_in_some_seeds"
        readiness = "consolidation_strength_needs_reduction"
        next_step = "reduce_protection_or_accelerate_deconsolidation_before_native_integration"
    else:
        verdict = "success_consolidation_does_not_yet_add_clear_population_level_value"
        readiness = "temporal_credit_remains_primary_native_candidate"
        next_step = "retain_temporal_credit_and_revisit_consolidation_only_for_restore_failures"

    payload = {
        "experiment": "Core Growth Binding v77 — Success Consolidation / Policy Stabilization",
        "contract": {
            "paired_temporal_vs_consolidated": True,
            "success_requires_repetition_before_protection": True,
            "consolidation_is_reversible": True,
            "persistent_failure_erodes_stability": True,
            "no_permanent_policy_lock": True,
            "same_v75_environment_suite": True,
            "same_seed_per_mode": True,
            "production_brain_json_saved": False,
        },
        "environment_rows": environment_rows,
        "restore": {"aggregates": restore_aggs, "seed_rows": restore_seed_rows},
        "summary": {
            "temporal_overall_success_rate": temporal_rate,
            "consolidated_overall_success_rate": consolidated_rate,
            "consolidated_environment_wins": consolidated_wins,
            "temporal_environment_wins": temporal_wins,
            "temporal_to_consolidated_improved_seed_trials": total_improved,
            "temporal_to_consolidated_worsened_seed_trials": total_worsened,
            "restore_temporal_success_rate": restore_aggs["temporal_credit"]["restored_success_rate"],
            "restore_consolidated_success_rate": restore_aggs["consolidated_temporal"]["restored_success_rate"],
            "restore_temporal_optimal_rate": restore_aggs["temporal_credit"]["restored_optimal_rate"],
            "restore_consolidated_optimal_rate": restore_aggs["consolidated_temporal"]["restored_optimal_rate"],
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v77.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v77</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v77：Success Consolidation / Policy Stabilization</h1><p class="lead">Temporal Creditを土台に、成功の反復で一時的なstabilityを形成し、単発矛盾では壊れにくく、持続矛盾では再び可塑化するSuccess Consolidationをpaired比較する。</p><section class="panel"><div class="controls"><button id="run">Success Consolidationを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>環境別結果</h2><pre id="env" class="raw">未実行</pre></section><section class="panel"><h2>復帰結果</h2><pre id="restore" class="raw">未実行</pre></section><script>
const pct=x=>(100*x).toFixed(1)+'%';function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Temporal 全体到達率',pct(s.temporal_overall_success_rate)),metric('Consolidated 到達率',pct(s.consolidated_overall_success_rate),s.consolidated_overall_success_rate>=s.temporal_overall_success_rate?'good':'warn'),metric('Consolidated勝利環境',s.consolidated_environment_wins),metric('Temporal勝利環境',s.temporal_environment_wins),metric('改善seed試行',s.temporal_to_consolidated_improved_seed_trials,'good'),metric('悪化seed試行',s.temporal_to_consolidated_worsened_seed_trials,s.temporal_to_consolidated_worsened_seed_trials===0?'good':'warn'),metric('復帰 Temporal',pct(s.restore_temporal_success_rate)),metric('復帰 Consolidated',pct(s.restore_consolidated_success_rate),s.restore_consolidated_success_rate>=s.restore_temporal_success_rate?'good':'warn'),metric('復帰最短 Temporal',pct(s.restore_temporal_optimal_rate)),metric('復帰最短 Consolidated',pct(s.restore_consolidated_optimal_rate)),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('env').textContent=JSON.stringify(d.environment_rows,null,2);document.getElementById('restore').textContent=JSON.stringify(d.restore,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v77: http://{HOST}:{PORT}")
    print("Temporal Credit vs Success Consolidation / paired seeds / reversible stability / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
