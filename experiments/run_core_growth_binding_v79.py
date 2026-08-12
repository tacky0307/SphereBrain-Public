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
import run_core_growth_binding_v77 as v77
import run_core_growth_binding_v78 as v78

HOST = "127.0.0.1"
START_PORT = 5125
OUT = ROOT / "data" / "core_growth_binding_v79" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEED_COUNT = v78.SEED_COUNT
BASE_SEED = 790100
EPISODES_PER_PHASE = v78.EPISODES_PER_PHASE
PHASES = v78.PHASES
MODES = ["temporal_credit", "legacy_consolidation", "homeostatic_consolidation"]

SUCCESS_GAIN = 0.26
PROVISIONAL_GAIN_SCALE = 0.45
PASSIVE_DECAY = 0.992
CONTRADICTION_LOSS = 0.18
MAX_STABILITY = 1.0
CONSOLIDATE_AFTER = 2
EXTRA_REPLAY_GATE = 0.55
MAX_EXTRA_REPLAY = 1
BASE_RETURN_WEIGHT_DECAY = v73.RETURN_WEIGHT_DECAY
BASE_RETURN_USAGE_DECAY = v73.RETURN_USAGE_DECAY
ENTRY_WEIGHT_DECAY = v73.ENTRY_WEIGHT_DECAY
ENTRY_USAGE_DECAY = v73.ENTRY_USAGE_DECAY
LOOP_GATE = v73.LOOP_GATE
MIN_FAILURE_EVIDENCE = v73.MIN_FAILURE_EVIDENCE
CEILING_EPS = 1e-9


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


def decay_unused(stability: dict[str, float], used_keys: set[str]) -> int:
    changed = 0
    for key in list(stability):
        if key in used_keys:
            continue
        before = float(stability[key])
        after = max(0.0, before * PASSIVE_DECAY)
        stability[key] = after
        changed += int(after != before)
    return changed


def homeostatic_success_update(
    brain: SphereBrain,
    ep: dict,
    success_counts: dict[str, int],
    stability: dict[str, float],
) -> dict:
    promoted = 0
    extra_replayed = 0
    used_keys: set[str] = set()
    for source, target in successful_pairs(ep):
        key = pair_key(source, target)
        used_keys.add(key)
        success_counts[key] += 1
        before = float(stability.get(key, 0.0))
        scale = 1.0 if success_counts[key] >= CONSOLIDATE_AFTER else PROVISIONAL_GAIN_SCALE
        gain = SUCCESS_GAIN * scale * (1.0 - before)
        after = min(MAX_STABILITY, before + gain)
        stability[key] = after
        if before < v77.PROTECTION_STABILITY_GATE <= after:
            promoted += 1
        if after >= EXTRA_REPLAY_GATE:
            for _ in range(MAX_EXTRA_REPLAY):
                v70.transition_route(brain, source, target, learn=True, relative=False)
            extra_replayed += 1
    passive = decay_unused(stability, used_keys)
    return {"promoted": promoted, "extra_replayed": extra_replayed, "passive_decayed": passive}


def homeostatic_failure_decay(ep: dict, stability: dict[str, float]) -> int:
    changed = 0
    used_keys = set()
    for source, target in successful_pairs(ep):
        key = pair_key(source, target)
        used_keys.add(key)
        before = float(stability.get(key, 0.0))
        if before <= 0:
            continue
        loss = CONTRADICTION_LOSS * (0.5 + before)
        after = max(0.0, before - loss)
        stability[key] = after
        changed += int(after != before)
    changed += decay_unused(stability, used_keys)
    return changed


def homeostatic_temporal_attribution(
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
            "stability_before": stable,
            "protection_strength": stable,
            "credited": False,
        }
        if count >= LOOP_GATE and failure_evidence >= MIN_FAILURE_EVIDENCE:
            # Continuous protection: high stability softens, but never blocks, temporal credit.
            wf = BASE_RETURN_WEIGHT_DECAY + (1.0 - BASE_RETURN_WEIGHT_DECAY) * 0.55 * stable
            uf = BASE_RETURN_USAGE_DECAY + (1.0 - BASE_RETURN_USAGE_DECAY) * 0.55 * stable
            event["return_decay"] = v73.decay_specific_credit(
                brain, source, target, weight_factor=wf, usage_factor=uf
            )
            es, et = motif["entry"]
            event["entry_decay"] = v73.decay_specific_credit(
                brain, es, et, weight_factor=ENTRY_WEIGHT_DECAY, usage_factor=ENTRY_USAGE_DECAY
            )
            if stable > 0:
                loss = CONTRADICTION_LOSS * (0.5 + stable)
                stability[key] = max(0.0, stable - loss)
            event["stability_after"] = float(stability.get(key, 0.0))
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
        brain, case, relative_mode=True, assist=False, rng=rng,
        explore=v70.EXPLORATION, max_steps=v70.TRAIN_MAX_STEPS,
    )
    v70.observe_episode(brain, ep, include_relative=True)
    events = []
    consolidation = {"promoted": 0, "extra_replayed": 0, "passive_decayed": 0, "eroded": 0}

    if ep["success"]:
        v70.reinforce_success(brain, ep, include_relative=True)
        if mode == "legacy_consolidation":
            consolidation.update(v77.update_success_stability(brain, ep, success_counts, stability))
        elif mode == "homeostatic_consolidation":
            consolidation.update(homeostatic_success_update(brain, ep, success_counts, stability))
    else:
        if mode == "legacy_consolidation":
            consolidation["eroded"] = v77.erode_stability_for_failed_episode(ep, stability)
            events = v77.consolidation_temporal_attribution(
                brain, ep, motif_counts, stability, episode_index=episode_index
            )
        elif mode == "homeostatic_consolidation":
            consolidation["eroded"] = homeostatic_failure_decay(ep, stability)
            events = homeostatic_temporal_attribution(
                brain, ep, motif_counts, stability, episode_index=episode_index
            )
        else:
            events = v73.temporal_attribution(brain, ep, motif_counts, episode_index=episode_index)
    return {"episode": ep, "events": events, "consolidation": consolidation}


def eval_episode(brain: SphereBrain, case: dict) -> dict:
    ep = v70.run_episode(brain, case, relative_mode=True, assist=False)
    optimal = v75.shortest_steps(case)
    return {**ep, "optimal": bool(ep["success"] and optimal is not None and int(ep["steps"]) == optimal)}


def run_long_horizon(pretrained: SphereBrain, mode: str, seed: int) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    success_counts: dict[str, int] = defaultdict(int)
    stability: dict[str, float] = defaultdict(float)

    phase_rows = []
    total_loops = 0
    total_credited = 0
    total_extra_replays = 0
    stability_ceiling_hits = 0
    stability_observations = 0
    prior_first_optimal: dict[str, int | None] = {}
    global_episode = 0

    for phase_index, case in enumerate(PHASES, start=1):
        start_eval = eval_episode(brain, case)
        first_success = None
        first_optimal = None
        phase_loops = 0
        success_episodes = 0
        optimal_episodes = 0
        phase_credited = 0
        phase_passive_decay = 0

        for local_episode in range(1, EPISODES_PER_PHASE + 1):
            global_episode += 1
            row = training_episode(
                brain, case, mode, rng, motif_counts, success_counts, stability, global_episode
            )
            ep = row["episode"]
            events = row["events"]
            c = row["consolidation"]
            optimal_steps = v75.shortest_steps(case)
            is_optimal = bool(ep["success"] and optimal_steps is not None and int(ep["steps"]) == optimal_steps)

            loops = int(ep.get("loop_steps", 0))
            phase_loops += loops
            total_loops += loops
            credited = sum(1 for e in events if e.get("credited"))
            phase_credited += credited
            total_credited += credited
            total_extra_replays += int(c.get("extra_replayed", 0))
            phase_passive_decay += int(c.get("passive_decayed", 0))

            if ep["success"]:
                success_episodes += 1
                if first_success is None:
                    first_success = local_episode
            if is_optimal:
                optimal_episodes += 1
                if first_optimal is None:
                    first_optimal = local_episode

            if mode != "temporal_credit":
                values = list(stability.values())
                stability_observations += max(1, len(values))
                if values:
                    stability_ceiling_hits += sum(1 for x in values if float(x) >= MAX_STABILITY - CEILING_EPS)

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
            "passive_decay_events": phase_passive_decay,
            "consolidated_transition_count": sum(1 for x in stability.values() if float(x) >= v77.PROTECTION_STABILITY_GATE),
            "max_stability": max([float(x) for x in stability.values()] or [0.0]),
            "revisit_faster_than_first": revisit_faster,
            "end_path": end_eval["path"],
            "end_steps": int(end_eval["steps"]),
        })

    final_a = eval_episode(brain, v78.A)
    ceiling_rate = 0.0 if stability_observations == 0 else stability_ceiling_hits / stability_observations
    return {
        "phase_rows": phase_rows,
        "final_A_success": bool(final_a["success"]),
        "final_A_optimal": bool(final_a["optimal"]),
        "final_A_path": final_a["path"],
        "mean_loop_steps": total_loops / (len(PHASES) * EPISODES_PER_PHASE),
        "total_credited_loop_events": total_credited,
        "total_extra_replays": total_extra_replays,
        "consolidated_transition_count": sum(1 for x in stability.values() if float(x) >= v77.PROTECTION_STABILITY_GATE),
        "max_stability": max([float(x) for x in stability.values()] or [0.0]),
        "stability_ceiling_rate": ceiling_rate,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    all_phases = [p for r in rows for p in r["phase_rows"]]
    first_opts = [int(p["first_optimal_episode"]) for p in all_phases if p["first_optimal_episode"] is not None]
    returns = [p for p in all_phases if p["environment"] == "base_A" and int(p["phase"]) > 1]
    switch_phases = [p for p in all_phases if int(p["phase"]) > 1]
    return {
        "seed_count": n,
        "phase_count": len(all_phases),
        "phase_end_success_rate": sum(1 for p in all_phases if p["end_success"]) / len(all_phases),
        "phase_end_optimal_rate": sum(1 for p in all_phases if p["end_optimal"]) / len(all_phases),
        "switch_phase_failure_rate": sum(1 for p in switch_phases if not p["end_success"]) / len(switch_phases),
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
    paired = {
        "homeostatic_better_than_legacy": 0,
        "homeostatic_worse_than_legacy": 0,
        "homeostatic_better_than_temporal": 0,
        "homeostatic_worse_than_temporal": 0,
    }

    for i in range(SEED_COUNT):
        seed = BASE_SEED + i
        results = {m: run_long_horizon(pretrained, m, seed) for m in MODES}
        for m in MODES:
            by_mode[m].append(results[m])
        if metric(results["homeostatic_consolidation"]) > metric(results["legacy_consolidation"]):
            paired["homeostatic_better_than_legacy"] += 1
        elif metric(results["homeostatic_consolidation"]) < metric(results["legacy_consolidation"]):
            paired["homeostatic_worse_than_legacy"] += 1
        if metric(results["homeostatic_consolidation"]) > metric(results["temporal_credit"]):
            paired["homeostatic_better_than_temporal"] += 1
        elif metric(results["homeostatic_consolidation"]) < metric(results["temporal_credit"]):
            paired["homeostatic_worse_than_temporal"] += 1
        seed_rows.append({"seed": seed, **results})

    aggregates = {m: aggregate(by_mode[m]) for m in MODES}
    t = aggregates["temporal_credit"]
    h = aggregates["homeostatic_consolidation"]
    legacy = aggregates["legacy_consolidation"]

    rigidity_not_worse = h["switch_phase_failure_rate"] <= t["switch_phase_failure_rate"] + 1e-12
    return_memory_not_worse = h["return_A_end_optimal_rate"] >= t["return_A_end_optimal_rate"] - 1e-12
    final_not_worse = h["final_A_optimal_rate"] >= t["final_A_optimal_rate"] - 1e-12
    ceiling_safe = h["mean_stability_ceiling_rate"] < 0.10 and h["mean_stability_ceiling_rate"] < legacy["mean_stability_ceiling_rate"]
    better_than_legacy_ceiling = h["mean_stability_ceiling_rate"] < legacy["mean_stability_ceiling_rate"]
    brain_unchanged = before_hash == file_hash(BRAIN_PATH)

    pass_all = rigidity_not_worse and return_memory_not_worse and final_not_worse and ceiling_safe and brain_unchanged
    if pass_all:
        verdict = "homeostatic_consolidation_preserves_temporal_plasticity_without_stability_saturation"
        readiness = "homeostatic_temporal_candidate_ready_for_native_integration"
        next_step = "v80_native_core_learning_integration_and_equivalence_validation"
    elif better_than_legacy_ceiling and final_not_worse:
        verdict = "homeostasis_reduces_stability_saturation_but_long_horizon_tradeoffs_remain"
        readiness = "homeostatic_consolidation_promising_not_final"
        next_step = "audit_phase_specific_regressions_before_native_integration"
    else:
        verdict = "homeostatic_consolidation_does_not_yet_preserve_temporal_long_horizon_behavior"
        readiness = "homeostatic_consolidation_not_ready"
        next_step = "retune_homeostatic_gain_decay_balance_before_core_integration"

    payload = {
        "experiment": "Core Growth Binding v79 — Homeostatic Consolidation / Stability Equilibrium",
        "contract": {
            "seed_count": SEED_COUNT,
            "phase_sequence": [p["name"] for p in PHASES],
            "episodes_per_phase": EPISODES_PER_PHASE,
            "total_episodes_per_seed": len(PHASES) * EPISODES_PER_PHASE,
            "paired_rng_seed_per_mode": True,
            "same_pretrained_core_per_mode": True,
            "assist_used": False,
            "stability_growth": "gain_times_one_minus_stability",
            "passive_stability_decay": True,
            "continuous_protection_strength": True,
            "permanent_lock": False,
            "production_brain_json_saved": False,
        },
        "aggregates": aggregates,
        "paired_comparison": paired,
        "summary": {
            "rigidity_not_worse": rigidity_not_worse,
            "return_memory_not_worse": return_memory_not_worse,
            "final_A_not_worse": final_not_worse,
            "stability_ceiling_safe": ceiling_safe,
            "homeostasis_reduces_legacy_saturation": better_than_legacy_ceiling,
            "long_horizon_pass": pass_all,
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "seed_rows": seed_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v79.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v79</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v79：Homeostatic Consolidation / Stability Equilibrium</h1><p class="lead">Temporal / v77 Consolidation / Homeostatic Consolidationを同じ長期環境切替で比較し、stability飽和を抑えながら適応・復帰・最短化を維持できるかを検証する。</p><section class="panel"><div class="controls"><button id="run">Homeostatic長期検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>集計</h2><pre id="agg" class="raw">未実行</pre></section><section class="panel"><h2>Seed別</h2><pre id="rows" class="raw">未実行</pre></section><script>
const pct=x=>(100*x).toFixed(1)+'%';const yn=x=>x?'YES':'NO';function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const t=d.aggregates.temporal_credit,l=d.aggregates.legacy_consolidation,h=d.aggregates.homeostatic_consolidation,s=d.summary,p=d.paired_comparison;document.getElementById('metrics').innerHTML=[metric('Temporal phase成功',pct(t.phase_end_success_rate)),metric('Legacy Consolidated',pct(l.phase_end_success_rate)),metric('Homeostatic phase成功',pct(h.phase_end_success_rate),h.phase_end_success_rate>=t.phase_end_success_rate?'good':'warn'),metric('Homeostatic phase最短',pct(h.phase_end_optimal_rate),h.phase_end_optimal_rate>=t.phase_end_optimal_rate?'good':'warn'),metric('Homeostatic A復帰',pct(h.return_A_end_optimal_rate),h.return_A_end_optimal_rate>=t.return_A_end_optimal_rate?'good':'warn'),metric('Homeostatic 最終A',pct(h.final_A_optimal_rate),h.final_A_optimal_rate>=t.final_A_optimal_rate?'good':'warn'),metric('Legacy飽和率',pct(l.mean_stability_ceiling_rate)),metric('Homeostatic飽和率',pct(h.mean_stability_ceiling_rate),s.stability_ceiling_safe?'good':'warn'),metric('Homeostatic>Legacy seed',p.homeostatic_better_than_legacy),metric('Homeostatic<Legacy seed',p.homeostatic_worse_than_legacy),metric('硬直悪化なし',yn(s.rigidity_not_worse),s.rigidity_not_worse?'good':'warn'),metric('Long-Horizon PASS',yn(s.long_horizon_pass),s.long_horizon_pass?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('agg').textContent=JSON.stringify({aggregates:d.aggregates,paired_comparison:d.paired_comparison,summary:d.summary},null,2);document.getElementById('rows').textContent=JSON.stringify(d.seed_rows,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v79: http://{HOST}:{PORT}")
    print("Temporal vs legacy consolidation vs homeostatic consolidation / long horizon / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
