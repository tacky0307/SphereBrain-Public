from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import sys
import threading
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

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
START_PORT = 5122
OUT = ROOT / "data" / "core_growth_binding_v76" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"


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


def diagnostic_episode(brain: SphereBrain, case: dict, max_steps: int = v70.MAX_STEPS) -> dict:
    current = int(case["start"])
    goal = int(case["goal"])
    blocked = set(case["blocked"])
    path = [current]
    rows = []
    transitions = []
    for step in range(max_steps):
        if current == goal:
            break
        ranked = v70.candidate_rows(brain, current, blocked, relative_mode=True)
        if not ranked:
            break
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        rows.append({
            "step": step + 1,
            "from": current,
            "top_target": int(top["target"]),
            "top_specific_score": float(top["specific_score"]),
            "top_relative_score": float(top["relative_score"]),
            "top_total_score": float(top["score"]),
            "second_target": None if second is None else int(second["target"]),
            "second_specific_score": None if second is None else float(second["specific_score"]),
            "second_relative_score": None if second is None else float(second["relative_score"]),
            "second_total_score": None if second is None else float(second["score"]),
            "margin": None if second is None else float(top["score"] - second["score"]),
        })
        target = int(top["target"])
        transitions.append([current, target])
        current = target
        path.append(current)
        if current == goal:
            break
    episode = {
        "success": current == goal,
        "steps": len(transitions),
        "path": path,
        "transitions": transitions,
        "loop_steps": max(0, len(path) - len(set(path))),
    }
    return {
        "episode": episode,
        "choice_rows": rows,
        "immediate_return_motifs": v73.immediate_return_motifs(episode),
        "longer_repeat_fragments": v73.longer_repeat_fragments(episode),
    }


def rerun_temporal(pretrained: SphereBrain, case: dict, seed: int, episodes: int) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    training_rows = []
    all_events = []
    first_success = None
    success_count = 0
    immediate_count = 0
    longer_count = 0

    for episode_index in range(1, episodes + 1):
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
        immediate = v73.immediate_return_motifs(ep)
        longer = v73.longer_repeat_fragments(ep)
        immediate_count += len(immediate)
        longer_count += len(longer)
        events = []
        if ep["success"]:
            success_count += 1
            if first_success is None:
                first_success = episode_index
            v70.reinforce_success(brain, ep, include_relative=True)
        else:
            events = v73.temporal_attribution(brain, ep, motif_counts, episode_index=episode_index)
            all_events.extend(events)
        training_rows.append({
            "episode": episode_index,
            "success": bool(ep["success"]),
            "steps": int(ep["steps"]),
            "path": ep["path"],
            "loop_steps": int(ep.get("loop_steps", 0)),
            "immediate_return_count": len(immediate),
            "longer_loop_count": len(longer),
            "credited_events": sum(1 for x in events if x.get("credited")),
        })

    diag = diagnostic_episode(brain, case)
    return {
        "brain": brain,
        "first_success_episode": first_success,
        "success_episode_count": success_count,
        "training_immediate_return_count": immediate_count,
        "training_longer_loop_count": longer_count,
        "credited_loop_events": sum(1 for x in all_events if x.get("credited")),
        "unique_immediate_loop_motifs": len(motif_counts),
        "motif_counts": dict(sorted(motif_counts.items(), key=lambda x: (-x[1], x[0]))),
        "training_rows": training_rows,
        "temporal_events": all_events,
        "final_diagnostic": diag,
    }


def classify_failure(run: dict, kind: str = "environment") -> str:
    success_count = int(run["success_episode_count"])
    credited = int(run["credited_loop_events"])
    final = run["final_diagnostic"]["episode"]
    immediate = len(run["final_diagnostic"]["immediate_return_motifs"])
    longer = len(run["final_diagnostic"]["longer_repeat_fragments"])
    if kind == "restore" and not final["success"]:
        if success_count > 0:
            return "restore_interference"
    if success_count == 0:
        if longer > 0 and immediate == 0:
            return "longer_loop_not_attributed"
        return "no_success_experience"
    if credited > 0 and not final["success"]:
        if longer > immediate:
            return "longer_loop_not_attributed"
        return "immediate_loop_credit_insufficient"
    if success_count > 0 and not final["success"]:
        return "success_seen_but_not_stabilized"
    return "other"


def compact_run(run: dict) -> dict:
    return {k: v for k, v in run.items() if k != "brain"}


def audit_restore(pretrained: SphereBrain, seed: int) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    all_events = []
    change_successes = 0
    restore_successes = 0

    for i in range(1, v75.RESTORE_CHANGE_EPISODES + 1):
        ep = v70.run_episode(brain, v75.ENVIRONMENTS[0], relative_mode=True, assist=False, rng=rng, explore=v70.EXPLORATION, max_steps=v70.TRAIN_MAX_STEPS)
        v70.observe_episode(brain, ep, include_relative=True)
        if ep["success"]:
            change_successes += 1
            v70.reinforce_success(brain, ep, include_relative=True)
        else:
            all_events.extend(v73.temporal_attribution(brain, ep, motif_counts, episode_index=i))

    after_change_base = diagnostic_episode(brain, v70.BASE)

    restore_rows = []
    for j in range(1, v75.RESTORE_BASE_EPISODES + 1):
        idx = v75.RESTORE_CHANGE_EPISODES + j
        ep = v70.run_episode(brain, v70.BASE, relative_mode=True, assist=False, rng=rng, explore=v70.EXPLORATION, max_steps=v70.TRAIN_MAX_STEPS)
        v70.observe_episode(brain, ep, include_relative=True)
        if ep["success"]:
            restore_successes += 1
            v70.reinforce_success(brain, ep, include_relative=True)
        else:
            events = v73.temporal_attribution(brain, ep, motif_counts, episode_index=idx)
            all_events.extend(events)
        restore_rows.append({"episode": j, "success": bool(ep["success"]), "path": ep["path"], "loop_steps": int(ep.get("loop_steps", 0))})

    final_diag = diagnostic_episode(brain, v70.BASE)
    run_like = {
        "success_episode_count": restore_successes,
        "credited_loop_events": sum(1 for x in all_events if x.get("credited")),
        "final_diagnostic": final_diag,
    }
    return {
        "change_success_episode_count": change_successes,
        "restore_success_episode_count": restore_successes,
        "credited_loop_events": run_like["credited_loop_events"],
        "after_change_base": after_change_base,
        "restore_rows": restore_rows,
        "final_diagnostic": final_diag,
        "classification": classify_failure(run_like, kind="restore"),
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = v70.pretrain(base)

    failure_rows = []
    environment_failure_counts = Counter()
    classifications = Counter()
    success_controls = []

    # Reproduce all v75 independent environments, but retain rich traces only for Temporal failures.
    for env_index, case in enumerate(v75.ENVIRONMENTS):
        for i in range(v75.SEEDS_PER_ENV):
            seed = v75.BASE_SEED + env_index * 1000 + i
            run = rerun_temporal(pretrained, case, seed, v75.ADAPT_EPISODES)
            final_success = bool(run["final_diagnostic"]["episode"]["success"])
            if not final_success:
                classification = classify_failure(run)
                classifications[classification] += 1
                environment_failure_counts[case["name"]] += 1
                failure_rows.append({
                    "kind": "environment",
                    "environment": case["name"],
                    "seed": seed,
                    "classification": classification,
                    **compact_run(run),
                })
            elif len(success_controls) < 8:
                success_controls.append({
                    "environment": case["name"],
                    "seed": seed,
                    "first_success_episode": run["first_success_episode"],
                    "success_episode_count": run["success_episode_count"],
                    "credited_loop_events": run["credited_loop_events"],
                    "training_immediate_return_count": run["training_immediate_return_count"],
                    "training_longer_loop_count": run["training_longer_loop_count"],
                    "final_path": run["final_diagnostic"]["episode"]["path"],
                })

    restore_failures = []
    for i in range(v75.SEEDS_PER_ENV):
        seed = v75.BASE_SEED + 9000 + i
        row = audit_restore(pretrained, seed)
        if not row["final_diagnostic"]["episode"]["success"]:
            classifications[row["classification"]] += 1
            restore_failures.append({"kind": "restore", "environment": "return_to_base", "seed": seed, **row})

    total_independent = len(v75.ENVIRONMENTS) * v75.SEEDS_PER_ENV
    independent_failure_count = len(failure_rows)
    restore_failure_count = len(restore_failures)
    top_classification = classifications.most_common(1)[0][0] if classifications else "none"

    # Score-margin view of failed final policies.
    margins = []
    for row in failure_rows:
        for choice in row["final_diagnostic"]["choice_rows"]:
            if choice["margin"] is not None:
                margins.append(float(choice["margin"]))
    mean_failed_margin = sum(margins) / len(margins) if margins else None

    brain_unchanged = before_hash == file_hash(BRAIN_PATH)
    if top_classification == "no_success_experience":
        verdict = "remaining_temporal_failures_are_mainly_exploration_or_success_discovery_failures"
        next_step = "improve_exploration_or_intrinsic_novelty_without_weakening_temporal_credit"
        readiness = "temporal_credit_ready_but_exploration_gap_remains"
    elif top_classification == "longer_loop_not_attributed":
        verdict = "remaining_temporal_failures_are_mainly_longer_loops_outside_immediate_return_credit"
        next_step = "extend_temporal_credit_to_repeated_longer_fragments_with_same_attribution_safety"
        readiness = "temporal_credit_needs_longer_loop_extension"
    elif top_classification in {"immediate_loop_credit_insufficient", "success_seen_but_not_stabilized"}:
        verdict = "remaining_temporal_failures_receive_credit_but_policy_margin_or_stability_still_prevents_escape"
        next_step = "audit_failed_choice_score_components_before_changing_credit_strength"
        readiness = "temporal_credit_strength_or_policy_stability_needs_audit"
    elif top_classification == "restore_interference":
        verdict = "remaining_failures_are_concentrated_in_restore_interference_after_environment_switching"
        next_step = "add_contextual_preservation_for_previously_successful_temporal_fragments"
        readiness = "temporal_credit_needs_restore_preservation"
    else:
        verdict = "temporal_failures_are_sparse_and_mixed_without_one_dominant_driver"
        next_step = "retain_temporal_credit_and_native_integrate_only_after_targeted_sparse_failure_tests"
        readiness = "temporal_credit_sparse_failures_mixed"

    payload = {
        "experiment": "Core Growth Binding v76 — Temporal Credit Failure Attribution",
        "contract": {
            "v75_temporal_logic_unchanged": True,
            "same_v75_environment_seeds_replayed": True,
            "failure_seeds_only_receive_deep_attribution": True,
            "credit_rules_modified": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "independent_trial_count": total_independent,
            "independent_failure_count": independent_failure_count,
            "independent_failure_rate": independent_failure_count / total_independent,
            "restore_failure_count": restore_failure_count,
            "restore_failure_rate": restore_failure_count / v75.SEEDS_PER_ENV,
            "environment_failure_counts": dict(environment_failure_counts),
            "classification_counts": dict(classifications),
            "dominant_failure_driver": top_classification,
            "mean_failed_final_choice_margin": mean_failed_margin,
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "failure_rows": failure_rows,
        "restore_failure_rows": restore_failures,
        "success_controls": success_controls,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v76.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v76</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v76：Temporal Credit Failure Attribution</h1><p class="lead">v75のTemporal失敗seedだけを同じseedで再現し、探索不足・未帰属long-loop・Credit不足・復帰干渉へ分類する。学習ルール自体は変更しない。</p><section class="panel"><div class="controls"><button id="run">失敗seedを解剖</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>失敗Seed</h2><pre id="fail" class="raw">未実行</pre></section><section class="panel"><h2>復帰失敗Seed</h2><pre id="restore" class="raw">未実行</pre></section><section class="panel"><h2>成功Control</h2><pre id="control" class="raw">未実行</pre></section><script>
const pct=x=>(100*x).toFixed(1)+'%';function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('独立試行失敗',`${s.independent_failure_count}/${s.independent_trial_count}`),metric('独立失敗率',pct(s.independent_failure_rate)),metric('復帰失敗',`${s.restore_failure_count}/${20}`),metric('復帰失敗率',pct(s.restore_failure_rate)),metric('最多原因',s.dominant_failure_driver),metric('環境別失敗',JSON.stringify(s.environment_failure_counts)),metric('原因分類',JSON.stringify(s.classification_counts)),metric('失敗時平均margin',s.mean_failed_final_choice_margin===null?'なし':s.mean_failed_final_choice_margin.toFixed(6)),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('fail').textContent=JSON.stringify(d.failure_rows,null,2);document.getElementById('restore').textContent=JSON.stringify(d.restore_failure_rows,null,2);document.getElementById('control').textContent=JSON.stringify(d.success_controls,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v76: http://{HOST}:{PORT}")
    print("Replay v75 failure seeds only / no credit-rule changes / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
