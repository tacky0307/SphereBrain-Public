from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

HOST = "127.0.0.1"
START_PORT = 5114
OUT = ROOT / "data" / "core_growth_binding_v67" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
GRID = 3
SEED = 6701
TRAIN_EPISODES = 30
ADAPT_EPISODES = 18
EXPLORATION = 0.35
MAX_STEPS = 16
REPLAY_REPEATS = 2
TIE_MARGIN = 0.0025
ASSIST_ABS_CAP = 5e-5
ASSIST_REL_CAP = 0.55
MIN_CONFIDENCE = 0.80
DIRS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}

BASE = {"name": "trained_board", "start": 0, "goal": 8, "blocked": {3, 4}}
CASES = [
    BASE,
    {"name": "new_start", "start": 1, "goal": 8, "blocked": {3, 4}},
    {"name": "new_goal", "start": 0, "goal": 5, "blocked": {3, 4}},
    {"name": "reverse_direction", "start": 8, "goal": 0, "blocked": {3, 4}},
    {"name": "route_blocked", "start": 0, "goal": 8, "blocked": {2, 4}},
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


def rc(node: int) -> tuple[int, int]:
    return divmod(node, GRID)


def node_at(r: int, c: int) -> int | None:
    if 0 <= r < GRID and 0 <= c < GRID:
        return r * GRID + c
    return None


def legal_moves(node: int, blocked: set[int]) -> list[tuple[str, int]]:
    r, c = rc(node)
    result = []
    for name, (dr, dc) in DIRS.items():
        target = node_at(r + dr, c + dc)
        if target is None or target in blocked:
            continue
        result.append((name, target))
    return result


def opaque_transition(source: int, target: int) -> str:
    return "t_" + hashlib.sha256(f"{source}>{target}".encode()).hexdigest()[:12]


def transition_universe() -> list[str]:
    tokens = []
    for node in range(GRID * GRID):
        for blocked in ({3, 4}, {2, 4}):
            if node in blocked:
                continue
            for _, target in legal_moves(node, set(blocked)):
                tokens.append(opaque_transition(node, target))
    return sorted(set(tokens))

EXPECTED = transition_universe()


def transition_sources(brain: SphereBrain, source: int, target: int) -> list[int]:
    return brain.text_to_sources(opaque_transition(source, target), count=3)


def transition_route(brain: SphereBrain, source: int, target: int, *, learn: bool) -> dict:
    result = brain.propagate(
        transition_sources(brain, source, target),
        steps=8,
        threshold=0.18,
        noise=0.0,
        learn=learn,
    )
    edges = [tuple(x) for x in result.traversed_edges]
    if edges:
        mean_weight = sum(float(brain.weights[a, b]) for a, b in edges) / len(edges)
        mean_usage = sum(float(brain.usage[a, b]) for a, b in edges) / len(edges)
    else:
        mean_weight = 0.0
        mean_usage = 0.0
    score = float(mean_weight + 0.012 * math.log1p(mean_usage) + 0.0005 * len(result.activated_nodes))
    return {"score": score, "edges": [list(x) for x in result.traversed_edges], "nodes": list(result.activated_nodes)}


def evidence_probability(brain: SphereBrain, source: int, target: int) -> float:
    token = opaque_transition(source, target)
    row = brain.experience_state.condition_evidence.get(token, {})
    t = float(row.get("true_weight", 0.0))
    f = float(row.get("false_weight", 0.0))
    total = t + f
    return 0.5 if total <= 1e-12 else t / total


def assist_rank(brain: SphereBrain, candidates: list[dict]) -> tuple[list[dict], dict]:
    ranked = [dict(x) for x in candidates]
    trace = {
        "eligible": False,
        "acted": False,
        "top_changed": False,
        "baseline_margin": None,
        "cap": 0.0,
        "confidence": float(brain.experience_state.confidence),
        "drift": bool(brain.experience_state.drift_suspected),
    }
    if len(ranked) < 2:
        return ranked, trace
    margin = float(ranked[0]["score"] - ranked[1]["score"])
    trace["baseline_margin"] = margin
    if margin > TIE_MARGIN:
        return ranked, trace
    if brain.experience_state.confidence < MIN_CONFIDENCE or brain.experience_state.drift_suspected:
        return ranked, trace

    trace["eligible"] = True
    cap = ASSIST_ABS_CAP if margin <= 1e-12 else min(ASSIST_ABS_CAP, ASSIST_REL_CAP * margin)
    trace["cap"] = cap
    probs = [evidence_probability(brain, int(x["source"]), int(x["target"])) for x in ranked]
    center = sum(probs) / len(probs)
    for item, p in zip(ranked, probs):
        pref = p - center
        modulation = max(-cap, min(cap, pref * 2.0 * cap))
        item["assist_probability"] = p
        item["assist_modulation"] = modulation
        item["assisted_score"] = float(item["score"] + modulation)
    baseline_top = int(ranked[0]["target"])
    ranked.sort(key=lambda x: (-float(x["assisted_score"]), int(x["target"])))
    assisted_top = int(ranked[0]["target"])
    trace["acted"] = any(abs(float(x.get("assist_modulation", 0.0))) > 0 for x in ranked)
    trace["top_changed"] = baseline_top != assisted_top
    return ranked, trace


def choose_move(brain: SphereBrain, current: int, blocked: set[int], *, assist: bool, rng: random.Random | None = None, explore: float = 0.0) -> tuple[str, int, list[dict], dict]:
    candidates = []
    for label, target in legal_moves(current, blocked):
        route = transition_route(brain, current, target, learn=False)
        candidates.append({"label": label, "source": current, "target": target, **route})
    candidates.sort(key=lambda x: (-float(x["score"]), int(x["target"])))
    trace = {"eligible": False, "acted": False, "top_changed": False}
    ranked = candidates
    if assist:
        ranked, trace = assist_rank(brain, candidates)
    if not ranked:
        raise RuntimeError("移動候補がありません。")
    if rng is not None and len(ranked) > 1 and rng.random() < explore:
        chosen = rng.choice(ranked)
    else:
        chosen = ranked[0]
    return str(chosen["label"]), int(chosen["target"]), ranked, trace


def run_episode(brain: SphereBrain, case: dict, *, assist: bool, rng: random.Random | None = None, explore: float = 0.0, max_steps: int = MAX_STEPS) -> dict:
    current = int(case["start"])
    goal = int(case["goal"])
    blocked = set(case["blocked"])
    path = [current]
    transitions = []
    traces = []
    for step in range(max_steps):
        if current == goal:
            break
        label, target, _ranked, trace = choose_move(brain, current, blocked, assist=assist, rng=rng, explore=explore)
        transitions.append((current, target))
        traces.append({"step": step + 1, "from": current, "to": target, "action": label, **trace})
        current = target
        path.append(current)
        if current == goal:
            break
    return {
        "success": current == goal,
        "steps": len(transitions),
        "path": path,
        "transitions": [list(x) for x in transitions],
        "loop_steps": max(0, len(path) - len(set(path))),
        "assist_eligible_steps": sum(1 for x in traces if x.get("eligible")),
        "assist_acted_steps": sum(1 for x in traces if x.get("acted")),
        "assist_top_changes": sum(1 for x in traces if x.get("top_changed")),
        "assist_trace": traces,
    }


def update_native_state(brain: SphereBrain, episode: dict) -> None:
    success = bool(episode["success"])
    seen = set()
    for source, target in episode["transitions"]:
        token = opaque_transition(int(source), int(target))
        if token in seen:
            continue
        seen.add(token)
        brain.experience_state.observe(condition=token, present=success, motif="m67", expected_conditions=EXPECTED)


def reinforce_success(brain: SphereBrain, episode: dict) -> None:
    if not episode["success"]:
        return
    seen = set()
    for source, target in episode["transitions"]:
        pair = (int(source), int(target))
        if pair in seen:
            continue
        seen.add(pair)
        for _ in range(REPLAY_REPEATS):
            transition_route(brain, pair[0], pair[1], learn=True)


def train_on_case(brain: SphereBrain, case: dict, *, episodes: int, assist: bool, seed: int) -> dict:
    rng = random.Random(seed)
    successes = 0
    for _ in range(episodes):
        ep = run_episode(brain, case, assist=assist, rng=rng, explore=EXPLORATION, max_steps=24)
        update_native_state(brain, ep)
        if ep["success"]:
            successes += 1
            reinforce_success(brain, ep)
    return {"successes": successes, "state": brain.snapshot_experience_state()}


def shortest_steps(case: dict) -> int | None:
    start, goal, blocked = int(case["start"]), int(case["goal"]), set(case["blocked"])
    q = [(start, 0)]
    seen = {start}
    for node, dist in q:
        if node == goal:
            return dist
        for _, nxt in legal_moves(node, blocked):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return None


def score_result(ep: dict, optimal: int | None) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def evaluate_suite(brain: SphereBrain, *, assist: bool) -> list[dict]:
    out = []
    for case in CASES:
        ep = run_episode(brain, case, assist=assist)
        optimal = shortest_steps(case)
        out.append({
            "case": case["name"],
            "start": case["start"],
            "goal": case["goal"],
            "blocked": sorted(case["blocked"]),
            "optimal_steps": optimal,
            "optimal": bool(ep["success"] and optimal is not None and ep["steps"] == optimal),
            **ep,
        })
    return out


def summarize(rows: list[dict]) -> dict:
    successes = [r for r in rows if r["success"]]
    return {
        "success_count": len(successes),
        "success_rate": len(successes) / len(rows),
        "optimal_count": sum(1 for r in rows if r["optimal"]),
        "mean_steps_success": None if not successes else sum(int(r["steps"]) for r in successes) / len(successes),
        "assist_eligible_steps": sum(int(r.get("assist_eligible_steps", 0)) for r in rows),
        "assist_acted_steps": sum(int(r.get("assist_acted_steps", 0)) for r in rows),
        "assist_top_changes": sum(int(r.get("assist_top_changes", 0)) for r in rows),
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()

    trained = copy.deepcopy(base)
    train_info = train_on_case(trained, BASE, episodes=TRAIN_EPISODES, assist=False, seed=SEED)

    core_rows = evaluate_suite(copy.deepcopy(trained), assist=False)
    assist_rows = evaluate_suite(copy.deepcopy(trained), assist=True)
    core_summary = summarize(core_rows)
    assist_summary = summarize(assist_rows)

    # Adaptation stress: old route is blocked. Compare equal additional experience budgets.
    changed_case = CASES[-1]
    core_adapt = copy.deepcopy(trained)
    assist_adapt = copy.deepcopy(trained)
    pre_core = run_episode(core_adapt, changed_case, assist=False)
    pre_assist = run_episode(assist_adapt, changed_case, assist=True)
    core_adapt_info = train_on_case(core_adapt, changed_case, episodes=ADAPT_EPISODES, assist=False, seed=SEED + 100)
    assist_adapt_info = train_on_case(assist_adapt, changed_case, episodes=ADAPT_EPISODES, assist=True, seed=SEED + 100)
    post_core = run_episode(core_adapt, changed_case, assist=False)
    post_assist = run_episode(assist_adapt, changed_case, assist=True)

    core_metric = (core_summary["success_count"], core_summary["optimal_count"], -(core_summary["mean_steps_success"] or 999.0))
    assist_metric = (assist_summary["success_count"], assist_summary["optimal_count"], -(assist_summary["mean_steps_success"] or 999.0))
    if assist_metric > core_metric:
        winner = "bounded_assist"
    elif core_metric > assist_metric:
        winner = "core_only"
    else:
        winner = "tie"

    assist_better_adaptation = score_result(post_assist, shortest_steps(changed_case)) > score_result(post_core, shortest_steps(changed_case))
    core_better_adaptation = score_result(post_core, shortest_steps(changed_case)) > score_result(post_assist, shortest_steps(changed_case))

    if winner == "bounded_assist" or assist_better_adaptation:
        verdict = "bounded_assist_improves_generalization_or_adaptation_over_core_only"
        next_step = "audit_where_assist_helped_then_integrate_only_the_verified_boundary_mechanism"
        readiness = "assist_has_measured_value"
    elif winner == "core_only" or core_better_adaptation:
        verdict = "core_only_generalizes_or_adapts_better_under_current_assist_design"
        next_step = "improve_assist_design_without_weakening_native_core_then_repeat_generalization"
        readiness = "core_leads_assist_still_open"
    else:
        verdict = "core_and_bounded_assist_are_tied_on_current_generalization_suite"
        next_step = "expand_generalization_cases_and_credit_assignment_before_deciding_assist_role"
        readiness = "assist_role_undecided"

    payload = {
        "experiment": "Core Growth Binding v67 — P/E Puzzle Generalization Trial",
        "contract": {
            "trained_only_on_base_board_before_generalization": True,
            "shortest_path_given": False,
            "assist_compared_fairly": True,
            "assist_only_within_tie_margin": TIE_MARGIN,
            "assist_min_confidence": MIN_CONFIDENCE,
            "assist_absolute_cap": ASSIST_ABS_CAP,
            "assist_relative_cap": ASSIST_REL_CAP,
            "production_brain_json_saved": False,
        },
        "base_training": {"successful_episodes": train_info["successes"], "episodes": TRAIN_EPISODES},
        "generalization": {
            "core_only": {"summary": core_summary, "cases": core_rows},
            "bounded_assist": {"summary": assist_summary, "cases": assist_rows},
            "winner": winner,
        },
        "changed_environment_adaptation": {
            "case": changed_case["name"],
            "core_only": {"before": pre_core, "training_successes": core_adapt_info["successes"], "after": post_core},
            "bounded_assist": {"before": pre_assist, "training_successes": assist_adapt_info["successes"], "after": post_assist},
            "assist_better": assist_better_adaptation,
            "core_better": core_better_adaptation,
        },
        "summary": {
            "core_success_rate": core_summary["success_rate"],
            "assist_success_rate": assist_summary["success_rate"],
            "core_optimal_count": core_summary["optimal_count"],
            "assist_optimal_count": assist_summary["optimal_count"],
            "assist_eligible_steps": assist_summary["assist_eligible_steps"],
            "assist_acted_steps": assist_summary["assist_acted_steps"],
            "assist_top_changes": assist_summary["assist_top_changes"],
            "generalization_winner": winner,
            "core_changed_env_after_success": post_core["success"],
            "core_changed_env_after_steps": post_core["steps"],
            "assist_changed_env_after_success": post_assist["success"],
            "assist_changed_env_after_steps": post_assist["steps"],
            "brain_file_unchanged": before_hash == file_hash(BRAIN_PATH),
            "overall_verdict": verdict,
            "core_readiness": readiness,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v67.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v67</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:950px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v67：P/E Puzzle Generalization Trial</h1><p class="lead">v66の学習済みCoreを、未経験の開始位置・E位置・逆方向・旧正解経路が塞がれた盤面で評価する。Core単体とNative Experience State由来のBounded Assistを公平に比較し、より良い結果を採る。</p><section class="panel"><div class="controls"><button id="run">一般化 + Assist比較を実行</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">まだ実行していません。</pre></section></main><script>
function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(3)}function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.onclick=async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const r=await fetch('/api/observe',{method:'POST'});if(!r.ok)throw new Error('HTTP '+r.status+' '+await r.text());const d=await r.json(),s=d.summary;m.innerHTML=`<div class="metric">Core到達率<b>${f(s.core_success_rate)}</b></div><div class="metric">Assist到達率<b>${f(s.assist_success_rate)}</b></div><div class="metric">Core最短Case<b>${s.core_optimal_count}</b></div><div class="metric">Assist最短Case<b>${s.assist_optimal_count}</b></div><div class="metric">Assist Eligible Step<b>${s.assist_eligible_steps}</b></div><div class="metric">Assist作動Step<b>${s.assist_acted_steps}</b></div><div class="metric">Assist Top変更<b>${s.assist_top_changes}</b></div><div class="metric">一般化Winner<b class="blue">${s.generalization_winner}</b></div><div class="metric">変更盤面 Core<b class="${s.core_changed_env_after_success?'good':'warn'}">${yn(s.core_changed_env_after_success)} / ${s.core_changed_env_after_steps}手</b></div><div class="metric">変更盤面 Assist<b class="${s.assist_changed_env_after_success?'good':'warn'}">${yn(s.assist_changed_env_after_success)} / ${s.assist_changed_env_after_steps}手</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}};
</script></body></html>'''

@app.get("/")
def index():
    return PAGE

@app.post("/api/observe")
def api_observe():
    return jsonify(observe())


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v67: http://{HOST}:{PORT}")
    print("P/E generalization / Core-only vs bounded Native-State Assist / no production Core save")
    serve(app, host=HOST, port=PORT)
