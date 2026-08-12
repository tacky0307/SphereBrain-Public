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

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

HOST = "127.0.0.1"
START_PORT = 5113
OUT = ROOT / "data" / "core_growth_binding_v66" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
GRID = 3
START = 0
GOAL = 8
BLOCKED = {3, 4}
MAX_EVAL_STEPS = 16
TRAIN_EPISODES = 30
EXPLORATION = 0.35
REPLAY_REPEATS = 2
SEED = 6601
DIRS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}


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


def legal_moves(node: int) -> list[tuple[str, int]]:
    r, c = rc(node)
    result = []
    for name, (dr, dc) in DIRS.items():
        target = node_at(r + dr, c + dc)
        if target is None or target in BLOCKED:
            continue
        result.append((name, target))
    return result


def opaque_transition(source: int, target: int) -> str:
    return "t_" + hashlib.sha256(f"{source}>{target}".encode()).hexdigest()[:12]


def all_transition_tokens() -> list[str]:
    out = []
    for node in range(GRID * GRID):
        if node in BLOCKED:
            continue
        for _, target in legal_moves(node):
            out.append(opaque_transition(node, target))
    return sorted(set(out))

EXPECTED = all_transition_tokens()


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
        weights = [float(brain.weights[a, b]) for a, b in edges]
        usages = [float(brain.usage[a, b]) for a, b in edges]
        mean_weight = float(sum(weights) / len(weights))
        mean_usage = float(sum(usages) / len(usages))
    else:
        mean_weight = 0.0
        mean_usage = 0.0
    score = mean_weight + 0.012 * math.log1p(mean_usage) + 0.0005 * len(result.activated_nodes)
    return {
        "score": float(score),
        "nodes": list(result.activated_nodes),
        "edges": [list(x) for x in result.traversed_edges],
        "mean_weight": mean_weight,
        "mean_usage": mean_usage,
    }


def choose_move(brain: SphereBrain, current: int, rng: random.Random | None = None, explore: float = 0.0) -> tuple[str, int, list[dict]]:
    candidates = []
    for label, target in legal_moves(current):
        route = transition_route(brain, current, target, learn=False)
        candidates.append({"label": label, "target": target, **route})
    candidates.sort(key=lambda x: (-float(x["score"]), int(x["target"])))
    if not candidates:
        raise RuntimeError("移動候補がありません。")
    if rng is not None and len(candidates) > 1 and rng.random() < explore:
        chosen = rng.choice(candidates)
    else:
        chosen = candidates[0]
    return str(chosen["label"]), int(chosen["target"]), candidates


def run_episode(brain: SphereBrain, *, rng: random.Random | None = None, explore: float = 0.0, max_steps: int = MAX_EVAL_STEPS) -> dict:
    current = START
    path = [current]
    transitions = []
    choices = []
    for step in range(max_steps):
        if current == GOAL:
            break
        label, target, candidates = choose_move(brain, current, rng=rng, explore=explore)
        transitions.append((current, target))
        choices.append({"step": step + 1, "from": current, "to": target, "action": label, "candidates": candidates})
        current = target
        path.append(current)
        if current == GOAL:
            break
    return {
        "success": current == GOAL,
        "steps": len(transitions),
        "path": path,
        "transitions": [list(x) for x in transitions],
        "choices": choices,
        "loop_steps": max(0, len(path) - len(set(path))),
    }


def update_native_state(brain: SphereBrain, episode: dict) -> None:
    success = bool(episode["success"])
    seen = set()
    for source, target in episode["transitions"]:
        token = opaque_transition(int(source), int(target))
        if token in seen:
            continue
        seen.add(token)
        brain.experience_state.observe(
            condition=token,
            present=success,
            motif="m66",
            expected_conditions=EXPECTED,
        )


def reinforce_success(brain: SphereBrain, episode: dict) -> None:
    if not episode["success"]:
        return
    # Reward only the transitions that actually occurred in a successful episode.
    # No shortest-path or direction label is supplied.
    unique = []
    seen = set()
    for source, target in episode["transitions"]:
        pair = (int(source), int(target))
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    for source, target in unique:
        for _ in range(REPLAY_REPEATS):
            transition_route(brain, source, target, learn=True)


def train_branch(base: SphereBrain, *, learn_weights: bool) -> dict:
    brain = copy.deepcopy(base)
    rng = random.Random(SEED + (1 if learn_weights else 0))
    timeline = []
    evals = {0: run_episode(brain)}
    successes = 0
    for episode_index in range(1, TRAIN_EPISODES + 1):
        ep = run_episode(brain, rng=rng, explore=EXPLORATION, max_steps=24)
        update_native_state(brain, ep)
        if ep["success"]:
            successes += 1
            if learn_weights:
                reinforce_success(brain, ep)
        timeline.append({
            "episode": episode_index,
            "success": ep["success"],
            "steps": ep["steps"],
            "loop_steps": ep["loop_steps"],
            "experience_confidence": float(brain.experience_state.confidence),
            "drift": bool(brain.experience_state.drift_suspected),
        })
        if episode_index in {10, 30}:
            evals[episode_index] = run_episode(brain)
    return {
        "successes": successes,
        "timeline": timeline,
        "evals": {str(k): v for k, v in evals.items()},
        "experience_state": brain.snapshot_experience_state(),
        "brain": brain,
    }


def behavior_metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()

    state_only = train_branch(base, learn_weights=False)
    learned = train_branch(base, learn_weights=True)

    state0 = state_only["evals"]["0"]
    state30 = state_only["evals"]["30"]
    learn0 = learned["evals"]["0"]
    learn10 = learned["evals"]["10"]
    learn30 = learned["evals"]["30"]

    state_behavior_changed = behavior_metric(state30) != behavior_metric(state0) or state30["path"] != state0["path"]
    learned_behavior_changed = behavior_metric(learn30) != behavior_metric(learn0) or learn30["path"] != learn0["path"]
    learned_improved = behavior_metric(learn30) > behavior_metric(learn0)
    native_state_formed = int(learned["experience_state"].get("evidence_experiences", 0)) > 0
    no_position_labels = not learned["brain"].experience_state.contains_forbidden_position_label()
    brain_file_unchanged = before_hash == file_hash(BRAIN_PATH)

    if learned_improved:
        verdict = "native_core_route_learning_improves_pe_puzzle_behavior_without_assist"
        next_step = "stress_test_native_puzzle_learning_and_environment_change_before_assist"
        readiness = "native_puzzle_learning_observed"
    elif learned["successes"] == 0:
        verdict = "no_successful_experience_was_discovered_so_assist_need_cannot_be_judged"
        next_step = "audit_exploration_and_transition_scoring_before_assist"
        readiness = "puzzle_learning_inconclusive"
    elif native_state_formed:
        verdict = "native_experience_state_forms_but_existing_core_learning_does_not_improve_choice"
        next_step = "compare_credit_assignment_vs_bounded_assist_as_behavior_bridge"
        readiness = "behavior_bridge_candidate"
    else:
        verdict = "native_experience_state_failed_to_form_in_puzzle"
        next_step = "fix_native_experience_observation_before_assist"
        readiness = "native_state_not_ready"

    payload = {
        "experiment": "Core Growth Binding v66 — P/E 3x3 Puzzle Native Core Trial",
        "board": {"grid": 3, "start": START, "goal": GOAL, "blocked": sorted(BLOCKED)},
        "contract": {
            "assist_used": False,
            "shortest_path_given": False,
            "direction_reward_given": False,
            "training_exploration": EXPLORATION,
            "successful_episode_transition_replay_only": True,
            "native_experience_state_used": True,
            "production_brain_json_saved": False,
        },
        "state_only_control": {k: v for k, v in state_only.items() if k != "brain"},
        "native_route_learning": {k: v for k, v in learned.items() if k != "brain"},
        "summary": {
            "native_state_formed": native_state_formed,
            "state_only_behavior_changed": state_behavior_changed,
            "learning_behavior_changed": learned_behavior_changed,
            "learning_improved": learned_improved,
            "successful_training_episodes": learned["successes"],
            "before_success": learn0["success"],
            "before_steps": learn0["steps"],
            "after10_success": learn10["success"],
            "after10_steps": learn10["steps"],
            "after30_success": learn30["success"],
            "after30_steps": learn30["steps"],
            "before_path": learn0["path"],
            "after30_path": learn30["path"],
            "no_position_labels_in_native_state": no_position_labels,
            "brain_file_unchanged": brain_file_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v66.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v66</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.layout{display:grid;grid-template-columns:360px 1fr;gap:20px}.board{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.cell{aspect-ratio:1;background:#203654;border:1px solid #49698d;border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:42px;font-weight:900}.blocked{background:#08101c}.goal{color:var(--green)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.path{font-size:22px;color:var(--blue);overflow-wrap:anywhere}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.layout{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}}
</style></head><body><main><h1>v66：P/E 3×3 Puzzle — Native Core Trial</h1><p class="lead">Assistなし。Native Experience StateだけのControlと、成功した試行で実際に通ったCore遷移経路だけを再経験するNative route learningを比較する。最短経路は教えない。</p><section class="panel"><div class="controls"><button id="run">Native Coreを3×3世界へ</button></div></section><div class="layout"><section class="panel"><h2>3×3世界</h2><div class="board"><div class="cell">P</div><div class="cell"></div><div class="cell"></div><div class="cell blocked"></div><div class="cell blocked"></div><div class="cell"></div><div class="cell"></div><div class="cell"></div><div class="cell goal">E</div></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div><h3>行動経路</h3><div id="paths" class="path">まだ実行していません。</div></section></div><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">まだ実行していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}const btn=document.getElementById('run');btn.onclick=async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const r=await fetch('/api/observe',{method:'POST'});if(!r.ok)throw new Error('HTTP '+r.status+' '+await r.text());const d=await r.json(),s=d.summary;m.innerHTML=`<div class="metric">Native State形成<b class="${s.native_state_formed?'good':'warn'}">${yn(s.native_state_formed)}</b></div><div class="metric">Stateだけで行動変化<b>${yn(s.state_only_behavior_changed)}</b></div><div class="metric">Core学習で行動変化<b class="${s.learning_behavior_changed?'good':'warn'}">${yn(s.learning_behavior_changed)}</b></div><div class="metric">Core学習で改善<b class="${s.learning_improved?'good':'warn'}">${yn(s.learning_improved)}</b></div><div class="metric">成功経験数<b>${s.successful_training_episodes}</b></div><div class="metric">初回<b>${s.before_success?'E到達':'未到達'} / ${s.before_steps}手</b></div><div class="metric">10回後<b>${s.after10_success?'E到達':'未到達'} / ${s.after10_steps}手</b></div><div class="metric">30回後<b>${s.after30_success?'E到達':'未到達'} / ${s.after30_steps}手</b></div><div class="metric">位置ラベルなし<b class="${s.no_position_labels_in_native_state?'good':'warn'}">${yn(s.no_position_labels_in_native_state)}</b></div><div class="metric">brain.json<b class="good">${s.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div>`;document.getElementById('paths').innerHTML=`初回：${s.before_path.join(' → ')}<br>30回後：${s.after30_path.join(' → ')}`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}};
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
    print(f"Core Growth Binding v66: http://{HOST}:{PORT}")
    print("P/E 3x3 Native Core Trial / no Assist / no shortest-path teaching / production brain.json unchanged")
    serve(app, host=HOST, port=PORT)
