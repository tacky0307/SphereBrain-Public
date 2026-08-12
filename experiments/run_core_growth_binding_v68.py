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
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

HOST = "127.0.0.1"
START_PORT = 5115
OUT = ROOT / "data" / "core_growth_binding_v68" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
GRID = 3
SEED = 6801
PRETRAIN_EPISODES = 30
ADAPT_EPISODES = 20
EXPLORATION = 0.35
MAX_STEPS = 16
TRAIN_MAX_STEPS = 24
REPLAY_REPEATS = 2
FAIL_STREAK_GATE = 3
MILD_DECAY = 0.985
DRIFT_DECAY = 0.965
MIN_FAILURE_EVIDENCE = 0.55
TIE_MARGIN = 0.0025
ASSIST_ABS_CAP = 5e-5
ASSIST_REL_CAP = 0.55
MIN_CONFIDENCE = 0.80
DIRS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}
BASE = {"name": "trained_board", "start": 0, "goal": 8, "blocked": {3, 4}}
CHANGED = {"name": "route_blocked", "start": 0, "goal": 8, "blocked": {2, 4}}
CHECKPOINTS = {0, 5, 10, 20}


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
    out = []
    for name, (dr, dc) in DIRS.items():
        target = node_at(r + dr, c + dc)
        if target is None or target in blocked:
            continue
        out.append((name, target))
    return out


def opaque_transition(source: int, target: int) -> str:
    return "t_" + hashlib.sha256(f"{source}>{target}".encode()).hexdigest()[:12]


def transition_universe() -> list[str]:
    tokens = []
    for blocked in ({3, 4}, {2, 4}):
        blocked = set(blocked)
        for node in range(GRID * GRID):
            if node in blocked:
                continue
            for _, target in legal_moves(node, blocked):
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
    return {
        "score": score,
        "edges": [list(x) for x in result.traversed_edges],
        "nodes": list(result.activated_nodes),
        "mean_weight": float(mean_weight),
        "mean_usage": float(mean_usage),
    }


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
    probs = [evidence_probability(brain, int(x["source"]), int(x["target"])) for x in ranked]
    center = sum(probs) / len(probs)
    for item, p in zip(ranked, probs):
        modulation = max(-cap, min(cap, (p - center) * 2.0 * cap))
        item["assist_probability"] = float(p)
        item["assist_modulation"] = float(modulation)
        item["assisted_score"] = float(item["score"] + modulation)
    baseline_top = int(ranked[0]["target"])
    ranked.sort(key=lambda x: (-float(x.get("assisted_score", x["score"])), int(x["target"])))
    trace["acted"] = any(abs(float(x.get("assist_modulation", 0.0))) > 0 for x in ranked)
    trace["top_changed"] = baseline_top != int(ranked[0]["target"])
    return ranked, trace


def choose_move(brain: SphereBrain, current: int, blocked: set[int], *, assist: bool, rng: random.Random | None = None, explore: float = 0.0) -> tuple[str, int, list[dict], dict]:
    candidates = []
    for label, target in legal_moves(current, blocked):
        route = transition_route(brain, current, target, learn=False)
        candidates.append({"label": label, "source": current, "target": target, **route})
    candidates.sort(key=lambda x: (-float(x["score"]), int(x["target"])))
    ranked = candidates
    trace = {"eligible": False, "acted": False, "top_changed": False}
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
        brain.experience_state.observe(
            condition=token,
            present=success,
            motif="m68",
            expected_conditions=EXPECTED,
        )


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


def route_weight_snapshot(brain: SphereBrain, source: int, target: int) -> float:
    route = transition_route(brain, source, target, learn=False)
    return float(route["mean_weight"])


def decay_transition_credit(brain: SphereBrain, source: int, target: int, factor: float) -> dict:
    route = transition_route(brain, source, target, learn=False)
    changed = []
    for a, b in [tuple(x) for x in route["edges"]]:
        before = float(brain.weights[a, b])
        after = max(0.0, min(1.0, before * factor))
        brain.weights[a, b] = after
        brain.weights[b, a] = after
        changed.append({"edge": [int(a), int(b)], "before": before, "after": after})
    return {"source": source, "target": target, "factor": factor, "changed_edges": changed}


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


def pretrain_base(base: SphereBrain) -> SphereBrain:
    brain = copy.deepcopy(base)
    rng = random.Random(SEED)
    for _ in range(PRETRAIN_EPISODES):
        ep = run_episode(brain, BASE, assist=False, rng=rng, explore=EXPLORATION, max_steps=TRAIN_MAX_STEPS)
        update_native_state(brain, ep)
        if ep["success"]:
            reinforce_success(brain, ep)
    return brain


def adapt_branch(pretrained: SphereBrain, *, mode: str) -> dict:
    brain = copy.deepcopy(pretrained)
    assist = mode == "adaptive_plus_assist"
    adaptive = mode in {"adaptive_credit", "adaptive_plus_assist"}
    rng = random.Random(SEED + {"success_only": 10, "adaptive_credit": 20, "adaptive_plus_assist": 20}[mode])
    fail_streak: dict[tuple[int, int], int] = defaultdict(int)
    credit_events = []
    checkpoints = {"0": run_episode(brain, CHANGED, assist=assist)}
    timeline = []
    total_success = 0
    old_transition = (0, 1)
    new_transition = (0, 3)
    old_weight_before = route_weight_snapshot(brain, *old_transition)
    new_weight_before = route_weight_snapshot(brain, *new_transition)

    for episode_index in range(1, ADAPT_EPISODES + 1):
        ep = run_episode(brain, CHANGED, assist=assist, rng=rng, explore=EXPLORATION, max_steps=TRAIN_MAX_STEPS)
        update_native_state(brain, ep)
        unique_pairs = []
        seen = set()
        for source, target in ep["transitions"]:
            pair = (int(source), int(target))
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        if ep["success"]:
            total_success += 1
            reinforce_success(brain, ep)
            for pair in unique_pairs:
                fail_streak[pair] = 0
        else:
            for pair in unique_pairs:
                fail_streak[pair] += 1
                if not adaptive or fail_streak[pair] < FAIL_STREAK_GATE:
                    continue
                probability = evidence_probability(brain, pair[0], pair[1])
                failure_evidence = 1.0 - probability
                if failure_evidence < MIN_FAILURE_EVIDENCE:
                    continue
                factor = DRIFT_DECAY if brain.experience_state.drift_suspected else MILD_DECAY
                event = decay_transition_credit(brain, pair[0], pair[1], factor)
                event.update({
                    "episode": episode_index,
                    "fail_streak": fail_streak[pair],
                    "success_probability": probability,
                    "failure_evidence": failure_evidence,
                    "drift": bool(brain.experience_state.drift_suspected),
                })
                credit_events.append(event)

        timeline.append({
            "episode": episode_index,
            "success": bool(ep["success"]),
            "steps": int(ep["steps"]),
            "loop_steps": int(ep["loop_steps"]),
            "confidence": float(brain.experience_state.confidence),
            "drift": bool(brain.experience_state.drift_suspected),
            "old_streak": int(fail_streak.get(old_transition, 0)),
            "credit_events_total": len(credit_events),
            "assist_eligible": int(ep.get("assist_eligible_steps", 0)),
            "assist_acted": int(ep.get("assist_acted_steps", 0)),
            "assist_top_changes": int(ep.get("assist_top_changes", 0)),
        })
        if episode_index in CHECKPOINTS:
            checkpoints[str(episode_index)] = run_episode(brain, CHANGED, assist=assist)

    old_weight_after = route_weight_snapshot(brain, *old_transition)
    new_weight_after = route_weight_snapshot(brain, *new_transition)
    return {
        "mode": mode,
        "checkpoints": checkpoints,
        "timeline": timeline,
        "successful_adaptation_episodes": total_success,
        "credit_events": credit_events,
        "credit_event_count": len(credit_events),
        "old_transition": list(old_transition),
        "new_transition": list(new_transition),
        "old_weight_before": old_weight_before,
        "old_weight_after": old_weight_after,
        "old_weight_delta": old_weight_after - old_weight_before,
        "new_weight_before": new_weight_before,
        "new_weight_after": new_weight_after,
        "new_weight_delta": new_weight_after - new_weight_before,
        "final_state": brain.snapshot_experience_state(),
    }


def metric(ep: dict, optimal: int | None) -> tuple[int, int, int, int]:
    return (
        1 if ep["success"] else 0,
        1 if ep["success"] and optimal is not None and ep["steps"] == optimal else 0,
        -int(ep["steps"]),
        -int(ep["loop_steps"]),
    )


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = pretrain_base(base)

    branches = {
        "success_only": adapt_branch(pretrained, mode="success_only"),
        "adaptive_credit": adapt_branch(pretrained, mode="adaptive_credit"),
        "adaptive_plus_assist": adapt_branch(pretrained, mode="adaptive_plus_assist"),
    }
    optimal = shortest_steps(CHANGED)
    final_metrics = {name: metric(row["checkpoints"]["20"], optimal) for name, row in branches.items()}
    winner = max(final_metrics, key=final_metrics.get)
    tied = [name for name, value in final_metrics.items() if value == final_metrics[winner]]
    if len(tied) > 1:
        winner = "tie:" + ",".join(tied)

    adaptive = branches["adaptive_credit"]
    assist = branches["adaptive_plus_assist"]
    assist_eligible = sum(x["assist_eligible"] for x in assist["timeline"])
    assist_acted = sum(x["assist_acted"] for x in assist["timeline"])
    assist_top = sum(x["assist_top_changes"] for x in assist["timeline"])

    if final_metrics["adaptive_plus_assist"] > final_metrics["adaptive_credit"]:
        assist_role = "measured_positive"
    elif final_metrics["adaptive_plus_assist"] < final_metrics["adaptive_credit"]:
        assist_role = "measured_negative"
    elif assist_eligible == 0:
        assist_role = "not_exercised"
    else:
        assist_role = "neutral_on_this_suite"

    payload = {
        "experiment": "Core Growth Binding v68 — Adaptive Credit Assignment & Recovery",
        "board": {"base": {**BASE, "blocked": sorted(BASE["blocked"])}, "changed": {**CHANGED, "blocked": sorted(CHANGED["blocked"])}},
        "contract": {
            "single_failure_penalty": False,
            "failure_streak_gate": FAIL_STREAK_GATE,
            "native_failure_evidence_required": MIN_FAILURE_EVIDENCE,
            "mild_decay": MILD_DECAY,
            "drift_decay": DRIFT_DECAY,
            "success_reinforcement_preserved": True,
            "assist_compared_not_assumed": True,
            "production_brain_json_saved": False,
        },
        "branches": branches,
        "summary": {
            "optimal_steps_changed_board": optimal,
            "winner": winner,
            "success_only_after20": branches["success_only"]["checkpoints"]["20"],
            "adaptive_after20": adaptive["checkpoints"]["20"],
            "assist_after20": assist["checkpoints"]["20"],
            "adaptive_credit_events": adaptive["credit_event_count"],
            "assist_credit_events": assist["credit_event_count"],
            "adaptive_old_weight_delta": adaptive["old_weight_delta"],
            "adaptive_new_weight_delta": adaptive["new_weight_delta"],
            "assist_eligible_steps": assist_eligible,
            "assist_acted_steps": assist_acted,
            "assist_top_changes": assist_top,
            "assist_role": assist_role,
            "brain_file_unchanged": before_hash == file_hash(BRAIN_PATH),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v68.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v68</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v68：Adaptive Credit Assignment & Recovery</h1><p class="lead">Success-only / Adaptive Credit / Adaptive Credit + Bounded Assist を同じ変更盤面・同じ経験予算で比較する。単発失敗では古い経路を弱めず、持続失敗とNative Evidenceが揃ったときだけcreditを下げる。</p><section class="panel"><div class="controls"><button id="run">Adaptive Creditを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="summary" class="metrics"></div></section><section class="panel"><h2>5 / 10 / 20経験後</h2><div id="checkpoints" class="raw">まだ実行していません。</div></section><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">まだ実行していません。</pre></section></main><script>
const el=id=>document.getElementById(id);function yes(v){return v?'YES':'NO'}function metric(k,v,c=''){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function epText(x){return `${x.success?'E到達':'未到達'} / ${x.steps}手 / ${x.path.join(' → ')}`}
async function run(){el('run').disabled=true;el('run').textContent='検証中...';try{const r=await fetch('/api/observe',{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.error||'error');const s=d.summary;el('summary').innerHTML=metric('Winner',s.winner,'blue')+metric('Success-only 20',epText(s.success_only_after20),s.success_only_after20.success?'good':'warn')+metric('Adaptive 20',epText(s.adaptive_after20),s.adaptive_after20.success?'good':'warn')+metric('Adaptive+Assist 20',epText(s.assist_after20),s.assist_after20.success?'good':'warn')+metric('Adaptive credit event',s.adaptive_credit_events)+metric('旧経路 weight Δ',Number(s.adaptive_old_weight_delta).toFixed(6),s.adaptive_old_weight_delta<0?'good':'')+metric('新経路 weight Δ',Number(s.adaptive_new_weight_delta).toFixed(6),s.adaptive_new_weight_delta>0?'good':'')+metric('Assist Eligible',s.assist_eligible_steps)+metric('Assist作動',s.assist_acted_steps)+metric('Assist Top変更',s.assist_top_changes)+metric('Assist role',s.assist_role,'blue')+metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn');let lines=[];for(const [name,b] of Object.entries(d.branches)){lines.push(`\n[${name}]`);for(const c of ['0','5','10','20'])lines.push(`${c}経験後: ${epText(b.checkpoints[c])}`);lines.push(`credit events=${b.credit_event_count} oldΔ=${b.old_weight_delta.toFixed(6)} newΔ=${b.new_weight_delta.toFixed(6)}`)}el('checkpoints').textContent=lines.join('\n');el('raw').textContent=JSON.stringify(d,null,2)}catch(e){el('summary').innerHTML=metric('エラー',e.message,'warn')}finally{el('run').disabled=false;el('run').textContent='Adaptive Creditを検証'}}el('run').onclick=run;</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/observe")
def api_observe():
    try:
        return jsonify(observe())
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v68: http://{HOST}:{PORT}")
    print("Adaptive Credit Assignment / Recovery / Assist comparison / production brain.json no-save")
    serve(app, host=HOST, port=PORT)
