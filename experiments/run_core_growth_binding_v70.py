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
START_PORT = 5117
OUT = ROOT / "data" / "core_growth_binding_v70" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
GRID = 3
SEED = 7001
PRETRAIN_EPISODES = 30
ADAPT_EPISODES = 20
EXPLORATION = 0.35
MAX_STEPS = 16
TRAIN_MAX_STEPS = 24
REPLAY_REPEATS = 2
FAIL_STREAK_GATE = 3
MILD_DECAY = 0.985
DRIFT_DECAY = 0.965
RELATIVE_DECAY = 0.988
MIN_FAILURE_EVIDENCE = 0.55
RELATIVE_FAILURE_POSITIONS = 2
RELATIVE_SCORE_SHARE = 0.20
TIE_MARGIN = 0.0025
ASSIST_ABS_CAP = 5e-5
ASSIST_REL_CAP = 0.55
MIN_CONFIDENCE = 0.80
CHECKPOINTS = {0, 5, 10, 20}
DIRS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}
BASE = {"name": "trained_board", "start": 0, "goal": 8, "blocked": {3, 4}}
CHANGED = {"name": "route_blocked", "start": 0, "goal": 8, "blocked": {2, 4}}
MODES = ["success_only", "specific_adaptive", "relative_credit", "relative_plus_assist"]


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


def delta_of(source: int, target: int) -> tuple[int, int]:
    sr, sc = rc(source)
    tr, tc = rc(target)
    return tr - sr, tc - sc


def legal_moves(node: int, blocked: set[int]) -> list[tuple[str, int]]:
    r, c = rc(node)
    out = []
    for name, (dr, dc) in DIRS.items():
        target = node_at(r + dr, c + dc)
        if target is None or target in blocked:
            continue
        out.append((name, target))
    return out


def specific_token(source: int, target: int) -> str:
    return "s_" + hashlib.sha256(f"{source}>{target}".encode()).hexdigest()[:14]


def relative_token(delta: tuple[int, int]) -> str:
    return "r_" + hashlib.sha256(f"{delta[0]},{delta[1]}".encode()).hexdigest()[:14]


def specific_universe() -> list[str]:
    tokens = []
    for blocked in ({3, 4}, {2, 4}):
        b = set(blocked)
        for node in range(GRID * GRID):
            if node in b:
                continue
            for _, target in legal_moves(node, b):
                tokens.append(specific_token(node, target))
    return sorted(set(tokens))


DELTAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
EXPECTED = specific_universe() + [relative_token(d) for d in DELTAS]


def unique_sources(*groups: list[int]) -> list[int]:
    out: list[int] = []
    for group in groups:
        for node in group:
            if int(node) not in out:
                out.append(int(node))
    return out


def transition_sources(brain: SphereBrain, source: int, target: int, *, relative: bool) -> list[int]:
    specific = brain.text_to_sources(specific_token(source, target), count=3)
    if not relative:
        return specific
    shared = brain.text_to_sources(relative_token(delta_of(source, target)), count=3)
    return unique_sources(specific, shared)


def relative_sources(brain: SphereBrain, delta: tuple[int, int]) -> list[int]:
    return brain.text_to_sources(relative_token(delta), count=3)


def route_from_sources(brain: SphereBrain, sources: list[int], *, learn: bool) -> dict:
    result = brain.propagate(sources, steps=8, threshold=0.18, noise=0.0, learn=learn)
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


def transition_route(brain: SphereBrain, source: int, target: int, *, learn: bool, relative: bool) -> dict:
    return route_from_sources(brain, transition_sources(brain, source, target, relative=relative), learn=learn)


def relative_route(brain: SphereBrain, delta: tuple[int, int], *, learn: bool) -> dict:
    return route_from_sources(brain, relative_sources(brain, delta), learn=learn)


def evidence_probability(brain: SphereBrain, token: str) -> float:
    row = brain.experience_state.condition_evidence.get(token, {})
    t = float(row.get("true_weight", 0.0))
    f = float(row.get("false_weight", 0.0))
    total = t + f
    return 0.5 if total <= 1e-12 else t / total


def candidate_rows(brain: SphereBrain, current: int, blocked: set[int], *, relative_mode: bool) -> list[dict]:
    rows = []
    for label, target in legal_moves(current, blocked):
        specific = transition_route(brain, current, target, learn=False, relative=False)
        delta = delta_of(current, target)
        relative = relative_route(brain, delta, learn=False)
        score = float(specific["score"])
        if relative_mode:
            # Relative structure is supportive only; the concrete route remains dominant.
            score += RELATIVE_SCORE_SHARE * float(relative["score"])
        rows.append({
            "label": label,
            "source": current,
            "target": target,
            "delta": list(delta),
            "specific_score": float(specific["score"]),
            "relative_score": float(relative["score"]),
            "score": score,
        })
    rows.sort(key=lambda x: (-float(x["score"]), int(x["target"])))
    return rows


def assist_rank(brain: SphereBrain, candidates: list[dict]) -> tuple[list[dict], dict]:
    ranked = [dict(x) for x in candidates]
    trace = {"eligible": False, "acted": False, "top_changed": False, "baseline_margin": None}
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
    prefs = []
    for item in ranked:
        s = evidence_probability(brain, specific_token(int(item["source"]), int(item["target"])))
        r = evidence_probability(brain, relative_token(tuple(item["delta"])))
        prefs.append(0.75 * s + 0.25 * r)
    center = sum(prefs) / len(prefs)
    baseline_top = int(ranked[0]["target"])
    for item, pref in zip(ranked, prefs):
        modulation = max(-cap, min(cap, (pref - center) * 2.0 * cap))
        item["assist_preference"] = float(pref)
        item["assist_modulation"] = float(modulation)
        item["assisted_score"] = float(item["score"] + modulation)
    ranked.sort(key=lambda x: (-float(x.get("assisted_score", x["score"])), int(x["target"])))
    trace["acted"] = any(abs(float(x.get("assist_modulation", 0.0))) > 0 for x in ranked)
    trace["top_changed"] = baseline_top != int(ranked[0]["target"])
    return ranked, trace


def choose_move(brain: SphereBrain, current: int, blocked: set[int], *, relative_mode: bool, assist: bool, rng: random.Random | None = None, explore: float = 0.0):
    ranked = candidate_rows(brain, current, blocked, relative_mode=relative_mode)
    trace = {"eligible": False, "acted": False, "top_changed": False}
    if assist:
        ranked, trace = assist_rank(brain, ranked)
    if not ranked:
        raise RuntimeError("移動候補がありません。")
    if rng is not None and len(ranked) > 1 and rng.random() < explore:
        chosen = rng.choice(ranked)
    else:
        chosen = ranked[0]
    return str(chosen["label"]), int(chosen["target"]), ranked, trace


def run_episode(brain: SphereBrain, case: dict, *, relative_mode: bool, assist: bool, rng: random.Random | None = None, explore: float = 0.0, max_steps: int = MAX_STEPS) -> dict:
    current = int(case["start"])
    goal = int(case["goal"])
    blocked = set(case["blocked"])
    path = [current]
    transitions = []
    traces = []
    for step in range(max_steps):
        if current == goal:
            break
        label, target, _ranked, trace = choose_move(brain, current, blocked, relative_mode=relative_mode, assist=assist, rng=rng, explore=explore)
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
    }


def observe_episode(brain: SphereBrain, episode: dict, *, include_relative: bool) -> None:
    success = bool(episode["success"])
    seen_specific = set()
    seen_relative = set()
    for source, target in episode["transitions"]:
        source, target = int(source), int(target)
        st = specific_token(source, target)
        if st not in seen_specific:
            seen_specific.add(st)
            brain.experience_state.observe(condition=st, present=success, motif="m70", expected_conditions=EXPECTED)
        if include_relative:
            rt = relative_token(delta_of(source, target))
            if rt not in seen_relative:
                seen_relative.add(rt)
                brain.experience_state.observe(condition=rt, present=success, motif="m70", expected_conditions=EXPECTED)


def reinforce_success(brain: SphereBrain, episode: dict, *, include_relative: bool) -> None:
    if not episode["success"]:
        return
    seen_specific = set()
    seen_relative = set()
    for source, target in episode["transitions"]:
        source, target = int(source), int(target)
        pair = (source, target)
        if pair not in seen_specific:
            seen_specific.add(pair)
            for _ in range(REPLAY_REPEATS):
                transition_route(brain, source, target, learn=True, relative=False)
        if include_relative:
            delta = delta_of(source, target)
            if delta not in seen_relative:
                seen_relative.add(delta)
                for _ in range(REPLAY_REPEATS):
                    relative_route(brain, delta, learn=True)


def decay_route(brain: SphereBrain, route: dict, factor: float) -> int:
    changed = 0
    for a, b in [tuple(x) for x in route["edges"]]:
        before = float(brain.weights[a, b])
        after = max(0.0, min(1.0, before * factor))
        brain.weights[a, b] = after
        brain.weights[b, a] = after
        changed += 1
    return changed


def route_weight(brain: SphereBrain, source: int, target: int) -> float:
    return float(transition_route(brain, source, target, learn=False, relative=False)["mean_weight"])


def relative_weight(brain: SphereBrain, delta: tuple[int, int]) -> float:
    return float(relative_route(brain, delta, learn=False)["mean_weight"])


def pretrain(base: SphereBrain) -> SphereBrain:
    brain = copy.deepcopy(base)
    rng = random.Random(SEED)
    for _ in range(PRETRAIN_EPISODES):
        ep = run_episode(brain, BASE, relative_mode=False, assist=False, rng=rng, explore=EXPLORATION, max_steps=TRAIN_MAX_STEPS)
        observe_episode(brain, ep, include_relative=True)
        reinforce_success(brain, ep, include_relative=True)
    return brain


def adapt_branch(pretrained: SphereBrain, mode: str) -> dict:
    brain = copy.deepcopy(pretrained)
    adaptive_specific = mode in {"specific_adaptive", "relative_credit", "relative_plus_assist"}
    relative_mode = mode in {"relative_credit", "relative_plus_assist"}
    assist = mode == "relative_plus_assist"
    rng = random.Random(SEED + {"success_only": 10, "specific_adaptive": 20, "relative_credit": 30, "relative_plus_assist": 30}[mode])
    fail_streak: dict[tuple[int, int], int] = defaultdict(int)
    relative_failure_positions: dict[tuple[int, int], set[int]] = defaultdict(set)
    specific_events = 0
    relative_events = 0
    assist_eligible = assist_acted = assist_top = 0
    checkpoints = {"0": run_episode(brain, CHANGED, relative_mode=relative_mode, assist=assist)}
    old_pair = (0, 1)
    alt_pair = (0, 3)
    old_before = route_weight(brain, *old_pair)
    alt_before = route_weight(brain, *alt_pair)
    horizontal_before = relative_weight(brain, (0, 1))
    vertical_before = relative_weight(brain, (1, 0))

    for episode_index in range(1, ADAPT_EPISODES + 1):
        ep = run_episode(brain, CHANGED, relative_mode=relative_mode, assist=assist, rng=rng, explore=EXPLORATION, max_steps=TRAIN_MAX_STEPS)
        observe_episode(brain, ep, include_relative=relative_mode)
        assist_eligible += int(ep["assist_eligible_steps"])
        assist_acted += int(ep["assist_acted_steps"])
        assist_top += int(ep["assist_top_changes"])
        unique_pairs = []
        seen = set()
        for source, target in ep["transitions"]:
            pair = (int(source), int(target))
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        if ep["success"]:
            reinforce_success(brain, ep, include_relative=relative_mode)
            for pair in unique_pairs:
                fail_streak[pair] = 0
        else:
            for source, target in unique_pairs:
                pair = (source, target)
                fail_streak[pair] += 1
                delta = delta_of(source, target)
                relative_failure_positions[delta].add(source)
                if adaptive_specific and fail_streak[pair] >= FAIL_STREAK_GATE:
                    failure_evidence = 1.0 - evidence_probability(brain, specific_token(source, target))
                    if failure_evidence >= MIN_FAILURE_EVIDENCE:
                        factor = DRIFT_DECAY if brain.experience_state.drift_suspected else MILD_DECAY
                        specific_events += decay_route(brain, transition_route(brain, source, target, learn=False, relative=False), factor) > 0
                if relative_mode and len(relative_failure_positions[delta]) >= RELATIVE_FAILURE_POSITIONS:
                    failure_evidence = 1.0 - evidence_probability(brain, relative_token(delta))
                    if failure_evidence >= MIN_FAILURE_EVIDENCE:
                        relative_events += decay_route(brain, relative_route(brain, delta, learn=False), RELATIVE_DECAY) > 0
                        relative_failure_positions[delta].clear()

        if episode_index in CHECKPOINTS:
            checkpoints[str(episode_index)] = run_episode(brain, CHANGED, relative_mode=relative_mode, assist=assist)

    return {
        "mode": mode,
        "checkpoints": checkpoints,
        "specific_credit_events": int(specific_events),
        "relative_credit_events": int(relative_events),
        "assist_eligible": int(assist_eligible),
        "assist_acted": int(assist_acted),
        "assist_top_changes": int(assist_top),
        "old_specific_weight_delta": route_weight(brain, *old_pair) - old_before,
        "new_specific_weight_delta": route_weight(brain, *alt_pair) - alt_before,
        "horizontal_relative_weight_delta": relative_weight(brain, (0, 1)) - horizontal_before,
        "vertical_relative_weight_delta": relative_weight(brain, (1, 0)) - vertical_before,
        "confidence": float(brain.experience_state.confidence),
        "drift": bool(brain.experience_state.drift_suspected),
    }


def metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = pretrain(base)
    branches = {mode: adapt_branch(pretrained, mode) for mode in MODES}
    finals = {mode: branches[mode]["checkpoints"]["20"] for mode in MODES}
    best_metric = max(metric(ep) for ep in finals.values())
    winners = [mode for mode, ep in finals.items() if metric(ep) == best_metric]

    relative_better_than_specific = metric(finals["relative_credit"]) > metric(finals["specific_adaptive"])
    assist_better_than_relative = metric(finals["relative_plus_assist"]) > metric(finals["relative_credit"])
    relative_recovered = bool(finals["relative_credit"]["success"])
    assist_recovered = bool(finals["relative_plus_assist"]["success"])
    brain_unchanged = before_hash == file_hash(BRAIN_PATH)

    if relative_better_than_specific:
        verdict = "relative_credit_improves_puzzle_recovery_over_specific_credit"
        readiness = "relative_credit_has_behavioral_value"
    elif relative_recovered:
        verdict = "relative_credit_recovers_but_does_not_outperform_specific_credit_under_current_budget"
        readiness = "relative_credit_recovery_observed"
    elif assist_better_than_relative or assist_recovered:
        verdict = "bounded_assist_adds_recovery_value_after_relative_credit"
        readiness = "relative_assist_has_behavioral_value"
    else:
        verdict = "relative_representation_is_valid_but_credit_bridge_still_does_not_escape_old_policy"
        readiness = "credit_bridge_needs_reanalysis"

    payload = {
        "experiment": "Core Growth Binding v70 — Relative Credit Assignment & Puzzle Recovery",
        "contract": {
            "direction_words_given_to_core": False,
            "relative_input": "numeric_delta_only",
            "relative_credit_cannot_choose_alone": True,
            "relative_failure_requires_multiple_concrete_positions": True,
            "assist_only_after_relative_credit": True,
            "production_brain_json_saved": False,
        },
        "branches": branches,
        "summary": {
            "winner": "tie:" + ",".join(winners) if len(winners) > 1 else winners[0],
            "success_only_20": finals["success_only"],
            "specific_adaptive_20": finals["specific_adaptive"],
            "relative_credit_20": finals["relative_credit"],
            "relative_plus_assist_20": finals["relative_plus_assist"],
            "relative_better_than_specific": relative_better_than_specific,
            "relative_recovered": relative_recovered,
            "assist_better_than_relative": assist_better_than_relative,
            "assist_recovered": assist_recovered,
            "assist_eligible": branches["relative_plus_assist"]["assist_eligible"],
            "assist_acted": branches["relative_plus_assist"]["assist_acted"],
            "assist_top_changes": branches["relative_plus_assist"]["assist_top_changes"],
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": "audit_relative_credit_contribution_and_local_context_if_recovery_is_still_incomplete",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v70.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v70</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v70：Relative Credit Assignment & Puzzle Recovery</h1><p class="lead">Success-only / Specific Adaptive / Relative Credit / Relative Credit + Bounded Assist を同じ変更盤面・同じ経験予算で比較する。Relativeは数値Δのみで、単独では勝者を決めない。</p><section class="panel"><div class="controls"><button id="run">Relative Creditを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">未実行</pre></section><script>
const b=v=>v?'YES':'NO';const cls=v=>v?'good':'warn';const ep=e=>`${e.success?'E到達':'未到達'} / ${e.steps}手 / ${e.path.join(' → ')}`;function metric(k,v,c=''){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…','blue');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary,br=d.branches;document.getElementById('metrics').innerHTML=[metric('Winner',s.winner,'blue'),metric('Success-only 20',ep(s.success_only_20),cls(s.success_only_20.success)),metric('Specific Adaptive 20',ep(s.specific_adaptive_20),cls(s.specific_adaptive_20.success)),metric('Relative Credit 20',ep(s.relative_credit_20),cls(s.relative_credit_20.success)),metric('Relative+Assist 20',ep(s.relative_plus_assist_20),cls(s.relative_plus_assist_20.success)),metric('Relative > Specific',b(s.relative_better_than_specific),cls(s.relative_better_than_specific)),metric('Specific credit event',br.specific_adaptive.specific_credit_events,'blue'),metric('Relative credit event',br.relative_credit.relative_credit_events,'blue'),metric('旧具体weight Δ',br.relative_credit.old_specific_weight_delta.toFixed(6),br.relative_credit.old_specific_weight_delta<0?'good':'warn'),metric('新具体weight Δ',br.relative_credit.new_specific_weight_delta.toFixed(6),br.relative_credit.new_specific_weight_delta>0?'good':'warn'),metric('横Relative weight Δ',br.relative_credit.horizontal_relative_weight_delta.toFixed(6),'blue'),metric('縦Relative weight Δ',br.relative_credit.vertical_relative_weight_delta.toFixed(6),'blue'),metric('Assist Eligible',s.assist_eligible,'blue'),metric('Assist作動',s.assist_acted,'blue'),metric('Assist Top変更',s.assist_top_changes,'blue'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',cls(s.brain_file_unchanged)),metric('Core readiness',s.core_readiness,'blue'),metric('総合判定',s.overall_verdict,'blue')].join('');document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v70: http://{HOST}:{PORT}")
    print("Relative Credit / numeric delta only / bounded assist comparison / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
