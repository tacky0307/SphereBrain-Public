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
START_PORT = 5118
OUT = ROOT / "data" / "core_growth_binding_v72" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
GRID = 3
SEED = 7201
PRETRAIN_EPISODES = 30
ADAPT_EPISODES = 20
EXPLORATION = 0.35
MAX_STEPS = 16
TRAIN_MAX_STEPS = 24
REPLAY_REPEATS = 2
CHECKPOINTS = {0, 5, 10, 20}

FAIL_STREAK_GATE = 3
MIN_FAILURE_EVIDENCE = 0.55
SPECIFIC_WEIGHT_DECAY = 0.985
SPECIFIC_USAGE_DECAY = 0.78
CONTEXT_WEIGHT_DECAY = 0.988
CONTEXT_USAGE_DECAY = 0.84
RELATIVE_WEIGHT_DECAY = 0.994
RELATIVE_USAGE_DECAY = 0.92
RELATIVE_FAILURE_CONTEXTS = 3

LEGACY_RELATIVE_SHARE = 0.20
CONTEXT_RELATIVE_SHARE = 0.12
CONTEXT_SCORE_SHARE = 0.18

TIE_MARGIN = 0.0025
ASSIST_ABS_CAP = 5e-5
ASSIST_REL_CAP = 0.55
MIN_CONFIDENCE = 0.80

DIRS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}
BASE = {"name": "trained_board", "start": 0, "goal": 8, "blocked": {3, 4}}
CHANGED = {"name": "route_blocked", "start": 0, "goal": 8, "blocked": {2, 4}}
MODES = [
    "success_only",
    "specific_adaptive",
    "legacy_relative",
    "contextual_relative",
    "contextual_plus_assist",
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


def local_delta_signature(node: int, blocked: set[int]) -> tuple[tuple[int, int], ...]:
    sig = [delta_of(node, target) for _, target in legal_moves(node, blocked)]
    return tuple(sorted(sig))


def goal_vector(node: int, goal: int) -> tuple[int, int]:
    nr, nc = rc(node)
    gr, gc = rc(goal)
    return gr - nr, gc - nc


def specific_token(source: int, target: int) -> str:
    return "s_" + hashlib.sha256(f"{source}>{target}".encode()).hexdigest()[:14]


def relative_token(delta: tuple[int, int]) -> str:
    return "r_" + hashlib.sha256(f"{delta[0]},{delta[1]}".encode()).hexdigest()[:14]


def context_token(source: int, target: int, case: dict) -> str:
    blocked = set(case["blocked"])
    goal = int(case["goal"])
    payload = {
        "delta": list(delta_of(source, target)),
        "goal_before": list(goal_vector(source, goal)),
        "goal_after": list(goal_vector(target, goal)),
        "local_before": [list(x) for x in local_delta_signature(source, blocked)],
        "local_after": [list(x) for x in local_delta_signature(target, blocked)],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "c_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def experience_universe() -> list[str]:
    values: set[str] = set(relative_token(d) for d in [(0, 1), (0, -1), (1, 0), (-1, 0)])
    for case in (BASE, CHANGED):
        blocked = set(case["blocked"])
        for node in range(GRID * GRID):
            if node in blocked:
                continue
            for _, target in legal_moves(node, blocked):
                values.add(specific_token(node, target))
                values.add(context_token(node, target, case))
    return sorted(values)


EXPECTED = experience_universe()


def unique_sources(*groups: list[int]) -> list[int]:
    out: list[int] = []
    for group in groups:
        for node in group:
            if int(node) not in out:
                out.append(int(node))
    return out


def route_from_sources(brain: SphereBrain, sources: list[int], *, learn: bool) -> dict:
    result = brain.propagate(sources, steps=8, threshold=0.18, noise=0.0, learn=learn)
    edges = [tuple(x) for x in result.traversed_edges]
    if edges:
        mean_weight = sum(float(brain.weights[a, b]) for a, b in edges) / len(edges)
        mean_usage = sum(float(brain.usage[a, b]) for a, b in edges) / len(edges)
    else:
        mean_weight = 0.0
        mean_usage = 0.0
    usage_bonus = float(0.012 * math.log1p(mean_usage))
    node_bonus = float(0.0005 * len(result.activated_nodes))
    score = float(mean_weight + usage_bonus + node_bonus)
    return {
        "score": score,
        "edges": [list(x) for x in result.traversed_edges],
        "nodes": list(result.activated_nodes),
        "mean_weight": float(mean_weight),
        "mean_usage": float(mean_usage),
        "usage_bonus": usage_bonus,
        "node_bonus": node_bonus,
    }


def token_route(brain: SphereBrain, token: str, *, learn: bool) -> dict:
    return route_from_sources(brain, brain.text_to_sources(token, count=3), learn=learn)


def specific_route(brain: SphereBrain, source: int, target: int, *, learn: bool) -> dict:
    return token_route(brain, specific_token(source, target), learn=learn)


def relative_route(brain: SphereBrain, source: int, target: int, *, learn: bool) -> dict:
    return token_route(brain, relative_token(delta_of(source, target)), learn=learn)


def contextual_route(brain: SphereBrain, source: int, target: int, case: dict, *, learn: bool) -> dict:
    return token_route(brain, context_token(source, target, case), learn=learn)


def evidence_probability(brain: SphereBrain, token: str) -> float:
    row = brain.experience_state.condition_evidence.get(token, {})
    t = float(row.get("true_weight", 0.0))
    f = float(row.get("false_weight", 0.0))
    total = t + f
    return 0.5 if total <= 1e-12 else t / total


def candidate_rows(brain: SphereBrain, current: int, case: dict, mode: str) -> list[dict]:
    blocked = set(case["blocked"])
    rows = []
    for label, target in legal_moves(current, blocked):
        spec = specific_route(brain, current, target, learn=False)
        rel = relative_route(brain, current, target, learn=False)
        ctx = contextual_route(brain, current, target, case, learn=False)
        score = float(spec["score"])
        if mode == "legacy_relative":
            score += LEGACY_RELATIVE_SHARE * float(rel["score"])
        elif mode in {"contextual_relative", "contextual_plus_assist"}:
            score += CONTEXT_RELATIVE_SHARE * float(rel["score"])
            score += CONTEXT_SCORE_SHARE * float(ctx["score"])
        rows.append({
            "label": label,
            "source": current,
            "target": target,
            "delta": list(delta_of(current, target)),
            "specific_score": float(spec["score"]),
            "specific_weight": float(spec["mean_weight"]),
            "specific_usage": float(spec["mean_usage"]),
            "specific_usage_bonus": float(spec["usage_bonus"]),
            "relative_score": float(rel["score"]),
            "context_score": float(ctx["score"]),
            "score": float(score),
            "specific_evidence": evidence_probability(brain, specific_token(current, target)),
            "relative_evidence": evidence_probability(brain, relative_token(delta_of(current, target))),
            "context_evidence": evidence_probability(brain, context_token(current, target, case)),
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
    prefs = [
        0.55 * float(x["specific_evidence"])
        + 0.15 * float(x["relative_evidence"])
        + 0.30 * float(x["context_evidence"])
        for x in ranked
    ]
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


def choose_move(brain: SphereBrain, current: int, case: dict, *, mode: str, rng: random.Random | None = None, explore: float = 0.0):
    ranked = candidate_rows(brain, current, case, mode)
    trace = {"eligible": False, "acted": False, "top_changed": False}
    if mode == "contextual_plus_assist":
        ranked, trace = assist_rank(brain, ranked)
    if not ranked:
        raise RuntimeError("移動候補がありません。")
    if rng is not None and len(ranked) > 1 and rng.random() < explore:
        chosen = rng.choice(ranked)
    else:
        chosen = ranked[0]
    return str(chosen["label"]), int(chosen["target"]), ranked, trace


def run_episode(brain: SphereBrain, case: dict, *, mode: str, rng: random.Random | None = None, explore: float = 0.0, max_steps: int = MAX_STEPS) -> dict:
    current = int(case["start"])
    goal = int(case["goal"])
    path = [current]
    transitions = []
    traces = []
    choice_rows = []
    for step in range(max_steps):
        if current == goal:
            break
        label, target, ranked, trace = choose_move(brain, current, case, mode=mode, rng=rng, explore=explore)
        transitions.append((current, target))
        traces.append({"step": step + 1, "from": current, "to": target, "action": label, **trace})
        choice_rows.append({"step": step + 1, "from": current, "candidates": ranked})
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
        "choice_rows": choice_rows,
    }


def observe_episode(brain: SphereBrain, episode: dict, case: dict, mode: str) -> None:
    success = bool(episode["success"])
    seen: set[str] = set()
    for source, target in episode["transitions"]:
        source, target = int(source), int(target)
        tokens = [specific_token(source, target)]
        if mode in {"legacy_relative", "contextual_relative", "contextual_plus_assist"}:
            tokens.append(relative_token(delta_of(source, target)))
        if mode in {"contextual_relative", "contextual_plus_assist"}:
            tokens.append(context_token(source, target, case))
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            brain.experience_state.observe(condition=token, present=success, motif="m72", expected_conditions=EXPECTED)


def reinforce_success(brain: SphereBrain, episode: dict, case: dict, mode: str) -> None:
    if not episode["success"]:
        return
    seen_specific: set[tuple[int, int]] = set()
    seen_relative: set[tuple[int, int]] = set()
    seen_context: set[str] = set()
    for source, target in episode["transitions"]:
        source, target = int(source), int(target)
        pair = (source, target)
        if pair not in seen_specific:
            seen_specific.add(pair)
            for _ in range(REPLAY_REPEATS):
                specific_route(brain, source, target, learn=True)
        if mode in {"legacy_relative", "contextual_relative", "contextual_plus_assist"}:
            delta = delta_of(source, target)
            if delta not in seen_relative:
                seen_relative.add(delta)
                for _ in range(REPLAY_REPEATS):
                    relative_route(brain, source, target, learn=True)
        if mode in {"contextual_relative", "contextual_plus_assist"}:
            token = context_token(source, target, case)
            if token not in seen_context:
                seen_context.add(token)
                for _ in range(REPLAY_REPEATS):
                    contextual_route(brain, source, target, case, learn=True)


def decay_route(brain: SphereBrain, route: dict, *, weight_factor: float, usage_factor: float) -> dict:
    changed = []
    for a, b in [tuple(x) for x in route["edges"]]:
        wb = float(brain.weights[a, b])
        wa = max(0.0, min(1.0, wb * weight_factor))
        ub = float(brain.usage[a, b])
        ua = max(0, int(round(ub * usage_factor)))
        brain.weights[a, b] = wa
        brain.weights[b, a] = wa
        brain.usage[a, b] = ua
        brain.usage[b, a] = ua
        changed.append({"edge": [int(a), int(b)], "weight_before": wb, "weight_after": wa, "usage_before": ub, "usage_after": ua})
    return {"changed_edges": changed, "weight_factor": weight_factor, "usage_factor": usage_factor}


def route_snapshot(route: dict) -> dict:
    return {"weight": float(route["mean_weight"]), "usage": float(route["mean_usage"]), "score": float(route["score"]), "usage_bonus": float(route["usage_bonus"])}


def pretrain(base: SphereBrain) -> SphereBrain:
    brain = copy.deepcopy(base)
    rng = random.Random(SEED)
    # Pretraining always gives the Core the richer representation, but choice is concrete-only.
    for _ in range(PRETRAIN_EPISODES):
        ep = run_episode(brain, BASE, mode="success_only", rng=rng, explore=EXPLORATION, max_steps=TRAIN_MAX_STEPS)
        observe_episode(brain, ep, BASE, "contextual_relative")
        if ep["success"]:
            reinforce_success(brain, ep, BASE, "contextual_relative")
    return brain


def adapt_branch(pretrained: SphereBrain, mode: str) -> dict:
    brain = copy.deepcopy(pretrained)
    rng = random.Random(SEED + {m: (i + 1) * 100 for i, m in enumerate(MODES)}[mode])
    fail_streak: dict[tuple[int, int], int] = defaultdict(int)
    context_fail_streak: dict[str, int] = defaultdict(int)
    relative_failure_contexts: dict[tuple[int, int], set[str]] = defaultdict(set)
    specific_events = context_events = relative_events = 0
    assist_eligible = assist_acted = assist_top = 0

    checkpoints = {"0": run_episode(brain, CHANGED, mode=mode)}
    old_pair = (0, 1)
    new_pair = (0, 3)
    old_before = route_snapshot(specific_route(brain, *old_pair, learn=False))
    new_before = route_snapshot(specific_route(brain, *new_pair, learn=False))

    timeline = []
    for episode_index in range(1, ADAPT_EPISODES + 1):
        ep = run_episode(brain, CHANGED, mode=mode, rng=rng, explore=EXPLORATION, max_steps=TRAIN_MAX_STEPS)
        observe_episode(brain, ep, CHANGED, mode)
        assist_eligible += int(ep.get("assist_eligible_steps", 0))
        assist_acted += int(ep.get("assist_acted_steps", 0))
        assist_top += int(ep.get("assist_top_changes", 0))

        unique_pairs: list[tuple[int, int]] = []
        seen_pairs = set()
        for source, target in ep["transitions"]:
            pair = (int(source), int(target))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_pairs.append(pair)

        if ep["success"]:
            reinforce_success(brain, ep, CHANGED, mode)
            for pair in unique_pairs:
                fail_streak[pair] = 0
                context_fail_streak[context_token(pair[0], pair[1], CHANGED)] = 0
        else:
            for pair in unique_pairs:
                source, target = pair
                fail_streak[pair] += 1
                ctx_token = context_token(source, target, CHANGED)
                context_fail_streak[ctx_token] += 1
                relative_failure_contexts[delta_of(source, target)].add(ctx_token)

                if mode in {"specific_adaptive", "legacy_relative", "contextual_relative", "contextual_plus_assist"} and fail_streak[pair] >= FAIL_STREAK_GATE:
                    failure_evidence = 1.0 - evidence_probability(brain, specific_token(source, target))
                    if failure_evidence >= MIN_FAILURE_EVIDENCE:
                        result = decay_route(
                            brain,
                            specific_route(brain, source, target, learn=False),
                            weight_factor=SPECIFIC_WEIGHT_DECAY,
                            usage_factor=SPECIFIC_USAGE_DECAY if mode in {"contextual_relative", "contextual_plus_assist"} else 1.0,
                        )
                        if result["changed_edges"]:
                            specific_events += 1

                if mode in {"contextual_relative", "contextual_plus_assist"} and context_fail_streak[ctx_token] >= FAIL_STREAK_GATE:
                    failure_evidence = 1.0 - evidence_probability(brain, ctx_token)
                    if failure_evidence >= MIN_FAILURE_EVIDENCE:
                        result = decay_route(
                            brain,
                            contextual_route(brain, source, target, CHANGED, learn=False),
                            weight_factor=CONTEXT_WEIGHT_DECAY,
                            usage_factor=CONTEXT_USAGE_DECAY,
                        )
                        if result["changed_edges"]:
                            context_events += 1

                if mode in {"legacy_relative", "contextual_relative", "contextual_plus_assist"}:
                    delta = delta_of(source, target)
                    broad_gate = len(relative_failure_contexts[delta]) >= (2 if mode == "legacy_relative" else RELATIVE_FAILURE_CONTEXTS)
                    if broad_gate:
                        failure_evidence = 1.0 - evidence_probability(brain, relative_token(delta))
                        if failure_evidence >= MIN_FAILURE_EVIDENCE:
                            result = decay_route(
                                brain,
                                relative_route(brain, source, target, learn=False),
                                weight_factor=0.988 if mode == "legacy_relative" else RELATIVE_WEIGHT_DECAY,
                                usage_factor=1.0 if mode == "legacy_relative" else RELATIVE_USAGE_DECAY,
                            )
                            if result["changed_edges"]:
                                relative_events += 1

        timeline.append({
            "episode": episode_index,
            "success": bool(ep["success"]),
            "steps": int(ep["steps"]),
            "loop_steps": int(ep["loop_steps"]),
            "confidence": float(brain.experience_state.confidence),
            "drift": bool(brain.experience_state.drift_suspected),
            "specific_events": specific_events,
            "context_events": context_events,
            "relative_events": relative_events,
        })
        if episode_index in CHECKPOINTS:
            checkpoints[str(episode_index)] = run_episode(brain, CHANGED, mode=mode)

    old_after = route_snapshot(specific_route(brain, *old_pair, learn=False))
    new_after = route_snapshot(specific_route(brain, *new_pair, learn=False))

    return {
        "mode": mode,
        "checkpoints": checkpoints,
        "timeline": timeline,
        "specific_credit_events": specific_events,
        "context_credit_events": context_events,
        "relative_credit_events": relative_events,
        "assist_eligible": assist_eligible,
        "assist_acted": assist_acted,
        "assist_top_changes": assist_top,
        "old_specific_before": old_before,
        "old_specific_after": old_after,
        "new_specific_before": new_before,
        "new_specific_after": new_after,
        "experience_state": brain.snapshot_experience_state(),
    }


def episode_metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = pretrain(base)

    branches = {mode: adapt_branch(pretrained, mode) for mode in MODES}
    finals = {mode: branches[mode]["checkpoints"]["20"] for mode in MODES}
    metrics = {mode: episode_metric(finals[mode]) for mode in MODES}
    best = max(metrics.values())
    winners = [mode for mode, metric in metrics.items() if metric == best]

    contextual_better_than_legacy = metrics["contextual_relative"] > metrics["legacy_relative"]
    assist_better_than_contextual = metrics["contextual_plus_assist"] > metrics["contextual_relative"]
    contextual_recovered = bool(finals["contextual_relative"]["success"])
    assist_recovered = bool(finals["contextual_plus_assist"]["success"])

    old = branches["contextual_relative"]
    old_weight_delta = old["old_specific_after"]["weight"] - old["old_specific_before"]["weight"]
    old_usage_delta = old["old_specific_after"]["usage"] - old["old_specific_before"]["usage"]
    new_weight_delta = old["new_specific_after"]["weight"] - old["new_specific_before"]["weight"]
    new_usage_delta = old["new_specific_after"]["usage"] - old["new_specific_before"]["usage"]

    if contextual_recovered:
        verdict = "contextual_relative_credit_and_usage_adaptation_escape_old_policy"
        readiness = "contextual_credit_behavior_bridge_candidate"
        next_step = "stress_test_contextual_credit_generalization_and_then_measure_assist_incremental_value"
    elif assist_recovered:
        verdict = "contextual_credit_needs_bounded_assist_to_escape_old_policy"
        readiness = "assist_adds_behavioral_value"
        next_step = "audit_assist_changed_boundaries_and_generalize_across_multiple_puzzles"
    elif contextual_better_than_legacy:
        verdict = "context_and_usage_adaptation_improve_over_v70_but_do_not_yet_recover"
        readiness = "contextual_credit_partial"
        next_step = "audit_remaining_choice_score_inertia_and_credit_timing"
    else:
        verdict = "contextual_credit_and_usage_adaptation_still_do_not_escape_old_policy"
        readiness = "behavior_bridge_needs_reanalysis"
        next_step = "inspect_checkpoint_candidate_score_components_before_more_learning_strength"

    payload = {
        "experiment": "Core Growth Binding v72 — Contextual Relative Credit & Usage Adaptation",
        "contract": {
            "direction_words_given_to_core": False,
            "absolute_position_labels_saved": False,
            "context_uses_numeric_delta": True,
            "context_uses_goal_relative_vectors_without_reward": True,
            "context_uses_local_topology": True,
            "specific_weight_adaptation": True,
            "specific_usage_adaptation": True,
            "contextual_credit": True,
            "pure_relative_decay_requires_multiple_contexts": True,
            "assist_only_in_final_comparison_branch": True,
            "production_brain_json_saved": False,
        },
        "branches": branches,
        "summary": {
            "winner": "tie:" + ",".join(winners) if len(winners) > 1 else winners[0],
            "success_only_20": finals["success_only"],
            "specific_adaptive_20": finals["specific_adaptive"],
            "legacy_relative_20": finals["legacy_relative"],
            "contextual_relative_20": finals["contextual_relative"],
            "contextual_plus_assist_20": finals["contextual_plus_assist"],
            "contextual_better_than_legacy": contextual_better_than_legacy,
            "assist_better_than_contextual": assist_better_than_contextual,
            "contextual_recovered": contextual_recovered,
            "assist_recovered": assist_recovered,
            "specific_credit_events": old["specific_credit_events"],
            "context_credit_events": old["context_credit_events"],
            "relative_credit_events": old["relative_credit_events"],
            "old_specific_weight_delta": old_weight_delta,
            "old_specific_usage_delta": old_usage_delta,
            "new_specific_weight_delta": new_weight_delta,
            "new_specific_usage_delta": new_usage_delta,
            "assist_eligible": branches["contextual_plus_assist"]["assist_eligible"],
            "assist_acted": branches["contextual_plus_assist"]["assist_acted"],
            "assist_top_changes": branches["contextual_plus_assist"]["assist_top_changes"],
            "brain_file_unchanged": before_hash == file_hash(BRAIN_PATH),
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v72.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v72</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}
</style></head><body><main><h1>v72：Contextual Relative Credit & Usage Adaptation</h1><p class="lead">v70型Relative Creditを対照に残し、Specific + Relative + Contextの階層Creditとusage適応を比較する。Contextは数値Δ・Goal相対ベクトル・局所接続形のみ。方向語や正解方向は与えない。</p><section class="panel"><div class="controls"><button id="run">Contextual Creditを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">未実行</pre></section><script>
const b=v=>v?'YES':'NO';const cls=v=>v?'good':'warn';const ep=e=>(e.success?'E到達':'未到達')+' / '+e.steps+'手 / '+e.path.join(' → ');function metric(k,v,c=''){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…','blue');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Winner',s.winner,'blue'),metric('Success-only 20',ep(s.success_only_20),cls(s.success_only_20.success)),metric('Specific Adaptive 20',ep(s.specific_adaptive_20),cls(s.specific_adaptive_20.success)),metric('v70 Relative 20',ep(s.legacy_relative_20),cls(s.legacy_relative_20.success)),metric('Contextual Relative 20',ep(s.contextual_relative_20),cls(s.contextual_relative_20.success)),metric('Contextual+Assist 20',ep(s.contextual_plus_assist_20),cls(s.contextual_plus_assist_20.success)),metric('Context > v70',b(s.contextual_better_than_legacy),cls(s.contextual_better_than_legacy)),metric('Assist > Context',b(s.assist_better_than_contextual),cls(s.assist_better_than_contextual)),metric('Specific credit event',s.specific_credit_events,'blue'),metric('Context credit event',s.context_credit_events,'blue'),metric('Relative credit event',s.relative_credit_events,'blue'),metric('旧specific weight Δ',s.old_specific_weight_delta.toFixed(6),s.old_specific_weight_delta<0?'good':'warn'),metric('旧specific usage Δ',s.old_specific_usage_delta.toFixed(3),s.old_specific_usage_delta<0?'good':'warn'),metric('新specific weight Δ',s.new_specific_weight_delta.toFixed(6),s.new_specific_weight_delta>0?'good':'warn'),metric('新specific usage Δ',s.new_specific_usage_delta.toFixed(3),s.new_specific_usage_delta>0?'good':'warn'),metric('Assist Eligible',s.assist_eligible,'blue'),metric('Assist作動',s.assist_acted,'blue'),metric('Assist Top変更',s.assist_top_changes,'blue'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',cls(s.brain_file_unchanged)),metric('Core readiness',s.core_readiness,'blue'),metric('総合判定',s.overall_verdict,'blue')].join('');document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v72: http://{HOST}:{PORT}")
    print("Contextual Relative Credit / usage adaptation / v70 legacy control / bounded assist comparison")
    serve(app, host=HOST, port=PORT)
