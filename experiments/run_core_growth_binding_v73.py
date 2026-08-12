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

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v70 as v70

HOST = "127.0.0.1"
START_PORT = 5119
OUT = ROOT / "data" / "core_growth_binding_v73" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEED = 7301
ADAPT_EPISODES = 20
CHECKPOINTS = {0, 5, 10, 20}
LOOP_GATE = 2
RETURN_WEIGHT_DECAY = 0.965
RETURN_USAGE_DECAY = 0.72
ENTRY_WEIGHT_DECAY = 0.997
ENTRY_USAGE_DECAY = 0.97
MIN_FAILURE_EVIDENCE = 0.55
MODES = ["legacy_relative", "temporal_credit", "temporal_plus_assist"]


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


def decay_specific_credit(brain: SphereBrain, source: int, target: int, *, weight_factor: float, usage_factor: float) -> dict:
    route = v70.transition_route(brain, source, target, learn=False, relative=False)
    changed = []
    for a, b in [tuple(x) for x in route["edges"]]:
        before_w = float(brain.weights[a, b])
        before_u = float(brain.usage[a, b])
        after_w = max(0.0, min(1.0, before_w * weight_factor))
        after_u = max(0.0, before_u * usage_factor)
        brain.weights[a, b] = after_w
        brain.weights[b, a] = after_w
        brain.usage[a, b] = after_u
        brain.usage[b, a] = after_u
        changed.append({
            "edge": [int(a), int(b)],
            "weight_before": before_w,
            "weight_after": after_w,
            "usage_before": before_u,
            "usage_after": after_u,
        })
    return {
        "source": int(source),
        "target": int(target),
        "weight_factor": float(weight_factor),
        "usage_factor": float(usage_factor),
        "changed_edges": changed,
    }


def route_usage(brain: SphereBrain, source: int, target: int) -> float:
    route = v70.transition_route(brain, source, target, learn=False, relative=False)
    edges = [tuple(x) for x in route["edges"]]
    if not edges:
        return 0.0
    return sum(float(brain.usage[a, b]) for a, b in edges) / len(edges)


def immediate_return_motifs(episode: dict) -> list[dict]:
    path = [int(x) for x in episode.get("path", [])]
    motifs = []
    for i in range(len(path) - 2):
        a, b, c = path[i], path[i + 1], path[i + 2]
        if a != b and a == c:
            motifs.append({
                "start_index": i,
                "nodes": [a, b, c],
                "entry": [a, b],
                "return": [b, a],
                "key": f"{a}>{b}>{a}",
            })
    return motifs


def longer_repeat_fragments(episode: dict, max_len: int = 4) -> list[dict]:
    path = [int(x) for x in episode.get("path", [])]
    rows = []
    for span in range(3, max_len + 1):
        for i in range(len(path) - span):
            frag = path[i:i + span + 1]
            if frag[0] == frag[-1] and len(set(frag[:-1])) > 1:
                rows.append({"nodes": frag, "span": span, "key": ">".join(map(str, frag))})
    return rows


def temporal_attribution(
    brain: SphereBrain,
    episode: dict,
    motif_counts: dict[str, int],
    *,
    episode_index: int,
) -> list[dict]:
    events = []
    for motif in immediate_return_motifs(episode):
        motif_counts[motif["key"]] += 1
        count = motif_counts[motif["key"]]
        source, target = motif["return"]
        failure_evidence = 1.0 - v70.evidence_probability(brain, v70.specific_token(source, target))
        event = {
            "episode": episode_index,
            "motif": motif["nodes"],
            "motif_count": count,
            "entry": motif["entry"],
            "return": motif["return"],
            "failure_evidence": failure_evidence,
            "credited": False,
        }
        if count >= LOOP_GATE and failure_evidence >= MIN_FAILURE_EVIDENCE:
            # Attribute most negative credit to the transition that completes the return.
            event["return_decay"] = decay_specific_credit(
                brain,
                source,
                target,
                weight_factor=RETURN_WEIGHT_DECAY,
                usage_factor=RETURN_USAGE_DECAY,
            )
            # Preserve the loop-entry transition almost entirely. A tiny decay prevents immortal stale paths.
            es, et = motif["entry"]
            event["entry_decay"] = decay_specific_credit(
                brain,
                es,
                et,
                weight_factor=ENTRY_WEIGHT_DECAY,
                usage_factor=ENTRY_USAGE_DECAY,
            )
            event["credited"] = True
        events.append(event)
    return events


def adapt_branch(pretrained: SphereBrain, mode: str) -> dict:
    brain = copy.deepcopy(pretrained)
    temporal = mode in {"temporal_credit", "temporal_plus_assist"}
    assist = mode == "temporal_plus_assist"
    rng = random.Random(SEED + {"legacy_relative": 10, "temporal_credit": 20, "temporal_plus_assist": 20}[mode])
    motif_counts: dict[str, int] = defaultdict(int)
    temporal_events = []
    longer_fragments = defaultdict(int)
    assist_eligible = assist_acted = assist_top = 0

    checkpoints = {
        "0": v70.run_episode(brain, v70.CHANGED, relative_mode=True, assist=assist)
    }

    watched = [(0, 1), (1, 0), (0, 3), (3, 0), (3, 6), (6, 7), (7, 8)]
    before = {
        f"{a}>{b}": {
            "weight": v70.route_weight(brain, a, b),
            "usage": route_usage(brain, a, b),
        }
        for a, b in watched
    }

    successes = 0
    timeline = []
    for episode_index in range(1, ADAPT_EPISODES + 1):
        ep = v70.run_episode(
            brain,
            v70.CHANGED,
            relative_mode=True,
            assist=assist,
            rng=rng,
            explore=v70.EXPLORATION,
            max_steps=v70.TRAIN_MAX_STEPS,
        )
        v70.observe_episode(brain, ep, include_relative=True)
        assist_eligible += int(ep.get("assist_eligible_steps", 0))
        assist_acted += int(ep.get("assist_acted_steps", 0))
        assist_top += int(ep.get("assist_top_changes", 0))

        for row in longer_repeat_fragments(ep):
            longer_fragments[row["key"]] += 1

        events = []
        if ep["success"]:
            successes += 1
            v70.reinforce_success(brain, ep, include_relative=True)
        elif temporal:
            events = temporal_attribution(brain, ep, motif_counts, episode_index=episode_index)
            temporal_events.extend(events)

        timeline.append({
            "episode": episode_index,
            "success": bool(ep["success"]),
            "steps": int(ep["steps"]),
            "path": ep["path"],
            "loop_steps": int(ep["loop_steps"]),
            "return_motifs": len(immediate_return_motifs(ep)),
            "credited_loop_events": sum(1 for x in events if x.get("credited")),
            "confidence": float(brain.experience_state.confidence),
            "drift": bool(brain.experience_state.drift_suspected),
        })

        if episode_index in CHECKPOINTS:
            checkpoints[str(episode_index)] = v70.run_episode(
                brain,
                v70.CHANGED,
                relative_mode=True,
                assist=assist,
            )

    after = {
        f"{a}>{b}": {
            "weight": v70.route_weight(brain, a, b),
            "usage": route_usage(brain, a, b),
        }
        for a, b in watched
    }
    deltas = {
        key: {
            "weight_delta": after[key]["weight"] - before[key]["weight"],
            "usage_delta": after[key]["usage"] - before[key]["usage"],
        }
        for key in before
    }

    return {
        "mode": mode,
        "checkpoints": checkpoints,
        "successes": successes,
        "temporal_event_count": len(temporal_events),
        "credited_loop_event_count": sum(1 for x in temporal_events if x.get("credited")),
        "unique_loop_motifs": len(motif_counts),
        "motif_counts": dict(sorted(motif_counts.items(), key=lambda x: (-x[1], x[0]))),
        "top_longer_fragments": sorted(
            ({"fragment": k, "count": v} for k, v in longer_fragments.items()),
            key=lambda x: (-x["count"], x["fragment"]),
        )[:10],
        "watched_credit_deltas": deltas,
        "assist_eligible": assist_eligible,
        "assist_acted": assist_acted,
        "assist_top_changes": assist_top,
        "timeline": timeline,
        "temporal_events": temporal_events,
    }


def metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = v70.pretrain(base)

    branches = {mode: adapt_branch(pretrained, mode) for mode in MODES}
    finals = {mode: branches[mode]["checkpoints"]["20"] for mode in MODES}
    best_metric = max(metric(x) for x in finals.values())
    winners = [mode for mode, x in finals.items() if metric(x) == best_metric]

    temporal_better = metric(finals["temporal_credit"]) > metric(finals["legacy_relative"])
    assist_better = metric(finals["temporal_plus_assist"]) > metric(finals["temporal_credit"])
    loop_escape = bool(finals["temporal_credit"]["success"])
    brain_unchanged = before_hash == file_hash(BRAIN_PATH)

    if temporal_better:
        verdict = "temporal_loop_attribution_improves_recovery_over_episode_level_relative_credit"
        readiness = "temporal_credit_has_behavioral_value"
        next_step = "stress_test_temporal_credit_across_multiple_loop_shapes_and_environment_changes"
    elif loop_escape:
        verdict = "temporal_credit_recovers_but_does_not_outperform_legacy_relative_under_current_seed"
        readiness = "temporal_credit_recovery_observed"
        next_step = "repeat_across_seeds_and_loop_shapes_before_core_native_integration"
    elif assist_better:
        verdict = "bounded_assist_adds_value_after_temporal_credit"
        readiness = "temporal_assist_candidate"
        next_step = "audit_assist_activated_boundaries_after_loop_attribution"
    else:
        verdict = "loop_attribution_changes_credit_but_does_not_escape_policy_under_current_budget"
        readiness = "temporal_credit_needs_reanalysis"
        next_step = "inspect_return_edge_vs_entry_edge_score_components_before_more_credit_rules"

    payload = {
        "experiment": "Core Growth Binding v73 — Temporal Credit Assignment / Loop Attribution",
        "contract": {
            "base_behavior": "v70_relative",
            "episode_failure_blanket_penalty": False,
            "immediate_return_motif_detected": True,
            "return_completing_transition_receives_primary_negative_credit": True,
            "loop_entry_transition_preserved": True,
            "direction_words_given_to_core": False,
            "goal_proximity_used_as_reward": False,
            "assist_compared_but_not_required": True,
            "production_brain_json_saved": False,
        },
        "branches": branches,
        "summary": {
            "winner": "tie:" + ",".join(winners) if len(winners) > 1 else winners[0],
            "legacy_relative_20": finals["legacy_relative"],
            "temporal_credit_20": finals["temporal_credit"],
            "temporal_plus_assist_20": finals["temporal_plus_assist"],
            "temporal_better_than_legacy": temporal_better,
            "temporal_loop_escape": loop_escape,
            "assist_better_than_temporal": assist_better,
            "credited_loop_events": branches["temporal_credit"]["credited_loop_event_count"],
            "unique_loop_motifs": branches["temporal_credit"]["unique_loop_motifs"],
            "assist_eligible": branches["temporal_plus_assist"]["assist_eligible"],
            "assist_acted": branches["temporal_plus_assist"]["assist_acted"],
            "assist_top_changes": branches["temporal_plus_assist"]["assist_top_changes"],
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v73.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v73</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v73：Temporal Credit Assignment / Loop Attribution</h1><p class="lead">v70 Relativeを基準に、失敗Episode全体ではなくA→B→Aの短期loop motifへCreditを帰属する。loopを完成させた戻り遷移を主に弱め、入口遷移は保持する。Assistも比較する。</p><section class="panel"><div class="controls"><button id="run">Temporal Creditを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Temporal Credit 生データ</h2><pre id="detail" class="raw">未実行</pre></section><section class="panel"><h2>全データ</h2><pre id="raw" class="raw">未実行</pre></section><script>
const b=v=>v?'YES':'NO';const cls=v=>v?'good':'warn';function metric(k,v,c=''){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function ep(x){return `${x.success?'E到達':'未到達'} / ${x.steps}手 / ${x.path.join(' → ')}`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…','blue');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Winner',s.winner,'blue'),metric('v70 Relative 20',ep(s.legacy_relative_20),s.legacy_relative_20.success?'good':'warn'),metric('Temporal Credit 20',ep(s.temporal_credit_20),s.temporal_credit_20.success?'good':'warn'),metric('Temporal+Assist 20',ep(s.temporal_plus_assist_20),s.temporal_plus_assist_20.success?'good':'warn'),metric('Temporal > v70',b(s.temporal_better_than_legacy),cls(s.temporal_better_than_legacy)),metric('Loop脱出',b(s.temporal_loop_escape),cls(s.temporal_loop_escape)),metric('Credit loop event',s.credited_loop_events,'blue'),metric('固有loop motif',s.unique_loop_motifs,'blue'),metric('Assist Eligible',s.assist_eligible,'blue'),metric('Assist作動',s.assist_acted,'blue'),metric('Assist Top変更',s.assist_top_changes,'blue'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',cls(s.brain_file_unchanged)),metric('Core readiness',s.core_readiness,'blue'),metric('総合判定',s.overall_verdict,'blue')].join('');document.getElementById('detail').textContent=JSON.stringify(d.branches.temporal_credit,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v73: http://{HOST}:{PORT}")
    print("Temporal credit / loop attribution / v70 Relative base / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
