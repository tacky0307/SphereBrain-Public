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
START_PORT = 5120
OUT = ROOT / "data" / "core_growth_binding_v74" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEED_COUNT = 30
BASE_SEED = 740100
ADAPT_EPISODES = 20
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


def metric(ep: dict) -> tuple[int, int, int]:
    return (1 if ep["success"] else 0, -int(ep["steps"]), -int(ep["loop_steps"]))


def paired_branch(pretrained: SphereBrain, mode: str, seed: int) -> dict:
    brain = copy.deepcopy(pretrained)
    temporal = mode in {"temporal_credit", "temporal_plus_assist"}
    assist = mode == "temporal_plus_assist"
    rng = random.Random(seed)
    motif_counts: dict[str, int] = defaultdict(int)
    temporal_events = []
    assist_eligible = assist_acted = assist_top = 0
    first_success_episode = None
    success_episodes = 0
    total_loop_steps = 0

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
        total_loop_steps += int(ep.get("loop_steps", 0))

        if ep["success"]:
            success_episodes += 1
            if first_success_episode is None:
                first_success_episode = episode_index
            v70.reinforce_success(brain, ep, include_relative=True)
        elif temporal:
            temporal_events.extend(
                v73.temporal_attribution(brain, ep, motif_counts, episode_index=episode_index)
            )

    final_ep = v70.run_episode(
        brain,
        v70.CHANGED,
        relative_mode=True,
        assist=assist,
    )
    optimal = bool(final_ep["success"] and int(final_ep["steps"]) == 4)
    return {
        "final": final_ep,
        "final_success": bool(final_ep["success"]),
        "final_optimal": optimal,
        "first_success_episode": first_success_episode,
        "success_episodes": success_episodes,
        "mean_loop_steps": total_loop_steps / ADAPT_EPISODES,
        "credited_loop_events": sum(1 for e in temporal_events if e.get("credited")),
        "assist_eligible": assist_eligible,
        "assist_acted": assist_acted,
        "assist_top_changes": assist_top,
    }


def aggregate(rows: list[dict]) -> dict:
    successes = [r for r in rows if r["final_success"]]
    firsts = [int(r["first_success_episode"]) for r in rows if r["first_success_episode"] is not None]
    return {
        "final_success_count": len(successes),
        "final_success_rate": len(successes) / len(rows),
        "optimal_count": sum(1 for r in rows if r["final_optimal"]),
        "optimal_rate": sum(1 for r in rows if r["final_optimal"]) / len(rows),
        "ever_success_count": sum(1 for r in rows if r["first_success_episode"] is not None),
        "ever_success_rate": sum(1 for r in rows if r["first_success_episode"] is not None) / len(rows),
        "median_first_success_episode": None if not firsts else float(median(firsts)),
        "mean_success_episodes": sum(int(r["success_episodes"]) for r in rows) / len(rows),
        "mean_loop_steps": sum(float(r["mean_loop_steps"]) for r in rows) / len(rows),
        "assist_eligible_seeds": sum(1 for r in rows if int(r["assist_eligible"]) > 0),
        "assist_acted_seeds": sum(1 for r in rows if int(r["assist_acted"]) > 0),
        "assist_top_change_seeds": sum(1 for r in rows if int(r["assist_top_changes"]) > 0),
        "assist_eligible_steps": sum(int(r["assist_eligible"]) for r in rows),
        "assist_acted_steps": sum(int(r["assist_acted"]) for r in rows),
        "assist_top_changes": sum(int(r["assist_top_changes"]) for r in rows),
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    pretrained = v70.pretrain(base)

    seed_rows = []
    by_mode = {m: [] for m in MODES}
    for i in range(SEED_COUNT):
        seed = BASE_SEED + i
        results = {m: paired_branch(pretrained, m, seed) for m in MODES}
        for m in MODES:
            by_mode[m].append(results[m])
        seed_rows.append({"seed": seed, **results})

    aggregates = {m: aggregate(by_mode[m]) for m in MODES}

    a_to_b_improved = a_to_b_worsened = 0
    b_to_c_improved = b_to_c_worsened = 0
    assist_helped_when_acted = assist_hurt_when_acted = 0
    for row in seed_rows:
        a = row["legacy_relative"]["final"]
        b = row["temporal_credit"]["final"]
        c = row["temporal_plus_assist"]["final"]
        if metric(b) > metric(a):
            a_to_b_improved += 1
        elif metric(b) < metric(a):
            a_to_b_worsened += 1
        if metric(c) > metric(b):
            b_to_c_improved += 1
        elif metric(c) < metric(b):
            b_to_c_worsened += 1
        if int(row["temporal_plus_assist"]["assist_acted"]) > 0:
            if metric(c) > metric(b):
                assist_helped_when_acted += 1
            elif metric(c) < metric(b):
                assist_hurt_when_acted += 1

    ranking = sorted(
        MODES,
        key=lambda m: (
            aggregates[m]["final_success_rate"],
            aggregates[m]["optimal_rate"],
            aggregates[m]["ever_success_rate"],
            -aggregates[m]["mean_loop_steps"],
        ),
        reverse=True,
    )
    winner = ranking[0]
    brain_unchanged = before_hash == file_hash(BRAIN_PATH)

    if aggregates["temporal_credit"]["final_success_rate"] > aggregates["legacy_relative"]["final_success_rate"]:
        verdict = "temporal_credit_improves_recovery_rate_across_paired_seeds"
        readiness = "temporal_credit_population_value_observed"
    elif aggregates["temporal_credit"]["final_success_rate"] < aggregates["legacy_relative"]["final_success_rate"]:
        verdict = "temporal_credit_reduces_recovery_rate_across_paired_seeds"
        readiness = "legacy_relative_preferred_over_temporal_credit"
    elif aggregates["temporal_plus_assist"]["final_success_rate"] > aggregates["temporal_credit"]["final_success_rate"]:
        verdict = "bounded_assist_adds_population_level_recovery_value"
        readiness = "assist_population_value_observed"
    else:
        verdict = "paired_seed_benchmark_shows_no_recovery_rate_advantage_yet"
        readiness = "mechanism_role_still_undecided"

    payload = {
        "experiment": "Core Growth Binding v74 — Paired-Seed Recovery Benchmark",
        "contract": {
            "seed_count": SEED_COUNT,
            "paired_rng_seed_per_mode": True,
            "same_pretrained_core_per_seed_and_mode": True,
            "same_exploration_rate": True,
            "same_episode_budget": True,
            "single_seed_winner_not_primary": True,
            "production_brain_json_saved": False,
        },
        "aggregates": aggregates,
        "paired_comparison": {
            "legacy_to_temporal_improved_seeds": a_to_b_improved,
            "legacy_to_temporal_worsened_seeds": a_to_b_worsened,
            "temporal_to_assist_improved_seeds": b_to_c_improved,
            "temporal_to_assist_worsened_seeds": b_to_c_worsened,
            "assist_helped_when_acted_seeds": assist_helped_when_acted,
            "assist_hurt_when_acted_seeds": assist_hurt_when_acted,
        },
        "summary": {
            "winner": winner,
            "brain_file_unchanged": brain_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": "choose_core_mechanism_from_population_results_then_retest_on_multiple_environment_changes",
        },
        "seed_rows": seed_rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v74.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v74</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v74：Paired-Seed Recovery Benchmark</h1><p class="lead">30 seedを同じ初期Core・同じ探索乱数seedで並走し、v70 Relative / Temporal Credit / Temporal+Assistを集団成績で比較する。</p><section class="panel"><div class="controls"><button id="run">30 seed Benchmarkを実行</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>集計</h2><pre id="agg" class="raw">未実行</pre></section><section class="panel"><h2>Seed別結果</h2><pre id="rows" class="raw">未実行</pre></section><script>
const pct=x=>(100*x).toFixed(1)+'%';function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const a=d.aggregates.legacy_relative,b=d.aggregates.temporal_credit,c=d.aggregates.temporal_plus_assist,p=d.paired_comparison,s=d.summary;document.getElementById('metrics').innerHTML=[metric('Winner',s.winner),metric('v70 最終到達率',pct(a.final_success_rate)),metric('Temporal 最終到達率',pct(b.final_success_rate)),metric('Assist 最終到達率',pct(c.final_success_rate)),metric('v70 最短率',pct(a.optimal_rate)),metric('Temporal 最短率',pct(b.optimal_rate)),metric('Assist 最短率',pct(c.optimal_rate)),metric('A→B 改善seed',p.legacy_to_temporal_improved_seeds),metric('A→B 悪化seed',p.legacy_to_temporal_worsened_seeds),metric('Assist作動seed',c.assist_acted_seeds),metric('Assist Top変更seed',c.assist_top_change_seeds),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('agg').textContent=JSON.stringify({aggregates:d.aggregates,paired_comparison:d.paired_comparison},null,2);document.getElementById('rows').textContent=JSON.stringify(d.seed_rows,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v74: http://{HOST}:{PORT}")
    print("30 paired seeds / same initial Core and RNG seed per mode / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
