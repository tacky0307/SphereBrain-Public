from __future__ import annotations

import copy
import hashlib
import json
import socket
import statistics
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
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v99 as v99
import run_core_growth_binding_v99b as v99b

HOST = "127.0.0.1"
START_PORT = 5158
OUT = ROOT / "data" / "core_growth_binding_v100" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"

# Frozen before running v100.
SEEDS = [
    42, 73, 101, 131, 173, 211, 257, 293, 337, 379,
    421, 463, 509, 557, 601, 647, 691, 733, 787, 829,
    877, 919, 967, 1013, 1061, 1103, 1151, 1201, 1249, 1291,
    1327, 1367, 1423, 1471, 1511, 1559, 1601, 1657, 1699, 1741,
]
FORMATION_CYCLES = v99.FORMATION_CYCLES
AFTERMATH_CYCLES = v99.AFTERMATH_CYCLES

TRIGGER_SETS = {
    "T1": [
        v83.EpisodeSpec("旅人", "空港", "歩く"),
        v83.EpisodeSpec("旅人", "港", "歩く"),
        v83.EpisodeSpec("旅人", "道", "歩く"),
        v83.EpisodeSpec("旅人", "工房", "歩く"),
    ],
    "T2": [
        v83.EpisodeSpec("配達員", "駅", "運ぶ"),
        v83.EpisodeSpec("配達員", "港", "運ぶ"),
        v83.EpisodeSpec("配達員", "駐車場", "運ぶ"),
        v83.EpisodeSpec("配達員", "工具箱", "運ぶ"),
    ],
    "T3": [
        v83.EpisodeSpec("観察者", "森", "見る"),
        v83.EpisodeSpec("観察者", "海", "見る"),
        v83.EpisodeSpec("観察者", "道", "見る"),
        v83.EpisodeSpec("観察者", "工房", "見る"),
    ],
    "T4": [
        v83.EpisodeSpec("記録者", "空", "記録する"),
        v83.EpisodeSpec("記録者", "川", "記録する"),
        v83.EpisodeSpec("記録者", "駅", "記録する"),
        v83.EpisodeSpec("記録者", "台所", "記録する"),
    ],
}

MATCHED_NOVEL_EPISODES = list(v99.MATCHED_NOVEL_EPISODES)


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


def prepare_seed(seed: int) -> SphereBrain:
    brain = v99.v84.clean_primary_seed(seed)
    for _ in range(FORMATION_CYCLES):
        v99.v98.train_one_cycle(brain)
    return brain


def run_branch(base_brain: SphereBrain, episodes: list[v83.EpisodeSpec]) -> dict:
    brain = copy.deepcopy(base_brain)
    history: dict[tuple[int, int], int] = {}
    before = v99.structural_snapshot(brain, history)
    pre_quorum = v99b.quorum_edges(before)

    trigger_edges = v99.train_normal(brain, episodes)
    immediate = v99.structural_snapshot(brain, history)
    immediate_quorum = v99b.quorum_edges(immediate)
    immediate_new = immediate_quorum - pre_quorum
    overlap = v99b.module_overlap_stats(trigger_edges, before["modules"])

    for _ in range(AFTERMATH_CYCLES):
        v99.v98.train_one_cycle(brain)
        v99.structural_snapshot(brain, history)

    final = v99.structural_snapshot(brain, history)
    final_quorum = v99b.quorum_edges(final)
    persistent = immediate_new & final_quorum

    return {
        **overlap,
        "trigger_edge_count": len(trigger_edges),
        "before_module_count": len(before["modules"]),
        "immediate_new_quorum": len(immediate_new),
        "persistent_new_quorum": len(persistent),
        "persistent_fraction": len(persistent) / max(1, len(immediate_new)),
        "support_delta": final["consensus"]["mean_support_sources"] - before["consensus"]["mean_support_sources"],
        "consensus_delta": final["consensus"]["mean_consensus_strength"] - before["consensus"]["mean_consensus_strength"],
        "module_delta": len(final["modules"]) - len(before["modules"]),
        "brain": brain,
    }


def median_split(rows: list[dict], key: str) -> dict:
    if not rows:
        return {"median": 0.0, "high_n": 0, "low_n": 0, "high_success_rate": 0.0, "low_success_rate": 0.0, "margin": 0.0}
    med = statistics.median(float(x[key]) for x in rows)
    high = [x for x in rows if float(x[key]) > med]
    low = [x for x in rows if float(x[key]) <= med]
    hr = sum(1 for x in high if x["success"]) / len(high) if high else 0.0
    lr = sum(1 for x in low if x["success"]) / len(low) if low else 0.0
    return {
        "median": float(med),
        "high_n": len(high),
        "low_n": len(low),
        "high_success_rate": hr,
        "low_success_rate": lr,
        "margin": hr - lr,
    }


def trigger_summary(rows: list[dict]) -> list[dict]:
    out = []
    for trigger_id in TRIGGER_SETS:
        subset = [x for x in rows if x["trigger_id"] == trigger_id]
        crossing = median_split(subset, "touched_module_count")
        persistence = median_split(subset, "persistent_fraction")
        out.append({
            "trigger_id": trigger_id,
            "trials": len(subset),
            "success": sum(1 for x in subset if x["success"]),
            "success_rate": sum(1 for x in subset if x["success"]) / len(subset) if subset else 0.0,
            "crossing_margin": crossing["margin"],
            "persistence_margin": persistence["margin"],
            "crossing_positive": crossing["margin"] > 0,
            "persistence_positive": persistence["margin"] > 0,
        })
    return out


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    rows = []
    representatives: list[SphereBrain] = []

    for seed_index, seed in enumerate(SEEDS, start=1):
        print(f"[v100] seed {seed_index}/{len(SEEDS)}: {seed}", flush=True)
        base = prepare_seed(seed)
        matched = run_branch(base, MATCHED_NOVEL_EPISODES)
        matched_persistent = int(matched["persistent_new_quorum"])
        print(f"[v100]   matched done: persistent={matched_persistent}", flush=True)

        for trigger_index, (trigger_id, episodes) in enumerate(TRIGGER_SETS.items(), start=1):
            print(f"[v100]   trigger {trigger_index}/{len(TRIGGER_SETS)}: {trigger_id}", flush=True)
            trial = run_branch(base, episodes)
            success = int(trial["persistent_new_quorum"]) > matched_persistent
            rows.append({
                "seed": seed,
                "trigger_id": trigger_id,
                "matched_persistent_new_quorum": matched_persistent,
                "success": bool(success),
                **{k: v for k, v in trial.items() if k != "brain"},
            })
            if not representatives:
                representatives.append(trial["brain"])

    crossing = median_split(rows, "touched_module_count")
    persistence = median_split(rows, "persistent_fraction")
    cross_edges = median_split(rows, "cross_module_edge_count")
    by_trigger = trigger_summary(rows)

    crossing_positive_triggers = sum(1 for x in by_trigger if x["crossing_positive"])
    persistence_positive_triggers = sum(1 for x in by_trigger if x["persistence_positive"])
    success_count = sum(1 for x in rows if x["success"])

    precursor_signal = (
        len(rows) == len(SEEDS) * len(TRIGGER_SETS)
        and success_count >= 8
        and crossing["margin"] >= 0.08
        and persistence["margin"] >= 0.20
        and crossing_positive_triggers >= 3
        and persistence_positive_triggers >= 3
    )

    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    if representatives:
        temp = OUT / "structural_precursor_replication_roundtrip.json"
        before_weights = representatives[0].weights.tolist()
        representatives[0].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
        native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    else:
        native_present = False
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    measurement_pass = saveload_equal and production_unchanged and native_present and len(rows) == 160

    if precursor_signal:
        readiness = "structural_reorganization_precursor_replicated"
        verdict = "broader_realized_module_crossing_and_persistent_new_quorum_replicate_as_predictors_of_triggered_reorganization_across_multiple_triggers"
        next_step = "test_pre_trigger_prediction_without_using_trigger_semantics_and then evaluate whether the precursor can guide experience scheduling"
    else:
        readiness = "structural_reorganization_precursor_not_yet_replicated"
        verdict = "v99b_precursor_does_not_yet_replicate strongly enough across expanded seeds and trigger streams"
        next_step = "inspect trigger-specific and seed-specific failure modes before changing core learning or consensus guidance"

    payload = {
        "experiment": "Core Growth Binding v100 — Structural Precursor Replication",
        "contract": {
            "primary_core_modified": False,
            "seed_count": len(SEEDS),
            "trigger_count": len(TRIGGER_SETS),
            "trigger_trial_count": len(rows),
            "formation_cycles": FORMATION_CYCLES,
            "aftermath_cycles": AFTERMATH_CYCLES,
            "success_definition": "trigger_persistent_new_quorum_greater_than_same_seed_matched_novel",
            "semantic_labels_used_for_precursor_scoring": False,
            "thresholds_frozen_before_run": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "seed_count": len(SEEDS),
            "trigger_count": len(TRIGGER_SETS),
            "trial_count": len(rows),
            "success_count": success_count,
            "success_rate": success_count / len(rows) if rows else 0.0,
            "crossing_median": crossing["median"],
            "high_crossing_success_rate": crossing["high_success_rate"],
            "low_crossing_success_rate": crossing["low_success_rate"],
            "crossing_success_margin": crossing["margin"],
            "cross_edge_success_margin": cross_edges["margin"],
            "high_persistence_success_rate": persistence["high_success_rate"],
            "low_persistence_success_rate": persistence["low_success_rate"],
            "persistence_success_margin": persistence["margin"],
            "crossing_positive_triggers": crossing_positive_triggers,
            "persistence_positive_triggers": persistence_positive_triggers,
            "precursor_signal": precursor_signal,
            "saveload_equal": saveload_equal,
            "brain_file_unchanged": production_unchanged,
            "primary_native_learning_present": native_present,
            "measurement_pass": measurement_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "trigger_summary": by_trigger,
        "trials": rows,
    }
    (OUT / "latest_binding_v100.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v100</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v100：Structural Precursor Replication</h1><p class="lead">v99Bで候補になった「実際のModule横断量」と「新規Quorumの持続」が、40 Seed × 4種類の具体的Triggerでも再編成成功と結びつくかを再現検証する。意味ラベルは判定に使わない。</p><section class="panel"><div class="controls"><button id="run">Replicationを実行</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Trigger別</h2><pre id="triggers" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="trials" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Seed',s.seed_count),metric('Trigger',s.trigger_count),metric('Trial',s.trial_count),metric('Success',`${s.success_count}/${s.trial_count}`),metric('High crossing success',`${(s.high_crossing_success_rate*100).toFixed(1)}%`),metric('Low crossing success',`${(s.low_crossing_success_rate*100).toFixed(1)}%`),metric('Crossing margin',`${(s.crossing_success_margin*100).toFixed(1)}pt`,s.crossing_success_margin>0?'good':'warn'),metric('Cross-edge margin',`${(s.cross_edge_success_margin*100).toFixed(1)}pt`,s.cross_edge_success_margin>0?'good':'warn'),metric('High persistence success',`${(s.high_persistence_success_rate*100).toFixed(1)}%`),metric('Low persistence success',`${(s.low_persistence_success_rate*100).toFixed(1)}%`),metric('Persistence margin',`${(s.persistence_success_margin*100).toFixed(1)}pt`,s.persistence_success_margin>0?'good':'warn'),metric('Crossing + triggers',`${s.crossing_positive_triggers}/${s.trigger_count}`),metric('Persistence + triggers',`${s.persistence_positive_triggers}/${s.trigger_count}`),metric('Precursor signal',yn(s.precursor_signal),s.precursor_signal?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('triggers').textContent=JSON.stringify(d.trigger_summary,null,2);document.getElementById('trials').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v100: http://{HOST}:{PORT}")
    print(f"Replication: {len(SEEDS)} seeds x {len(TRIGGER_SETS)} triggers = {len(SEEDS) * len(TRIGGER_SETS)} trigger trials")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
