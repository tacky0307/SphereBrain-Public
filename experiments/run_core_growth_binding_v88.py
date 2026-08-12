from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
from semantic_bridge_delayed_selective import DelayedSelectiveConsolidation
from semantic_bridge_event_triggered import EventTriggeredConsolidation
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v87 as v87

HOST = "127.0.0.1"
START_PORT = 5139
OUT = ROOT / "data" / "core_growth_binding_v88" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v85.CHECKPOINTS
SEEDS = v84.SEEDS
DOMAINS = v84.DOMAINS
MODES = ["primary", "delayed_selective", "event_triggered", "event_triggered_assist"]


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


def edge_tuple(value) -> tuple[int, int]:
    a, b = [int(x) for x in value]
    return tuple(sorted((a, b)))


def run_trial(domain, seed: int, mode: str) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "event_triggered_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)

    exp_baseline = v84.make_baseline(experimental, domain)
    ctrl_baseline = v84.make_baseline(control, domain)
    exp_episodes = v84.episodes_for(domain, True)
    ctrl_episodes = v84.episodes_for(domain, False)

    delayed_exp = DelayedSelectiveConsolidation()
    delayed_ctrl = DelayedSelectiveConsolidation()
    event_exp = EventTriggeredConsolidation()
    event_ctrl = EventTriggeredConsolidation()

    records = []
    protected_edges: set[tuple[int, int]] = set()
    hub_protected_edges: set[tuple[int, int]] = set()
    trigger_cycles: list[int] = []
    trigger_events = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v85.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle)
            row["mode"] = mode
            records.append(row)

        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

            if mode == "delayed_selective":
                exp_rows = v87.selective_candidates(experimental, domain, exp_baseline["direct"], True)
                ctrl_rows = v87.selective_candidates(control, domain, ctrl_baseline["direct"], False)
                result = delayed_exp.observe(experimental, exp_rows)
                delayed_ctrl.observe(control, ctrl_rows)
                for item in result.get("protected", []):
                    edge = edge_tuple(item["edge"])
                    protected_edges.add(edge)
                    if v87.hub_score(experimental, edge) >= 0.88:
                        hub_protected_edges.add(edge)

            elif mode in {"event_triggered", "event_triggered_assist"}:
                exp_rows = v87.selective_candidates(experimental, domain, exp_baseline["direct"], True)
                ctrl_rows = v87.selective_candidates(control, domain, ctrl_baseline["direct"], False)
                result = event_exp.observe(experimental, exp_rows)
                event_ctrl.observe(control, ctrl_rows)
                for item in result.get("triggered", []):
                    trigger_cycles.append(int(item["cycle"]))
                    trigger_events.append(item)
                for item in result.get("protected", []):
                    edge = edge_tuple(item["edge"])
                    protected_edges.add(edge)
                    if v87.hub_score(experimental, edge) >= 0.88:
                        hub_protected_edges.add(edge)

    cls = v86.classify(records)
    active = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    final = records[-1]
    return {
        "domain": domain.name,
        "seed": int(seed),
        "mode": mode,
        "classification": cls,
        "ever_bridge": bool(active),
        "stable": cls == "stable",
        "never": cls == "never",
        "first_bridge_cycle": active[0] if active else None,
        "final_direct_similarity": float(final["experimental"]["direct"]["edge_similarity"]),
        "final_candidate_count": int(final["context_linked_candidate_count"]),
        "protected_edge_count": len(protected_edges),
        "hub_protected_count": len(hub_protected_edges),
        "trigger_count": len(trigger_events),
        "first_trigger_cycle": min(trigger_cycles) if trigger_cycles else None,
        "trigger_cycles": sorted(set(trigger_cycles)),
        "trigger_events": trigger_events,
        "records": records,
        "experimental": experimental,
    }


def summarize(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    protected = sum(x["protected_edge_count"] for x in subset)
    hub_protected = sum(x["hub_protected_count"] for x in subset)
    trigger_cycles = [x["first_trigger_cycle"] for x in subset if x["first_trigger_cycle"] is not None]
    classes: dict[str, int] = {}
    for row in subset:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    return {
        "mode": mode,
        "trials": len(subset),
        "ever_bridge": sum(1 for x in subset if x["ever_bridge"]),
        "stable": sum(1 for x in subset if x["stable"]),
        "never": sum(1 for x in subset if x["never"]),
        "class_counts": classes,
        "protected_edges": protected,
        "hub_protected_edges": hub_protected,
        "hub_protection_rate": hub_protected / protected if protected else 0.0,
        "triggered_trials": sum(1 for x in subset if x["first_trigger_cycle"] is not None),
        "unique_first_trigger_cycles": sorted(set(int(x) for x in trigger_cycles)),
        "mean_final_direct_similarity": sum(x["final_direct_similarity"] for x in subset) / len(subset),
    }


def paired(rows: list[dict], mode: str) -> dict:
    p = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == "primary"}
    t = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == mode}
    deltas = [t[k]["final_direct_similarity"] - p[k]["final_direct_similarity"] for k in p]
    return {
        "stable_gain": sum(1 for k in p if t[k]["stable"] and not p[k]["stable"]),
        "stable_loss": sum(1 for k in p if p[k]["stable"] and not t[k]["stable"]),
        "never_gain": sum(1 for k in p if t[k]["never"] and not p[k]["never"]),
        "never_loss": sum(1 for k in p if p[k]["never"] and not t[k]["never"]),
        "max_similarity_delta": max(deltas),
        "mean_similarity_delta": sum(deltas) / len(deltas),
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed, mode) for mode in MODES for domain in DOMAINS for seed in SEEDS]
    summaries = {m: summarize(trials, m) for m in MODES}
    event_pair = paired(trials, "event_triggered")
    assist_pair = paired(trials, "event_triggered_assist")

    primary = summaries["primary"]
    delayed = summaries["delayed_selective"]
    event = summaries["event_triggered"]
    assist = summaries["event_triggered_assist"]

    event_distinctness = event_pair["max_similarity_delta"] <= 0.12
    assist_distinctness = assist_pair["max_similarity_delta"] <= 0.12
    event_safe = event["never"] <= primary["never"] and event_distinctness
    assist_safe = assist["never"] <= primary["never"] and assist_distinctness
    event_improves = event["stable"] > primary["stable"] and event_safe
    assist_improves = assist["stable"] > primary["stable"] and assist_safe

    trigger_adaptive = len(event["unique_first_trigger_cycles"]) >= 2 or len(assist["unique_first_trigger_cycles"]) >= 2
    event_hub_safe = event["hub_protection_rate"] <= delayed["hub_protection_rate"]
    assist_hub_safe = assist["hub_protection_rate"] <= delayed["hub_protection_rate"]

    representative = next((x for x in trials if x["mode"] == "event_triggered" and x["stable"]), next(x for x in trials if x["mode"] == "event_triggered"))
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "event_triggered_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    overall_pass = (
        (event_improves or assist_improves)
        and trigger_adaptive
        and (event_hub_safe or assist_hub_safe)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if event_improves and assist_improves:
        winner = "event_triggered_assist" if assist["stable"] > event["stable"] else "event_triggered"
    elif event_improves:
        winner = "event_triggered"
    elif assist_improves:
        winner = "event_triggered_assist"
    else:
        winner = "primary"

    if overall_pass:
        verdict = "structural_maturity_events_trigger_semantic_consolidation_more_effectively_than_fixed_time_gating"
        readiness = "semantic_event_triggered_consolidation_candidate"
        next_step = "stress_test_event_triggered_winner_under_context_switching_and_contradictory_episodes"
    else:
        verdict = "event_triggered_consolidation_does_not_yet_outperform_primary_core_safely"
        readiness = "semantic_event_triggered_consolidation_not_yet_validated"
        next_step = "attribute_maturity_trigger_false_positive_and_false_negative_cases_before_primary_core_change"

    payload = {
        "experiment": "Core Growth Binding v88 — Event-Triggered Consolidation",
        "contract": {
            "primary_core_modified": False,
            "modes": MODES,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "checkpoints": CHECKPOINTS,
            "fixed_protection_start_used_by_event_mode": False,
            "semantic_answer_labels_used": False,
            "event_basis": ["recurrence", "consecutive_persistence", "age", "context_proximity", "non_hub"],
            "production_brain_json_saved": False,
        },
        "summary": {
            "primary_stable": primary["stable"],
            "delayed_stable": delayed["stable"],
            "event_stable": event["stable"],
            "event_assist_stable": assist["stable"],
            "primary_never": primary["never"],
            "delayed_never": delayed["never"],
            "event_never": event["never"],
            "event_assist_never": assist["never"],
            "event_hub_protection_rate": event["hub_protection_rate"],
            "event_assist_hub_protection_rate": assist["hub_protection_rate"],
            "event_triggered_trials": event["triggered_trials"],
            "assist_triggered_trials": assist["triggered_trials"],
            "event_first_trigger_cycles": event["unique_first_trigger_cycles"],
            "assist_first_trigger_cycles": assist["unique_first_trigger_cycles"],
            "adaptive_trigger_timing_observed": trigger_adaptive,
            "event_distinctness_safe": event_distinctness,
            "assist_distinctness_safe": assist_distinctness,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "event_triggered_pass": overall_pass,
            "winner": winner,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "mode_summaries": summaries,
        "paired": {"event_triggered": event_pair, "event_triggered_assist": assist_pair},
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in trials],
    }
    (OUT / "latest_binding_v88.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v88</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v88：Event-Triggered Consolidation</h1><p class="lead">固定Episode数ではなく、候補Edgeの再出現・連続持続・生存期間・Context近接・非Hub性から構造成熟イベントを検出し、その瞬間にだけConsolidationを発火する。</p><section class="panel"><div class="controls"><button id="run">成熟イベント定着を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode比較</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Primary Stable',s.primary_stable),metric('v87 Stable',s.delayed_stable),metric('Event Stable',s.event_stable,s.event_stable>s.primary_stable?'good':'warn'),metric('Event+Assist Stable',s.event_assist_stable,s.event_assist_stable>s.primary_stable?'good':'warn'),metric('Primary Never',s.primary_never),metric('Event Never',s.event_never,s.event_never<=s.primary_never?'good':'warn'),metric('Event+Assist Never',s.event_assist_never,s.event_assist_never<=s.primary_never?'good':'warn'),metric('Event Hub保護率',(100*s.event_hub_protection_rate).toFixed(1)+'%',s.event_hub_protection_rate===0?'good':'blue'),metric('発火Trial',s.event_triggered_trials),metric('発火cycle',JSON.stringify(s.event_first_trigger_cycles)),metric('Adaptive timing',yn(s.adaptive_trigger_timing_observed),s.adaptive_trigger_timing_observed?'good':'warn'),metric('Distinctness',yn(s.event_distinctness_safe&&s.assist_distinctness_safe),(s.event_distinctness_safe&&s.assist_distinctness_safe)?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変更',s.brain_file_unchanged?'good':'warn'),metric('v88 PASS',yn(s.event_triggered_pass),s.event_triggered_pass?'good':'warn'),metric('Winner',s.winner),metric('Core readiness',s.core_readiness)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post('/api/run')
def api_run():
    return jsonify(observe())


@app.get('/')
def index():
    return PAGE


def main() -> None:
    print(f"v88: http://{HOST}:{PORT}")
    threading.Timer(0.8, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    serve(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
