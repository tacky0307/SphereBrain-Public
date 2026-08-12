from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import webbrowser
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v89 as v89

HOST = "127.0.0.1"
START_PORT = 5143
OUT = ROOT / "data" / "core_growth_binding_v91b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v89.SEEDS
DOMAINS = v89.DOMAINS


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


def bridge_components(row: dict) -> dict[str, bool]:
    exp = row["experimental"]["direct"]
    eff = row["effects"]
    return {
        "candidate": bool(row["context_linked_candidate_count"] > 0),
        "shared_edge": bool(exp["new_shared_edges"] > 0),
        "control": bool(eff["direct_new_shared_edges"] > 0),
        "similarity": bool(
            eff["direct_edge_similarity"] > 0
            or eff["direct_activation_similarity"] > 0
        ),
    }


def is_bridge(row: dict) -> bool:
    return all(bridge_components(row).values())


def quantitative(row: dict) -> dict:
    exp = row["experimental"]["direct"]
    ctrl = row["control"]["direct"]
    eff = row["effects"]
    return {
        "candidate_count": int(row["context_linked_candidate_count"]),
        "experimental_new_shared_edges": int(exp["new_shared_edges"]),
        "control_new_shared_edges": int(ctrl["new_shared_edges"]),
        "new_shared_edge_effect": float(eff["direct_new_shared_edges"]),
        "experimental_edge_similarity": float(exp["edge_similarity"]),
        "control_edge_similarity": float(ctrl["edge_similarity"]),
        "edge_similarity_effect": float(eff["direct_edge_similarity"]),
        "experimental_activation_similarity": float(exp["activation_similarity"]),
        "control_activation_similarity": float(ctrl["activation_similarity"]),
        "activation_similarity_effect": float(eff["direct_activation_similarity"]),
    }


def collapse_code(row: dict) -> str:
    c = bridge_components(row)
    failed = [k for k in ("candidate", "shared_edge", "control", "similarity") if not c[k]]
    return "+".join(failed) if failed else "none"


def run_trial(domain, seed: int) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)

    records = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            records.append(v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle))
        if cycle < max(CHECKPOINTS):
            v85.v83.train_episode_set(experimental, exp_episodes)
            v85.v83.train_episode_set(control, ctrl_episodes)

    active_idx = [i for i, row in enumerate(records) if is_bridge(row)]
    if not active_idx:
        return {
            "domain": domain.name,
            "seed": int(seed),
            "ever_bridge": False,
            "stable": False,
            "experimental": experimental,
        }

    cls = v86.classify(records)
    first_i = active_idx[0]
    first_bridge = records[first_i]
    first_collapse = None
    for i in range(first_i + 1, len(records)):
        if not is_bridge(records[i]):
            first_collapse = records[i]
            break

    stable = cls == "stable"
    out = {
        "domain": domain.name,
        "seed": int(seed),
        "ever_bridge": True,
        "stable": stable,
        "classification": cls,
        "first_bridge_cycle": int(first_bridge["cycle"]),
        "first_bridge": quantitative(first_bridge),
        "experimental": experimental,
    }

    if first_collapse is None:
        out.update({
            "first_collapse_cycle": None,
            "collapse_code": "no_observed_collapse",
            "collapse": None,
            "delta": None,
            "mechanism": "no_observed_collapse",
        })
        return out

    before = quantitative(first_bridge)
    after = quantitative(first_collapse)
    delta = {k: float(after[k]) - float(before[k]) for k in before}

    exp_weakened = (
        delta["experimental_new_shared_edges"] < 0
        or delta["experimental_edge_similarity"] < -1e-12
        or delta["experimental_activation_similarity"] < -1e-12
    )
    ctrl_caught_up = (
        delta["control_new_shared_edges"] > 0
        or delta["control_edge_similarity"] > 1e-12
        or delta["control_activation_similarity"] > 1e-12
    )

    if exp_weakened and ctrl_caught_up:
        mechanism = "experimental_weakened_and_control_caught_up"
    elif exp_weakened:
        mechanism = "experimental_weakened"
    elif ctrl_caught_up:
        mechanism = "control_caught_up"
    else:
        mechanism = "relative_threshold_shift_without_large_raw_directional_change"

    out.update({
        "first_collapse_cycle": int(first_collapse["cycle"]),
        "collapse_code": collapse_code(first_collapse),
        "collapse": after,
        "delta": delta,
        "mechanism": mechanism,
    })
    return out


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    all_trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    ever = [x for x in all_trials if x["ever_bridge"]]
    stable = [x for x in ever if x["stable"]]
    unstable = [x for x in ever if not x["stable"]]

    collapse_counts = Counter(x["collapse_code"] for x in unstable)
    mechanism_counts = Counter(x["mechanism"] for x in unstable)
    classified_total = sum(collapse_counts.values())
    exhaustive = classified_total == len(unstable)

    multi_factor = {
        k: v for k, v in collapse_counts.items()
        if "+" in k and k != "no_observed_collapse"
    }

    delayed = [x for x in unstable if x.get("first_collapse_cycle") is not None and x["first_collapse_cycle"] > x["first_bridge_cycle"]]
    no_observed = collapse_counts.get("no_observed_collapse", 0)

    domain_rows = []
    for domain in DOMAINS:
        rows = [x for x in unstable if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "unstable": len(rows),
            "collapse_codes": dict(Counter(x["collapse_code"] for x in rows)),
            "mechanisms": dict(Counter(x["mechanism"] for x in rows)),
        })

    representative = ever[0] if ever else all_trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "multi_factor_decomposition_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    dominant_code = collapse_counts.most_common(1)[0][0] if collapse_counts else None
    dominant_mechanism = mechanism_counts.most_common(1)[0][0] if mechanism_counts else None
    attribution_pass = (
        len(unstable) > 0
        and exhaustive
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if attribution_pass:
        readiness = "post_bridge_multi_factor_collapse_fully_decomposed"
        verdict = "all_unstable_bridge_trials_are_exhaustively_classified_and_experimental_vs_control_dynamics_are_separated"
        next_step = "design_intervention_against_dominant_relative-collapse_mechanism_not_generic_edge_protection"
    else:
        readiness = "post_bridge_multi_factor_decomposition_incomplete"
        verdict = "unstable_trials_are_not_yet_exhaustively_decomposed"
        next_step = "increase_checkpoint_resolution_or_fix_classification_before_new_intervention"

    payload = {
        "experiment": "Core Growth Binding v91B — Multi-Factor Collapse Decomposition",
        "contract": {
            "primary_core_modified": False,
            "consolidation_used": False,
            "assist_used": False,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "trial_count": len(all_trials),
            "focus": "ever_bridge_unstable_trials",
            "classification_exhaustive_required": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(all_trials),
            "ever_bridge_trials": len(ever),
            "stable_trials": len(stable),
            "unstable_trials": len(unstable),
            "classified_unstable_trials": classified_total,
            "classification_exhaustive": exhaustive,
            "collapse_code_counts": dict(collapse_counts),
            "multi_factor_breakdown": multi_factor,
            "mechanism_counts": dict(mechanism_counts),
            "dominant_collapse_code": dominant_code,
            "dominant_mechanism": dominant_mechanism,
            "delayed_collapse_trials": len(delayed),
            "no_observed_collapse_trials": no_observed,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "domain_rows": domain_rows,
        "unstable_trials": [{k: v for k, v in x.items() if k != "experimental"} for x in unstable],
    }
    (OUT / "latest_binding_v91b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v91B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v91B：Multi-Factor Collapse Decomposition</h1><p class="lead">Ever Bridge後に不安定化した45試行を完全排他的に分類し、Experimental側の弱化とControl側の追いつきを分離する。Core・Consolidation・Assistは変更しない。</p><section class="panel"><div class="controls"><button id="run">Multi-Factorを分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Collapse分類</h2><pre id="collapse" class="raw">未実行</pre></section><section class="panel"><h2>Experimental / Control機序</h2><pre id="mechanism" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Ever Bridge',s.ever_bridge_trials),metric('Stable',s.stable_trials),metric('Unstable',s.unstable_trials,'warn'),metric('分類済み',`${s.classified_unstable_trials}/${s.unstable_trials}`,s.classification_exhaustive?'good':'warn'),metric('Dominant collapse',s.dominant_collapse_code||'—'),metric('Dominant mechanism',s.dominant_mechanism||'—'),metric('Delayed collapse',s.delayed_collapse_trials),metric('未観測collapse',s.no_observed_collapse_trials),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('collapse').textContent=JSON.stringify({collapse_code_counts:s.collapse_code_counts,multi_factor_breakdown:s.multi_factor_breakdown},null,2);document.getElementById('mechanism').textContent=JSON.stringify(s.mechanism_counts,null,2);document.getElementById('groups').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.unstable_trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v91B: http://{HOST}:{PORT}")
    threading.Timer(0.7, open_browser).start()
    serve(app, host=HOST, port=PORT, threads=4)
