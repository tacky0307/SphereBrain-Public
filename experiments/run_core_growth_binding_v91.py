from __future__ import annotations

import hashlib
import json
import socket
import statistics
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
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v89 as v89

HOST = "127.0.0.1"
START_PORT = 5142
OUT = ROOT / "data" / "core_growth_binding_v91" / "results"
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


def row_bridge(row: dict) -> bool:
    return bool(row["direct_success_now"])


def bridge_components(row: dict) -> dict:
    exp = row["experimental"]["direct"]
    eff = row["effects"]
    return {
        "candidate": row["context_linked_candidate_count"] > 0,
        "shared_edge": exp["new_shared_edges"] > 0,
        "control_beaten": eff["direct_new_shared_edges"] > 0,
        "similarity_beaten": (
            eff["direct_edge_similarity"] > 0
            or eff["direct_activation_similarity"] > 0
        ),
    }


def failure_mode(prev: dict, cur: dict) -> str:
    p = bridge_components(prev)
    c = bridge_components(cur)
    lost = [k for k in p if p[k] and not c[k]]
    if len(lost) >= 2:
        return "multi_factor_collapse"
    if lost == ["candidate"]:
        return "candidate_lost"
    if lost == ["shared_edge"]:
        return "shared_edge_lost"
    if lost == ["control_beaten"]:
        return "control_advantage_lost"
    if lost == ["similarity_beaten"]:
        return "similarity_advantage_lost"
    if not cur["direct_success_now"]:
        # One condition may have been absent already in the previous bridge checkpoint
        # due to numerical ties around the threshold. Classify by current failed gates.
        failed = [k for k, ok in c.items() if not ok]
        if len(failed) >= 2:
            return "multi_factor_collapse"
        mapping = {
            "candidate": "candidate_lost",
            "shared_edge": "shared_edge_lost",
            "control_beaten": "control_advantage_lost",
            "similarity_beaten": "similarity_advantage_lost",
        }
        if failed:
            return mapping[failed[0]]
    return "bridge_persisted"


def snapshot_metrics(row: dict) -> dict:
    exp = row["experimental"]["direct"]
    eff = row["effects"]
    return {
        "candidate_count": int(row["context_linked_candidate_count"]),
        "experimental_new_shared_edges": int(exp["new_shared_edges"]),
        "new_shared_edge_effect": float(eff["direct_new_shared_edges"]),
        "edge_similarity_effect": float(eff["direct_edge_similarity"]),
        "activation_similarity_effect": float(eff["direct_activation_similarity"]),
    }


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

    active_idx = [i for i, r in enumerate(records) if row_bridge(r)]
    if not active_idx:
        return {
            "domain": domain.name,
            "seed": int(seed),
            "ever_bridge": False,
            "stable": False,
            "records": records,
            "experimental": experimental,
        }

    cls = v86.classify(records)
    first_i = active_idx[0]
    first = records[first_i]
    transitions = []
    collapse_modes = []
    for i in range(first_i, len(records) - 1):
        cur = records[i]
        nxt = records[i + 1]
        mode = failure_mode(cur, nxt) if row_bridge(cur) else None
        if row_bridge(cur):
            transitions.append({
                "from_cycle": int(cur["cycle"]),
                "to_cycle": int(nxt["cycle"]),
                "from_bridge": bool(cur["direct_success_now"]),
                "to_bridge": bool(nxt["direct_success_now"]),
                "failure_mode": mode,
                "from_metrics": snapshot_metrics(cur),
                "to_metrics": snapshot_metrics(nxt),
            })
            if not row_bridge(nxt) and mode:
                collapse_modes.append(mode)

    first_loss = next((x for x in transitions if not x["to_bridge"]), None)
    final = records[-1]
    return {
        "domain": domain.name,
        "seed": int(seed),
        "ever_bridge": True,
        "stable": cls == "stable",
        "classification": cls,
        "first_bridge_cycle": int(first["cycle"]),
        "first_bridge_metrics": snapshot_metrics(first),
        "first_loss": first_loss,
        "collapse_modes": collapse_modes,
        "bridge_checkpoint_count": len(active_idx),
        "final_bridge": bool(final["direct_success_now"]),
        "final_metrics": snapshot_metrics(final),
        "transitions": transitions,
        "records": records,
        "experimental": experimental,
    }


def mean_metric(rows: list[dict], key: str) -> float | None:
    vals = [x["first_bridge_metrics"].get(key) for x in rows if x.get("first_bridge_metrics") is not None]
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    all_trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    ever = [x for x in all_trials if x["ever_bridge"]]
    stable = [x for x in ever if x["stable"]]
    unstable = [x for x in ever if not x["stable"]]

    mode_counts: dict[str, int] = {}
    for row in unstable:
        mode = row["first_loss"]["failure_mode"] if row.get("first_loss") else "no_observed_loss_after_first_bridge"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    metrics = [
        "candidate_count",
        "experimental_new_shared_edges",
        "new_shared_edge_effect",
        "edge_similarity_effect",
        "activation_similarity_effect",
    ]
    stable_vs_unstable = []
    for key in metrics:
        s = mean_metric(stable, key)
        u = mean_metric(unstable, key)
        stable_vs_unstable.append({
            "metric": key,
            "stable_mean": s,
            "unstable_mean": u,
            "difference_stable_minus_unstable": None if s is None or u is None else s - u,
        })

    ranked = sorted(
        [x for x in stable_vs_unstable if x["difference_stable_minus_unstable"] is not None],
        key=lambda x: abs(float(x["difference_stable_minus_unstable"])),
        reverse=True,
    )

    domain_rows = []
    for domain in DOMAINS:
        rows = [x for x in ever if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "ever_bridge": len(rows),
            "stable": sum(1 for x in rows if x["stable"]),
            "unstable": sum(1 for x in rows if not x["stable"]),
            "first_loss_modes": {
                k: sum(1 for x in rows if x.get("first_loss") and x["first_loss"]["failure_mode"] == k)
                for k in [
                    "candidate_lost",
                    "shared_edge_lost",
                    "control_advantage_lost",
                    "similarity_advantage_lost",
                    "multi_factor_collapse",
                ]
            },
        })

    representative = ever[0] if ever else all_trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "post_bridge_dynamics_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else None
    attribution_pass = (
        len(ever) > 0
        and len(unstable) > 0
        and bool(mode_counts)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if attribution_pass:
        readiness = "post_bridge_stability_failure_mode_attributed"
        verdict = "post_bridge_collapse_is_measurable_and_can_be_separated_into_specific_gate_failures"
        next_step = "design_targeted_stability_intervention_for_the_dominant_post_bridge_failure_mode"
    else:
        readiness = "post_bridge_stability_attribution_inconclusive"
        verdict = "post_bridge_dynamics_could_not_be_reliably_attributed"
        next_step = "increase_checkpoint_resolution_around_first_bridge_before_new_core_change"

    payload = {
        "experiment": "Core Growth Binding v91 — Post-Bridge Stability Dynamics Microscope",
        "contract": {
            "primary_core_modified": False,
            "consolidation_used": False,
            "assist_used": False,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "trial_count": len(all_trials),
            "focus": "ever_bridge_trials_only",
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(all_trials),
            "ever_bridge_trials": len(ever),
            "stable_trials": len(stable),
            "unstable_trials": len(unstable),
            "first_loss_mode_counts": mode_counts,
            "dominant_first_loss_mode": dominant_mode,
            "top_stable_vs_unstable_metric": ranked[0]["metric"] if ranked else None,
            "top_stable_vs_unstable_difference": ranked[0]["difference_stable_minus_unstable"] if ranked else None,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "stable_vs_unstable_first_bridge": ranked,
        "domain_rows": domain_rows,
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in ever],
    }
    (OUT / "latest_binding_v91.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v91</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v91：Post-Bridge Stability Dynamics Microscope</h1><p class="lead">Primary Coreで一度Bridgeに到達した試行だけを追い、次checkpointで何が失われてStable / Unstableに分かれるのかを観察する。Core・Consolidation・Assistは変更しない。</p><section class="panel"><div class="controls"><button id="run">Bridge崩壊を追跡</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Stable vs Unstable</h2><pre id="compare" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Trial詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Ever Bridge',s.ever_bridge_trials),metric('Stable',s.stable_trials),metric('Unstable',s.unstable_trials,'warn'),metric('Candidate lost',s.first_loss_mode_counts.candidate_lost||0),metric('Shared edge lost',s.first_loss_mode_counts.shared_edge_lost||0),metric('Control差消失',s.first_loss_mode_counts.control_advantage_lost||0),metric('Similarity差消失',s.first_loss_mode_counts.similarity_advantage_lost||0),metric('Multi-factor',s.first_loss_mode_counts.multi_factor_collapse||0),metric('Dominant',s.dominant_first_loss_mode||'-'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変更',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('compare').textContent=JSON.stringify(d.stable_vs_unstable_first_bridge,null,2);document.getElementById('groups').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"v91: {url}")
    serve(app, host=HOST, port=PORT, threads=4)


if __name__ == "__main__":
    main()
