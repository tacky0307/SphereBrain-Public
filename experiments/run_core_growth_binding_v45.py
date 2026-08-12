from __future__ import annotations

import json
import math
import socket
import sys
import threading
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v42 as v42
import run_core_growth_binding_v43 as v43
import run_core_growth_binding_v44 as v44

HOST = "127.0.0.1"
START_PORT = 5091
OUT = ROOT / "data" / "core_growth_binding_v45" / "results"
POSITIONS = ["左", "中央"]
WINDOW = 1
REL_TOL = float(v43.REL_TOL)


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


def rel_raw(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-9)
    return (a - b) / scale


def rel_boundary_distance(a: float, b: float) -> float:
    raw = rel_raw(a, b)
    return min(abs(raw - REL_TOL), abs(raw + REL_TOL))


def ratio_value(a: float, b: float) -> float:
    return float(a) / max(abs(float(b)), 1e-9)


def ratio_boundary_distance(a: float, b: float) -> float:
    ratio = ratio_value(a, b)
    boundaries = [0.8, 0.95, 1.05, 1.25]
    return min(abs(ratio - x) for x in boundaries)


def local_topology_components(event: dict) -> list[dict]:
    brain = v3.base.CORE
    e = int(event["echo_node"])
    p = int(event["position_node"])
    n = int(event["shared_neighbor"])
    e_neighbors = set(int(x) for x in np.flatnonzero(brain.adjacency[e]))
    p_neighbors = set(int(x) for x in np.flatnonzero(brain.adjacency[p]))
    n_neighbors = set(int(x) for x in np.flatnonzero(brain.adjacency[n]))
    shared = e_neighbors & p_neighbors
    triangles = sum(1 for x in n_neighbors if x in e_neighbors or x in p_neighbors)
    mean_ep = (len(e_neighbors) + len(p_neighbors)) / 2.0
    values = [
        ("local_topology.degree_relation_E_vs_position", "relation", float(len(e_neighbors)), float(len(p_neighbors)), v43.relation(float(len(e_neighbors)), float(len(p_neighbors)))),
        ("local_topology.neighbor_vs_mean_degree", "relation", float(len(n_neighbors)), float(mean_ep), v43.relation(float(len(n_neighbors)), float(mean_ep))),
        ("local_topology.shared_neighbor_exists", "binary", float(bool(shared)), 0.0, 1.0 if shared else 0.0),
        ("local_topology.triangle_exists", "binary", float(triangles > 0), 0.0, 1.0 if triangles > 0 else 0.0),
        ("local_topology.degree_ratio_band", "ratio_band", float(len(e_neighbors) + 1), float(len(p_neighbors) + 1), v43.ratio_band(float(len(e_neighbors) + 1), float(len(p_neighbors) + 1))),
    ]
    rows = []
    for name, family, a, b, category in values:
        rows.append(component_row(name, family, a, b, category))
    return rows


def component_row(name: str, family: str, a: float, b: float, category: float) -> dict:
    row = {
        "name": name,
        "family": family,
        "a": float(a),
        "b": float(b),
        "category": float(category),
        "raw_relation": None,
        "ratio": None,
        "boundary_distance": None,
    }
    if family == "relation":
        row["raw_relation"] = rel_raw(a, b)
        row["boundary_distance"] = rel_boundary_distance(a, b)
    elif family == "ratio_band":
        row["ratio"] = ratio_value(a, b)
        row["boundary_distance"] = ratio_boundary_distance(a, b)
    return row


def labeled_identity(report: dict) -> dict:
    if not report.get("events"):
        return {"components": [], "vector": None, "matches_v43": False}
    event = report["events"][0]
    echo_trace = report["traces"]["echo_only"]
    position_trace = report["traces"]["position_only"]

    components: list[dict] = [
        component_row("event.time_gap", "event_structure", float(event["time_gap"]), 0.0, float(event["time_gap"])),
        component_row("event.graph_distance", "event_structure", float(event["graph_distance"]), 0.0, float(event["graph_distance"])),
        component_row(
            "arrival_order",
            "arrival_order",
            float(event["echo_step"]),
            float(event["position_step"]),
            1.0 if int(event["echo_step"]) < int(event["position_step"]) else (-1.0 if int(event["echo_step"]) > int(event["position_step"]) else 0.0),
        ),
    ]

    echo_groups = v42.step_groups(echo_trace, int(event["echo_step"]), "")
    pos_groups = v42.step_groups(position_trace, int(event["position_step"]), "")
    for key in v43.STEP_GROUPS:
        ev = list(echo_groups[key])
        pv = list(pos_groups[key])
        width = max(len(ev), len(pv))
        ev += [0.0] * (width - len(ev))
        pv += [0.0] * (width - len(pv))
        for i, (a, b) in enumerate(zip(ev, pv)):
            suffix = "" if width == 1 else f"[{i}]"
            components.append(component_row(
                f"{key}{suffix}.relation",
                "relation",
                float(a), float(b),
                v43.relation(float(a), float(b)),
            ))
            components.append(component_row(
                f"{key}{suffix}.ratio_band",
                "ratio_band",
                float(a) + 1e-9, float(b) + 1e-9,
                v43.ratio_band(float(a) + 1e-9, float(b) + 1e-9),
            ))

    components.extend(local_topology_components(event))
    vector = [float(row["category"]) for row in components]
    reference = v43.relative_identity(report, WINDOW)
    matches = reference is not None and len(reference) == len(vector) and all(float(a) == float(b) for a, b in zip(reference, vector))
    return {
        "components": components,
        "vector": vector,
        "matches_v43": matches,
    }


def audit_position(position: str) -> dict:
    runs = v44.condition_runs(position)
    baseline = next(row for row in runs if row["condition"] == "baseline")
    baseline_labeled = labeled_identity(baseline["report"])
    baseline_by_name = {row["name"]: row for row in baseline_labeled["components"]}

    events = []
    flip_counts = Counter()
    family_flip_counts = Counter()
    near_boundary_flips = 0
    all_flips = 0
    condition_flip_counts = {}

    for run in runs:
        labeled = labeled_identity(run["report"]) if run["event_formed"] else {"components": [], "vector": None, "matches_v43": False}
        current_by_name = {row["name"]: row for row in labeled["components"]}
        flips = []
        if run["condition"] != "baseline" and run["event_formed"]:
            for name, base in baseline_by_name.items():
                current = current_by_name.get(name)
                if current is None or float(current["category"]) == float(base["category"]):
                    continue
                all_flips += 1
                flip_counts[name] += 1
                family_flip_counts[base["family"]] += 1
                boundary = None
                if base["boundary_distance"] is not None or current["boundary_distance"] is not None:
                    values = [x for x in [base["boundary_distance"], current["boundary_distance"]] if x is not None]
                    boundary = min(values) if values else None
                near = boundary is not None and boundary <= 0.03
                if near:
                    near_boundary_flips += 1
                flips.append({
                    "component": name,
                    "family": base["family"],
                    "baseline_category": base["category"],
                    "condition_category": current["category"],
                    "baseline_a": base["a"],
                    "baseline_b": base["b"],
                    "condition_a": current["a"],
                    "condition_b": current["b"],
                    "baseline_raw_relation": base["raw_relation"],
                    "condition_raw_relation": current["raw_relation"],
                    "baseline_ratio": base["ratio"],
                    "condition_ratio": current["ratio"],
                    "nearest_boundary_distance": boundary,
                    "near_boundary_0.03": near,
                })
        condition_flip_counts[run["condition"]] = len(flips)
        events.append({
            "condition": run["condition"],
            "echo_scale": run["echo_scale"],
            "position_scale": run["position_scale"],
            "event_formed": run["event_formed"],
            "identity_matches_v43": labeled.get("matches_v43", False),
            "flip_count_vs_baseline": len(flips),
            "flips": flips,
        })

    component_count = len(baseline_labeled["components"])
    comparisons = max(1, len(runs) - 1)
    possible = component_count * comparisons
    top_components = [
        {"component": name, "flip_count": count, "rate_across_conditions": count / comparisons}
        for name, count in flip_counts.most_common()
    ]
    family_summary = [
        {"family": name, "flip_count": count}
        for name, count in family_flip_counts.most_common()
    ]
    near_fraction = 0.0 if all_flips == 0 else near_boundary_flips / all_flips
    return {
        "position": position,
        "component_count": component_count,
        "condition_count": len(runs),
        "possible_component_comparisons": possible,
        "total_flips": all_flips,
        "overall_flip_rate": 0.0 if possible == 0 else all_flips / possible,
        "near_boundary_flip_count": near_boundary_flips,
        "near_boundary_flip_fraction": near_fraction,
        "condition_flip_counts": condition_flip_counts,
        "family_flip_counts": family_summary,
        "top_flipped_components": top_components,
        "baseline_components": baseline_labeled["components"],
        "conditions": events,
        "trend_component_count": 0,
        "trend_note": "WINDOW=1なのでstep-to-step trend成分はRelative Identityに含まれない。",
    }


def observe() -> dict:
    audits = {position: audit_position(position) for position in POSITIONS}
    total_flips = sum(row["total_flips"] for row in audits.values())
    total_near = sum(row["near_boundary_flip_count"] for row in audits.values())
    near_fraction = 0.0 if total_flips == 0 else total_near / total_flips

    aggregate_components = Counter()
    aggregate_families = Counter()
    for row in audits.values():
        for item in row["top_flipped_components"]:
            aggregate_components[item["component"]] += int(item["flip_count"])
        for item in row["family_flip_counts"]:
            aggregate_families[item["family"]] += int(item["flip_count"])

    if total_flips == 0:
        verdict = "no_relative_identity_flips_detected"
        next_step = "recheck_live_separation_logic"
    elif near_fraction >= 0.70:
        verdict = "identity_flips_are_mostly_hard_boundary_crossings"
        next_step = "test_hysteresis_or_soft_stability_bands"
    elif near_fraction >= 0.40:
        verdict = "identity_flips_mix_boundary_crossings_and_real_context_changes"
        next_step = "separate_boundary_sensitive_and_structurally_stable_components"
    else:
        verdict = "identity_flips_reflect_broader_live_context_changes"
        next_step = "reconsider_relative_identity_definition_before_core_integration"

    payload = {
        "experiment": "Core Growth Binding v45",
        "purpose": "Audit every 1-step Relative Context Identity component that flips under v44 live input perturbations and determine whether flips come from hard category boundaries or broader pathway-context changes.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_perturbation_source": "v44 reruns Core propagation with scaled initial activation",
            "audit_only": True,
        },
        "identity_parameters": {
            "window": WINDOW,
            "relation_tolerance": REL_TOL,
            "near_boundary_audit_band": 0.03,
            "trend_included": False,
        },
        "positions": audits,
        "summary": {
            "total_flip_count": total_flips,
            "near_boundary_flip_count": total_near,
            "near_boundary_flip_fraction": near_fraction,
            "most_flipped_components": [
                {"component": name, "flip_count": count}
                for name, count in aggregate_components.most_common(10)
            ],
            "most_flipped_families": [
                {"family": name, "flip_count": count}
                for name, count in aggregate_families.most_common()
            ],
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v45.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v45</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v45</h1><p class="lead">v44のlive perturbationで1Step Relative Context Identityのどの成分が反転したかを監査する。relation/ratio bandの分類境界までの距離も記録し、硬い境界の問題か実際の経路文脈変化かを切り分ける。</p><section class="panel"><div class="controls"><button id="run">Identity Flipを監査</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Flip Audit 生データ</h2><pre id="raw" class="raw">まだ監査していません。</pre></section></main><script>
function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,L=d.positions['左'],C=d.positions['中央'];const top=(s.most_flipped_components||[]).slice(0,4).map(x=>x.component+'('+x.flip_count+')').join(', ')||'なし';const fam=(s.most_flipped_families||[]).map(x=>x.family+'('+x.flip_count+')').join(', ')||'なし';document.getElementById('metrics').innerHTML=`<div class="metric">全Flip数<b>${s.total_flip_count}</b></div><div class="metric">境界近傍Flip率<b class="${s.near_boundary_flip_fraction>=0.7?'good':'warn'}">${f(s.near_boundary_flip_fraction)}</b></div><div class="metric">左 Flip率<b>${f(L.overall_flip_rate)}</b></div><div class="metric">中央 Flip率<b>${f(C.overall_flip_rate)}</b></div><div class="metric">左 境界近傍率<b>${f(L.near_boundary_flip_fraction)}</b></div><div class="metric">中央 境界近傍率<b>${f(C.near_boundary_flip_fraction)}</b></div><div class="metric">最多Flip成分<b class="blue">${top}</b></div><div class="metric">Flip family<b>${fam}</b></div><div class="metric">trend成分<b>0 / 1Stepでは対象外</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v45: http://{HOST}:{PORT}")
    print("Relative Identity Flip Audit / live perturbation audit / no Core changes")
    serve(app, host=HOST, port=PORT)
