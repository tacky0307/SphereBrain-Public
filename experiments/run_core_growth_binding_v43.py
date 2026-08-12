from __future__ import annotations

import json
import math
import random
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v36 as v36
import run_core_growth_binding_v42 as v42

HOST = "127.0.0.1"
START_PORT = 5089
OUT = ROOT / "data" / "core_growth_binding_v43" / "results"
POSITIONS = ["左", "中央", "右"]
REPEATS = 3
WINDOWS = [1, 2, 3]
PERTURBATION = 0.03
RANDOM_SEED = 4301
REL_TOL = 0.08

STEP_GROUPS = [
    "activity_width",
    "accepted_edges",
    "candidate_width",
    "local_top_width",
    "activation_shape",
    "weight_shape",
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


def relation(a: float, b: float, tol: float = REL_TOL) -> float:
    scale = max(abs(a), abs(b), 1e-9)
    diff = (a - b) / scale
    if diff > tol:
        return 1.0
    if diff < -tol:
        return -1.0
    return 0.0


def trend(previous: float, current: float, tol: float = REL_TOL) -> float:
    return relation(current, previous, tol)


def safe_ratio(a: float, b: float) -> float:
    return a / max(abs(b), 1e-9)


def ratio_band(a: float, b: float) -> float:
    ratio = safe_ratio(a, b)
    if ratio < 0.8:
        return -2.0
    if ratio < 0.95:
        return -1.0
    if ratio <= 1.05:
        return 0.0
    if ratio <= 1.25:
        return 1.0
    return 2.0


def event_report(position: str) -> dict:
    return v36.make_events(position)


def step_groups(trace: dict, step_index: int) -> dict[str, list[float]]:
    return v42.step_groups(trace, step_index, "")


def local_topology_relative(event: dict) -> list[float]:
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
    return [
        relation(float(len(e_neighbors)), float(len(p_neighbors))),
        relation(float(len(n_neighbors)), float(mean_ep)),
        1.0 if shared else 0.0,
        1.0 if triangles > 0 else 0.0,
        ratio_band(float(len(e_neighbors) + 1), float(len(p_neighbors) + 1)),
    ]


def perturb_groups(groups: dict[str, list[float]], rng: random.Random) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, values in groups.items():
        if key in {"activation_shape", "weight_shape"}:
            out[key] = [float(v * (1.0 + rng.uniform(-PERTURBATION, PERTURBATION))) for v in values]
        else:
            out[key] = [float(v) for v in values]
    return out


def relative_identity(report: dict, window: int, *, perturb_seed: int | None = None) -> list[float] | None:
    if not report.get("events"):
        return None
    event = report["events"][0]
    echo_trace = report["traces"]["echo_only"]
    position_trace = report["traces"]["position_only"]
    rng = random.Random(perturb_seed) if perturb_seed is not None else None

    identity: list[float] = [
        float(event["time_gap"]),
        float(event["graph_distance"]),
        1.0 if int(event["echo_step"]) < int(event["position_step"]) else (-1.0 if int(event["echo_step"]) > int(event["position_step"]) else 0.0),
    ]

    previous_echo: dict[str, list[float]] | None = None
    previous_pos: dict[str, list[float]] | None = None

    for offset in range(window - 1, -1, -1):
        echo_groups = step_groups(echo_trace, int(event["echo_step"]) - offset)
        pos_groups = step_groups(position_trace, int(event["position_step"]) - offset)
        if rng is not None:
            echo_groups = perturb_groups(echo_groups, rng)
            pos_groups = perturb_groups(pos_groups, rng)

        for key in STEP_GROUPS:
            ev = echo_groups[key]
            pv = pos_groups[key]
            width = max(len(ev), len(pv))
            e_pad = ev + [0.0] * (width - len(ev))
            p_pad = pv + [0.0] * (width - len(pv))
            for a, b in zip(e_pad, p_pad):
                identity.append(relation(float(a), float(b)))
                identity.append(ratio_band(float(a) + 1e-9, float(b) + 1e-9))

            if previous_echo is not None and previous_pos is not None:
                pe = previous_echo[key]
                pp = previous_pos[key]
                for prev, cur in zip(pe, ev):
                    identity.append(trend(float(prev), float(cur)))
                for prev, cur in zip(pp, pv):
                    identity.append(trend(float(prev), float(cur)))

        previous_echo = echo_groups
        previous_pos = pos_groups

    identity.extend(local_topology_relative(event))
    return identity


def absolute_reference(report: dict, window: int, *, perturb_seed: int | None = None) -> list[float] | None:
    groups = v42.context_groups(report, window)
    if groups is None:
        return None
    vector = v42.flatten(groups, list(v42.FEATURE_GROUPS))
    if perturb_seed is None:
        return vector
    return v42.perturb_vector(vector, perturb_seed)


def hamming_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    width = max(len(a), len(b))
    aa = a + [0.0] * (width - len(a))
    bb = b + [0.0] * (width - len(b))
    return sum(1.0 for x, y in zip(aa, bb) if x != y) / width


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def separation_relative(left: list[list[float]], center: list[list[float]]) -> dict:
    within = []
    for rows in (left, center):
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                within.append(hamming_distance(a, b))
    between = [hamming_distance(a, b) for a in left for b in center]
    max_within = max(within, default=0.0)
    min_between = min(between, default=0.0)
    return {
        "max_same_position_distance": max_within,
        "min_left_center_distance": min_between,
        "separation_margin": min_between - max_within,
        "separated": bool(left and center and min_between > max_within),
    }


def separation_absolute(left: list[list[float]], center: list[list[float]]) -> dict:
    all_vectors = left + center
    norm = v42.normalized_vectors(all_vectors)
    left_n = norm[: len(left)]
    center_n = norm[len(left):]
    within = []
    for rows in (left_n, center_n):
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                within.append(euclidean(a, b))
    between = [euclidean(a, b) for a in left_n for b in center_n]
    max_within = max(within, default=0.0)
    min_between = min(between, default=0.0)
    return {
        "max_same_position_distance": max_within,
        "min_left_center_distance": min_between,
        "separation_margin": min_between - max_within,
        "separated": bool(left and center and min_between > max_within),
    }


def repeated_reports(position: str) -> list[dict]:
    return [event_report(position) for _ in range(REPEATS)]


def vectors(reports: dict[str, list[dict]], window: int, mode: str, perturb: bool) -> tuple[list[list[float]], list[list[float]]]:
    left: list[list[float]] = []
    center: list[list[float]] = []
    for position, target, base_seed in (("左", left, RANDOM_SEED), ("中央", center, RANDOM_SEED + 1000)):
        for i, report in enumerate(reports[position]):
            seed = base_seed + i if perturb else None
            if mode == "relative":
                vec = relative_identity(report, window, perturb_seed=seed)
            else:
                vec = absolute_reference(report, window, perturb_seed=seed)
            if vec is not None:
                target.append(vec)
                if perturb:
                    # 無摂動と摂動の両方を同一位置内へ入れて、本当にIdentityが保たれるか測る。
                    base = relative_identity(report, window) if mode == "relative" else absolute_reference(report, window)
                    if base is not None:
                        target.append(base)
    return left, center


def analyze_window(reports: dict[str, list[dict]], window: int) -> dict:
    rel_left, rel_center = vectors(reports, window, "relative", False)
    rel_p_left, rel_p_center = vectors(reports, window, "relative", True)
    abs_left, abs_center = vectors(reports, window, "absolute", False)
    abs_p_left, abs_p_center = vectors(reports, window, "absolute", True)

    relative_full = separation_relative(rel_left, rel_center)
    relative_perturbed = separation_relative(rel_p_left, rel_p_center)
    absolute_full = separation_absolute(abs_left, abs_center)
    absolute_perturbed = separation_absolute(abs_p_left, abs_p_center)

    return {
        "window": window,
        "relative_vector_length": len(rel_left[0]) if rel_left else 0,
        "absolute_vector_length": len(abs_left[0]) if abs_left else 0,
        "relative_full": relative_full,
        "relative_perturbed": relative_perturbed,
        "absolute_reference_full": absolute_full,
        "absolute_reference_perturbed": absolute_perturbed,
        "relative_improves_perturbation_margin": relative_perturbed["separation_margin"] > absolute_perturbed["separation_margin"],
    }


def observe() -> dict:
    reports = {position: repeated_reports(position) for position in POSITIONS}
    analyses = {str(window): analyze_window(reports, window) for window in WINDOWS}

    robust_relative = [
        int(key) for key, row in analyses.items()
        if row["relative_full"]["separated"] and row["relative_perturbed"]["separated"]
    ]
    preferred = min(robust_relative) if robust_relative else None

    if preferred is not None:
        verdict = "relative_context_identity_robust_under_perturbation"
        recommended = {
            "window": preferred,
            "identity": "relative pathway context",
        }
    elif any(row["relative_full"]["separated"] for row in analyses.values()):
        verdict = "relative_context_identity_separates_but_not_yet_robust"
        recommended = None
    else:
        verdict = "relative_context_identity_does_not_separate"
        recommended = None

    repeatability = {
        position: [int(report["event_count"]) for report in rows]
        for position, rows in reports.items()
    }

    payload = {
        "experiment": "Core Growth Binding v43",
        "purpose": "Replace absolute Context Identity values with relative pathway relations, trends, dominance bands, and topology relations, then test whether left/center identity survives small feature-level perturbations.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "perturbation_scope": "Only continuous activation/weight-shape context features are perturbed by ±3%; Core propagation is not rerun.",
        },
        "relative_identity_definition": {
            "uses": [
                "E-vs-position ordering",
                "coarse E/position dominance band",
                "step-to-step increase/stable/decrease",
                "relative arrival order",
                "relative local topology",
            ],
            "excludes": [
                "Node IDs",
                "position labels",
                "absolute Event signal",
                "TTL",
                "event age",
                "raw continuous context values as identity",
            ],
            "relation_tolerance": REL_TOL,
        },
        "windows": analyses,
        "repeatability": repeatability,
        "summary": {
            "left_repeatable": len(set(repeatability["左"])) == 1 and repeatability["左"][0] > 0,
            "center_repeatable": len(set(repeatability["中央"])) == 1 and repeatability["中央"][0] > 0,
            "right_absent": all(value == 0 for value in repeatability["右"]),
            "robust_relative_windows": robust_relative,
            "preferred_window": preferred,
            "recommended_context_identity": recommended,
            "overall_verdict": verdict,
            "all_routes_unchanged": all(
                bool(report["route_unchanged_by_event_logging"])
                for rows in reports.values() for report in rows
            ),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v43.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v43</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v43</h1><p class="lead">絶対Context値をIdentityにせず、E/位置の優劣・粗い比率帯・Step間の増減・到着順・局所トポロジーの相対関係だけで左/中央を識別できるかを検証する。</p><section class="panel"><div class="controls"><button id="run">Relative Identityを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Relative Context 生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,w=d.windows||{};const cards=Object.entries(w).map(([k,r])=>`<div class="metric">${k}Step Relative margin<b class="${r.relative_full.separated?'good':'warn'}">${f(r.relative_full.separation_margin)}</b></div><div class="metric">${k}Step Relative摂動margin<b class="${r.relative_perturbed.separated?'good':'warn'}">${f(r.relative_perturbed.separation_margin)}</b></div><div class="metric">${k}Step Absolute摂動margin<b>${f(r.absolute_reference_perturbed.separation_margin)}</b></div><div class="metric">${k}Step 相対表現改善<b>${yn(r.relative_improves_perturbation_margin)}</b></div>`).join('');document.getElementById('metrics').innerHTML=`<div class="metric">左 再現性<b>${yn(s.left_repeatable)}</b></div><div class="metric">中央 再現性<b>${yn(s.center_repeatable)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_absent)}</b></div><div class="metric">頑健window<b class="blue">${s.robust_relative_windows.length?s.robust_relative_windows.join(', '):'なし'}</b></div><div class="metric">推奨window<b class="blue">${s.preferred_window||'まだなし'}</b></div>${cards}<div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">経路不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v43: http://{HOST}:{PORT}")
    print("Relative Context Identity / diagnostic only / no Core changes")
    serve(app, host=HOST, port=PORT)
