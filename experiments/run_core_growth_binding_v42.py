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

HOST = "127.0.0.1"
START_PORT = 5088
OUT = ROOT / "data" / "core_growth_binding_v42" / "results"
POSITIONS = ["左", "中央", "右"]
REPEATS = 3
WINDOWS = [1, 2, 3]
PERTURBATION = 0.03
RANDOM_SEED = 4201

FEATURE_GROUPS = [
    "event_structure",
    "activity_width",
    "accepted_edges",
    "candidate_width",
    "local_top_width",
    "activation_shape",
    "weight_shape",
    "local_topology",
    "arrival_order",
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


def stats(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    arr = np.asarray(values, dtype=float)
    maximum = float(np.max(np.abs(arr)))
    normalized = arr / maximum if maximum > 0 else arr
    return [float(np.mean(normalized)), float(np.std(normalized)), float(np.max(normalized))]


def step_groups(trace: dict, step_index: int, prefix: str) -> dict[str, list[float]]:
    if step_index < 0 or step_index >= len(trace.get("steps", [])):
        return {
            "activity_width": [0.0],
            "accepted_edges": [0.0],
            "candidate_width": [0.0],
            "local_top_width": [0.0],
            "activation_shape": [0.0, 0.0, 0.0],
            "weight_shape": [0.0, 0.0, 0.0],
        }
    step = trace["steps"][step_index]
    active = [float(x) for x in step.get("active_values", {}).values()]
    accepted = step.get("accepted_edges", [])
    records = step.get("records", [])
    weights = [float(row.get("weight", 0.0)) for row in records]
    local_top = sum(1 for row in records if row.get("local_top"))
    return {
        "activity_width": [float(len(step.get("active_sources", [])))],
        "accepted_edges": [float(len(accepted))],
        "candidate_width": [float(len(records))],
        "local_top_width": [float(local_top)],
        "activation_shape": stats(active),
        "weight_shape": stats(weights),
    }


def local_topology(event: dict) -> list[float]:
    brain = v3.base.CORE
    e = int(event["echo_node"])
    p = int(event["position_node"])
    n = int(event["shared_neighbor"])
    e_neighbors = set(int(x) for x in np.flatnonzero(brain.adjacency[e]))
    p_neighbors = set(int(x) for x in np.flatnonzero(brain.adjacency[p]))
    n_neighbors = set(int(x) for x in np.flatnonzero(brain.adjacency[n]))
    shared_ep = e_neighbors & p_neighbors
    triangles = sum(1 for x in n_neighbors if x in e_neighbors or x in p_neighbors)
    return [
        float(len(e_neighbors)),
        float(len(p_neighbors)),
        float(len(n_neighbors)),
        float(len(shared_ep)),
        float(triangles),
    ]


def context_groups(report: dict, window: int) -> dict[str, list[float]] | None:
    if not report.get("events"):
        return None
    event = report["events"][0]
    echo_trace = report["traces"]["echo_only"]
    position_trace = report["traces"]["position_only"]

    groups: dict[str, list[float]] = {name: [] for name in FEATURE_GROUPS}
    groups["event_structure"] = [float(event["time_gap"]), float(event["graph_distance"])]
    groups["arrival_order"] = [float(event["echo_step"] <= event["position_step"])]
    groups["local_topology"] = local_topology(event)

    for offset in range(window - 1, -1, -1):
        for trace, step_index in (
            (echo_trace, int(event["echo_step"]) - offset),
            (position_trace, int(event["position_step"]) - offset),
        ):
            row = step_groups(trace, step_index, "")
            for key, values in row.items():
                groups[key].extend(values)
    return groups


def flatten(groups: dict[str, list[float]], included: list[str]) -> list[float]:
    out: list[float] = []
    for key in included:
        out.extend(groups[key])
    return out


def normalized_vectors(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        return []
    arr = np.asarray(vectors, dtype=float)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std[std < 1e-12] = 1.0
    norm = (arr - mean) / std
    scale = math.sqrt(max(1, arr.shape[1]))
    return [[float(x / scale) for x in row] for row in norm]


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def separation(left: list[list[float]], center: list[list[float]]) -> dict:
    all_vectors = left + center
    norm = normalized_vectors(all_vectors)
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
        "separated": bool(left_n and center_n and min_between > max_within),
    }


def perturb_vector(vector: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    return [float(value * (1.0 + rng.uniform(-PERTURBATION, PERTURBATION))) for value in vector]


def repeated_reports(position: str) -> list[dict]:
    return [v36.make_events(position) for _ in range(REPEATS)]


def vectors_for(reports: dict[str, list[dict]], window: int, included: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    left = []
    center = []
    for position, target in (("左", left), ("中央", center)):
        for report in reports[position]:
            groups = context_groups(report, window)
            if groups is not None:
                target.append(flatten(groups, included))
    return left, center


def robustness_with_perturbation(left: list[list[float]], center: list[list[float]]) -> dict:
    left_p = []
    center_p = []
    for i, vector in enumerate(left):
        left_p.extend([vector, perturb_vector(vector, RANDOM_SEED + i)])
    for i, vector in enumerate(center):
        center_p.extend([vector, perturb_vector(vector, RANDOM_SEED + 100 + i)])
    return separation(left_p, center_p)


def analyze_window(reports: dict[str, list[dict]], window: int) -> dict:
    included = list(FEATURE_GROUPS)
    left, center = vectors_for(reports, window, included)
    full = separation(left, center)
    perturbed = robustness_with_perturbation(left, center)

    ablations = {}
    for removed in FEATURE_GROUPS:
        kept = [key for key in FEATURE_GROUPS if key != removed]
        l, c = vectors_for(reports, window, kept)
        ablations[removed] = separation(l, c)

    single_groups = {}
    for only in FEATURE_GROUPS:
        l, c = vectors_for(reports, window, [only])
        single_groups[only] = separation(l, c)

    critical = [name for name, row in ablations.items() if not row["separated"]]
    sufficient = [name for name, row in single_groups.items() if row["separated"]]
    return {
        "window": window,
        "vector_length": len(left[0]) if left else 0,
        "full_identity": full,
        "perturbed_identity": perturbed,
        "ablations": ablations,
        "single_group_tests": single_groups,
        "critical_groups": critical,
        "individually_sufficient_groups": sufficient,
    }


def observe() -> dict:
    reports = {position: repeated_reports(position) for position in POSITIONS}
    analyses = {str(window): analyze_window(reports, window) for window in WINDOWS}

    robust_windows = [
        int(key) for key, row in analyses.items()
        if row["full_identity"]["separated"] and row["perturbed_identity"]["separated"]
    ]
    preferred_window = min(robust_windows) if robust_windows else None
    preferred = analyses.get(str(preferred_window)) if preferred_window is not None else None

    if preferred is None:
        verdict = "context_identity_not_robust_under_ablation_or_perturbation"
        recommended = None
    elif preferred["critical_groups"]:
        verdict = "context_identity_robust_but_depends_on_critical_feature_groups"
        recommended = {
            "window": preferred_window,
            "critical_groups": preferred["critical_groups"],
        }
    else:
        verdict = "context_identity_robust_and_redundantly_supported"
        recommended = {
            "window": preferred_window,
            "critical_groups": [],
        }

    repeatability = {
        position: [int(report["event_count"]) for report in rows]
        for position, rows in reports.items()
    }
    payload = {
        "experiment": "Core Growth Binding v42",
        "purpose": "Test pre-contact Context Identity across 1/2/3-step windows, feature-group ablations, normalized distance, and small signature-level perturbations.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "perturbation_scope": "Context-feature vectors only; Core propagation is not rerun with altered activation or threshold.",
        },
        "feature_groups": FEATURE_GROUPS,
        "distance_normalization": "per-feature z-score over left+center samples, then Euclidean distance divided by sqrt(vector length)",
        "windows": analyses,
        "repeatability": repeatability,
        "summary": {
            "left_repeatable": len(set(repeatability["左"])) == 1 and repeatability["左"][0] > 0,
            "center_repeatable": len(set(repeatability["中央"])) == 1 and repeatability["中央"][0] > 0,
            "right_absent": all(value == 0 for value in repeatability["右"]),
            "robust_windows": robust_windows,
            "preferred_window": preferred_window,
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
    (OUT / "latest_binding_v42.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v42</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v42</h1><p class="lead">接触前Context Identityを、1/2/3Step窓、特徴除去、特徴単独、正規化距離、±3%署名摂動で監査する。Core伝播・Edge・weightは変更しない。</p><section class="panel"><div class="controls"><button id="run">Context Identityを監査</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Robustness / Ablation 生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,w=d.windows||{};const cards=Object.entries(w).map(([k,r])=>`<div class="metric">${k}Step full margin<b class="${r.full_identity.separated?'good':'warn'}">${f(r.full_identity.separation_margin)}</b></div><div class="metric">${k}Step 摂動margin<b class="${r.perturbed_identity.separated?'good':'warn'}">${f(r.perturbed_identity.separation_margin)}</b></div><div class="metric">${k}Step critical groups<b>${r.critical_groups.length?r.critical_groups.join(', '):'なし'}</b></div><div class="metric">${k}Step 単独十分特徴<b>${r.individually_sufficient_groups.length?r.individually_sufficient_groups.join(', '):'なし'}</b></div>`).join('');document.getElementById('metrics').innerHTML=`<div class="metric">左 再現性<b>${yn(s.left_repeatable)}</b></div><div class="metric">中央 再現性<b>${yn(s.center_repeatable)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_absent)}</b></div><div class="metric">頑健window<b class="blue">${s.robust_windows.length?s.robust_windows.join(', '):'なし'}</b></div><div class="metric">推奨window<b class="blue">${s.preferred_window===null?'まだなし':s.preferred_window+'Step'}</b></div>${cards}<div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">経路不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v42: http://{HOST}:{PORT}")
    print("Context Identity robustness and ablation / no Core changes")
    serve(app, host=HOST, port=PORT)
