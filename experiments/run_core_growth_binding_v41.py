from __future__ import annotations

import json
import math
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
START_PORT = 5087
OUT = ROOT / "data" / "core_growth_binding_v41" / "results"
POSITIONS = ["左", "中央", "右"]
REPEATS = 3
WINDOW = 2


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


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def stats(values: list[float]) -> list[float]:
    if not values:
        return [0.0, 0.0, 0.0]
    arr = np.asarray(values, dtype=float)
    maximum = float(np.max(np.abs(arr)))
    normalized = arr / maximum if maximum > 0 else arr
    return [float(np.mean(normalized)), float(np.std(normalized)), float(np.max(normalized))]


def step_shape(trace: dict, step_index: int) -> list[float]:
    if step_index < 0 or step_index >= len(trace.get("steps", [])):
        return [0.0] * 8
    step = trace["steps"][step_index]
    active = [float(x) for x in step.get("active_values", {}).values()]
    accepted = step.get("accepted_edges", [])
    records = step.get("records", [])
    weights = [float(row.get("weight", 0.0)) for row in records]
    local_top = sum(1 for row in records if row.get("local_top"))
    return [
        float(len(step.get("active_sources", []))),
        float(len(accepted)),
        float(len(records)),
        float(local_top),
        *stats(active),
        float(np.mean(weights)) if weights else 0.0,
    ]


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
        float(event["time_gap"]),
        float(event["graph_distance"]),
        float(event["echo_step"] <= event["position_step"]),
    ]


def context_identity(report: dict, mode: str) -> list[float] | None:
    if not report.get("events"):
        return None
    event = report["events"][0]
    echo_trace = report["traces"]["echo_only"]
    position_trace = report["traces"]["position_only"]
    combined_trace = report["traces"]["combined"]
    created = int(event["created_step"])

    base = [
        float(event["time_gap"]),
        float(event["graph_distance"]),
        float(event["echo_step"] <= event["position_step"]),
    ]
    if mode == "event_only_structure":
        return base

    pre = []
    for offset in range(WINDOW - 1, -1, -1):
        pre.extend(step_shape(echo_trace, int(event["echo_step"]) - offset))
        pre.extend(step_shape(position_trace, int(event["position_step"]) - offset))
    topology = local_topology(event)
    if mode == "pre_context_shape":
        return base + pre + topology

    post = []
    for offset in range(0, WINDOW):
        post.extend(step_shape(combined_trace, created + offset))
    return base + pre + topology + post


def event_state(report: dict) -> dict | None:
    if not report.get("events"):
        return None
    event = report["events"][0]
    return {
        "echo_signal": float(event["echo_signal"]),
        "position_signal": float(event["position_signal"]),
        "signal_total": float(event["echo_signal"]) + float(event["position_signal"]),
        "ttl": int(event["ttl"]),
        "age": 0,
    }


def repeated_reports(position: str) -> list[dict]:
    return [v36.make_events(position) for _ in range(REPEATS)]


def analyze_mode(mode: str, reports: dict[str, list[dict]]) -> dict:
    vectors = {
        position: [context_identity(report, mode) for report in rows]
        for position, rows in reports.items()
    }
    left = [v for v in vectors["左"] if v is not None]
    center = [v for v in vectors["中央"] if v is not None]

    within_values = []
    for rows in (left, center):
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                within_values.append(distance(a, b))
    between_values = [distance(a, b) for a in left for b in center]
    max_within = max(within_values, default=0.0)
    min_between = min(between_values, default=0.0)
    margin = min_between - max_within
    return {
        "mode": mode,
        "vector_length": len(left[0]) if left else 0,
        "max_same_position_distance": max_within,
        "min_left_center_distance": min_between,
        "separation_margin": margin,
        "separated": bool(left and center and margin > 0.0),
        "left_vectors": left,
        "center_vectors": center,
    }


def observe() -> dict:
    reports = {position: repeated_reports(position) for position in POSITIONS}
    modes = ["event_only_structure", "pre_context_shape", "pre_post_context_identity"]
    analyses = {mode: analyze_mode(mode, reports) for mode in modes}

    if analyses["event_only_structure"]["separated"]:
        verdict = "event_structure_alone_context_identity_robust"
        recommended = "event structure"
    elif analyses["pre_context_shape"]["separated"]:
        verdict = "pre_contact_context_adds_robust_identity"
        recommended = "event structure + pre-contact pathway context"
    elif analyses["pre_post_context_identity"]["separated"]:
        verdict = "pre_post_context_adds_robust_identity"
        recommended = "event structure + pre/post pathway context"
    else:
        verdict = "context_identity_not_yet_robust"
        recommended = None

    repeatability = {
        position: [int(report["event_count"]) for report in rows]
        for position, rows in reports.items()
    }
    representative = {
        position: rows[0] for position, rows in reports.items()
    }
    payload = {
        "experiment": "Core Growth Binding v41",
        "purpose": "Test whether Node-ID-free pathway context before and after a Cross-Lineage Contact Event yields a robust relation identity without absolute signal values or TTL.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "identity_excludes": ["Node IDs", "position labels", "absolute event signals", "TTL", "event age"],
        },
        "identity_features": {
            "event_only_structure": ["time gap", "graph distance", "relative arrival order"],
            "pre_context_shape": ["event structure", "two-step lineage activity shape", "branch width", "accepted-edge count", "normalized activation distribution", "local topology"],
            "pre_post_context_identity": ["pre-context shape", "two-step combined post-contact candidate shape"],
        },
        "analyses": analyses,
        "representative_states": {
            position: event_state(report) for position, report in representative.items()
        },
        "repeatability": repeatability,
        "summary": {
            "left_repeatable": len(set(repeatability["左"])) == 1 and repeatability["左"][0] > 0,
            "center_repeatable": len(set(repeatability["中央"])) == 1 and repeatability["中央"][0] > 0,
            "right_absent": all(value == 0 for value in repeatability["右"]),
            "event_only_separated": analyses["event_only_structure"]["separated"],
            "pre_context_separated": analyses["pre_context_shape"]["separated"],
            "pre_post_context_separated": analyses["pre_post_context_identity"]["separated"],
            "recommended_core_identity": recommended,
            "overall_verdict": verdict,
            "all_routes_unchanged": all(
                bool(report["route_unchanged_by_event_logging"])
                for rows in reports.values() for report in rows
            ),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v41.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v41</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v41</h1><p class="lead">Contact Event単体ではなく、接触前2Stepの両系統経路と接触後2Stepの局所候補構造を、Node ID・位置名・signal絶対値・TTLなしでIdentity候補として比較する。</p><section class="panel"><div class="controls"><button id="run">Context Identityを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Context生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,a=d.analyses||{};const cards=Object.entries(a).map(([k,r])=>`<div class="metric">${k} 同位置最大<b>${f(r.max_same_position_distance)}</b></div><div class="metric">${k} 異位置最小<b>${f(r.min_left_center_distance)}</b></div><div class="metric">${k} margin<b class="${r.separated?'good':'warn'}">${f(r.separation_margin)}</b></div><div class="metric">${k} 分離<b>${yn(r.separated)}</b></div>`).join('');document.getElementById('metrics').innerHTML=`<div class="metric">左 再現性<b>${yn(s.left_repeatable)}</b></div><div class="metric">中央 再現性<b>${yn(s.center_repeatable)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_absent)}</b></div><div class="metric">推奨Core Identity<b class="blue">${s.recommended_core_identity||'まだなし'}</b></div>${cards}<div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">経路不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v41: http://{HOST}:{PORT}")
    print("Contact Event Context Identity / diagnostic only / no Core changes")
    serve(app, host=HOST, port=PORT)
