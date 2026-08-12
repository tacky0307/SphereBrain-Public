from __future__ import annotations

import copy
import json
import math
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v36 as v36

HOST = "127.0.0.1"
START_PORT = 5086
OUT = ROOT / "data" / "core_growth_binding_v40" / "results"
POSITIONS = ["左", "中央", "右"]
SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
TTL_VALUES = [1, 2, 3]
RATIO_BANDS = [(0.0, 0.90, 0.0), (0.90, 1.10, 1.0), (1.10, float("inf"), 2.0)]


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


def first_event(position: str) -> dict | None:
    report = v36.make_events(position)
    return dict(report["events"][0]) if report["events"] else None


def ratio_band(echo: float, position: float) -> float:
    ratio = 0.0 if position <= 0.0 else echo / position
    for lower, upper, code in RATIO_BANDS:
        if lower <= ratio < upper:
            return code
    return 2.0


def identity_features(raw: dict, mode: str) -> list[float]:
    echo = float(raw["echo_signal"])
    position = float(raw["position_signal"])
    temporal = [
        float(raw["time_gap"]),
        float(raw["created_step"] - min(raw["echo_step"], raw["position_step"])),
    ]
    spatial = [float(raw["graph_distance"])]
    if mode == "structure_only":
        return temporal + spatial
    if mode == "structure_plus_ratio_band":
        return temporal + spatial + [ratio_band(echo, position)]
    if mode == "structure_plus_continuous_ratio":
        ratio = 0.0 if position <= 0.0 else echo / position
        return temporal + spatial + [ratio]
    if mode == "legacy_mixed":
        total = echo + position
        ratio = 0.0 if position <= 0.0 else echo / position
        return temporal + spatial + [echo / 0.18, position / 0.18, total / 0.18, ratio, float(raw["ttl"]) / 2.0]
    raise ValueError(mode)


def state_features(raw: dict) -> dict:
    echo = float(raw["echo_signal"])
    position = float(raw["position_signal"])
    return {
        "echo_signal": echo,
        "position_signal": position,
        "signal_total": echo + position,
        "ttl": int(raw["ttl"]),
        "age_at_creation": 0,
    }


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def transformed(base: dict, *, echo_scale: float = 1.0, position_scale: float = 1.0, ttl: int | None = None) -> dict:
    row = copy.deepcopy(base)
    row["echo_signal"] = float(row["echo_signal"]) * echo_scale
    row["position_signal"] = float(row["position_signal"]) * position_scale
    if ttl is not None:
        row["ttl"] = int(ttl)
    return row


def samples(base: dict) -> list[dict]:
    rows = []
    for echo_scale in SCALES:
        rows.append(transformed(base, echo_scale=echo_scale))
    for position_scale in SCALES:
        rows.append(transformed(base, position_scale=position_scale))
    for common_scale in SCALES:
        rows.append(transformed(base, echo_scale=common_scale, position_scale=common_scale))
    for ttl in TTL_VALUES:
        rows.append(transformed(base, ttl=ttl))
    # 重複除去
    unique = {}
    for row in rows:
        key = (round(float(row["echo_signal"]), 12), round(float(row["position_signal"]), 12), int(row["ttl"]))
        unique[key] = row
    return list(unique.values())


def max_within(rows: list[dict], mode: str) -> float:
    vectors = [identity_features(row, mode) for row in rows]
    values = [distance(a, b) for i, a in enumerate(vectors) for b in vectors[i + 1:]]
    return max(values, default=0.0)


def min_between(left: list[dict], center: list[dict], mode: str) -> float:
    return min(
        (distance(identity_features(a, mode), identity_features(b, mode)) for a in left for b in center),
        default=0.0,
    )


def analyze_mode(mode: str, left_base: dict, center_base: dict) -> dict:
    left = samples(left_base)
    center = samples(center_base)
    left_within = max_within(left, mode)
    center_within = max_within(center, mode)
    within = max(left_within, center_within)
    between = min_between(left, center, mode)
    margin = between - within
    return {
        "mode": mode,
        "left_sample_count": len(left),
        "center_sample_count": len(center),
        "max_left_within_distance": left_within,
        "max_center_within_distance": center_within,
        "max_same_position_distance": within,
        "min_left_center_distance": between,
        "separation_margin": margin,
        "separated": margin > 0.0,
        "left_base_identity": identity_features(left_base, mode),
        "center_base_identity": identity_features(center_base, mode),
    }


def observe() -> dict:
    bases = {position: first_event(position) for position in POSITIONS}
    left, center = bases["左"], bases["中央"]
    modes = [
        "structure_only",
        "structure_plus_ratio_band",
        "structure_plus_continuous_ratio",
        "legacy_mixed",
    ]
    if left is None or center is None:
        analyses = {}
        verdict = "required_left_or_center_event_missing"
        recommended_identity = None
    else:
        analyses = {mode: analyze_mode(mode, left, center) for mode in modes}
        if analyses["structure_only"]["separated"]:
            verdict = "structure_only_identity_robust"
            recommended_identity = "temporal + spatial"
        elif analyses["structure_plus_ratio_band"]["separated"]:
            verdict = "ratio_band_adds_stable_specificity"
            recommended_identity = "temporal + spatial + coarse ratio band"
        else:
            verdict = "identity_not_robust_without_continuous_signal"
            recommended_identity = None

    repeatability = {
        position: [v36.make_events(position)["event_count"] for _ in range(3)]
        for position in POSITIONS
    }
    payload = {
        "experiment": "Core Growth Binding v40",
        "purpose": "Separate Contact Event Identity from transient Event State and test whether temporal/spatial structure or a coarse signal-ratio band yields a robust identity suitable for future Core integration.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "analysis_scope": "Event-feature transformations only; Core is not rerun under altered signals or TTL.",
        },
        "identity_definition": {
            "structure_only": ["time_gap", "relative_contact_timing", "graph_distance"],
            "structure_plus_ratio_band": ["time_gap", "relative_contact_timing", "graph_distance", "coarse E/position dominance band"],
            "continuous_ratio_reference": ["time_gap", "relative_contact_timing", "graph_distance", "continuous E/position ratio"],
            "excluded_from_identity": ["absolute signal strength", "TTL", "event age", "Node IDs", "position labels"],
        },
        "state_definition": ["echo signal", "position signal", "signal total", "TTL", "event age"],
        "baseline_events": bases,
        "baseline_states": {
            position: None if event is None else state_features(event)
            for position, event in bases.items()
        },
        "analyses": analyses,
        "repeatability": repeatability,
        "summary": {
            "left_repeatable": len(set(repeatability["左"])) == 1 and repeatability["左"][0] > 0,
            "center_repeatable": len(set(repeatability["中央"])) == 1 and repeatability["中央"][0] > 0,
            "right_absent": all(value == 0 for value in repeatability["右"]),
            "structure_only_separated": analyses.get("structure_only", {}).get("separated", False),
            "ratio_band_separated": analyses.get("structure_plus_ratio_band", {}).get("separated", False),
            "continuous_ratio_separated": analyses.get("structure_plus_continuous_ratio", {}).get("separated", False),
            "legacy_mixed_separated": analyses.get("legacy_mixed", {}).get("separated", False),
            "recommended_core_identity": recommended_identity,
            "overall_verdict": verdict,
            "all_routes_unchanged": True,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v40.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v40</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v40</h1><p class="lead">Contact Eventを、関係として残すIdentityと、その瞬間だけ変わるStateへ分離する。時間・空間構造だけ、粗いsignal比率帯、連続比率、旧混合表現を比較する。</p><section class="panel"><div class="controls"><button id="run">Identityを分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Identity / State 生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
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
    print(f"Core Growth Binding v40: http://{HOST}:{PORT}")
    print("Contact Event Identity decomposition / no Core changes")
    serve(app, host=HOST, port=PORT)
