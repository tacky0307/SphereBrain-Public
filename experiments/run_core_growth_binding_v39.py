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
START_PORT = 5085
OUT = ROOT / "data" / "core_growth_binding_v39" / "results"
POSITIONS = ["左", "中央", "右"]
SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
ROUND_DIGITS = [4, 3, 2]
TTL_VALUES = [1, 2, 3]


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


def features(raw: dict) -> dict[str, list[float]]:
    echo = float(raw["echo_signal"])
    pos = float(raw["position_signal"])
    total = echo + pos
    ratio = 0.0 if pos == 0.0 else echo / pos
    return {
        "temporal": [
            float(raw["time_gap"]),
            float(raw["created_step"] - min(raw["echo_step"], raw["position_step"])),
        ],
        "spatial": [float(raw["graph_distance"])],
        "absolute_signal": [echo / 0.18, pos / 0.18, total / 0.18],
        "signal_ratio": [ratio],
        "lifetime": [float(raw["ttl"]) / 2.0],
    }


def flatten(groups: dict[str, list[float]], included: list[str] | None = None) -> list[float]:
    keys = included or ["temporal", "spatial", "absolute_signal", "signal_ratio", "lifetime"]
    result: list[float] = []
    for key in keys:
        result.extend(groups[key])
    return result


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def component_distances(a: dict, b: dict) -> dict[str, float]:
    af, bf = features(a), features(b)
    return {key: distance(af[key], bf[key]) for key in af}


def transformed(base: dict, kind: str, value: float | int) -> dict:
    row = copy.deepcopy(base)
    echo = float(row["echo_signal"])
    pos = float(row["position_signal"])
    if kind == "echo_scale":
        row["echo_signal"] = echo * float(value)
    elif kind == "position_scale":
        row["position_signal"] = pos * float(value)
    elif kind == "common_scale":
        row["echo_signal"] = echo * float(value)
        row["position_signal"] = pos * float(value)
    elif kind == "ratio_shift":
        # 合計量をおおむね保ちつつ、E/位置比だけを変える。
        total = echo + pos
        ratio_factor = float(value)
        new_echo = echo * ratio_factor
        new_pos = max(1e-12, total - new_echo)
        row["echo_signal"] = new_echo
        row["position_signal"] = new_pos
    elif kind == "round":
        digits = int(value)
        row["echo_signal"] = round(echo, digits)
        row["position_signal"] = round(pos, digits)
    elif kind == "ttl":
        row["ttl"] = int(value)
    return row


def scenario_samples(base: dict, kind: str) -> list[dict]:
    if kind == "baseline":
        return [copy.deepcopy(base)]
    if kind in {"echo_scale", "position_scale", "common_scale", "ratio_shift"}:
        return [transformed(base, kind, scale) for scale in SCALES]
    if kind == "round":
        return [transformed(base, kind, digits) for digits in ROUND_DIGITS]
    if kind == "ttl":
        return [transformed(base, kind, ttl) for ttl in TTL_VALUES]
    raise ValueError(kind)


def max_within(samples: list[dict], included: list[str] | None = None) -> float:
    values = []
    vectors = [flatten(features(row), included) for row in samples]
    for i, a in enumerate(vectors):
        for b in vectors[i + 1:]:
            values.append(distance(a, b))
    return max(values, default=0.0)


def min_between(left: list[dict], center: list[dict], included: list[str] | None = None) -> float:
    return min(
        (
            distance(flatten(features(a), included), flatten(features(b), included))
            for a in left for b in center
        ),
        default=0.0,
    )


def analyze_scenario(kind: str, left_base: dict, center_base: dict) -> dict:
    left = scenario_samples(left_base, kind)
    center = scenario_samples(center_base, kind)
    within_left = max_within(left)
    within_center = max_within(center)
    within = max(within_left, within_center)
    between = min_between(left, center)
    margin = between - within

    component_results = {}
    for component in ["temporal", "spatial", "absolute_signal", "signal_ratio", "lifetime"]:
        w = max(max_within(left, [component]), max_within(center, [component]))
        b = min_between(left, center, [component])
        component_results[component] = {
            "max_same_position_distance": w,
            "min_left_center_distance": b,
            "separation_margin": b - w,
        }

    return {
        "kind": kind,
        "left_sample_count": len(left),
        "center_sample_count": len(center),
        "max_left_within_distance": within_left,
        "max_center_within_distance": within_center,
        "max_same_position_distance": within,
        "min_left_center_distance": between,
        "separation_margin": margin,
        "separated": margin > 0.0,
        "component_attribution": component_results,
        "left_samples": left,
        "center_samples": center,
    }


def observe() -> dict:
    bases = {position: first_event(position) for position in POSITIONS}
    left, center = bases["左"], bases["中央"]
    if left is None or center is None:
        scenarios = {}
        verdict = "required_left_or_center_event_missing"
        weakest = None
        strongest_component = None
    else:
        kinds = ["baseline", "round", "echo_scale", "position_scale", "common_scale", "ratio_shift", "ttl"]
        scenarios = {kind: analyze_scenario(kind, left, center) for kind in kinds}
        perturbed = [row for kind, row in scenarios.items() if kind != "baseline"]
        weakest_row = min(perturbed, key=lambda row: row["separation_margin"])
        weakest = weakest_row["kind"]

        # ベースラインで左右差を最も大きく支える特徴成分。
        baseline_components = scenarios["baseline"]["component_attribution"]
        strongest_component = max(
            baseline_components,
            key=lambda key: baseline_components[key]["min_left_center_distance"],
        )
        all_separated = all(row["separated"] for row in scenarios.values())
        verdict = (
            "event_specificity_robust_and_attributed"
            if all_separated
            else "event_specificity_fragile_driver_identified"
        )

    repeatability = {
        position: [v36.make_events(position)["event_count"] for _ in range(3)]
        for position in POSITIONS
    }
    payload = {
        "experiment": "Core Growth Binding v39",
        "purpose": "Attribute Contact Event robustness failures by isolating rounding, echo scaling, position scaling, common scaling, ratio shifts, and TTL changes, and by decomposing distance into temporal, spatial, absolute-signal, ratio, and lifetime components.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "attribution_scope": "Event-feature transformations only; Core is not rerun under altered signals or TTL.",
        },
        "baseline_events": bases,
        "repeatability": repeatability,
        "scenarios": scenarios,
        "summary": {
            "left_repeatable": len(set(repeatability["左"])) == 1 and repeatability["左"][0] > 0,
            "center_repeatable": len(set(repeatability["中央"])) == 1 and repeatability["中央"][0] > 0,
            "right_absent": all(value == 0 for value in repeatability["右"]),
            "weakest_perturbation": weakest,
            "strongest_baseline_separation_component": strongest_component,
            "overall_verdict": verdict,
            "all_routes_unchanged": True,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v39.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v39</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v39</h1><p class="lead">Contact Event分離を壊した要因を、丸め・E信号倍率・位置信号倍率・共通倍率・比率変化・TTLへ分離し、時間・空間・絶対signal・比率・寿命の各成分へ帰属する。</p><section class="panel"><div class="controls"><button id="run">頑健性要因を分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Attribution生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,sc=d.scenarios||{};const cards=Object.entries(sc).map(([k,r])=>`<div class="metric">${k} 同位置最大<b>${f(r.max_same_position_distance)}</b></div><div class="metric">${k} 異位置最小<b>${f(r.min_left_center_distance)}</b></div><div class="metric">${k} margin<b class="${r.separated?'good':'warn'}">${f(r.separation_margin)}</b></div><div class="metric">${k} 分離<b>${yn(r.separated)}</b></div>`).join('');document.getElementById('metrics').innerHTML=`<div class="metric">左 再現性<b>${yn(s.left_repeatable)}</b></div><div class="metric">中央 再現性<b>${yn(s.center_repeatable)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_absent)}</b></div><div class="metric">最弱摂動<b class="warn">${s.weakest_perturbation||'なし'}</b></div><div class="metric">主要分離成分<b class="blue">${s.strongest_baseline_separation_component||'なし'}</b></div>${cards}<div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v39: http://{HOST}:{PORT}")
    print("Contact Event robustness attribution / feature-level diagnostic / no Core changes")
    serve(app, host=HOST, port=PORT)
