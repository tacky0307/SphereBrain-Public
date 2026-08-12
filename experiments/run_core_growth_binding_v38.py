from __future__ import annotations

import json
import math
import random
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
START_PORT = 5084
OUT = ROOT / "data" / "core_growth_binding_v38" / "results"
POSITIONS = ["左", "中央", "右"]
REPEATS = 5
RANDOM_SEED = 3801
NOISE_LEVELS = [0.01, 0.03, 0.05]
ECHO_SCALES = [0.90, 1.00, 1.10]
TTL_VALUES = [1, 2, 3]
ROUND_DIGITS = [4, 3, 2]


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


def vector(raw: dict, *, include_ttl: bool = False) -> list[float]:
    echo = float(raw["echo_signal"])
    pos = float(raw["position_signal"])
    total = echo + pos
    ratio = 0.0 if pos == 0.0 else echo / pos
    values = [
        float(raw["time_gap"]),
        float(raw["graph_distance"]),
        float(raw["created_step"] - min(raw["echo_step"], raw["position_step"])),
        echo / 0.18,
        pos / 0.18,
        total / 0.18,
        ratio,
    ]
    if include_ttl:
        values.append(float(raw["ttl"]) / 2.0)
    return values


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rounded_signature(raw: dict, digits: int) -> tuple:
    return (
        int(raw["time_gap"]),
        int(raw["graph_distance"]),
        int(raw["created_step"] - min(raw["echo_step"], raw["position_step"])),
        round(float(raw["echo_signal"]), digits),
        round(float(raw["position_signal"]), digits),
        round(float(raw["echo_signal"]) + float(raw["position_signal"]), digits),
        round(float(raw["echo_signal"]) / max(float(raw["position_signal"]), 1e-12), digits),
    )


def perturb(raw: dict, rng: random.Random, noise: float, echo_scale: float, ttl: int) -> dict:
    row = dict(raw)
    row["echo_signal"] = max(0.0, float(raw["echo_signal"]) * echo_scale * (1.0 + rng.uniform(-noise, noise)))
    row["position_signal"] = max(0.0, float(raw["position_signal"]) * (1.0 + rng.uniform(-noise, noise)))
    row["ttl"] = int(ttl)
    return row


def repeatability(position: str) -> dict:
    runs = [v36.make_events(position) for _ in range(REPEATS)]
    signatures = []
    for run in runs:
        if not run["events"]:
            signatures.append(None)
        else:
            signatures.append(rounded_signature(run["events"][0], 9))
    return {
        "event_counts": [int(run["event_count"]) for run in runs],
        "signatures": [None if s is None else list(s) for s in signatures],
        "repeatable": all(s == signatures[0] for s in signatures[1:]),
        "all_expire": all((not run["events"]) or bool(run["expires_after_ttl"]) for run in runs),
        "all_routes_unchanged": all(bool(run["route_unchanged_by_event_logging"]) for run in runs),
    }


def robustness_samples(base: dict, position: str) -> list[dict]:
    rng = random.Random(RANDOM_SEED + sum(ord(ch) for ch in position))
    samples = []
    for noise in [0.0] + NOISE_LEVELS:
        for scale in ECHO_SCALES:
            for ttl in TTL_VALUES:
                repetitions = 1 if noise == 0.0 else 5
                for index in range(repetitions):
                    raw = perturb(base, rng, noise, scale, ttl)
                    samples.append({
                        "position": position,
                        "noise": noise,
                        "echo_scale": scale,
                        "ttl": ttl,
                        "sample": index,
                        "raw": raw,
                        "vector_without_ttl": vector(raw, include_ttl=False),
                        "vector_with_ttl": vector(raw, include_ttl=True),
                    })
    return samples


def pair_distances(rows: list[dict]) -> list[float]:
    values = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            values.append(distance(left["vector_without_ttl"], right["vector_without_ttl"]))
    return values


def observe() -> dict:
    repeats = {position: repeatability(position) for position in POSITIONS}
    bases = {position: first_event(position) for position in POSITIONS}
    left_base, center_base = bases["左"], bases["中央"]

    if left_base is None or center_base is None:
        verdict = "required_left_or_center_event_missing"
        robustness = {}
    else:
        left_samples = robustness_samples(left_base, "左")
        center_samples = robustness_samples(center_base, "中央")
        left_within = pair_distances(left_samples)
        center_within = pair_distances(center_samples)
        between = [
            distance(a["vector_without_ttl"], b["vector_without_ttl"])
            for a in left_samples for b in center_samples
        ]
        max_within = max(left_within + center_within, default=0.0)
        mean_within = sum(left_within + center_within) / max(1, len(left_within) + len(center_within))
        min_between = min(between, default=0.0)
        mean_between = sum(between) / max(1, len(between))
        margin = min_between - max_within

        rounding = {}
        for digits in ROUND_DIGITS:
            left_sig = rounded_signature(left_base, digits)
            center_sig = rounded_signature(center_base, digits)
            rounding[str(digits)] = {
                "left": list(left_sig),
                "center": list(center_sig),
                "separated": left_sig != center_sig,
            }

        ttl_invariant = all(
            vector({**left_base, "ttl": ttl}, include_ttl=False) == vector(left_base, include_ttl=False)
            and vector({**center_base, "ttl": ttl}, include_ttl=False) == vector(center_base, include_ttl=False)
            for ttl in TTL_VALUES
        )

        robust = margin > 0.0 and all(row["separated"] for row in rounding.values())
        verdict = (
            "event_structure_robust_with_between_position_margin"
            if robust
            else "event_structure_separation_not_robust_under_perturbation"
        )
        robustness = {
            "sample_scope": "signature-level perturbation; Core propagation is not rerun with injected noise or altered echo strength",
            "left_sample_count": len(left_samples),
            "center_sample_count": len(center_samples),
            "max_same_position_distance": max_within,
            "mean_same_position_distance": mean_within,
            "min_left_center_distance": min_between,
            "mean_left_center_distance": mean_between,
            "separation_margin": margin,
            "same_position_max_less_than_between_min": max_within < min_between,
            "rounding_tests": rounding,
            "ttl_invariant_when_ttl_excluded_from_identity": ttl_invariant,
            "noise_levels": NOISE_LEVELS,
            "echo_scales": ECHO_SCALES,
            "ttl_values": TTL_VALUES,
            "left_samples": left_samples,
            "center_samples": center_samples,
        }

    payload = {
        "experiment": "Core Growth Binding v38",
        "purpose": "Test whether left and center Contact Event structures remain separated under rounding, small signature perturbations, modest echo-signal scaling, and TTL changes.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
        },
        "repeatability": repeats,
        "baseline_events": bases,
        "robustness": robustness,
        "summary": {
            "left_repeatable": repeats["左"]["repeatable"],
            "center_repeatable": repeats["中央"]["repeatable"],
            "right_absent": all(count == 0 for count in repeats["右"]["event_counts"]),
            "all_events_expire": all(row["all_expire"] for row in repeats.values()),
            "all_routes_unchanged": all(row["all_routes_unchanged"] for row in repeats.values()),
            "overall_verdict": verdict,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v38.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v38</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v38</h1><p class="lead">Contact Eventの左・中央分離が、丸め・微小摂動・E信号倍率・TTL変更に耐えるかを距離で検証する。摂動試験はEvent署名レベルで行い、Core伝播は変更しない。</p><section class="panel"><div class="controls"><button id="run">Event頑健性を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Event生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,r=d.robustness||{},round=r.rounding_tests||{};document.getElementById('metrics').innerHTML=`<div class="metric">左 再現性<b class="${s.left_repeatable?'good':'warn'}">${yn(s.left_repeatable)}</b></div><div class="metric">中央 再現性<b class="${s.center_repeatable?'good':'warn'}">${yn(s.center_repeatable)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_absent)}</b></div><div class="metric">同位置 最大距離<b>${f(r.max_same_position_distance)}</b></div><div class="metric">異位置 最小距離<b>${f(r.min_left_center_distance)}</b></div><div class="metric">分離margin<b class="${(r.separation_margin||0)>0?'good':'warn'}">${f(r.separation_margin)}</b></div><div class="metric">距離条件成立<b>${yn(r.same_position_max_less_than_between_min)}</b></div><div class="metric">小数4桁分離<b>${yn(round['4']&&round['4'].separated)}</b></div><div class="metric">小数3桁分離<b>${yn(round['3']&&round['3'].separated)}</b></div><div class="metric">小数2桁分離<b>${yn(round['2']&&round['2'].separated)}</b></div><div class="metric">TTL非依存<b>${yn(r.ttl_invariant_when_ttl_excluded_from_identity)}</b></div><div class="metric">経路不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v38: http://{HOST}:{PORT}")
    print("Contact Event robustness / signature-level perturbation / no Core changes")
    serve(app, host=HOST, port=PORT)
