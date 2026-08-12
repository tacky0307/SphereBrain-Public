from __future__ import annotations

import json
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
START_PORT = 5083
OUT = ROOT / "data" / "core_growth_binding_v37" / "results"
POSITIONS = list(v3.POSITIONS)
REPEATS = 3


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


def rounded(value: float) -> float:
    return round(float(value), 9)


def pure_structural_signature(raw: dict) -> tuple:
    """位置名とNode IDを除外したContact Eventの純構造署名。"""
    echo_signal = float(raw["echo_signal"])
    position_signal = float(raw["position_signal"])
    total = echo_signal + position_signal
    ratio = 0.0 if position_signal == 0 else echo_signal / position_signal
    return (
        "cross_lineage_contact",
        int(raw["time_gap"]),
        int(raw["graph_distance"]),
        int(raw["created_step"] - min(raw["echo_step"], raw["position_step"])),
        rounded(echo_signal),
        rounded(position_signal),
        rounded(total),
        rounded(ratio),
        int(raw["ttl"]),
    )


def labeled_signature(raw: dict) -> tuple:
    return (str(raw["position"]),) + pure_structural_signature(raw)


def run_repeats(position: str) -> dict:
    runs = [v36.make_events(position) for _ in range(REPEATS)]
    pure = []
    labeled = []
    event_counts = []
    for run in runs:
        event_counts.append(int(run["event_count"]))
        pure.append(sorted(pure_structural_signature(event) for event in run["events"]))
        labeled.append(sorted(labeled_signature(event) for event in run["events"]))
    return {
        "position": position,
        "event_counts": event_counts,
        "event_formed_each_run": [count > 0 for count in event_counts],
        "pure_signatures_by_run": [[list(x) for x in rows] for rows in pure],
        "labeled_signatures_by_run": [[list(x) for x in rows] for rows in labeled],
        "repeatable_pure_signature": all(rows == pure[0] for rows in pure[1:]) if pure else True,
        "repeatable_labeled_signature": all(rows == labeled[0] for rows in labeled[1:]) if labeled else True,
        "expires_after_ttl_each_run": [bool(run["expires_after_ttl"]) for run in runs],
        "interaction_edge_counts": [int(run["interaction_edge_count"]) for run in runs],
        "route_unchanged_each_run": [bool(run["route_unchanged_by_event_logging"]) for run in runs],
        "representative": runs[0],
    }


def signature_set(report: dict, key: str) -> set[tuple]:
    rows = report[key][0] if report[key] else []
    return {tuple(row) for row in rows}


def observe() -> dict:
    reports = {position: run_repeats(position) for position in POSITIONS}
    left = reports["左"]
    center = reports["中央"]
    right = reports["右"]

    left_pure = signature_set(left, "pure_signatures_by_run")
    center_pure = signature_set(center, "pure_signatures_by_run")
    left_labeled = signature_set(left, "labeled_signatures_by_run")
    center_labeled = signature_set(center, "labeled_signatures_by_run")

    pure_equal = left_pure == center_pure
    labeled_equal = left_labeled == center_labeled
    pure_union = left_pure | center_pure
    pure_overlap = 1.0 if not pure_union else len(left_pure & center_pure) / len(pure_union)

    # 実行順を入れ替えても各位置の署名が変わらないかを確認。
    swap_center = v36.make_events("中央")
    swap_left = v36.make_events("左")
    swap_center_pure = sorted(pure_structural_signature(event) for event in swap_center["events"])
    swap_left_pure = sorted(pure_structural_signature(event) for event in swap_left["events"])
    order_invariant = (
        swap_left_pure == [tuple(x) for x in left["pure_signatures_by_run"][0]]
        and swap_center_pure == [tuple(x) for x in center["pure_signatures_by_run"][0]]
    )

    left_center_repeatable = left["repeatable_pure_signature"] and center["repeatable_pure_signature"]
    right_absent = all(count == 0 for count in right["event_counts"])

    if left_center_repeatable and not pure_equal and right_absent:
        verdict = "position_specific_structure_detected_without_labels"
    elif left_center_repeatable and pure_equal and right_absent:
        verdict = "events_repeatable_but_left_center_not_structurally_distinct"
    elif not left_center_repeatable:
        verdict = "event_signature_not_repeatable"
    else:
        verdict = "specificity_inconclusive"

    payload = {
        "experiment": "Core Growth Binding v37",
        "purpose": "Test whether Cross-Lineage Contact Events are repeatable and position-specific after removing position labels and Node IDs from their structural signatures.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "structural_assist_used": False,
            "core_file_modified": False,
        },
        "signature_definition": {
            "excludes": ["position label", "echo Node ID", "position Node ID", "shared-neighbor Node ID"],
            "includes": ["time gap", "graph distance", "relative contact timing", "echo signal", "position signal", "signal sum", "signal ratio", "TTL"],
            "note": "This tests event-feature specificity, not full graph-isomorphism invariance under an actual Node permutation.",
        },
        "positions": reports,
        "comparison": {
            "left_center_pure_signature_equal": pure_equal,
            "left_center_labeled_signature_equal": labeled_equal,
            "left_center_pure_signature_jaccard": pure_overlap,
            "left_center_structurally_distinct_without_labels": not pure_equal,
            "execution_order_invariant": order_invariant,
            "same_position_repeatable": left_center_repeatable,
            "right_event_absent_all_runs": right_absent,
        },
        "summary": {
            "left_event_count": left["event_counts"][0],
            "center_event_count": center["event_counts"][0],
            "right_event_count": right["event_counts"][0],
            "left_repeatable": left["repeatable_pure_signature"],
            "center_repeatable": center["repeatable_pure_signature"],
            "right_absent": right_absent,
            "pure_signature_separates_left_center": not pure_equal,
            "overall_verdict": verdict,
            "all_routes_unchanged": all(
                all(report["route_unchanged_each_run"]) for report in reports.values()
            ),
            "all_events_expire": all(
                all(report["expires_after_ttl_each_run"])
                for report in reports.values()
                if any(report["event_formed_each_run"])
            ),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v37.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v37</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v37</h1><p class="lead">Contact Eventを反復し、位置名とNode IDを除いた純構造署名で左・中央を区別できるかを検証する。Eventは経路・activation・weightへ影響しない。</p><section class="panel"><div class="controls"><button id="run">Event固有性を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Event生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,c=d.comparison;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event数<b>${s.left_event_count}</b></div><div class="metric">中央 Event数<b>${s.center_event_count}</b></div><div class="metric">右 Event数<b>${s.right_event_count}</b></div><div class="metric">左 再現性<b class="${s.left_repeatable?'good':'warn'}">${yn(s.left_repeatable)}</b></div><div class="metric">中央 再現性<b class="${s.center_repeatable?'good':'warn'}">${yn(s.center_repeatable)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_absent)}</b></div><div class="metric">純構造署名一致<b>${yn(c.left_center_pure_signature_equal)}</b></div><div class="metric">純構造Jaccard<b>${f(c.left_center_pure_signature_jaccard)}</b></div><div class="metric">ラベルなし位置分離<b class="${s.pure_signature_separates_left_center?'good':'warn'}">${yn(s.pure_signature_separates_left_center)}</b></div><div class="metric">実行順不変<b>${yn(c.execution_order_invariant)}</b></div><div class="metric">TTL後消滅<b>${yn(s.all_events_expire)}</b></div><div class="metric">経路不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v37: http://{HOST}:{PORT}")
    print("Contact Event specificity / repeated diagnostic / no Core changes")
    serve(app, host=HOST, port=PORT)
