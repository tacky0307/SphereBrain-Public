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
import run_core_growth_binding_v31 as v31

HOST = "127.0.0.1"
START_PORT = 5078
OUT = ROOT / "data" / "core_growth_binding_v32" / "results"
TARGET_POSITIONS = ["左", "中央"]
EPSILON = 1e-9


def choose_port(start: int) -> int:
    for port in range(start, start + 40):
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


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-15:
        return None
    return float(numerator / denominator)


def feasibility(multiplier: float | None) -> str:
    if multiplier is None or not math.isfinite(multiplier):
        return "unreachable"
    if multiplier <= 1.05:
        return "already_or_nearly_ready"
    if multiplier <= 1.50:
        return "small_experience_strengthening"
    if multiplier <= 2.00:
        return "moderate_strengthening"
    if multiplier <= 3.00:
        return "large_strengthening"
    return "impractical_for_simple_weight_learning"


def rank_report(component: dict, neighbor: int) -> dict:
    brain = v3.base.CORE
    source = int(component["source_node"])
    activation = float(component["source_activation"])
    decay = float(component["signal_decay"])
    neighbors = [int(x) for x in np.flatnonzero(brain.adjacency[source])]

    rows = []
    for target in neighbors:
        weight = float(brain.weights[source, target])
        signal = activation * weight * decay
        rows.append({
            "target": target,
            "weight": weight,
            "signal": signal,
        })
    rows.sort(key=lambda row: (-row["signal"], row["target"]))

    target_row = next((row for row in rows if row["target"] == int(neighbor)), None)
    rank = None if target_row is None else rows.index(target_row) + 1
    branch_count = min(int(brain.max_branches), len(rows))
    cutoff_row = None if branch_count == 0 else rows[branch_count - 1]
    cutoff_signal = 0.0 if cutoff_row is None else float(cutoff_row["signal"])
    target_signal = 0.0 if target_row is None else float(target_row["signal"])

    if target_row is None:
        local_top_multiplier = None
    elif rank is not None and rank <= branch_count:
        local_top_multiplier = 1.0
    else:
        local_top_multiplier = safe_ratio(cutoff_signal + EPSILON, target_signal)

    threshold_multiplier = component.get("required_edge_weight_multiplier")
    factors = [x for x in (local_top_multiplier, threshold_multiplier) if x is not None]
    minimum_multiplier = max(factors) if factors else None

    current_weight = float(component["edge_weight"])
    required_weight_local_top = None if local_top_multiplier is None else current_weight * local_top_multiplier
    required_weight_threshold = component.get("required_edge_weight_at_current_activation")
    required_weight_both = None if minimum_multiplier is None else current_weight * minimum_multiplier

    stronger_count = 0 if rank is None else rank - 1
    local_top_gap = max(0.0, cutoff_signal - target_signal)

    return {
        "source_node": source,
        "target_neighbor": int(neighbor),
        "neighbor_count": len(rows),
        "max_branches": int(brain.max_branches),
        "local_top_cutoff_rank": branch_count,
        "current_rank": rank,
        "stronger_edge_count": stronger_count,
        "currently_local_top": bool(rank is not None and rank <= branch_count),
        "target_signal": target_signal,
        "target_weight": current_weight,
        "local_top_cutoff_target": None if cutoff_row is None else int(cutoff_row["target"]),
        "local_top_cutoff_signal": cutoff_signal,
        "local_top_signal_gap": float(local_top_gap),
        "local_top_required_multiplier": local_top_multiplier,
        "threshold_required_multiplier": threshold_multiplier,
        "minimum_multiplier_for_local_top_and_threshold": minimum_multiplier,
        "required_weight_for_local_top": required_weight_local_top,
        "required_weight_for_threshold": required_weight_threshold,
        "required_weight_for_both": required_weight_both,
        "feasibility": feasibility(minimum_multiplier),
        "top_edges": rows[: min(10, len(rows))],
    }


def diagnose_position(position: str) -> dict:
    base = v31.diagnose_position(position)
    if not base.get("candidate_found"):
        return {
            "position": position,
            "candidate_found": False,
            "base": base,
        }

    neighbor = int(base["neighbor"])
    echo_rank = rank_report(base["echo"], neighbor)
    position_rank = rank_report(base["position_component"], neighbor)

    minimums = [
        x for x in (
            echo_rank["minimum_multiplier_for_local_top_and_threshold"],
            position_rank["minimum_multiplier_for_local_top_and_threshold"],
        ) if x is not None
    ]
    pair_minimum = max(minimums) if minimums else None

    return {
        "position": position,
        "candidate_found": True,
        "neighbor": neighbor,
        "echo": echo_rank,
        "position_component": position_rank,
        "pair": {
            "both_currently_local_top": bool(echo_rank["currently_local_top"] and position_rank["currently_local_top"]),
            "minimum_shared_multiplier_if_scaled_together": pair_minimum,
            "shared_feasibility": feasibility(pair_minimum),
            "recommended_next_action": (
                "test_binding_limited_learning" if pair_minimum is not None and pair_minimum <= 1.5
                else "do_not_strengthen_blindly_reconsider_contact_or_integration_rule"
            ),
        },
        "base": base,
    }


def observe() -> dict:
    reports = {position: diagnose_position(position) for position in TARGET_POSITIONS}
    valid = [row for row in reports.values() if row.get("candidate_found")]
    practical = [
        row["position"] for row in valid
        if row["pair"]["minimum_shared_multiplier_if_scaled_together"] is not None
        and row["pair"]["minimum_shared_multiplier_if_scaled_together"] <= 1.5
    ]

    payload = {
        "experiment": "Core Growth Binding v32",
        "purpose": "Measure the current rank of each shared-neighbor contact edge and the minimum weight multiplier required to enter local top and pass threshold.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "diagnostic_only": True,
            "puzzle_specific_adjustment": False,
        },
        "positions": reports,
        "summary": {
            "positions_with_small_strengthening_path": practical,
            "binding_limited_learning_recommended": bool(practical),
            "decision_rule": "try limited learning only when both local-top and threshold conditions need <=1.5x",
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v32.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v32</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v32</h1><p class="lead">接触Edgeの現在順位、local top最下位との差、threshold到達倍率を測り、少量の経験強化で機能的接触へ育つかを診断する。</p><section class="panel"><div class="controls"><button id="run">接触Edge順位を診断</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function yn(v){return v?'YES':'NO'}function cards(position,label,r){return `<div class="metric">${position} ${label} Edge<b class="blue">${r.source_node} → ${r.target_neighbor}</b></div><div class="metric">${position} ${label} 現在順位<b>${r.current_rank} / ${r.neighbor_count}</b></div><div class="metric">${position} ${label} local top範囲<b>上位 ${r.local_top_cutoff_rank}</b></div><div class="metric">${position} ${label} 現在local top<b>${yn(r.currently_local_top)}</b></div><div class="metric">${position} ${label} cutoff signal<b>${f(r.local_top_cutoff_signal)}</b></div><div class="metric">${position} ${label} 接触signal<b>${f(r.target_signal)}</b></div><div class="metric">${position} ${label} local top必要倍率<b>${f(r.local_top_required_multiplier)}</b></div><div class="metric">${position} ${label} threshold必要倍率<b>${f(r.threshold_required_multiplier)}</b></div><div class="metric">${position} ${label} 両条件最小倍率<b>${f(r.minimum_multiplier_for_local_top_and_threshold)}</b></div><div class="metric">${position} ${label} 判定<b>${r.feasibility}</b></div>`}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),rows=Object.values(d.positions);document.getElementById('metrics').innerHTML=rows.map(r=>{if(!r.candidate_found)return `<div class="metric">${r.position}<b class="warn">候補なし</b></div>`;return `<div class="metric">${r.position} 共通neighbor<b class="blue">Node ${r.neighbor}</b></div>`+cards(r.position,'E側',r.echo)+cards(r.position,'位置側',r.position_component)+`<div class="metric">${r.position} 共通倍率<b>${f(r.pair.minimum_shared_multiplier_if_scaled_together)}</b></div><div class="metric">${r.position} 次の判断<b>${r.pair.recommended_next_action}</b></div>`}).join('')+`<div class="metric">限定学習推奨<b class="${d.summary.binding_limited_learning_recommended?'good':'warn'}">${yn(d.summary.binding_limited_learning_recommended)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v32: http://{HOST}:{PORT}")
    print("Contact-edge rank diagnostics / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
