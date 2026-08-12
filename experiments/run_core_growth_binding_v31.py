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
import run_core_growth_binding_v30 as v30

HOST = "127.0.0.1"
START_PORT = 5077
OUT = ROOT / "data" / "core_growth_binding_v31" / "results"
TARGET_POSITIONS = ["左", "中央"]
THRESHOLD = 0.18


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


def component_report(
    *,
    lineage: str,
    source_node: int,
    source_step: int,
    weight: float,
    signal: float,
    local_top: bool,
    passes_threshold: bool,
    trace: dict,
    initial_reference: float,
) -> dict:
    source_activation = v30.active_value(trace, source_step, source_node)
    decay = float(v3.base.CORE.signal_decay)
    after_weight = source_activation * weight
    reconstructed_signal = after_weight * decay

    history_loss = max(0.0, initial_reference - source_activation)
    weight_loss = max(0.0, source_activation - after_weight)
    decay_loss = max(0.0, after_weight - reconstructed_signal)
    shortage = max(0.0, THRESHOLD - signal)

    transmission_factor = weight * decay
    required_source_activation = None if transmission_factor <= 0 else THRESHOLD / transmission_factor
    required_weight = None if source_activation * decay <= 0 else THRESHOLD / (source_activation * decay)

    factors = {
        "history_retention": safe_ratio(source_activation, initial_reference),
        "edge_weight": weight,
        "signal_decay": decay,
    }
    finite_factors = {k: v for k, v in factors.items() if v is not None}
    strongest_attenuator = min(finite_factors, key=finite_factors.get) if finite_factors else "unknown"

    return {
        "lineage": lineage,
        "source_node": int(source_node),
        "source_step": int(source_step),
        "initial_reference_activation": float(initial_reference),
        "source_activation": float(source_activation),
        "history_retention_ratio": safe_ratio(source_activation, initial_reference),
        "history_loss": float(history_loss),
        "edge_weight": float(weight),
        "after_weight": float(after_weight),
        "weight_loss": float(weight_loss),
        "signal_decay": decay,
        "decay_loss": float(decay_loss),
        "reported_signal": float(signal),
        "reconstructed_signal": float(reconstructed_signal),
        "reconstruction_error": float(abs(signal - reconstructed_signal)),
        "threshold": THRESHOLD,
        "threshold_shortage": float(shortage),
        "threshold_fraction": safe_ratio(signal, THRESHOLD),
        "local_top": bool(local_top),
        "passes_threshold": bool(passes_threshold),
        "required_source_activation_at_current_weight": required_source_activation,
        "required_source_activation_multiplier": None if required_source_activation is None else safe_ratio(required_source_activation, source_activation),
        "required_edge_weight_at_current_activation": required_weight,
        "required_edge_weight_multiplier": None if required_weight is None else safe_ratio(required_weight, weight),
        "strongest_attenuator": strongest_attenuator,
        "attenuation_factors": factors,
    }


def diagnose_position(position: str) -> dict:
    base = v30.candidate_rows(position)
    best = base.get("best_candidate")
    if best is None:
        return {
            "position": position,
            "candidate_found": False,
            "base": base,
        }

    echo_trace = base["traces"]["echo_only"]
    position_trace = base["traces"]["position_only"]

    echo = component_report(
        lineage="E_residual",
        source_node=int(best["echo_node"]),
        source_step=int(best["echo_step"]),
        weight=float(best["echo_weight"]),
        signal=float(best["echo_signal"]),
        local_top=bool(best["echo_local_top"]),
        passes_threshold=bool(best["echo_passes_threshold"]),
        trace=echo_trace,
        initial_reference=float(v3.ECHO_STRENGTH),
    )
    position_component = component_report(
        lineage=f"position:{position}",
        source_node=int(best["position_node"]),
        source_step=int(best["position_step"]),
        weight=float(best["position_weight"]),
        signal=float(best["position_signal"]),
        local_top=bool(best["position_local_top"]),
        passes_threshold=bool(best["position_passes_threshold"]),
        trace=position_trace,
        initial_reference=1.0,
    )

    components = [echo, position_component]
    weakest_signal_component = min(components, key=lambda row: row["reported_signal"])["lineage"]
    larger_shortage_component = max(components, key=lambda row: row["threshold_shortage"])["lineage"]

    return {
        "position": position,
        "candidate_found": True,
        "neighbor": int(best["neighbor"]),
        "simultaneous": bool(best["simultaneous"]),
        "diagnostic_sum": float(best["sum_signal_diagnostic"]),
        "sum_threshold_shortage": float(max(0.0, THRESHOLD - best["sum_signal_diagnostic"])),
        "echo": echo,
        "position_component": position_component,
        "comparison": {
            "weakest_signal_component": weakest_signal_component,
            "larger_threshold_shortage_component": larger_shortage_component,
            "both_local_top": bool(echo["local_top"] and position_component["local_top"]),
            "both_subthreshold": bool(not echo["passes_threshold"] and not position_component["passes_threshold"]),
            "simple_sum_still_subthreshold": bool(best["sum_signal_diagnostic"] < THRESHOLD),
            "dominant_attenuator_by_lineage": {
                echo["lineage"]: echo["strongest_attenuator"],
                position_component["lineage"]: position_component["strongest_attenuator"],
            },
        },
        "base_candidate": best,
    }


def observe() -> dict:
    reports = {position: diagnose_position(position) for position in TARGET_POSITIONS}
    valid = [row for row in reports.values() if row.get("candidate_found")]

    payload = {
        "experiment": "Core Growth Binding v31",
        "purpose": "Decompose each shared-neighbor signal into source activation, edge weight, and signal decay, and identify the dominant attenuation before thresholding.",
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
            "candidate_positions": [row["position"] for row in valid],
            "all_both_local_top": bool(valid) and all(row["comparison"]["both_local_top"] for row in valid),
            "all_both_subthreshold": bool(valid) and all(row["comparison"]["both_subthreshold"] for row in valid),
            "all_simple_sums_subthreshold": bool(valid) and all(row["comparison"]["simple_sum_still_subthreshold"] for row in valid),
            "signal_formula": "source_activation * edge_weight * signal_decay",
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v31.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v31</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v31</h1><p class="lead">共通neighborへ届くsignalを、活動履歴・source activation・Edge weight・signal decayへ分解し、threshold前に最も強く減衰させた要素を診断する。</p><section class="panel"><div class="controls"><button id="run">signalを分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function yn(v){return v?'YES':'NO'}function componentCards(position,label,c){return `<div class="metric">${position} ${label} source<b class="blue">Node ${c.source_node} / Step ${c.source_step}</b></div><div class="metric">${position} ${label} source activation<b>${f(c.source_activation)}</b></div><div class="metric">${position} ${label} 履歴保持率<b>${f(c.history_retention_ratio)}</b></div><div class="metric">${position} ${label} Edge weight<b>${f(c.edge_weight)}</b></div><div class="metric">${position} ${label} weight後<b>${f(c.after_weight)}</b></div><div class="metric">${position} ${label} signal decay<b>${f(c.signal_decay)}</b></div><div class="metric">${position} ${label} 最終signal<b>${f(c.reported_signal)}</b></div><div class="metric">${position} ${label} threshold不足<b class="warn">${f(c.threshold_shortage)}</b></div><div class="metric">${position} ${label} local top<b>${yn(c.local_top)}</b></div><div class="metric">${position} ${label} 主減衰要因<b>${c.strongest_attenuator}</b></div><div class="metric">${position} ${label} 必要activation倍率<b>${f(c.required_source_activation_multiplier)}</b></div><div class="metric">${position} ${label} 必要weight倍率<b>${f(c.required_edge_weight_multiplier)}</b></div>`}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),rows=Object.values(d.positions);document.getElementById('metrics').innerHTML=rows.map(r=>{if(!r.candidate_found)return `<div class="metric">${r.position}<b class="warn">候補なし</b></div>`;return `<div class="metric">${r.position} 共通neighbor<b class="blue">Node ${r.neighbor}</b></div><div class="metric">${r.position} 診断用sum<b>${f(r.diagnostic_sum)}</b></div><div class="metric">${r.position} sum不足<b class="warn">${f(r.sum_threshold_shortage)}</b></div><div class="metric">${r.position} 両方local top<b>${yn(r.comparison.both_local_top)}</b></div>`+componentCards(r.position,'E側',r.echo)+componentCards(r.position,'位置側',r.position_component)}).join('')+`<div class="metric">全候補 両方local top<b>${yn(d.summary.all_both_local_top)}</b></div><div class="metric">全候補 両方subthreshold<b>${yn(d.summary.all_both_subthreshold)}</b></div><div class="metric">単純sumも全て不足<b>${yn(d.summary.all_simple_sums_subthreshold)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v31: http://{HOST}:{PORT}")
    print("Signal component decomposition / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
