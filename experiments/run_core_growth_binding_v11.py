from __future__ import annotations

import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v8 as v8
import run_core_growth_binding_v10 as v10

HOST = "127.0.0.1"
START_PORT = 5046
OUT = ROOT / "data" / "core_growth_binding_v11" / "results"
POSITIONS = v3.POSITIONS


def choose_port(start: int) -> int:
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("利用可能なローカルポートが見つかりません。")


PORT = choose_port(START_PORT)


def edge_tuple(edge):
    return tuple(sorted((int(edge[0]), int(edge[1]))))


def activation_for_source(record: dict | None) -> float | None:
    return None if record is None else float(record.get("source_activation", 0.0))


def chain_coverage(entity: str, position: str) -> dict:
    specific = sorted(v8.binding_specific_edges(entity, position))
    bound = v10.run_detailed(entity, position, threshold=0.18, binding=True)
    single = v10.run_detailed(entity, None, threshold=0.16, binding=False)

    reports = []
    for edge in specific:
        chain = v10.ordered_binding_chain(bound, edge)
        reproduced = 0
        first_stop = None
        rows = []
        for item in chain:
            current_edge = edge_tuple(item["edge"])
            single_record = v10.best_record(single, current_edge)
            status = v10.classify_record(single_record)
            if status == "selected" and first_stop is None:
                reproduced += 1
            elif first_stop is None:
                first_stop = {
                    "binding_step": int(item["step"]),
                    "edge": list(current_edge),
                    "status": status,
                    "single_record": single_record,
                }
            rows.append({
                "binding_step": int(item["step"]),
                "edge": list(current_edge),
                "status": status,
                "single_record": single_record,
            })
        reports.append({
            "specific_edge": list(edge),
            "binding_chain_edge_count": len(chain),
            "reproduced_edge_count": reproduced,
            "reproduction_ratio": 0.0 if not chain else reproduced / len(chain),
            "first_stop": first_stop,
            "chain_rows": rows,
        })

    total = sum(r["binding_chain_edge_count"] for r in reports)
    reproduced_total = sum(r["reproduced_edge_count"] for r in reports)
    return {
        "entity": entity,
        "position": position,
        "reports": reports,
        "total_chain_edges": total,
        "total_reproduced_edges": reproduced_total,
        "overall_reproduction_ratio": 0.0 if total == 0 else reproduced_total / total,
        "single_0_16_node_count": len(single["activated_nodes"]),
        "single_0_16_edge_count": len(single["traversed_edges"]),
    }


def entry_activation_compare(entity: str, position: str) -> dict:
    specific = sorted(v8.binding_specific_edges(entity, position))
    bound = v10.run_detailed(entity, position, threshold=0.18, binding=True)
    single_018 = v10.run_detailed(entity, None, threshold=0.18, binding=False)
    single_016 = v10.run_detailed(entity, None, threshold=0.16, binding=False)

    rows = []
    for edge in specific:
        chain = v10.ordered_binding_chain(bound, edge)
        first_edge = edge_tuple(chain[0]["edge"]) if chain else edge
        bound_record = v10.best_record(bound, first_edge)
        single_018_record = v10.best_record(single_018, first_edge)
        single_016_record = v10.best_record(single_016, first_edge)
        rows.append({
            "specific_edge": list(edge),
            "entry_edge": list(first_edge),
            "binding_source_activation": activation_for_source(bound_record),
            "single_0_18_source_activation": activation_for_source(single_018_record),
            "single_0_16_source_activation": activation_for_source(single_016_record),
            "binding_signal": None if bound_record is None else bound_record.get("signal"),
            "single_0_18_signal": None if single_018_record is None else single_018_record.get("signal"),
            "single_0_16_signal": None if single_016_record is None else single_016_record.get("signal"),
            "activation_gap_binding_vs_single_0_18": (
                None if bound_record is None or single_018_record is None
                else activation_for_source(bound_record) - activation_for_source(single_018_record)
            ),
            "activation_gap_binding_vs_single_0_16": (
                None if bound_record is None or single_016_record is None
                else activation_for_source(bound_record) - activation_for_source(single_016_record)
            ),
        })
    return {"entity": entity, "position": position, "rows": rows}


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v11",
        "world": {"P": player, "E": other},
        "purpose": "Quantify E-side chain reproduction and compare P-side Binding-entry activation with single-cue activation.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "puzzle_specific_rules": False,
        },
        "P_entry_activation": entry_activation_compare("P", player),
        "E_chain_coverage": chain_coverage("E", other),
        "P_chain_coverage": chain_coverage("P", player),
        "E_entry_activation": entry_activation_compare("E", other),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v11.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v11</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1450px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,58px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:23px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:720px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v11</h1><p class="lead">E側はBinding連鎖の再生率を、P側はBinding時と単独時の入口activation差を測る。学習・重み変更・新規Edge・専用規則はない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">確かめる</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ実行していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const pr=d.P_entry_activation.rows[0]||{},er=d.E_entry_activation.rows[0]||{},ec=d.E_chain_coverage,pc=d.P_chain_coverage;document.getElementById('metrics').innerHTML=`<div class="metric">P Binding入口activation<b>${f(pr.binding_source_activation)}</b></div><div class="metric">P 単独0.18入口activation<b>${f(pr.single_0_18_source_activation)}</b></div><div class="metric">P activation差<b class="${(pr.activation_gap_binding_vs_single_0_18||0)>0?'warn':'good'}">${f(pr.activation_gap_binding_vs_single_0_18)}</b></div><div class="metric">P連鎖再生率<b>${f(pc.overall_reproduction_ratio)}</b></div><div class="metric">E Binding入口activation<b>${f(er.binding_source_activation)}</b></div><div class="metric">E 単独0.16入口activation<b>${f(er.single_0_16_source_activation)}</b></div><div class="metric">E連鎖Edge<b>${ec.total_chain_edges}</b></div><div class="metric">E再生Edge<b>${ec.total_reproduced_edges}</b></div><div class="metric">E連鎖再生率<b class="${ec.overall_reproduction_ratio===1?'good':'warn'}">${f(ec.overall_reproduction_ratio)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/observe")
def api_observe():
    data = request.get_json(silent=True) or {}
    player = str(data.get("player", "左"))
    other = str(data.get("other", "右"))
    if player not in POSITIONS or other not in POSITIONS:
        return jsonify({"error": "位置が不正です。"}), 400
    return jsonify(observe(player, other))


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v11: http://{HOST}:{PORT}")
    print("Entry activation and chain coverage diagnosis / no learning / no changes")
    serve(app, host=HOST, port=PORT)
