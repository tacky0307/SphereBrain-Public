from __future__ import annotations

import copy
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

HOST = "127.0.0.1"
START_PORT = 5044
OUT = ROOT / "data" / "core_growth_binding_v9" / "results"
POSITIONS = v3.POSITIONS
BASE_THRESHOLD = 0.18
COMPARE_THRESHOLD = 0.16


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


def edge_set(edges):
    return {tuple(sorted((int(a), int(b)))) for a, b in edges}


def inspect_specific_edge(entity: str, edge: tuple[int, int], probe: dict, threshold: float) -> dict:
    brain = v3.base.CORE
    a, b = edge
    weight = float(brain.weights[a, b])
    observations = []

    accepted = edge in edge_set(probe["traversed_edges"])
    for step in probe["step_records"]:
        accepted_edges = edge_set(step["accepted_edges"])
        for source, target, signal in step["local_top_edges"]:
            if tuple(sorted((int(source), int(target)))) != edge:
                continue
            denominator = weight * float(brain.signal_decay)
            activation = 0.0 if denominator == 0 else float(signal) / denominator
            observations.append({
                "step": int(step["step"]),
                "source": int(source),
                "target": int(target),
                "source_activation": activation,
                "edge_weight": weight,
                "signal_decay": float(brain.signal_decay),
                "transmission_signal": float(signal),
                "threshold": float(threshold),
                "threshold_gap": float(signal) - float(threshold),
                "accepted_this_step": edge in accepted_edges,
            })

    best = max(observations, key=lambda row: row["transmission_signal"], default=None)
    return {
        "edge": [a, b],
        "seen_as_local_top": bool(observations),
        "selected_anywhere": accepted,
        "best_observation": best,
        "all_observations": observations,
        "exact_threshold_to_pass_best": None if best is None else best["transmission_signal"],
        "additional_source_activation_needed_at_0_18": (
            None if best is None or weight * brain.signal_decay == 0
            else max(0.0, BASE_THRESHOLD / (weight * brain.signal_decay) - best["source_activation"])
        ),
        "required_weight_at_best_activation_for_0_18": (
            None if best is None or best["source_activation"] * brain.signal_decay == 0
            else BASE_THRESHOLD / (best["source_activation"] * brain.signal_decay)
        ),
    }


def selectivity(entity: str, position: str, probe: dict) -> dict:
    refs = v8.binding_refs(entity)
    return v8.score_probe(probe, refs, position)


def compare_thresholds(entity: str, position: str, specific: set[tuple[int, int]]) -> dict:
    base_probe = v8.run_probe(entity, threshold=BASE_THRESHOLD, persistence=0.0)
    compare_probe = v8.run_probe(entity, threshold=COMPARE_THRESHOLD, persistence=0.0)

    base_nodes = set(base_probe["activated_nodes"])
    compare_nodes = set(compare_probe["activated_nodes"])
    base_edges = edge_set(base_probe["traversed_edges"])
    compare_edges = edge_set(compare_probe["traversed_edges"])
    added_nodes = compare_nodes - base_nodes
    added_edges = compare_edges - base_edges
    added_specific = added_edges & specific

    return {
        "base_threshold": BASE_THRESHOLD,
        "compare_threshold": COMPARE_THRESHOLD,
        "base_probe": base_probe,
        "compare_probe": compare_probe,
        "added_node_count": len(added_nodes),
        "added_edge_count": len(added_edges),
        "added_nodes": sorted(added_nodes),
        "added_edges": [list(x) for x in sorted(added_edges)],
        "added_specific_edge_count": len(added_specific),
        "added_specific_edges": [list(x) for x in sorted(added_specific)],
        "specific_share_of_added_edges": 0.0 if not added_edges else len(added_specific) / len(added_edges),
        "base_selectivity": selectivity(entity, position, base_probe),
        "compare_selectivity": selectivity(entity, position, compare_probe),
        "node_count_growth_ratio": (
            0.0 if not base_nodes else (len(compare_nodes) - len(base_nodes)) / len(base_nodes)
        ),
        "edge_count_growth_ratio": (
            0.0 if not base_edges else (len(compare_edges) - len(base_edges)) / len(base_edges)
        ),
    }


def diagnose(entity: str, position: str) -> dict:
    specific = v8.binding_specific_edges(entity, position)
    base_probe = v8.run_probe(entity, threshold=BASE_THRESHOLD, persistence=0.0)
    edge_diagnostics = [
        inspect_specific_edge(entity, edge, base_probe, BASE_THRESHOLD)
        for edge in sorted(specific)
    ]

    best_signals = [
        row["best_observation"]["transmission_signal"]
        for row in edge_diagnostics
        if row["best_observation"] is not None
    ]
    closest = max(best_signals, default=None)

    return {
        "entity": entity,
        "position": position,
        "specific_edge_count": len(specific),
        "specific_edges": [list(x) for x in sorted(specific)],
        "base_threshold": BASE_THRESHOLD,
        "closest_specific_signal": closest,
        "closest_gap_to_0_18": None if closest is None else closest - BASE_THRESHOLD,
        "edge_diagnostics": edge_diagnostics,
        "threshold_0_18_vs_0_16": compare_thresholds(entity, position, specific),
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v9",
        "world": {"P": player, "E": other},
        "purpose": "Directly measure why Binding-specific edges fail or pass the recall gate.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "puzzle_specific_rules": False,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v9.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v9</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:23px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:720px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v9</h1><p class="lead">Binding固有Edgeの手前で、活動値・重み・減衰・伝播値・閾値差を直接測る。0.18から0.16へ下げたときの活動拡大も同時に確認する。学習・重み変更・専用規則はない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">信号を測る</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ測定していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();const P=d.diagnostics.P,E=d.diagnostics.E,pc=P.threshold_0_18_vs_0_16,ec=E.threshold_0_18_vs_0_16;document.getElementById('metrics').innerHTML=`<div class="metric">P 最接近信号<b>${f(P.closest_specific_signal)}</b></div><div class="metric">P 0.18との差<b class="${P.closest_gap_to_0_18!==null&&P.closest_gap_to_0_18>=0?'good':'warn'}">${f(P.closest_gap_to_0_18)}</b></div><div class="metric">E 最接近信号<b>${f(E.closest_specific_signal)}</b></div><div class="metric">E 0.18との差<b class="${E.closest_gap_to_0_18!==null&&E.closest_gap_to_0_18>=0?'good':'warn'}">${f(E.closest_gap_to_0_18)}</b></div><div class="metric">P 0.16追加Node<b>${pc.added_node_count}</b></div><div class="metric">P 0.16追加Edge<b>${pc.added_edge_count}</b></div><div class="metric">E 0.16追加Node<b>${ec.added_node_count}</b></div><div class="metric">E 0.16追加Edge<b>${ec.added_edge_count}</b></div><div class="metric">E追加Edge中Binding比<b>${f(ec.specific_share_of_added_edges)}</b></div><div class="metric">P選択性 0.18→0.16<b>${f(pc.base_selectivity.target_node_margin)} → ${f(pc.compare_selectivity.target_node_margin)}</b></div><div class="metric">E選択性 0.18→0.16<b>${f(ec.base_selectivity.target_node_margin)} → ${f(ec.compare_selectivity.target_node_margin)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v9: http://{HOST}:{PORT}")
    print("Direct signal diagnosis / no learning / no weight changes")
    serve(app, host=HOST, port=PORT)
