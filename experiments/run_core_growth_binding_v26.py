from __future__ import annotations

import copy
import json
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

HOST = "127.0.0.1"
START_PORT = 5071
OUT = ROOT / "data" / "core_growth_binding_v26" / "results"
POSITIONS = list(v3.POSITIONS)
STEPS = 10


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


def edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def edge_set(edges) -> set[tuple[int, int]]:
    return {edge_key(a, b) for a, b in edges}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def run_position(position: str) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    source_nodes = [int(x) for x in v3.position_nodes(position)]

    # v3.propagate() と同じ初期条件を保ちながら、生の SignalResult を受け取る。
    original_initial = brain._initial_activation

    def initial(source_nodes_arg, context_nodes):
        sources = [int(x) for x in source_nodes_arg]
        activation = np.zeros(brain.node_count, dtype=float)
        for node in sources:
            activation[node] = 1.0
        return sources, activation

    brain._initial_activation = initial
    try:
        result = brain.propagate(
            source_nodes,
            context_nodes=None,
            steps=STEPS,
            threshold=0.18,
            noise=0.0,
            learn=False,
        )
    finally:
        brain._initial_activation = original_initial

    history = [set(int(x) for x in step) for step in result.activation_history]
    final = np.asarray(result.final_activation, dtype=float)
    return {
        "position": position,
        "source_nodes": source_nodes,
        "activated_nodes": sorted(int(x) for x in result.activated_nodes),
        "traversed_edges": [list(edge_key(a, b)) for a, b in result.traversed_edges],
        "activation_history": [sorted(step) for step in history],
        "history_sets": history,
        "final_activation": final,
        "final_nonzero": sorted(int(x) for x in np.flatnonzero(final > 0)),
        "activated_node_count": len(result.activated_nodes),
        "traversed_edge_count": len(result.traversed_edges),
        "depth": max(0, len(history) - 1),
    }


def first_divergence(a: list[set[int]], b: list[set[int]]) -> int | None:
    length = max(len(a), len(b))
    for i in range(length):
        left = a[i] if i < len(a) else set()
        right = b[i] if i < len(b) else set()
        if left != right:
            return i
    return None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 0.0
    if na == 0.0 or nb == 0.0:
        return 1.0
    cosine = float(np.dot(a, b) / (na * nb))
    return 1.0 - max(-1.0, min(1.0, cosine))


def pair_report(left: dict, right: dict) -> dict:
    source_a = set(left["source_nodes"])
    source_b = set(right["source_nodes"])
    nodes_a = set(left["activated_nodes"])
    nodes_b = set(right["activated_nodes"])
    edges_a = edge_set(left["traversed_edges"])
    edges_b = edge_set(right["traversed_edges"])
    final_a = set(left["final_nonzero"])
    final_b = set(right["final_nonzero"])

    max_steps = max(len(left["history_sets"]), len(right["history_sets"]))
    step_rows = []
    for step in range(max_steps):
        a = left["history_sets"][step] if step < len(left["history_sets"]) else set()
        b = right["history_sets"][step] if step < len(right["history_sets"]) else set()
        step_rows.append({
            "step": step,
            "left_count": len(a),
            "right_count": len(b),
            "node_jaccard": jaccard(a, b),
            "shared_nodes": sorted(a & b),
            "left_only_nodes": sorted(a - b),
            "right_only_nodes": sorted(b - a),
        })

    return {
        "pair": f"{left['position']}__{right['position']}",
        "source_node_jaccard": jaccard(source_a, source_b),
        "activated_node_jaccard": jaccard(nodes_a, nodes_b),
        "edge_jaccard": jaccard(edges_a, edges_b),
        "final_node_jaccard": jaccard(final_a, final_b),
        "final_activation_cosine_distance": cosine_distance(left["final_activation"], right["final_activation"]),
        "first_divergence_step": first_divergence(left["history_sets"], right["history_sets"]),
        "shared_activated_nodes": sorted(nodes_a & nodes_b),
        "left_unique_nodes": sorted(nodes_a - nodes_b),
        "right_unique_nodes": sorted(nodes_b - nodes_a),
        "shared_edges": [list(x) for x in sorted(edges_a & edges_b)],
        "left_unique_edges": [list(x) for x in sorted(edges_a - edges_b)],
        "right_unique_edges": [list(x) for x in sorted(edges_b - edges_a)],
        "step_comparison": step_rows,
    }


def structural_signature(run: dict) -> dict:
    sizes = [len(step) for step in run["history_sets"]]
    return {
        "depth": run["depth"],
        "history_widths": sizes,
        "peak_width": max(sizes, default=0),
        "activated_node_count": run["activated_node_count"],
        "traversed_edge_count": run["traversed_edge_count"],
    }


def observe() -> dict:
    runs = {p: run_position(p) for p in POSITIONS}
    pairwise = {}
    for i, left in enumerate(POSITIONS):
        for right in POSITIONS[i + 1:]:
            pairwise[f"{left}__{right}"] = pair_report(runs[left], runs[right])

    pair_values = list(pairwise.values())
    payload = {
        "experiment": "Core Growth Binding v26",
        "purpose": "Measure whether left, center, and right position inputs remain distinct inside Core before Binding.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "runs": {
            p: {
                k: v for k, v in run.items()
                if k not in {"history_sets", "final_activation"}
            }
            for p, run in runs.items()
        },
        "structural_signatures": {p: structural_signature(run) for p, run in runs.items()},
        "pairwise": pairwise,
        "summary": {
            "mean_source_jaccard": float(np.mean([x["source_node_jaccard"] for x in pair_values])),
            "mean_activated_node_jaccard": float(np.mean([x["activated_node_jaccard"] for x in pair_values])),
            "mean_edge_jaccard": float(np.mean([x["edge_jaccard"] for x in pair_values])),
            "mean_final_node_jaccard": float(np.mean([x["final_node_jaccard"] for x in pair_values])),
            "mean_final_activation_cosine_distance": float(np.mean([x["final_activation_cosine_distance"] for x in pair_values])),
            "all_sources_distinct": all(x["source_node_jaccard"] < 1.0 for x in pair_values),
            "all_routes_distinct": all(x["edge_jaccard"] < 1.0 for x in pair_values),
            "all_final_states_distinct": all(x["final_activation_cosine_distance"] > 0.0 for x in pair_values),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v26.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v26</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v26</h1><p class="lead">左・中央・右の位置単独入力を、入力Node・Step活動・通過Edge・最終activationまで比較し、位置表現がCore内で分離しているかを診断する。</p><section class="panel"><div class="controls"><button id="run">位置入力を比較</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return Number(x).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,p=d.pairwise;const cards=Object.values(p).map(x=>`<div class="metric">${x.pair} 入力Jaccard<b>${f(x.source_node_jaccard)}</b></div><div class="metric">${x.pair} Node Jaccard<b>${f(x.activated_node_jaccard)}</b></div><div class="metric">${x.pair} Edge Jaccard<b>${f(x.edge_jaccard)}</b></div><div class="metric">${x.pair} 最終距離<b>${f(x.final_activation_cosine_distance)}</b></div>`).join('');document.getElementById('metrics').innerHTML=cards+`<div class="metric">平均入力Jaccard<b>${f(s.mean_source_jaccard)}</b></div><div class="metric">平均Node Jaccard<b>${f(s.mean_activated_node_jaccard)}</b></div><div class="metric">平均Edge Jaccard<b>${f(s.mean_edge_jaccard)}</b></div><div class="metric">全経路分離<b class="${s.all_routes_distinct?'good':'warn'}">${s.all_routes_distinct?'YES':'NO'}</b></div><div class="metric">最終状態分離<b class="${s.all_final_states_distinct?'good':'warn'}">${s.all_final_states_distinct?'YES':'NO'}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v26: http://{HOST}:{PORT}")
    print("Position-input separation diagnostics / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
