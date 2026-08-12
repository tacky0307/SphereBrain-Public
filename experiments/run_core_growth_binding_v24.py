from __future__ import annotations

import copy
import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v12 as v12
import run_core_growth_binding_v19 as v19
import run_core_growth_binding_v23 as v23

HOST = "127.0.0.1"
START_PORT = 5059
OUT = ROOT / "data" / "core_growth_binding_v24" / "results"
POSITIONS = v3.POSITIONS
KEY_NODES = [80, 450]


def choose_port(start: int) -> int:
    for port in range(start, start + 40):
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


def chain_edges(chain: list[dict]) -> set[tuple[int, int]]:
    return {tuple(item["edge"]) for item in chain}


def jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def key_activations() -> dict[int, float]:
    trace, _ = v19.build_trace("E")
    return {node: float(trace[node]) for node in KEY_NODES}


def run_key_only(key_node: int, activation: float) -> dict:
    # position引数は初期化に使われないため、任意の既知位置を渡す。
    return v23.run_condition(POSITIONS[0], key_node, activation, include_position=False)


def compare_against_positions(key_node: int, activation: float) -> dict:
    trace = run_key_only(key_node, activation)
    traversed = edge_set(trace["traversed_edges"])
    per_position = {}

    for position in POSITIONS:
        reference = v12.binding_reference("E", position)
        chain = reference.get("chain", [])
        summary = v23.condition_summary(trace, chain)
        selected = traversed & chain_edges(chain)
        per_position[position] = {
            "reference_chain": chain,
            "chain_edge_count": len(chain),
            "replayed_edge_count": len(selected),
            "replay_ratio": summary["replay_ratio"],
            "first_visible": summary["first_visible"],
            "first_selected": summary["first_selected"],
            "selected_chain_edges": [list(x) for x in sorted(selected)],
            "rows": summary["rows"],
        }

    ratios = {position: row["replay_ratio"] for position, row in per_position.items()}
    max_ratio = max(ratios.values(), default=0.0)
    winners = [position for position, ratio in ratios.items() if ratio == max_ratio]
    positive_positions = [position for position, ratio in ratios.items() if ratio > 0]

    return {
        "node": key_node,
        "trace_activation": activation,
        "traversed_edge_count": len(traversed),
        "per_position": per_position,
        "winner_positions": winners,
        "positive_position_count": len(positive_positions),
        "position_specific": len(positive_positions) == 1,
        "all_positions_full": all(row["replay_ratio"] == 1.0 for row in per_position.values()),
        "trace": trace,
    }


def reference_comparison() -> dict:
    refs = {
        position: v12.binding_reference("E", position).get("chain", [])
        for position in POSITIONS
    }
    sets = {position: chain_edges(chain) for position, chain in refs.items()}
    common = set.intersection(*sets.values()) if sets else set()
    union = set.union(*sets.values()) if sets else set()
    pairwise = {}
    for i, left in enumerate(POSITIONS):
        for right in POSITIONS[i + 1:]:
            pairwise[f"{left}__{right}"] = {
                "jaccard": jaccard(sets[left], sets[right]),
                "shared_edges": [list(x) for x in sorted(sets[left] & sets[right])],
                "left_only": [list(x) for x in sorted(sets[left] - sets[right])],
                "right_only": [list(x) for x in sorted(sets[right] - sets[left])],
            }
    return {
        "chains": refs,
        "edge_sets": {p: [list(x) for x in sorted(s)] for p, s in sets.items()},
        "common_edge_count": len(common),
        "common_edges": [list(x) for x in sorted(common)],
        "union_edge_count": len(union),
        "pairwise": pairwise,
    }


def diagnose() -> dict:
    activations = key_activations()
    reports = [compare_against_positions(node, activations[node]) for node in KEY_NODES]
    return {
        "keys": reports,
        "references": reference_comparison(),
        "comparison": {
            "position_specific_key_count": sum(1 for row in reports if row["position_specific"]),
            "all_positions_full_key_count": sum(1 for row in reports if row["all_positions_full"]),
            "keys_replaying_multiple_positions": sum(1 for row in reports if row["positive_position_count"] > 1),
        },
    }


def observe() -> dict:
    payload = {
        "experiment": "Core Growth Binding v24",
        "purpose": "Cross-compare E Recall Keys 80/450 against E-left, E-center, and E-right reference chains without position input.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "position_input_during_key_probe": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "diagnostics": diagnose(),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v24.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v24</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:880px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v24</h1><p class="lead">Node 80 / 450を位置入力なしで走らせ、E→左・E→中央・E→右の参照連鎖へ横断照合する。Keyが位置固有か、E共通経路を再生しているかを確認する。</p><section class="panel"><div class="controls"><button onclick="run()">3位置を横断比較</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return Number(x).toFixed(6)}function keyCards(k,i){const p=k.per_position;return `<div class="metric">Key ${i+1}<b class="blue">Node ${k.node}</b></div><div class="metric">左連鎖 再生率<b>${f(p['左'].replay_ratio)}</b></div><div class="metric">中央連鎖 再生率<b>${f(p['中央'].replay_ratio)}</b></div><div class="metric">右連鎖 再生率<b>${f(p['右'].replay_ratio)}</b></div><div class="metric">再生した位置数<b>${k.positive_position_count}</b></div><div class="metric">位置固有Key<b class="${k.position_specific?'good':'warn'}">${k.position_specific?'YES':'NO'}</b></div>`}async function run(){const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),x=d.diagnostics;document.getElementById('metrics').innerHTML=x.keys.map(keyCards).join('')+`<div class="metric">3位置共通Edge<b>${x.references.common_edge_count}</b></div><div class="metric">位置固有Key数<b>${x.comparison.position_specific_key_count}</b></div><div class="metric">複数位置再生Key<b>${x.comparison.keys_replaying_multiple_positions}</b></div><div class="metric">3位置完全再生Key<b>${x.comparison.all_positions_full_key_count}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v24: http://{HOST}:{PORT}")
    print("Cross-position Recall Key comparison / no position input / no Core changes")
    serve(app, host=HOST, port=PORT)
