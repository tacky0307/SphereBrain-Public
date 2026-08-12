from __future__ import annotations

import copy
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

HOST = "127.0.0.1"
START_PORT = 5070
OUT = ROOT / "data" / "core_growth_binding_v25" / "results"
POSITIONS = list(v3.POSITIONS)
CHROME_UNSAFE_PORTS = {5060, 5061}


def choose_port(start: int) -> int:
    for port in range(start, start + 40):
        if port in CHROME_UNSAFE_PORTS:
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


def run_entity_only() -> set[tuple[int, int]]:
    brain = copy.deepcopy(v3.base.CORE)
    result = v3.propagate(brain, v3.entity_nodes("E"), learn=False, steps=8)
    return edge_set(result["traversed_edges"])


def run_position_only(position: str) -> set[tuple[int, int]]:
    brain = copy.deepcopy(v3.base.CORE)
    result = v3.propagate(brain, v3.position_nodes(position), learn=False, steps=10)
    return edge_set(result["traversed_edges"])


def run_binding(position: str) -> set[tuple[int, int]]:
    result = v3.make_binding(
        copy.deepcopy(v3.base.CORE), "E", position, learn=False, assist=False
    )
    return edge_set(result["bound_edges"])


def lists(values: set[tuple[int, int]]) -> list[list[int]]:
    return [list(x) for x in sorted(values)]


def decompose() -> dict:
    entity_only = run_entity_only()
    position_only = {p: run_position_only(p) for p in POSITIONS}
    bound = {p: run_binding(p) for p in POSITIONS}

    # まずE単独と各位置単独で説明できるEdgeを除き、関係候補だけを残す。
    relation_candidates = {
        p: bound[p] - entity_only - position_only[p]
        for p in POSITIONS
    }

    left, center, right = POSITIONS
    all_three = relation_candidates[left] & relation_candidates[center] & relation_candidates[right]
    left_center = (relation_candidates[left] & relation_candidates[center]) - all_three
    left_right = (relation_candidates[left] & relation_candidates[right]) - all_three
    center_right = (relation_candidates[center] & relation_candidates[right]) - all_three

    left_only = relation_candidates[left] - relation_candidates[center] - relation_candidates[right]
    center_only = relation_candidates[center] - relation_candidates[left] - relation_candidates[right]
    right_only = relation_candidates[right] - relation_candidates[left] - relation_candidates[center]

    explained_by_entity = {p: bound[p] & entity_only for p in POSITIONS}
    explained_by_position = {p: bound[p] & position_only[p] for p in POSITIONS}

    return {
        "positions": POSITIONS,
        "raw": {
            "entity_only": lists(entity_only),
            "position_only": {p: lists(s) for p, s in position_only.items()},
            "binding": {p: lists(s) for p, s in bound.items()},
            "relation_candidates": {p: lists(s) for p, s in relation_candidates.items()},
        },
        "counts": {
            "entity_only": len(entity_only),
            "binding": {p: len(bound[p]) for p in POSITIONS},
            "relation_candidates": {p: len(relation_candidates[p]) for p in POSITIONS},
            "all_three_common": len(all_three),
            "left_center_only": len(left_center),
            "left_right_only": len(left_right),
            "center_right_only": len(center_right),
            "left_only": len(left_only),
            "center_only": len(center_only),
            "right_only": len(right_only),
            "explained_by_entity": {p: len(explained_by_entity[p]) for p in POSITIONS},
            "explained_by_position": {p: len(explained_by_position[p]) for p in POSITIONS},
        },
        "groups": {
            "all_three_common": lists(all_three),
            "left_center_only": lists(left_center),
            "left_right_only": lists(left_right),
            "center_right_only": lists(center_right),
            "left_only": lists(left_only),
            "center_only": lists(center_only),
            "right_only": lists(right_only),
            "explained_by_entity": {p: lists(explained_by_entity[p]) for p in POSITIONS},
            "explained_by_position": {p: lists(explained_by_position[p]) for p in POSITIONS},
        },
        "interpretation": {
            "has_any_position_specific_edge": bool(left_only or center_only or right_only),
            "position_specific_counts": {
                left: len(left_only),
                center: len(center_only),
                right: len(right_only),
            },
        },
    }


def observe() -> dict:
    payload = {
        "experiment": "Core Growth Binding v25",
        "purpose": "Decompose E-left, E-center, and E-right Binding edge sets into common, pair-shared, and position-specific components after subtracting entity-only and position-only paths.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "diagnostics": decompose(),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v25.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v25</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v25</h1><p class="lead">E→左・中央・右の全Binding Edgeを、E単独・位置単独由来を差し引いたうえで、3位置共通・2位置共有・位置固有へ分解する。</p><section class="panel"><div class="controls"><button id="run">Binding経路を分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
const runButton=document.getElementById('run');runButton.addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),x=d.diagnostics,c=x.counts,p=x.positions;document.getElementById('metrics').innerHTML=`<div class="metric">3位置共通<b>${c.all_three_common}</b></div><div class="metric">左＋中央のみ<b>${c.left_center_only}</b></div><div class="metric">左＋右のみ<b>${c.left_right_only}</b></div><div class="metric">中央＋右のみ<b>${c.center_right_only}</b></div><div class="metric">左だけ<b class="${c.left_only?'good':'warn'}">${c.left_only}</b></div><div class="metric">中央だけ<b class="${c.center_only?'good':'warn'}">${c.center_only}</b></div><div class="metric">右だけ<b class="${c.right_only?'good':'warn'}">${c.right_only}</b></div><div class="metric">位置固有Edgeあり<b class="${x.interpretation.has_any_position_specific_edge?'good':'warn'}">${x.interpretation.has_any_position_specific_edge?'YES':'NO'}</b></div><div class="metric">左 関係候補<b>${c.relation_candidates[p[0]]}</b></div><div class="metric">中央 関係候補<b>${c.relation_candidates[p[1]]}</b></div><div class="metric">右 関係候補<b>${c.relation_candidates[p[2]]}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v25: http://{HOST}:{PORT}")
    print("Cross-position Binding decomposition / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
