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
import run_core_growth_binding_v27 as v27

HOST = "127.0.0.1"
START_PORT = 5074
OUT = ROOT / "data" / "core_growth_binding_v28" / "results"
POSITIONS = list(v3.POSITIONS)


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


def edge_list(values: set[tuple[int, int]]) -> list[list[int]]:
    return [list(edge) for edge in sorted(values)]


def first_selected_step(trace: dict, edge: tuple[int, int]) -> int | None:
    for step in trace.get("steps", []):
        if edge in edge_set(step.get("accepted_edges", [])):
            return int(step["step"])
    return None


def interaction_rows(trace: dict, edges: set[tuple[int, int]]) -> list[dict]:
    rows = []
    for edge in sorted(edges):
        rows.append({
            "edge": list(edge),
            "source": int(edge[0]),
            "target": int(edge[1]),
            "first_selected_step": first_selected_step(trace, edge),
        })
    return rows


def position_report(position: str, echo_only_trace: dict) -> dict:
    position_only_trace = v27.run_live(position=position, include_echo=False)
    binding_trace = v27.run_live(position=position, include_echo=True)

    echo_edges = edge_set(echo_only_trace["traversed_edges"])
    position_edges = edge_set(position_only_trace["traversed_edges"])
    binding_edges = edge_set(binding_trace["traversed_edges"])

    interaction = binding_edges - echo_edges - position_edges
    explained_by_echo = binding_edges & echo_edges
    explained_by_position = binding_edges & position_edges

    return {
        "position": position,
        "counts": {
            "binding_all": len(binding_edges),
            "echo_only": len(echo_edges),
            "position_only": len(position_edges),
            "explained_by_echo": len(explained_by_echo),
            "explained_by_position": len(explained_by_position),
            "strict_interaction": len(interaction),
        },
        "edges": {
            "binding_all": edge_list(binding_edges),
            "echo_only": edge_list(echo_edges),
            "position_only": edge_list(position_edges),
            "explained_by_echo": edge_list(explained_by_echo),
            "explained_by_position": edge_list(explained_by_position),
            "strict_interaction": edge_list(interaction),
        },
        "strict_interaction_rows": interaction_rows(binding_trace, interaction),
        "traces": {
            "position_only": position_only_trace,
            "binding": binding_trace,
        },
    }


def decompose_interactions(reports: dict[str, dict]) -> dict:
    sets = {
        position: edge_set(report["edges"]["strict_interaction"])
        for position, report in reports.items()
    }
    left, center, right = POSITIONS

    all_three = sets[left] & sets[center] & sets[right]
    left_center = (sets[left] & sets[center]) - all_three
    left_right = (sets[left] & sets[right]) - all_three
    center_right = (sets[center] & sets[right]) - all_three
    left_only = sets[left] - sets[center] - sets[right]
    center_only = sets[center] - sets[left] - sets[right]
    right_only = sets[right] - sets[left] - sets[center]
    union = sets[left] | sets[center] | sets[right]

    return {
        "counts": {
            "union": len(union),
            "all_three_common": len(all_three),
            "left_center_only": len(left_center),
            "left_right_only": len(left_right),
            "center_right_only": len(center_right),
            "left_only": len(left_only),
            "center_only": len(center_only),
            "right_only": len(right_only),
        },
        "groups": {
            "union": edge_list(union),
            "all_three_common": edge_list(all_three),
            "left_center_only": edge_list(left_center),
            "left_right_only": edge_list(left_right),
            "center_right_only": edge_list(center_right),
            "left_only": edge_list(left_only),
            "center_only": edge_list(center_only),
            "right_only": edge_list(right_only),
        },
        "has_any_strict_interaction": bool(union),
        "has_position_specific_interaction": bool(left_only or center_only or right_only),
    }


def diagnose() -> dict:
    echo_only_trace = v27.run_live(position=None, include_echo=True)
    reports = {
        position: position_report(position, echo_only_trace)
        for position in POSITIONS
    }
    decomposition = decompose_interactions(reports)

    if not decomposition["has_any_strict_interaction"]:
        verdict = "no_interaction_edges"
    elif decomposition["has_position_specific_interaction"]:
        verdict = "position_specific_interaction_found"
    else:
        verdict = "interaction_edges_exist_but_not_position_specific"

    return {
        "definition": {
            "strict_interaction": "BindingAllEdges - EResidualOnlyEdges - PositionOnlyEdges",
        },
        "echo_only": {
            "edge_count": len(echo_only_trace["traversed_edges"]),
            "edges": echo_only_trace["traversed_edges"],
            "trace": echo_only_trace,
        },
        "per_position": reports,
        "decomposition": decomposition,
        "verdict": verdict,
    }


def observe() -> dict:
    payload = {
        "experiment": "Core Growth Binding v28",
        "purpose": "Recalculate Binding interaction edges using the strict difference: Binding all edges minus E-residual-only edges minus position-only edges.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "diagnostics": diagnose(),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v28.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v28</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.formula{background:#0c1727;border-radius:14px;padding:16px;font-size:18px;color:var(--blue)}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v28</h1><p class="lead">E残響単独Edgeを明示的に除外し、現行Binding Windowに本当の相互作用Edgeが残るかを再計算する。</p><section class="panel"><div class="formula">真のBinding候補 = Binding全Edge − E残響単独Edge − 位置単独Edge</div><div class="controls"><button id="run">厳密差分を取る</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function card(label,value,cls=''){return `<div class="metric">${label}<b class="${cls}">${value}</b></div>`}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),x=d.diagnostics,dec=x.decomposition,c=dec.counts;let html='';for(const p of ['左','中央','右']){const n=x.per_position[p].counts;html+=card(`${p} Binding全Edge`,n.binding_all)+card(`${p} E残響で説明`,n.explained_by_echo)+card(`${p} 位置単独で説明`,n.explained_by_position)+card(`${p} 厳密相互作用`,n.strict_interaction,n.strict_interaction?'good':'warn')}html+=card('相互作用Edge総数',c.union,dec.has_any_strict_interaction?'good':'warn')+card('3位置共通',c.all_three_common)+card('左だけ',c.left_only,c.left_only?'good':'')+card('中央だけ',c.center_only,c.center_only?'good':'')+card('右だけ',c.right_only,c.right_only?'good':'')+card('位置固有相互作用',dec.has_position_specific_interaction?'YES':'NO',dec.has_position_specific_interaction?'good':'warn')+card('判定',x.verdict)+card('brain.json',d.brain_file_unchanged?'不変':'変化','good');document.getElementById('metrics').innerHTML=html;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v28: http://{HOST}:{PORT}")
    print("Strict Binding interaction decomposition / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
