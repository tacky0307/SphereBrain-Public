from __future__ import annotations

import copy
import itertools
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
import run_core_growth_binding_v15 as v15
import run_core_growth_binding_v17 as v17

HOST = "127.0.0.1"
START_PORT = 5054
OUT = ROOT / "data" / "core_growth_binding_v19" / "results"
POSITIONS = v3.POSITIONS
WINDOW = 4
DECAY = 0.95
GAP = 0
MODE = "capped_sum"
MAX_EXACT_NODES = 14


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


def build_trace(entity: str) -> tuple[np.ndarray, list[dict]]:
    brain = copy.deepcopy(v3.base.CORE)
    state = v15.propagate_state(
        brain,
        v15.initial_sources(brain, v3.entity_nodes(entity)),
        v15.ENTITY_STEPS,
    )
    trace = v17.temporal_trace(
        state["activation_values"], brain.node_count, WINDOW, DECAY, MODE, GAP
    )
    return trace, state["activation_values"]


def node_metadata(trace: np.ndarray, values: list[dict]) -> list[dict]:
    recent = values[-WINDOW:]
    rows = []
    for node in np.flatnonzero(trace > 0):
        appearances = []
        for relative_index, snapshot in enumerate(recent):
            value = float(snapshot.get(str(int(node)), 0.0))
            if value > 0:
                appearances.append({
                    "relative_step": relative_index,
                    "age_from_latest": len(recent) - 1 - relative_index,
                    "raw_activation": value,
                })
        newest_age = min((x["age_from_latest"] for x in appearances), default=999)
        oldest_age = max((x["age_from_latest"] for x in appearances), default=-1)
        rows.append({
            "node": int(node),
            "trace_activation": float(trace[node]),
            "appearance_count": len(appearances),
            "newest_age": newest_age,
            "oldest_age": oldest_age,
            "appearances": appearances,
        })
    return rows


def replay_with_trace(entity: str, position: str, trace: np.ndarray, reference: dict) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    initial = trace.copy()
    for node in v3.position_nodes(position):
        initial[int(node)] = max(initial[int(node)], 1.0)
    state = v15.propagate_state(brain, initial, v15.POSITION_STEPS)
    traversed = edge_set(state["traversed_edges"])
    chain = reference.get("chain", [])
    total = len(chain)
    replayed = sum(1 for item in chain if tuple(item["edge"]) in traversed)
    return {
        "replayed": replayed,
        "total": total,
        "ratio": 0.0 if total == 0 else replayed / total,
        "activated_node_count": len(state["activated_nodes"]),
        "traversed_edge_count": len(state["traversed_edges"]),
    }


def evaluate_keep(entity: str, position: str, full_trace: np.ndarray, keep_nodes: set[int], reference: dict) -> dict:
    trace = np.zeros_like(full_trace)
    if keep_nodes:
        idx = np.fromiter(sorted(keep_nodes), dtype=int)
        trace[idx] = full_trace[idx]
    report = replay_with_trace(entity, position, trace, reference)
    report["kept_nodes"] = sorted(keep_nodes)
    report["kept_count"] = len(keep_nodes)
    report["trace_energy"] = float(trace.sum())
    return report


def progressive_removal(entity: str, position: str, full_trace: np.ndarray, order: list[int], reference: dict) -> list[dict]:
    keep = set(int(x) for x in np.flatnonzero(full_trace > 0))
    rows = [evaluate_keep(entity, position, full_trace, keep, reference)]
    for node in order:
        keep.discard(int(node))
        row = evaluate_keep(entity, position, full_trace, keep, reference)
        row["removed_node"] = int(node)
        rows.append(row)
    return rows


def exact_minimum_sets(entity: str, position: str, full_trace: np.ndarray, reference: dict) -> dict:
    nodes = [int(x) for x in np.flatnonzero(full_trace > 0)]
    if len(nodes) > MAX_EXACT_NODES:
        return {"searched": False, "reason": "too_many_nodes", "node_count": len(nodes)}
    tested = 0
    for size in range(len(nodes) + 1):
        winners = []
        for combo in itertools.combinations(nodes, size):
            tested += 1
            report = evaluate_keep(entity, position, full_trace, set(combo), reference)
            if report["total"] > 0 and report["ratio"] == 1.0:
                winners.append(report)
        if winners:
            winners.sort(key=lambda x: (x["trace_energy"], x["kept_nodes"]))
            return {
                "searched": True,
                "tested_subsets": tested,
                "minimum_size": size,
                "minimum_set_count": len(winners),
                "minimum_sets": winners[:50],
            }
    return {
        "searched": True,
        "tested_subsets": tested,
        "minimum_size": None,
        "minimum_set_count": 0,
        "minimum_sets": [],
    }


def grouped_ablation(entity: str, position: str) -> dict:
    reference = v12.binding_reference(entity, position)
    trace, values = build_trace(entity)
    metadata = node_metadata(trace, values)
    nodes = [row["node"] for row in metadata]
    baseline = evaluate_keep(entity, position, trace, set(nodes), reference)

    weak_first = [r["node"] for r in sorted(metadata, key=lambda r: (r["trace_activation"], r["node"]))]
    strong_first = [r["node"] for r in sorted(metadata, key=lambda r: (-r["trace_activation"], r["node"]))]
    old_first = [r["node"] for r in sorted(metadata, key=lambda r: (-r["newest_age"], r["node"]))]
    new_first = [r["node"] for r in sorted(metadata, key=lambda r: (r["newest_age"], r["node"]))]

    pair_failures = []
    triple_failures = []
    for combo in itertools.combinations(nodes, 2):
        report = evaluate_keep(entity, position, trace, set(nodes) - set(combo), reference)
        if report["ratio"] < baseline["ratio"]:
            pair_failures.append({"removed": list(combo), **report})
    for combo in itertools.combinations(nodes, 3):
        report = evaluate_keep(entity, position, trace, set(nodes) - set(combo), reference)
        if report["ratio"] < baseline["ratio"]:
            triple_failures.append({"removed": list(combo), **report})

    pair_failures.sort(key=lambda r: (r["ratio"], r["removed"]))
    triple_failures.sort(key=lambda r: (r["ratio"], r["removed"]))

    return {
        "entity": entity,
        "position": position,
        "settings": {"mode": MODE, "window": WINDOW, "decay": DECAY, "gap": GAP},
        "trace_node_count": len(nodes),
        "node_metadata": metadata,
        "baseline": baseline,
        "progressive": {
            "weak_first": progressive_removal(entity, position, trace, weak_first, reference),
            "strong_first": progressive_removal(entity, position, trace, strong_first, reference),
            "old_first": progressive_removal(entity, position, trace, old_first, reference),
            "new_first": progressive_removal(entity, position, trace, new_first, reference),
        },
        "pair_failure_count": len(pair_failures),
        "pair_failures": pair_failures[:100],
        "triple_failure_count": len(triple_failures),
        "triple_failures": triple_failures[:200],
        "exact_minimum": exact_minimum_sets(entity, position, trace, reference),
    }


def first_break(rows: list[dict]) -> dict | None:
    baseline = rows[0]["ratio"] if rows else 0.0
    return next((row for row in rows[1:] if row["ratio"] < baseline), None)


def observe(player: str, other: str) -> dict:
    p = grouped_ablation("P", player)
    e = grouped_ablation("E", other)
    for item in (p, e):
        item["first_breaks"] = {
            name: first_break(rows)
            for name, rows in item["progressive"].items()
        }
    payload = {
        "experiment": "Core Growth Binding v19",
        "world": {"P": player, "E": other},
        "purpose": "Find distributed Node groups and minimum temporal-trace subsets required for Binding-chain replay.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "hand_selected_nodes_for_trace_creation": False,
            "core_file_modified": False,
        },
        "diagnostics": {"P": p, "E": e},
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v19.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v19</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:21px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.raw{white-space:pre-wrap;max-height:780px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v19</h1><p class="lead">短期TraceのNodeを群で段階的に削り、完全再生を保てる最小Node集合と、再生を壊すNode組を探す。Core本体は変更しない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Node群を削る</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function v(x){return x===null||x===undefined?'なし':String(x)}function br(d,name){const r=d.first_breaks[name];return r?`${r.kept_count}Node残 / 再生率${Number(r.ratio).toFixed(3)}`:'壊れず'}function cards(label,d){const m=d.exact_minimum;return `<div class="metric">${label}基準再生率<b>${Number(d.baseline.ratio).toFixed(3)}</b></div><div class="metric">${label}最小Node数<b class="${m.minimum_size!==null?'good':'warn'}">${v(m.minimum_size)}</b></div><div class="metric">${label}最小集合数<b>${v(m.minimum_set_count)}</b></div><div class="metric">${label}弱い順で破綻<b>${br(d,'weak_first')}</b></div><div class="metric">${label}強い順で破綻<b>${br(d,'strong_first')}</b></div><div class="metric">${label}古い順で破綻<b>${br(d,'old_first')}</b></div><div class="metric">${label}新しい順で破綻<b>${br(d,'new_first')}</b></div><div class="metric">${label}破綻Pair/Triple<b>${d.pair_failure_count} / ${d.triple_failure_count}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=cards('P',d.diagnostics.P)+cards('E',d.diagnostics.E)+`<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v19: http://{HOST}:{PORT}")
    print("Grouped temporal-trace ablation / exact minimum subset search")
    serve(app, host=HOST, port=PORT)
