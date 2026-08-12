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
import run_core_growth_binding_v20 as v20

HOST = "127.0.0.1"
START_PORT = 5058
OUT = ROOT / "data" / "core_growth_binding_v23" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
MAX_STEPS = 10
STATUS_RANK = {
    "not_visible": 0,
    "neighbor_seen_but_not_local_top": 1,
    "local_top_but_below_threshold": 2,
    "candidate_but_not_selected": 3,
    "selected": 4,
}


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


def run_condition(position: str, key_node: int | None, key_activation: float, include_position: bool) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    activation = np.zeros(brain.node_count, dtype=float)
    if key_node is not None:
        activation[int(key_node)] = float(key_activation)
    if include_position:
        for source in v3.position_nodes(position):
            activation[int(source)] = max(activation[int(source)], 1.0)

    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed: set[tuple[int, int]] = set()
    steps = []

    for step_index in range(MAX_STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break
        candidates: dict[int, tuple[float, int]] = {}
        records = []
        for source_raw in active_sources:
            source = int(source_raw)
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            local_top = {int(neighbors[i]) for i in best_indices}
            for idx, target_raw in enumerate(neighbors):
                target = int(target_raw)
                signal = float(scores[idx]) * float(brain.signal_decay)
                row = {
                    "source": source,
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "signal": signal,
                    "is_local_top": target in local_top,
                    "passes_threshold": signal >= THRESHOLD,
                }
                records.append(row)
                if not row["is_local_top"] or not row["passes_threshold"]:
                    continue
                old = candidates.get(target)
                if old is None or signal > old[0]:
                    candidates[target] = (signal, source)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining = max(0, brain.max_total_active_nodes - len(activated_nodes))
        selected = []
        new_count = 0
        for target, payload in ranked:
            is_new = target not in activated_nodes
            if is_new and new_count >= remaining:
                continue
            selected.append((target, payload))
            if is_new:
                new_count += 1
            if len(selected) >= min(brain.max_active_per_step, len(ranked)):
                break

        next_activation = np.zeros(brain.node_count, dtype=float)
        accepted = set()
        for target, (signal, source) in selected:
            next_activation[target] = max(next_activation[target], signal)
            accepted.add(edge_key(source, target))
            traversed.add(edge_key(source, target))

        candidate_edges = {edge_key(source, target) for target, (_, source) in candidates.items()}
        for row in records:
            key = tuple(row["edge"])
            row["became_candidate"] = key in candidate_edges
            row["accepted"] = key in accepted

        active_now = np.flatnonzero(next_activation > 0).tolist()
        steps.append({
            "step": step_index,
            "active_sources": [int(x) for x in active_sources],
            "accepted_edges": [list(x) for x in sorted(accepted)],
            "edge_records": records,
            "active_now": active_now,
        })
        if not active_now:
            break
        activated_nodes.update(active_now)
        activation = next_activation

    return {
        "key_node": key_node,
        "key_activation": key_activation,
        "include_position": include_position,
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "steps": steps,
    }


def classify(record: dict | None) -> str:
    if record is None:
        return "not_visible"
    if record.get("accepted"):
        return "selected"
    if record.get("became_candidate"):
        return "candidate_but_not_selected"
    if record.get("is_local_top") and not record.get("passes_threshold"):
        return "local_top_but_below_threshold"
    return "neighbor_seen_but_not_local_top"


def best_edge_state(trace: dict, edge: tuple[int, int]) -> dict:
    best = None
    for step in trace["steps"]:
        for record in step["edge_records"]:
            if tuple(record["edge"]) != edge:
                continue
            row = {"step": int(step["step"]), **record}
            status = classify(row)
            row["status"] = status
            if best is None or (STATUS_RANK[status], row["signal"]) > (STATUS_RANK[best["status"]], best["signal"]):
                best = row
    if best is None:
        return {"status": "not_visible", "step": None, "signal": 0.0}
    return best


def condition_summary(trace: dict, chain: list[dict]) -> dict:
    traversed = edge_set(trace["traversed_edges"])
    rows = []
    for index, item in enumerate(chain):
        edge = tuple(item["edge"])
        state = best_edge_state(trace, edge)
        rows.append({
            "chain_index": index,
            "binding_step": int(item["step"]),
            "edge": list(edge),
            "selected": edge in traversed,
            "best_state": state,
        })
    replayed = sum(1 for row in rows if row["selected"])
    first_contact = next((row for row in rows if row["best_state"]["status"] != "not_visible"), None)
    first_selected = next((row for row in rows if row["selected"]), None)
    return {
        "chain_edge_count": len(rows),
        "replayed_edge_count": replayed,
        "replay_ratio": 0.0 if not rows else replayed / len(rows),
        "first_visible": first_contact,
        "first_selected": first_selected,
        "rows": rows,
    }


def compare_key(position: str, key_node: int, key_activation: float, chain: list[dict]) -> dict:
    key_only_trace = run_condition(position, key_node, key_activation, include_position=False)
    position_only_trace = run_condition(position, None, 0.0, include_position=True)
    combined_trace = run_condition(position, key_node, key_activation, include_position=True)

    key_only = condition_summary(key_only_trace, chain)
    position_only = condition_summary(position_only_trace, chain)
    combined = condition_summary(combined_trace, chain)

    deltas = []
    for a, b, c in zip(key_only["rows"], position_only["rows"], combined["rows"]):
        ks = a["best_state"]["status"]
        ps = b["best_state"]["status"]
        cs = c["best_state"]["status"]
        baseline_rank = max(STATUS_RANK[ks], STATUS_RANK[ps])
        gain = STATUS_RANK[cs] - baseline_rank
        deltas.append({
            "chain_index": c["chain_index"],
            "edge": c["edge"],
            "key_only_status": ks,
            "position_only_status": ps,
            "combined_status": cs,
            "visibility_gain": gain,
            "synergy": gain > 0,
        })

    first_synergy = next((row for row in deltas if row["synergy"]), None)
    return {
        "node": key_node,
        "trace_activation": key_activation,
        "key_only": key_only,
        "position_only": position_only,
        "combined": combined,
        "visibility_deltas": deltas,
        "first_synergy": first_synergy,
        "synergy_edge_count": sum(1 for row in deltas if row["synergy"]),
        "traces": {
            "key_only": key_only_trace,
            "position_only": position_only_trace,
            "combined": combined_trace,
        },
    }


def diagnose(player: str, other: str) -> dict:
    base = v20.diagnose(player, other)
    reference = v12.binding_reference("E", other)
    chain = reference.get("chain", [])
    reports = [
        compare_key(other, int(role["node"]), float(role["trace_activation"]), chain)
        for role in base["E"]["roles"]
    ]
    return {
        "position": other,
        "binding_chain": chain,
        "keys": reports,
        "comparison": {
            "keys_full_without_position": sum(1 for r in reports if r["key_only"]["replay_ratio"] == 1.0),
            "keys_full_with_position": sum(1 for r in reports if r["combined"]["replay_ratio"] == 1.0),
            "keys_with_synergy": sum(1 for r in reports if r["synergy_edge_count"] > 0),
        },
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v23",
        "world": {"P": player, "E": other},
        "purpose": "Compare E Recall Key propagation with and without the simultaneous position input and locate the exact visibility transition for Binding-chain edges.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "core_file_modified": False,
            "puzzle_specific_adjustment": False,
        },
        "diagnostics": diagnose(player, other),
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v23.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v23</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:880px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v23</h1><p class="lead">E Recall Keyを、Keyだけ・位置だけ・Key＋位置の3条件でStep伝播させ、Binding連鎖がどの状態遷移で初めて可視化されたかを比較する。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">可視化差を調べる</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function first(x){return x?`${x.key_only_status} / ${x.position_only_status} → ${x.combined_status}`:'なし'}function cards(r,i){return `<div class="metric">Key ${i+1}<b class="blue">Node ${r.node}</b></div><div class="metric">Keyだけ再生率<b>${f(r.key_only.replay_ratio)}</b></div><div class="metric">位置だけ再生率<b>${f(r.position_only.replay_ratio)}</b></div><div class="metric">Key＋位置再生率<b class="${r.combined.replay_ratio===1?'good':'warn'}">${f(r.combined.replay_ratio)}</b></div><div class="metric">可視化上昇Edge<b>${r.synergy_edge_count}</b></div><div class="metric">最初の相乗変化<b>${first(r.first_synergy)}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const res=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await res.json(),x=d.diagnostics;document.getElementById('metrics').innerHTML=x.keys.map(cards).join('')+`<div class="metric">位置なし完全再生Key<b>${x.comparison.keys_full_without_position}</b></div><div class="metric">位置あり完全再生Key<b>${x.comparison.keys_full_with_position}</b></div><div class="metric">相乗効果ありKey<b>${x.comparison.keys_with_synergy}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v23: http://{HOST}:{PORT}")
    print("Recall Key visibility: key-only vs position-only vs combined")
    serve(app, host=HOST, port=PORT)
