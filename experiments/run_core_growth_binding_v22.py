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
import run_core_growth_binding_v20 as v20
import run_core_growth_binding_v21 as v21

HOST = "127.0.0.1"
START_PORT = 5057
OUT = ROOT / "data" / "core_growth_binding_v22" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
MAX_STEPS = 10


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


def run_live_trace(entity: str, position: str, node: int, node_activation: float) -> dict:
    brain = copy.deepcopy(v3.base.CORE)
    activation = np.zeros(brain.node_count, dtype=float)
    activation[int(node)] = float(node_activation)
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
                is_top = target in local_top
                row = {
                    "source": source,
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "signal": signal,
                    "is_local_top": is_top,
                    "passes_threshold": signal >= THRESHOLD,
                }
                records.append(row)
                if not is_top or signal < THRESHOLD:
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
            "active_values": {str(int(x)): float(activation[x]) for x in active_sources},
            "edge_records": records,
            "accepted_edges": [list(x) for x in sorted(accepted)],
            "active_now": active_now,
        })
        if not active_now:
            break
        activated_nodes.update(active_now)
        activation = next_activation

    return {
        "entity": entity,
        "position": position,
        "trace_node": int(node),
        "trace_activation": float(node_activation),
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed)],
        "steps": steps,
    }


def classify_record(record: dict | None) -> str:
    if record is None:
        return "not_visible"
    if record.get("accepted"):
        return "selected"
    if record.get("became_candidate"):
        return "candidate_but_not_selected"
    if record.get("is_local_top") and not record.get("passes_threshold"):
        return "local_top_but_below_threshold"
    if not record.get("is_local_top"):
        return "neighbor_seen_but_not_local_top"
    return "unclassified"


def first_chain_contact(trace: dict, chain: list[dict]) -> dict | None:
    chain_edges = {tuple(item["edge"]): item for item in chain}
    for step in trace["steps"]:
        accepted = edge_set(step["accepted_edges"])
        overlap = accepted & set(chain_edges)
        if overlap:
            edge = sorted(overlap)[0]
            return {
                "step": int(step["step"]),
                "edge": list(edge),
                "binding_step": int(chain_edges[edge]["step"]),
            }
    return None


def chain_diagnostics(trace: dict, chain: list[dict]) -> dict:
    traversed = edge_set(trace["traversed_edges"])
    rows = []
    first_failure = None
    for index, item in enumerate(chain):
        target_edge = tuple(item["edge"])
        selected = target_edge in traversed
        best = None
        for step in trace["steps"]:
            for record in step["edge_records"]:
                if tuple(record["edge"]) != target_edge:
                    continue
                candidate = {"step": int(step["step"]), **record}
                if best is None or candidate["signal"] > best["signal"]:
                    best = candidate
        status = "selected" if selected else classify_record(best)
        row = {
            "chain_index": index,
            "binding_step": int(item["step"]),
            "edge": list(target_edge),
            "selected": selected,
            "status": status,
            "best_observation": best,
        }
        rows.append(row)
        if first_failure is None and not selected:
            first_failure = row
    replayed = sum(1 for row in rows if row["selected"])
    return {
        "chain_edge_count": len(rows),
        "replayed_edge_count": replayed,
        "replay_ratio": 0.0 if not rows else replayed / len(rows),
        "first_contact": first_chain_contact(trace, chain),
        "first_failure": first_failure,
        "rows": rows,
    }


def step_zero_summary(trace: dict, chain: list[dict]) -> dict:
    if not trace["steps"]:
        return {"accepted_count": 0, "chain_edges_selected": [], "top_records": []}
    step = trace["steps"][0]
    chain_set = {tuple(item["edge"]) for item in chain}
    accepted = edge_set(step["accepted_edges"])
    top = sorted(
        [row for row in step["edge_records"] if row["is_local_top"]],
        key=lambda row: row["signal"],
        reverse=True,
    )[:20]
    return {
        "accepted_count": len(accepted),
        "chain_edges_selected": [list(x) for x in sorted(accepted & chain_set)],
        "top_records": top,
    }


def live_report(entity: str, position: str, node: int, activation: float, reference: dict) -> dict:
    trace = run_live_trace(entity, position, node, activation)
    chain = reference.get("chain", [])
    diagnostics = chain_diagnostics(trace, chain)
    return {
        "entity": entity,
        "position": position,
        "node": int(node),
        "trace_activation": float(activation),
        "step_zero": step_zero_summary(trace, chain),
        "chain": diagnostics,
        "selected_edge_count": len(trace["traversed_edges"]),
        "activated_node_count": len(trace["activated_nodes"]),
        "trace": trace,
    }


def diagnose(player: str, other: str) -> dict:
    base = v20.diagnose(player, other)
    v21_data = v21.diagnose(player, other)

    e_reference = v12.binding_reference("E", other)
    p_reference = v12.binding_reference("P", player)

    e_reports = [
        live_report("E", other, int(role["node"]), float(role["trace_activation"]), e_reference)
        for role in base["E"]["roles"]
    ]
    p_reports = [
        live_report("P", player, int(row["node"]), float(row["trace_activation"]), p_reference)
        for row in v21_data["P_candidates"]
    ]

    failure_counts: dict[str, int] = {}
    for report in p_reports:
        failure = report["chain"]["first_failure"]
        status = "none" if failure is None else str(failure["status"])
        failure_counts[status] = failure_counts.get(status, 0) + 1

    return {
        "E_keys": e_reports,
        "P_candidates": p_reports,
        "comparison": {
            "e_full_replay_count": sum(1 for x in e_reports if x["chain"]["replay_ratio"] == 1.0),
            "p_any_contact_count": sum(1 for x in p_reports if x["chain"]["first_contact"] is not None),
            "p_any_replay_count": sum(1 for x in p_reports if x["chain"]["replayed_edge_count"] > 0),
            "p_full_replay_count": sum(1 for x in p_reports if x["chain"]["replay_ratio"] == 1.0),
            "p_first_failure_status_counts": failure_counts,
        },
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v22",
        "world": {"P": player, "E": other},
        "purpose": "Compare successful E Recall Keys and failed P candidates using actual step-by-step Core propagation rather than static distance estimates.",
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
    (OUT / "latest_binding_v22.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v22</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:20px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:860px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v22</h1><p class="lead">E Key 80 / 450とP候補5Nodeを、同じ位置入力下で実際にStep伝播させる。local top・threshold・候補化・選択・Binding連鎖への接触点を直接比較する。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">Step伝播を比較</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function rep(r,label,i){const c=r.chain,fail=c.first_failure;return `<div class="metric">${label}${i+1}<b class="blue">Node ${r.node}</b></div><div class="metric">${label}${i+1} Step0選択<b>${r.step_zero.accepted_count}</b></div><div class="metric">${label}${i+1} 初接触<b>${c.first_contact?`Step ${c.first_contact.step}`:'なし'}</b></div><div class="metric">${label}${i+1} 再生率<b class="${c.replay_ratio>0?'good':'warn'}">${f(c.replay_ratio)}</b></div><div class="metric">${label}${i+1} 最初の失敗<b>${fail?fail.status:'なし'}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json(),x=d.diagnostics;document.getElementById('metrics').innerHTML=x.E_keys.map((r,i)=>rep(r,'E Key ',i)).join('')+x.P_candidates.map((r,i)=>rep(r,'P候補 ',i)).join('')+`<div class="metric">P連鎖接触候補<b>${x.comparison.p_any_contact_count}</b></div><div class="metric">P一部再生候補<b>${x.comparison.p_any_replay_count}</b></div><div class="metric">P完全再生候補<b>${x.comparison.p_full_replay_count}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v22: http://{HOST}:{PORT}")
    print("Actual step propagation comparison / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
