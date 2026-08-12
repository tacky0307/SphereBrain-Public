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
import run_core_growth_binding_v10 as v10
import run_core_growth_binding_v12 as v12
import run_core_growth_binding_v13 as v13

HOST = "127.0.0.1"
START_PORT = 5049
OUT = ROOT / "data" / "core_growth_binding_v14" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
MAX_STEPS = 12


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


def edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def run_variant(
    entity: str,
    position: str,
    *,
    position_sources: list[int],
    echo_nodes: list[int],
) -> dict:
    """Run the current Core propagation from an explicitly composed initial state.

    This is diagnostic-only. It does not learn, alter weights, create edges, or use
    Structural Assist. Position source nodes start at 1.0 and echo nodes at the
    existing Binding echo strength.
    """
    brain = copy.deepcopy(v3.base.CORE)
    activation = np.zeros(brain.node_count, dtype=float)
    for node in position_sources:
        activation[int(node)] = 1.0
    for node in echo_nodes:
        activation[int(node)] = max(activation[int(node)], float(v3.ECHO_STRENGTH))

    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed_edges: set[tuple[int, int]] = set()
    step_records = []

    for step_index in range(MAX_STEPS):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            break

        candidates: dict[int, tuple[float, int]] = {}
        edge_records = []
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            local_top = {int(neighbors[i]) for i in best_indices}

            for idx, target_raw in enumerate(neighbors):
                target = int(target_raw)
                signal = float(scores[idx]) * brain.signal_decay
                is_local_top = target in local_top
                row = {
                    "source": int(source),
                    "target": target,
                    "edge": list(edge_key(source, target)),
                    "source_activation": float(activation[source]),
                    "weight": float(brain.weights[source, target]),
                    "signal": signal,
                    "is_local_top": is_local_top,
                    "passes_threshold": bool(signal >= THRESHOLD),
                }
                edge_records.append(row)
                if not is_local_top or signal < THRESHOLD:
                    continue
                previous = candidates.get(target)
                if previous is None or signal > previous[0]:
                    candidates[target] = (signal, int(source))

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
        step_limit = min(brain.max_active_per_step, len(ranked))
        selected = []
        new_nodes = 0
        for target, payload in ranked:
            is_new = target not in activated_nodes
            if is_new and new_nodes >= remaining_capacity:
                continue
            selected.append((target, payload))
            if is_new:
                new_nodes += 1
            if len(selected) >= step_limit:
                break

        next_activation = np.zeros(brain.node_count, dtype=float)
        accepted = []
        for target, (signal, source) in selected:
            next_activation[target] = max(next_activation[target], signal)
            accepted.append(edge_key(source, target))
            traversed_edges.add(edge_key(source, target))

        accepted_set = set(accepted)
        candidate_set = {
            edge_key(source, target)
            for target, (_, source) in candidates.items()
        }
        for row in edge_records:
            key = tuple(row["edge"])
            row["became_candidate"] = key in candidate_set
            row["accepted"] = key in accepted_set

        active_now = np.flatnonzero(next_activation > 0).tolist()
        step_records.append({
            "step": step_index,
            "active_sources": [int(x) for x in active_sources],
            "active_values": {str(int(x)): float(activation[x]) for x in active_sources},
            "edge_records": edge_records,
            "accepted_edges": [list(x) for x in sorted(accepted_set)],
            "active_now": active_now,
        })
        if not active_now:
            break
        activated_nodes.update(active_now)
        activation = next_activation

    return {
        "entity": entity,
        "position": position,
        "position_sources": [int(x) for x in position_sources],
        "echo_nodes": [int(x) for x in echo_nodes],
        "activated_nodes": sorted(activated_nodes),
        "traversed_edges": [list(x) for x in sorted(traversed_edges)],
        "step_records": step_records,
    }


def entry_measure(trace: dict, entry_node: int) -> dict:
    value, step = v13.active_value_at(trace, entry_node)
    incoming = v13.incoming_breakdown(trace, entry_node)
    return {
        "entry_activation": float(value),
        "entry_activation_step": step,
        "incoming": incoming,
    }


def diagnose(entity: str, position: str) -> dict:
    reference = v12.binding_reference(entity, position)
    entry = reference.get("entry")
    if entry is None:
        return {"entity": entity, "position": position, "entry": None}

    entry_node = int(entry["source"])
    entity_stage = v3.propagate(
        copy.deepcopy(v3.base.CORE), v3.entity_nodes(entity), learn=False, steps=8
    )
    echo_nodes = [int(x) for x in entity_stage["final_active_nodes"][: v3.ECHO_LIMIT]]
    position_sources = [int(x) for x in v3.position_nodes(position)]

    variants_spec = {
        "full_binding_initial_state": (position_sources, echo_nodes),
        "position_only": (position_sources, []),
        "echo_only": ([], echo_nodes),
        "neither": ([], []),
        "full_without_entry_direct": (
            [x for x in position_sources if x != entry_node],
            [x for x in echo_nodes if x != entry_node],
        ),
        "position_without_entry_direct": (
            [x for x in position_sources if x != entry_node],
            [],
        ),
        "echo_without_entry_direct": (
            [],
            [x for x in echo_nodes if x != entry_node],
        ),
    }

    variants = {}
    for name, (pos_sources, echoes) in variants_spec.items():
        trace = run_variant(
            entity,
            position,
            position_sources=pos_sources,
            echo_nodes=echoes,
        )
        variants[name] = {
            "trace": trace,
            "measure": entry_measure(trace, entry_node),
        }

    full_value = variants["full_binding_initial_state"]["measure"]["entry_activation"]
    position_value = variants["position_only"]["measure"]["entry_activation"]
    echo_value = variants["echo_only"]["measure"]["entry_activation"]
    neither_value = variants["neither"]["measure"]["entry_activation"]
    without_direct_value = variants["full_without_entry_direct"]["measure"]["entry_activation"]

    position_ablation_drop = full_value - echo_value
    echo_ablation_drop = full_value - position_value
    direct_entry_drop = full_value - without_direct_value
    interaction_remainder = full_value - max(position_value, echo_value, neither_value)

    echo_node_ablation = []
    for node in echo_nodes:
        trace = run_variant(
            entity,
            position,
            position_sources=position_sources,
            echo_nodes=[x for x in echo_nodes if x != node],
        )
        measure = entry_measure(trace, entry_node)
        echo_node_ablation.append({
            "removed_echo_node": node,
            "entry_activation": measure["entry_activation"],
            "drop_from_full": full_value - measure["entry_activation"],
            "entry_activation_step": measure["entry_activation_step"],
        })

    position_node_ablation = []
    for node in position_sources:
        trace = run_variant(
            entity,
            position,
            position_sources=[x for x in position_sources if x != node],
            echo_nodes=echo_nodes,
        )
        measure = entry_measure(trace, entry_node)
        position_node_ablation.append({
            "removed_position_node": node,
            "entry_activation": measure["entry_activation"],
            "drop_from_full": full_value - measure["entry_activation"],
            "entry_activation_step": measure["entry_activation_step"],
        })

    largest_echo_source = max(
        echo_node_ablation,
        key=lambda row: row["drop_from_full"],
        default=None,
    )
    largest_position_source = max(
        position_node_ablation,
        key=lambda row: row["drop_from_full"],
        default=None,
    )

    return {
        "entity": entity,
        "position": position,
        "entry": entry,
        "entry_node": entry_node,
        "entry_is_position_source": entry_node in set(position_sources),
        "entry_is_echo_node": entry_node in set(echo_nodes),
        "position_sources": position_sources,
        "echo_nodes": echo_nodes,
        "variants": variants,
        "ablation_summary": {
            "full_entry_activation": full_value,
            "position_only_entry_activation": position_value,
            "echo_only_entry_activation": echo_value,
            "neither_entry_activation": neither_value,
            "full_without_direct_entry_activation": without_direct_value,
            "drop_when_position_group_removed": position_ablation_drop,
            "drop_when_echo_group_removed": echo_ablation_drop,
            "drop_when_direct_entry_initialization_removed": direct_entry_drop,
            "interaction_remainder_over_stronger_single_group": interaction_remainder,
            "largest_echo_source": largest_echo_source,
            "largest_position_source": largest_position_source,
        },
        "echo_node_ablation": echo_node_ablation,
        "position_node_ablation": position_node_ablation,
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v14",
        "world": {"P": player, "E": other},
        "purpose": "Locate the source of Binding entry activation by source-group and node ablation.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "structural_assist": False,
            "puzzle_specific_rules": False,
            "core_behavior_changed": False,
            "ablation_is_diagnostic_only": True,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v14.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v14</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:22px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:780px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v14</h1><p class="lead">Binding入口activationの出所を、位置入力群・主体echo群・入口Nodeへの直接初期化・各入力Nodeの除去で分解する。Core本体は変更しない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">出所を追う</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function cards(label,d){if(!d.entry)return `<div class="metric">${label}<b>入口なし</b></div>`;const a=d.ablation_summary;return `<div class="metric">${label} full入口<b>${f(a.full_entry_activation)}</b></div><div class="metric">${label} 位置だけ<b>${f(a.position_only_entry_activation)}</b></div><div class="metric">${label} echoだけ<b>${f(a.echo_only_entry_activation)}</b></div><div class="metric">${label} 直接初期化除去<b>${f(a.full_without_direct_entry_activation)}</b></div><div class="metric">${label} 位置群除去drop<b class="blue">${f(a.drop_when_position_group_removed)}</b></div><div class="metric">${label} echo群除去drop<b class="blue">${f(a.drop_when_echo_group_removed)}</b></div><div class="metric">${label} 直接入口drop<b class="warn">${f(a.drop_when_direct_entry_initialization_removed)}</b></div><div class="metric">${label} 相互作用残差<b>${f(a.interaction_remainder_over_stronger_single_group)}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=cards('P',d.diagnostics.P)+cards('E',d.diagnostics.E)+`<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v14: http://{HOST}:{PORT}")
    print("Binding entry source ablation / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
