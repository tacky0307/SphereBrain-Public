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
import run_core_growth_binding_v10 as v10
import run_core_growth_binding_v12 as v12

HOST = "127.0.0.1"
START_PORT = 5048
OUT = ROOT / "data" / "core_growth_binding_v13" / "results"
POSITIONS = v3.POSITIONS
THRESHOLD = 0.18
REQUIRED_ENTRY = 0.35


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


def edge_set(edges) -> set[tuple[int, int]]:
    return {edge_key(a, b) for a, b in edges}


def active_value_at(trace: dict, node: int) -> tuple[float, int | None]:
    best = 0.0
    best_step = None
    for step in trace.get("step_records", []):
        value = float(step.get("active_values", {}).get(str(node), 0.0))
        if value > best:
            best = value
            best_step = int(step["step"])
    return best, best_step


def incoming_breakdown(trace: dict, target_node: int) -> dict:
    rows = []
    best_actual = 0.0
    best_actual_step = None
    best_local_top_sum = 0.0
    best_local_top_sum_step = None
    best_all_sum = 0.0
    best_all_sum_step = None

    for step in trace.get("step_records", []):
        incoming = [
            row for row in step.get("edge_records", [])
            if int(row["target"]) == int(target_node)
        ]
        local_top = [row for row in incoming if row.get("is_local_top")]
        accepted = [row for row in incoming if row.get("accepted")]

        actual_max = max((float(row["signal"]) for row in accepted), default=0.0)
        local_top_sum = sum(float(row["signal"]) for row in local_top)
        all_sum = sum(float(row["signal"]) for row in incoming)

        if actual_max > best_actual:
            best_actual = actual_max
            best_actual_step = int(step["step"])
        if local_top_sum > best_local_top_sum:
            best_local_top_sum = local_top_sum
            best_local_top_sum_step = int(step["step"])
        if all_sum > best_all_sum:
            best_all_sum = all_sum
            best_all_sum_step = int(step["step"])

        rows.append({
            "step": int(step["step"]),
            "incoming_edge_count": len(incoming),
            "local_top_incoming_count": len(local_top),
            "accepted_incoming_count": len(accepted),
            "actual_accepted_max": actual_max,
            "hypothetical_local_top_sum": local_top_sum,
            "hypothetical_all_incoming_sum": all_sum,
            "incoming": incoming,
        })

    return {
        "rows": rows,
        "best_actual_accepted_max": best_actual,
        "best_actual_step": best_actual_step,
        "best_hypothetical_local_top_sum": best_local_top_sum,
        "best_hypothetical_local_top_sum_step": best_local_top_sum_step,
        "best_hypothetical_all_sum": best_all_sum,
        "best_hypothetical_all_sum_step": best_all_sum_step,
    }


def assist_shadow(entity: str, position: str) -> dict:
    off = v3.make_binding(copy.deepcopy(v3.base.CORE), entity, position, learn=False, assist=False)
    on = v3.make_binding(copy.deepcopy(v3.base.CORE), entity, position, learn=False, assist=True)
    off_nodes = set(off["bound_nodes"])
    on_nodes = set(on["bound_nodes"])
    off_edges = edge_set(off["bound_edges"])
    on_edges = edge_set(on["bound_edges"])
    node_union = off_nodes | on_nodes
    edge_union = off_edges | on_edges
    return {
        "off_rank_changes": off["bound_stage"]["assist_rank_changes"],
        "on_rank_changes": on["bound_stage"]["assist_rank_changes"],
        "node_jaccard": 1.0 if not node_union else len(off_nodes & on_nodes) / len(node_union),
        "edge_jaccard": 1.0 if not edge_union else len(off_edges & on_edges) / len(edge_union),
        "observed_path_change": off_nodes != on_nodes or off_edges != on_edges,
    }


def diagnose(entity: str, position: str) -> dict:
    reference = v12.binding_reference(entity, position)
    entry = reference.get("entry")
    if entry is None:
        return {
            "entity": entity,
            "position": position,
            "entry": None,
            "components": None,
        }

    entry_source = int(entry["source"])
    natural = v10.run_detailed(entity, None, threshold=THRESHOLD, binding=False)
    binding = reference["trace"]

    natural_activation, natural_step = active_value_at(natural, entry_source)
    binding_activation, binding_step = active_value_at(binding, entry_source)
    natural_incoming = incoming_breakdown(natural, entry_source)
    binding_incoming = incoming_breakdown(binding, entry_source)

    direct_input = 1.0 if entry_source in set(v3.entity_nodes(entity)) else 0.0
    echo_nodes = set(binding.get("echo_nodes", []))
    explicit_echo = v3.ECHO_STRENGTH if entry_source in echo_nodes else 0.0
    natural_gap = max(0.0, REQUIRED_ENTRY - natural_activation)

    potential_local_top = max(
        natural_activation,
        natural_incoming["best_hypothetical_local_top_sum"],
    )
    potential_all = max(
        natural_activation,
        natural_incoming["best_hypothetical_all_sum"],
    )

    return {
        "entity": entity,
        "position": position,
        "entry": entry,
        "required_entry_activation": REQUIRED_ENTRY,
        "components": {
            "direct_input_component": direct_input,
            "natural_entry_activation": natural_activation,
            "natural_entry_activation_step": natural_step,
            "binding_entry_activation": binding_activation,
            "binding_entry_activation_step": binding_step,
            "explicit_binding_echo_component": explicit_echo,
            "natural_gap_to_required": natural_gap,
            "natural_incoming": natural_incoming,
            "binding_incoming": binding_incoming,
            "hypothetical_if_local_top_inputs_summed": potential_local_top,
            "hypothetical_local_top_gap": max(0.0, REQUIRED_ENTRY - potential_local_top),
            "hypothetical_if_all_inputs_summed": potential_all,
            "hypothetical_all_gap": max(0.0, REQUIRED_ENTRY - potential_all),
            "current_aggregation_rule": "max incoming activation, not sum",
            "natural_short_term_residual_component": 0.0,
            "assist_shadow": assist_shadow(entity, position),
        },
        "interpretation_flags": {
            "directly_stimulated_entry": direct_input > 0,
            "binding_entry_was_explicit_echo_node": explicit_echo > 0,
            "local_top_sum_would_reach_required": potential_local_top >= REQUIRED_ENTRY,
            "all_incoming_sum_would_reach_required": potential_all >= REQUIRED_ENTRY,
        },
    }


def observe(player: str, other: str) -> dict:
    payload = {
        "experiment": "Core Growth Binding v13",
        "world": {"P": player, "E": other},
        "purpose": "Decompose the activation available at a Binding recall entry and identify which generic component is missing.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "puzzle_specific_rules": False,
            "core_behavior_changed": False,
            "hypothetical_sums_are_diagnostic_only": True,
        },
        "diagnostics": {
            "P": diagnose("P", player),
            "E": diagnose("E", other),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v13.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v13</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:rgba(23,37,60,.96);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}select,button{padding:14px;border-radius:12px;border:1px solid #466486;background:#0d1828;color:var(--text);font-size:16px}button{background:var(--orange);color:#101722;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:22px;margin-top:6px}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:760px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.controls,.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v13</h1><p class="lead">0.35へ届く入口activationを、直接入力・既存経路流入・合流・残響・Binding時のecho・Structural Assistに分けて診断する。Core本体は変更しない。</p><section class="panel"><div class="controls"><select id="p"><option>左</option><option>中央</option><option>右</option></select><select id="e"><option>左</option><option>中央</option><option selected>右</option></select><button onclick="run()">成分を測る</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Core生データ</h2><pre id="raw" class="raw">まだ測定していません。</pre></section></main><script>
function f(x){return x===null||x===undefined?'なし':Number(x).toFixed(6)}function cards(label,d){if(!d.components)return `<div class="metric">${label}<b>入口なし</b></div>`;const c=d.components;return `<div class="metric">${label}自然入口<b>${f(c.natural_entry_activation)}</b></div><div class="metric">${label}不足<b class="warn">${f(c.natural_gap_to_required)}</b></div><div class="metric">${label}local-top仮想合計<b class="${d.interpretation_flags.local_top_sum_would_reach_required?'good':'blue'}">${f(c.hypothetical_if_local_top_inputs_summed)}</b></div><div class="metric">${label}全流入仮想合計<b class="${d.interpretation_flags.all_incoming_sum_would_reach_required?'good':'blue'}">${f(c.hypothetical_if_all_inputs_summed)}</b></div><div class="metric">${label}Binding時入口<b>${f(c.binding_entry_activation)}</b></div><div class="metric">${label}明示echo<b>${f(c.explicit_binding_echo_component)}</b></div><div class="metric">${label}Assist経路変化<b>${c.assist_shadow.observed_path_change?'あり':'なし'}</b></div>`}async function run(){const p=document.getElementById('p').value,e=document.getElementById('e').value;const r=await fetch('/api/observe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player:p,other:e})});const d=await r.json();document.getElementById('metrics').innerHTML=cards('P',d.diagnostics.P)+cards('E',d.diagnostics.E)+`<div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}
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
    print(f"Core Growth Binding v13: http://{HOST}:{PORT}")
    print("Recall priming component diagnosis / no learning / no Core changes")
    serve(app, host=HOST, port=PORT)
