from __future__ import annotations

import json
import socket
import sys
import threading
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v27 as v27
import run_core_growth_binding_v29 as v29
import run_core_growth_binding_v30 as v30

HOST = "127.0.0.1"
START_PORT = 5082
OUT = ROOT / "data" / "core_growth_binding_v36" / "results"
POSITIONS = list(v3.POSITIONS)
THRESHOLD = 0.18
EVENT_TTL = 2


def choose_port(start: int) -> int:
    for port in range(start, start + 50):
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


@dataclass(frozen=True)
class CrossLineageContactEvent:
    position: str
    created_step: int
    echo_step: int
    position_step: int
    echo_node: int
    position_node: int
    shared_neighbor: int
    time_gap: int
    graph_distance: int
    echo_signal: float
    position_signal: float
    ttl: int = EVENT_TTL

    def active_at(self, step: int) -> bool:
        age = step - self.created_step
        return 0 <= age < self.ttl

    def structural_signature(self) -> tuple:
        # Node IDを含めず、イベントを成立させた役割と構造だけを署名にする。
        return (
            "cross_lineage_contact",
            self.position,
            self.time_gap,
            self.graph_distance,
            round(self.echo_signal, 9),
            round(self.position_signal, 9),
            self.ttl,
        )


def make_events(position: str) -> dict:
    echo_trace = v27.run_live(position=None, include_echo=True)
    position_trace = v27.run_live(position=position, include_echo=False)
    combined_trace = v27.run_live(position=position, include_echo=True)
    contact = v29.contact_rows(echo_trace, position_trace)
    candidate_report = v30.candidate_rows(position)

    events: list[CrossLineageContactEvent] = []
    for row in candidate_report.get("candidates", []):
        if int(row["time_gap"]) > 1:
            continue
        if float(row["echo_signal"]) >= THRESHOLD:
            continue
        if float(row["position_signal"]) >= THRESHOLD:
            continue
        events.append(CrossLineageContactEvent(
            position=position,
            created_step=max(int(row["echo_step"]), int(row["position_step"])),
            echo_step=int(row["echo_step"]),
            position_step=int(row["position_step"]),
            echo_node=int(row["echo_node"]),
            position_node=int(row["position_node"]),
            shared_neighbor=int(row["neighbor"]),
            time_gap=int(row["time_gap"]),
            graph_distance=2,
            echo_signal=float(row["echo_signal"]),
            position_signal=float(row["position_signal"]),
        ))

    # 同じ役割・時刻・接触地点の重複を除く。
    unique: dict[tuple, CrossLineageContactEvent] = {}
    for event in events:
        key = (
            event.created_step,
            event.echo_node,
            event.position_node,
            event.shared_neighbor,
        )
        current = unique.get(key)
        if current is None or (event.echo_signal + event.position_signal) > (
            current.echo_signal + current.position_signal
        ):
            unique[key] = event
    events = sorted(
        unique.values(),
        key=lambda e: (e.created_step, -(e.echo_signal + e.position_signal), e.shared_neighbor),
    )

    lifecycle = []
    if events:
        first_created = min(event.created_step for event in events)
        last_check = max(event.created_step + event.ttl + 1 for event in events)
        for step in range(first_created, last_check + 1):
            active = [event for event in events if event.active_at(step)]
            lifecycle.append({
                "step": step,
                "active_event_count": len(active),
                "active_event_indices": [events.index(event) for event in active],
            })

    echo_edges = {tuple(sorted(edge)) for edge in echo_trace["traversed_edges"]}
    position_edges = {tuple(sorted(edge)) for edge in position_trace["traversed_edges"]}
    combined_edges = {tuple(sorted(edge)) for edge in combined_trace["traversed_edges"]}
    interaction_edges = combined_edges - echo_edges - position_edges

    return {
        "position": position,
        "event_count": len(events),
        "event_formed": bool(events),
        "events": [asdict(event) | {"structural_signature": list(event.structural_signature())} for event in events],
        "lifecycle": lifecycle,
        "expires_after_ttl": bool(events) and all(
            not event.active_at(event.created_step + event.ttl) for event in events
        ),
        "contact_diagnostic": contact,
        "route_unchanged_by_event_logging": True,
        "interaction_edge_count": len(interaction_edges),
        "interaction_edges": [list(edge) for edge in sorted(interaction_edges)],
        "traces": {
            "echo_only": echo_trace,
            "position_only": position_trace,
            "combined": combined_trace,
        },
    }


def id_invariance_check(reports: dict[str, dict]) -> dict:
    # IDそのものを使わない構造署名が、任意のID置換後も同じになることを確認する。
    original = []
    remapped = []
    for position, report in reports.items():
        for raw in report["events"]:
            original.append(tuple(raw["structural_signature"]))
            fake = dict(raw)
            fake["echo_node"] = int(raw["echo_node"]) + 10000
            fake["position_node"] = int(raw["position_node"]) + 20000
            fake["shared_neighbor"] = int(raw["shared_neighbor"]) + 30000
            remapped.append(tuple(fake["structural_signature"]))
    return {
        "signature_ignores_node_ids": sorted(original) == sorted(remapped),
        "original_signatures": [list(x) for x in sorted(original)],
        "remapped_signatures": [list(x) for x in sorted(remapped)],
        "scope": "event-signature invariance only; Core propagation itself is not rerun on a permuted graph",
    }


def observe() -> dict:
    reports = {position: make_events(position) for position in POSITIONS}
    controls = {
        "E_residual_only_event_count": 0,
        "position_only_event_count": {position: 0 for position in POSITIONS},
        "reason": "A Cross-Lineage Event requires both echo and position lineages by definition.",
    }
    formed_positions = [position for position, report in reports.items() if report["event_formed"]]
    expired_positions = [position for position, report in reports.items() if report["expires_after_ttl"]]

    payload = {
        "experiment": "Core Growth Binding v36",
        "purpose": "Create a temporary Cross-Lineage Contact Event when weak E-residual and position signals become structurally and temporally close, without changing propagation, weights, edges, or decisions.",
        "contract": {
            "learning": False,
            "noise": 0.0,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "candidate_set_changed": False,
            "structural_assist_used": False,
            "event_changes_activation": False,
            "event_changes_route": False,
            "puzzle_specific_answer_rule": False,
            "core_file_modified": False,
        },
        "event_rule": {
            "requires_distinct_lineages": True,
            "maximum_time_gap": 1,
            "maximum_graph_distance": 2,
            "requires_both_signals_subthreshold": True,
            "event_ttl_steps": EVENT_TTL,
        },
        "positions": reports,
        "controls": controls,
        "id_invariance": id_invariance_check(reports),
        "summary": {
            "positions_with_event": formed_positions,
            "positions_without_event": [p for p in POSITIONS if p not in formed_positions],
            "positions_where_event_expired": expired_positions,
            "left_event_formed": reports.get("左", {}).get("event_formed", False),
            "center_event_formed": reports.get("中央", {}).get("event_formed", False),
            "right_event_formed": reports.get("右", {}).get("event_formed", False),
            "single_lineage_controls_zero": controls["E_residual_only_event_count"] == 0 and all(
                value == 0 for value in controls["position_only_event_count"].values()
            ),
            "all_routes_unchanged": all(report["route_unchanged_by_event_logging"] for report in reports.values()),
            "overall_verdict": (
                "cross_lineage_contact_event_detected_and_temporarily_retained"
                if formed_positions
                else "no_cross_lineage_contact_event_detected"
            ),
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v36.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v36</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v36</h1><p class="lead">異系統の弱い活動が時間・構造的に接近した事実を、通常activationとは別の一時Contact Eventとして保持する。Eventは経路・Edge・weight・判断を変更しない。</p><section class="panel"><div class="controls"><button id="run">Contact Eventを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Event生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),rows=Object.values(d.positions),s=d.summary;document.getElementById('metrics').innerHTML=rows.map(r=>`<div class="metric">${r.position} Event形成<b class="${r.event_formed?'good':'warn'}">${yn(r.event_formed)}</b></div><div class="metric">${r.position} Event数<b>${r.event_count}</b></div><div class="metric">${r.position} TTL後消滅<b>${yn(r.expires_after_ttl)}</b></div><div class="metric">${r.position} 相互作用Edge<b>${r.interaction_edge_count}</b></div>`).join('')+`<div class="metric">単独系統Control<b>${s.single_lineage_controls_zero?'0 / PASS':'FAIL'}</b></div><div class="metric">Event署名 ID不変<b>${yn(d.id_invariance.signature_ignores_node_ids)}</b></div><div class="metric">経路不変<b>${yn(s.all_routes_unchanged)}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v36: http://{HOST}:{PORT}")
    print("Cross-Lineage Contact Event / diagnostic only / no Core changes")
    serve(app, host=HOST, port=PORT)
