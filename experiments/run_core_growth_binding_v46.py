from __future__ import annotations

import json
import socket
import sys
import threading
import webbrowser
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v42 as v42
import run_core_growth_binding_v43 as v43
import run_core_growth_binding_v44 as v44

HOST = "127.0.0.1"
START_PORT = 5092
OUT = ROOT / "data" / "core_growth_binding_v46" / "results"
POSITIONS = ["左", "中央", "右"]
WINDOW = 1


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


def named_relative_components(report: dict) -> dict[str, float] | None:
    """v43 1Step Relative Context Identityを、監査可能な名前付き成分へ展開する。"""
    if not report.get("events"):
        return None

    event = report["events"][0]
    echo_trace = report["traces"]["echo_only"]
    position_trace = report["traces"]["position_only"]

    components: dict[str, float] = {
        "event.time_gap": float(event["time_gap"]),
        "event.graph_distance": float(event["graph_distance"]),
        "event.arrival_order": (
            1.0 if int(event["echo_step"]) < int(event["position_step"])
            else (-1.0 if int(event["echo_step"]) > int(event["position_step"]) else 0.0)
        ),
    }

    echo_groups = v42.step_groups(echo_trace, int(event["echo_step"]), "")
    position_groups = v42.step_groups(position_trace, int(event["position_step"]), "")

    for group in v43.STEP_GROUPS:
        ev = list(echo_groups[group])
        pv = list(position_groups[group])
        width = max(len(ev), len(pv))
        ev += [0.0] * (width - len(ev))
        pv += [0.0] * (width - len(pv))
        for index, (a, b) in enumerate(zip(ev, pv)):
            components[f"{group}[{index}].relation"] = float(v43.relation(float(a), float(b)))
            components[f"{group}[{index}].ratio_band"] = float(
                v43.ratio_band(float(a) + 1e-9, float(b) + 1e-9)
            )

    topology = v43.local_topology_relative(event)
    topology_names = [
        "topology.echo_vs_position_degree",
        "topology.neighbor_vs_source_mean_degree",
        "topology.shared_neighbor_presence",
        "topology.triangle_presence",
        "topology.echo_vs_position_degree_band",
    ]
    for name, value in zip(topology_names, topology):
        components[name] = float(value)

    return components


def stable_skeleton(rows: list[dict]) -> dict:
    component_rows = []
    for row in rows:
        if not row.get("event_formed"):
            continue
        named = named_relative_components(row["report"])
        if named is not None:
            component_rows.append(named)

    if not component_rows:
        return {
            "event_complete": False,
            "condition_count": len(rows),
            "observed_count": 0,
            "stable_components": {},
            "unstable_components": {},
            "stable_fraction": 0.0,
        }

    names = sorted(set.intersection(*(set(row.keys()) for row in component_rows)))
    stable: dict[str, float] = {}
    unstable: dict[str, dict] = {}

    for name in names:
        values = [row[name] for row in component_rows]
        unique = sorted(set(values))
        if len(unique) == 1:
            stable[name] = unique[0]
        else:
            counts = Counter(values)
            unstable[name] = {
                "values": values,
                "unique_values": unique,
                "mode": counts.most_common(1)[0][0],
                "mode_count": counts.most_common(1)[0][1],
            }

    total = len(names)
    return {
        "event_complete": len(component_rows) == len(rows),
        "condition_count": len(rows),
        "observed_count": len(component_rows),
        "stable_components": stable,
        "unstable_components": unstable,
        "stable_component_count": len(stable),
        "unstable_component_count": len(unstable),
        "total_component_count": total,
        "stable_fraction": 0.0 if total == 0 else len(stable) / total,
    }


def compare_skeletons(left: dict, center: dict) -> dict:
    ls = left.get("stable_components", {})
    cs = center.get("stable_components", {})
    common_names = sorted(set(ls) & set(cs))
    same = {name: ls[name] for name in common_names if ls[name] == cs[name]}
    discriminative = {
        name: {"left": ls[name], "center": cs[name]}
        for name in common_names if ls[name] != cs[name]
    }
    only_left = {name: ls[name] for name in sorted(set(ls) - set(cs))}
    only_center = {name: cs[name] for name in sorted(set(cs) - set(ls))}

    return {
        "shared_stable_name_count": len(common_names),
        "same_value_count": len(same),
        "discriminative_component_count": len(discriminative),
        "same_value_components": same,
        "discriminative_components": discriminative,
        "left_only_stable_components": only_left,
        "center_only_stable_components": only_center,
        "skeletons_distinct": bool(discriminative),
    }


def compact_condition_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "condition": row["condition"],
            "echo_scale": row["echo_scale"],
            "position_scale": row["position_scale"],
            "event_formed": row["event_formed"],
            "identity": row["identity"],
            "combined_edges": row["report"]["traces"]["combined"]["traversed_edges"],
        }
        for row in rows
    ]


def observe() -> dict:
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    skeletons = {
        position: stable_skeleton(rows)
        for position, rows in runs.items()
    }
    comparison = compare_skeletons(skeletons["左"], skeletons["中央"])

    v44_summaries = {
        position: v44.summarize_position(rows)
        for position, rows in runs.items()
    }
    right_absent = all(not row["event_formed"] for row in runs["右"])
    left_complete = skeletons["左"]["event_complete"]
    center_complete = skeletons["中央"]["event_complete"]

    if left_complete and center_complete and right_absent and comparison["skeletons_distinct"]:
        verdict = "distinct_invariant_context_skeletons_found"
        next_step = "validate_skeleton_across_broader_live_perturbations_before_core_integration"
    elif left_complete and center_complete and not comparison["skeletons_distinct"]:
        verdict = "invariant_skeleton_exists_but_does_not_separate_left_center"
        next_step = "seek_higher_order_invariants_or_longer_context"
    else:
        verdict = "invariant_skeleton_inconclusive_due_to_event_instability"
        next_step = "stabilize_contact_event_before_skeleton_integration"

    payload = {
        "experiment": "Core Growth Binding v46",
        "purpose": "Extract only Relative Context components that remain unchanged across all seven live input-strength conditions, then test whether left and center retain distinct invariant skeletons.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "identity_rule": "A component belongs to the invariant skeleton only if its categorical value is identical across every live condition for that position.",
            "excludes": ["Node IDs", "position labels", "absolute Event signal", "TTL", "event age"],
        },
        "conditions": [
            {"name": name, "echo_scale": e, "position_scale": p}
            for name, e, p in v44.CONDITIONS
        ],
        "skeletons": skeletons,
        "comparison": comparison,
        "live_route_stability": {
            position: {
                "minimum_route_jaccard_vs_baseline": summary["minimum_route_jaccard_vs_baseline"],
                "route_jaccards_vs_baseline": summary["route_jaccards_vs_baseline"],
            }
            for position, summary in v44_summaries.items()
        },
        "runs": {
            position: compact_condition_rows(rows)
            for position, rows in runs.items()
        },
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent": right_absent,
            "left_stable_fraction": skeletons["左"]["stable_fraction"],
            "center_stable_fraction": skeletons["中央"]["stable_fraction"],
            "shared_stable_component_count": comparison["shared_stable_name_count"],
            "discriminative_invariant_component_count": comparison["discriminative_component_count"],
            "left_center_skeleton_distinct": comparison["skeletons_distinct"],
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v46.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v46</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v46</h1><p class="lead">7つのlive input条件を横断し、毎回同じ値で残ったRelative Context成分だけをInvariant Skeletonとして抽出する。揺れる成分はIdentityから自動的に除外する。</p><section class="panel"><div class="controls"><button id="run">Invariant Skeletonを抽出</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Skeleton生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,c=d.comparison,sk=d.skeletons;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event全条件<b class="${s.left_event_all_conditions?'good':'warn'}">${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b class="${s.center_event_all_conditions?'good':'warn'}">${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">左 Skeleton保持率<b>${f(s.left_stable_fraction)}</b></div><div class="metric">中央 Skeleton保持率<b>${f(s.center_stable_fraction)}</b></div><div class="metric">左 Stable成分<b>${sk['左'].stable_component_count}</b></div><div class="metric">中央 Stable成分<b>${sk['中央'].stable_component_count}</b></div><div class="metric">共有Stable名<b>${s.shared_stable_component_count}</b></div><div class="metric">位置識別Invariant成分<b class="${s.discriminative_invariant_component_count>0?'good':'warn'}">${s.discriminative_invariant_component_count}</b></div><div class="metric">左右Skeleton分離<b class="${s.left_center_skeleton_distinct?'good':'warn'}">${yn(s.left_center_skeleton_distinct)}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'].minimum_route_jaccard_vs_baseline)}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'].minimum_route_jaccard_vs_baseline)}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v46: http://{HOST}:{PORT}")
    print("Invariant Context Skeleton / seven live conditions / no Core changes")
    serve(app, host=HOST, port=PORT)
