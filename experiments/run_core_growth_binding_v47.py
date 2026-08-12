from __future__ import annotations

import itertools
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
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v46 as v46

HOST = "127.0.0.1"
START_PORT = 5093
OUT = ROOT / "data" / "core_growth_binding_v47" / "results"
POSITIONS = ["左", "中央", "右"]


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


def relation(a: float, b: float) -> int:
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def named_rows(condition_rows: list[dict]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in condition_rows:
        if not row.get("event_formed"):
            continue
        named = v46.named_relative_components(row["report"])
        if named is not None:
            rows.append(named)
    return rows


def invariant_pair_relations(component_rows: list[dict[str, float]]) -> dict[str, int]:
    if not component_rows:
        return {}
    names = sorted(set.intersection(*(set(row.keys()) for row in component_rows)))
    stable: dict[str, int] = {}
    for a, b in itertools.combinations(names, 2):
        values = [relation(float(row[a]), float(row[b])) for row in component_rows]
        if len(set(values)) == 1:
            stable[f"{a}::{b}"] = int(values[0])
    return stable


def triple_order_signature(a: float, b: float, c: float) -> tuple[int, int, int]:
    return (relation(a, b), relation(a, c), relation(b, c))


def invariant_triple_relations(component_rows: list[dict[str, float]]) -> dict[str, list[int]]:
    if not component_rows:
        return {}
    names = sorted(set.intersection(*(set(row.keys()) for row in component_rows)))
    stable: dict[str, list[int]] = {}
    for a, b, c in itertools.combinations(names, 3):
        values = [
            triple_order_signature(float(row[a]), float(row[b]), float(row[c]))
            for row in component_rows
        ]
        if len(set(values)) == 1:
            stable[f"{a}::{b}::{c}"] = list(values[0])
    return stable


def invariant_cooccurrence(component_rows: list[dict[str, float]], order: int) -> dict[str, list[float]]:
    """個々のname=valueが全条件で安定して共存するpair/triple。比較用。"""
    if not component_rows:
        return {}
    names = sorted(set.intersection(*(set(row.keys()) for row in component_rows)))
    stable_names = []
    for name in names:
        values = [float(row[name]) for row in component_rows]
        if len(set(values)) == 1:
            stable_names.append(name)
    out: dict[str, list[float]] = {}
    for combo in itertools.combinations(stable_names, order):
        out["::".join(combo)] = [float(component_rows[0][name]) for name in combo]
    return out


def compare_map(left: dict, center: dict) -> dict:
    common = sorted(set(left) & set(center))
    same = []
    different = {}
    for key in common:
        if left[key] == center[key]:
            same.append(key)
        else:
            different[key] = {"left": left[key], "center": center[key]}
    only_left = sorted(set(left) - set(center))
    only_center = sorted(set(center) - set(left))
    return {
        "left_count": len(left),
        "center_count": len(center),
        "common_key_count": len(common),
        "same_value_count": len(same),
        "different_value_count": len(different),
        "left_only_count": len(only_left),
        "center_only_count": len(only_center),
        "different_values": different,
        "left_only_keys": only_left[:100],
        "center_only_keys": only_center[:100],
        "separates_by_shared_relation": bool(different),
        "separates_by_presence": bool(only_left or only_center),
        "separates": bool(different or only_left or only_center),
    }


def family_counts(keys: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for key in keys:
        parts = key.split("::")
        families = sorted({part.split("[")[0].split(".")[0] for part in parts})
        counter[" + ".join(families)] += 1
    return dict(counter.most_common())


def route_stability(runs: dict[str, list[dict]]) -> dict:
    summaries = {p: v44.summarize_position(rows) for p, rows in runs.items()}
    return {
        p: {
            "minimum_route_jaccard_vs_baseline": summary["minimum_route_jaccard_vs_baseline"],
            "route_jaccards_vs_baseline": summary["route_jaccards_vs_baseline"],
        }
        for p, summary in summaries.items()
    }


def observe() -> dict:
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: named_rows(condition_rows) for position, condition_rows in runs.items()}

    pair_rel = {position: invariant_pair_relations(component_rows) for position, component_rows in rows.items()}
    triple_rel = {position: invariant_triple_relations(component_rows) for position, component_rows in rows.items()}
    pair_co = {position: invariant_cooccurrence(component_rows, 2) for position, component_rows in rows.items()}
    triple_co = {position: invariant_cooccurrence(component_rows, 3) for position, component_rows in rows.items()}

    pair_cmp = compare_map(pair_rel["左"], pair_rel["中央"])
    triple_cmp = compare_map(triple_rel["左"], triple_rel["中央"])
    pair_co_cmp = compare_map(pair_co["左"], pair_co["中央"])
    triple_co_cmp = compare_map(triple_co["左"], triple_co["中央"])

    discriminative_pair_keys = list(pair_cmp["different_values"].keys()) + pair_cmp["left_only_keys"] + pair_cmp["center_only_keys"]
    discriminative_triple_keys = list(triple_cmp["different_values"].keys()) + triple_cmp["left_only_keys"] + triple_cmp["center_only_keys"]

    left_complete = len(rows["左"]) == len(v44.CONDITIONS)
    center_complete = len(rows["中央"]) == len(v44.CONDITIONS)
    right_absent = all(not row["event_formed"] for row in runs["右"])

    higher_order_separates = pair_cmp["separates"] or triple_cmp["separates"]
    shared_relation_separates = pair_cmp["separates_by_shared_relation"] or triple_cmp["separates_by_shared_relation"]

    if left_complete and center_complete and right_absent and shared_relation_separates:
        verdict = "higher_order_invariant_relations_separate_left_center"
        next_step = "validate_higher_order_relations_under_broader_live_perturbations"
    elif left_complete and center_complete and right_absent and higher_order_separates:
        verdict = "higher_order_presence_patterns_separate_left_center"
        next_step = "verify_presence_based_skeleton_is_not_feature_availability_artifact"
    elif left_complete and center_complete and right_absent:
        verdict = "higher_order_invariants_do_not_separate_left_center"
        next_step = "extend_temporal_context_or_seek_path_level_motifs"
    else:
        verdict = "higher_order_invariant_test_inconclusive"
        next_step = "stabilize_event_formation_before_higher_order_identity"

    payload = {
        "experiment": "Core Growth Binding v47",
        "purpose": "Search for higher-order invariant context structure across seven live perturbation conditions: stable pair/triple relations and co-occurrence patterns, including relations that remain invariant even when individual components fluctuate.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "higher_order_rule": "Pair/triple relations are retained only when their order relation is identical across every live condition for that position.",
            "human_selected_combinations": False,
            "excludes": ["Node IDs", "position labels", "absolute Event signal", "TTL", "event age"],
        },
        "conditions": [
            {"name": name, "echo_scale": e, "position_scale": p}
            for name, e, p in v44.CONDITIONS
        ],
        "pair_relations": pair_rel,
        "triple_relations": triple_rel,
        "stable_cooccurrence_pairs": pair_co,
        "stable_cooccurrence_triples": triple_co,
        "comparison": {
            "pair_relations": pair_cmp,
            "triple_relations": triple_cmp,
            "stable_pair_cooccurrence": pair_co_cmp,
            "stable_triple_cooccurrence": triple_co_cmp,
            "discriminative_pair_family_counts": family_counts(discriminative_pair_keys),
            "discriminative_triple_family_counts": family_counts(discriminative_triple_keys),
        },
        "live_route_stability": route_stability(runs),
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent": right_absent,
            "left_invariant_pair_relation_count": len(pair_rel["左"]),
            "center_invariant_pair_relation_count": len(pair_rel["中央"]),
            "discriminative_pair_relation_count": pair_cmp["different_value_count"],
            "left_only_pair_relation_count": pair_cmp["left_only_count"],
            "center_only_pair_relation_count": pair_cmp["center_only_count"],
            "left_invariant_triple_relation_count": len(triple_rel["左"]),
            "center_invariant_triple_relation_count": len(triple_rel["中央"]),
            "discriminative_triple_relation_count": triple_cmp["different_value_count"],
            "left_only_triple_relation_count": triple_cmp["left_only_count"],
            "center_only_triple_relation_count": triple_cmp["center_only_count"],
            "higher_order_separates_left_center": higher_order_separates,
            "shared_higher_order_relation_separates": shared_relation_separates,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v47.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v47</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v47</h1><p class="lead">単一Invariantではなく、7つのlive条件すべてで不変な成分間の大小関係・順序・共成立を抽出する。個々の成分が揺れていても、その関係が不変ならHigher-Order Invariantとして残す。</p><section class="panel"><div class="controls"><button id="run">Higher-Order Skeletonを抽出</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Higher-Order生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,c=d.comparison;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event全条件<b class="${s.left_event_all_conditions?'good':'warn'}">${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b class="${s.center_event_all_conditions?'good':'warn'}">${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">左 Invariant Pair<b>${s.left_invariant_pair_relation_count}</b></div><div class="metric">中央 Invariant Pair<b>${s.center_invariant_pair_relation_count}</b></div><div class="metric">共有Pair値差<b class="${s.discriminative_pair_relation_count>0?'good':'warn'}">${s.discriminative_pair_relation_count}</b></div><div class="metric">左のみPair<b>${s.left_only_pair_relation_count}</b></div><div class="metric">中央のみPair<b>${s.center_only_pair_relation_count}</b></div><div class="metric">左 Invariant Triple<b>${s.left_invariant_triple_relation_count}</b></div><div class="metric">中央 Invariant Triple<b>${s.center_invariant_triple_relation_count}</b></div><div class="metric">共有Triple値差<b class="${s.discriminative_triple_relation_count>0?'good':'warn'}">${s.discriminative_triple_relation_count}</b></div><div class="metric">Higher-Order分離<b class="${s.higher_order_separates_left_center?'good':'warn'}">${yn(s.higher_order_separates_left_center)}</b></div><div class="metric">共有関係だけで分離<b class="${s.shared_higher_order_relation_separates?'good':'warn'}">${yn(s.shared_higher_order_relation_separates)}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'].minimum_route_jaccard_vs_baseline)}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'].minimum_route_jaccard_vs_baseline)}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v47: http://{HOST}:{PORT}")
    print("Higher-Order Invariant Skeleton / pair-triple relational invariants / live propagation")
    serve(app, host=HOST, port=PORT)
