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
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v46 as v46
import run_core_growth_binding_v47 as v47

HOST = "127.0.0.1"
START_PORT = 5094
OUT = ROOT / "data" / "core_growth_binding_v48" / "results"
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


def named_rows(condition_rows: list[dict]) -> list[dict[str, float]]:
    rows = []
    for row in condition_rows:
        if not row.get("event_formed"):
            continue
        named = v46.named_relative_components(row["report"])
        if named is not None:
            rows.append(named)
    return rows


def feature_availability(rows: list[dict[str, float]], universe: list[str]) -> dict:
    counts = {name: sum(1 for row in rows if name in row) for name in universe}
    n = len(rows)
    return {
        "condition_count": n,
        "counts": counts,
        "fully_available": sorted(name for name, count in counts.items() if count == n and n > 0),
        "partially_available": sorted(name for name, count in counts.items() if 0 < count < n),
        "never_available": sorted(name for name, count in counts.items() if count == 0),
    }


def relation_series(rows: list[dict[str, float]], key: str) -> list[int | None]:
    parts = key.split("::")
    if len(parts) != 2:
        return []
    a, b = parts
    out = []
    for row in rows:
        if a not in row or b not in row:
            out.append(None)
        else:
            out.append(v47.relation(float(row[a]), float(row[b])))
    return out


def triple_series(rows: list[dict[str, float]], key: str) -> list[list[int] | None]:
    parts = key.split("::")
    if len(parts) != 3:
        return []
    a, b, c = parts
    out = []
    for row in rows:
        if a not in row or b not in row or c not in row:
            out.append(None)
        else:
            out.append(list(v47.triple_order_signature(float(row[a]), float(row[b]), float(row[c]))))
    return out


def key_components(keys: list[str]) -> list[str]:
    names = set()
    for key in keys:
        names.update(key.split("::"))
    return sorted(names)


def greedy_cover(keys: list[str]) -> list[dict]:
    remaining = set(keys)
    cover = []
    while remaining:
        counts = Counter()
        for key in remaining:
            for part in set(key.split("::")):
                counts[part] += 1
        if not counts:
            break
        name, count = counts.most_common(1)[0]
        covered = sorted(key for key in remaining if name in key.split("::"))
        cover.append({"component": name, "covers": len(covered), "covered_keys_sample": covered[:30]})
        remaining.difference_update(covered)
    return cover


def availability_reason(keys: list[str], left_av: dict, center_av: dict) -> dict:
    left_full = set(left_av["fully_available"])
    center_full = set(center_av["fully_available"])
    both_full = left_full & center_full
    all_components = key_components(keys)
    not_both = sorted(name for name in all_components if name not in both_full)
    keys_with_availability_issue = []
    keys_fully_supported = []
    for key in keys:
        parts = key.split("::")
        if all(part in both_full for part in parts):
            keys_fully_supported.append(key)
        else:
            keys_with_availability_issue.append(key)
    return {
        "source_component_count": len(all_components),
        "source_components": all_components,
        "components_not_fully_available_both_positions": not_both,
        "exclusive_key_count": len(keys),
        "exclusive_keys_fully_supported_both_positions": len(keys_fully_supported),
        "exclusive_keys_with_feature_availability_issue": len(keys_with_availability_issue),
        "fully_supported_fraction": 0.0 if not keys else len(keys_fully_supported) / len(keys),
        "availability_issue_key_sample": keys_with_availability_issue[:50],
    }


def audit_exclusive_pair(keys: list[str], other_rows: list[dict[str, float]]) -> dict:
    items = []
    for key in keys[:200]:
        series = relation_series(other_rows, key)
        nonnull = [x for x in series if x is not None]
        items.append({
            "key": key,
            "other_position_series": series,
            "missing_count": sum(x is None for x in series),
            "other_position_unique_relation_count": len(set(nonnull)),
            "other_position_relation_unstable": len(set(nonnull)) > 1,
        })
    return {"items": items}


def audit_exclusive_triple(keys: list[str], other_rows: list[dict[str, float]]) -> dict:
    items = []
    for key in keys[:200]:
        series = triple_series(other_rows, key)
        nonnull = [tuple(x) for x in series if x is not None]
        items.append({
            "key": key,
            "other_position_series": series,
            "missing_count": sum(x is None for x in series),
            "other_position_unique_relation_count": len(set(nonnull)),
            "other_position_relation_unstable": len(set(nonnull)) > 1,
        })
    return {"items": items}


def observe() -> dict:
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: named_rows(runs[position]) for position in POSITIONS}

    universe = sorted(set().union(*(set(row.keys()) for position in ("左", "中央") for row in rows[position])))
    availability = {
        "左": feature_availability(rows["左"], universe),
        "中央": feature_availability(rows["中央"], universe),
    }

    pair_rel = {p: v47.invariant_pair_relations(rows[p]) for p in ("左", "中央")}
    triple_rel = {p: v47.invariant_triple_relations(rows[p]) for p in ("左", "中央")}

    left_only_pairs = sorted(set(pair_rel["左"]) - set(pair_rel["中央"]))
    center_only_pairs = sorted(set(pair_rel["中央"]) - set(pair_rel["左"]))
    left_only_triples = sorted(set(triple_rel["左"]) - set(triple_rel["中央"]))
    center_only_triples = sorted(set(triple_rel["中央"]) - set(triple_rel["左"]))

    pair_av_left = availability_reason(left_only_pairs, availability["左"], availability["中央"])
    pair_av_center = availability_reason(center_only_pairs, availability["左"], availability["中央"])
    triple_av_left = availability_reason(left_only_triples, availability["左"], availability["中央"])
    triple_av_center = availability_reason(center_only_triples, availability["左"], availability["中央"])

    all_exclusive = left_only_pairs + center_only_pairs + left_only_triples + center_only_triples
    all_availability_issue = (
        pair_av_left["exclusive_keys_with_feature_availability_issue"]
        + pair_av_center["exclusive_keys_with_feature_availability_issue"]
        + triple_av_left["exclusive_keys_with_feature_availability_issue"]
        + triple_av_center["exclusive_keys_with_feature_availability_issue"]
    )
    all_fully_supported = len(all_exclusive) - all_availability_issue

    left_pair_cover = greedy_cover(left_only_pairs)
    center_pair_cover = greedy_cover(center_only_pairs)
    left_triple_cover = greedy_cover(left_only_triples)
    center_triple_cover = greedy_cover(center_only_triples)

    feature_universe_equal = (
        set(availability["左"]["fully_available"]) == set(availability["中央"]["fully_available"])
        and not availability["左"]["partially_available"]
        and not availability["中央"]["partially_available"]
    )

    pair_origin_is_relation_instability = all_fully_supported > 0 and all_availability_issue == 0
    source_union = set(key_components(all_exclusive))
    combination_expansion_ratio = 0.0 if not source_union else len(all_exclusive) / len(source_union)

    if feature_universe_equal and all_exclusive and all_availability_issue == 0:
        verdict = "presence_difference_is_relation_stability_not_feature_availability"
        next_step = "extract_minimal_relation_stability_signature_from_exclusive_origin_components"
    elif all_availability_issue > 0:
        verdict = "higher_order_presence_partly_explained_by_feature_availability"
        next_step = "normalize_feature_universe_before_using_presence_skeleton"
    elif not all_exclusive:
        verdict = "no_exclusive_higher_order_presence_to_audit"
        next_step = "seek_temporal_or_path_level_motifs"
    else:
        verdict = "presence_origin_mixed_or_inconclusive"
        next_step = "inspect_origin_components_before_core_integration"

    route = {p: v44.summarize_position(runs[p]) for p in ("左", "中央")}

    payload = {
        "experiment": "Core Growth Binding v48",
        "purpose": "Audit whether v47 left-only/center-only higher-order invariants arise from missing feature availability, relation-stability differences, or combinatorial expansion from a small set of origin components.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "common_feature_universe": True,
        },
        "feature_universe": {
            "component_count": len(universe),
            "components": universe,
            "left": availability["左"],
            "center": availability["中央"],
            "fully_available_universe_equal": feature_universe_equal,
        },
        "exclusive_higher_order": {
            "left_only_pairs": left_only_pairs,
            "center_only_pairs": center_only_pairs,
            "left_only_triples": left_only_triples,
            "center_only_triples": center_only_triples,
        },
        "availability_audit": {
            "left_only_pairs": pair_av_left,
            "center_only_pairs": pair_av_center,
            "left_only_triples": triple_av_left,
            "center_only_triples": triple_av_center,
            "all_exclusive_count": len(all_exclusive),
            "all_fully_supported_both_positions": all_fully_supported,
            "all_with_feature_availability_issue": all_availability_issue,
            "difference_is_relation_stability_not_missing_feature": pair_origin_is_relation_instability,
        },
        "origin_component_audit": {
            "unique_origin_component_count": len(source_union),
            "unique_origin_components": sorted(source_union),
            "exclusive_higher_order_count": len(all_exclusive),
            "combination_expansion_ratio": combination_expansion_ratio,
            "left_pair_greedy_cover": left_pair_cover,
            "center_pair_greedy_cover": center_pair_cover,
            "left_triple_greedy_cover": left_triple_cover,
            "center_triple_greedy_cover": center_triple_cover,
        },
        "relation_stability_examples": {
            "left_only_pairs_seen_in_center": audit_exclusive_pair(left_only_pairs, rows["中央"]),
            "center_only_pairs_seen_in_left": audit_exclusive_pair(center_only_pairs, rows["左"]),
            "left_only_triples_seen_in_center": audit_exclusive_triple(left_only_triples, rows["中央"]),
            "center_only_triples_seen_in_left": audit_exclusive_triple(center_only_triples, rows["左"]),
        },
        "live_route_stability": {
            "左": route["左"]["minimum_route_jaccard_vs_baseline"],
            "中央": route["中央"]["minimum_route_jaccard_vs_baseline"],
        },
        "summary": {
            "left_event_all_conditions": len(rows["左"]) == len(v44.CONDITIONS),
            "center_event_all_conditions": len(rows["中央"]) == len(v44.CONDITIONS),
            "right_event_absent": all(not row["event_formed"] for row in runs["右"]),
            "feature_universe_equal": feature_universe_equal,
            "exclusive_higher_order_count": len(all_exclusive),
            "exclusive_keys_with_availability_issue": all_availability_issue,
            "exclusive_keys_fully_supported_both_positions": all_fully_supported,
            "unique_origin_component_count": len(source_union),
            "combination_expansion_ratio": combination_expansion_ratio,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v48.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v48</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v48</h1><p class="lead">v47の左右固有Pair/Tripleを元成分まで逆追跡し、feature欠損・relation安定性差・組合せ爆発のどれがHigher-Order分離を生んだか監査する。</p><section class="panel"><div class="controls"><button id="run">Presence Skeleton起源を監査</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Origin Audit 生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event全条件<b>${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b>${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">Feature Universe一致<b class="${s.feature_universe_equal?'good':'warn'}">${yn(s.feature_universe_equal)}</b></div><div class="metric">Higher-Order固有総数<b>${s.exclusive_higher_order_count}</b></div><div class="metric">Feature欠損由来<b class="${s.exclusive_keys_with_availability_issue===0?'good':'warn'}">${s.exclusive_keys_with_availability_issue}</b></div><div class="metric">両位置でFeature存在<b>${s.exclusive_keys_fully_supported_both_positions}</b></div><div class="metric">元成分数<b>${s.unique_origin_component_count}</b></div><div class="metric">組合せ膨張率<b>${f(s.combination_expansion_ratio)}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'])}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'])}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v48: http://{HOST}:{PORT}")
    print("Presence Skeleton Origin Audit / live propagation / no Core changes")
    serve(app, host=HOST, port=PORT)
