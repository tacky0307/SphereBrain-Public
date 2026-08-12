from __future__ import annotations

import json
import socket
import sys
import threading
import webbrowser
from collections import defaultdict
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
import run_core_growth_binding_v49 as v49

HOST = "127.0.0.1"
START_PORT = 5096
OUT = ROOT / "data" / "core_growth_binding_v50" / "results"
POSITIONS = ["左", "中央", "右"]
TARGETS = [0.80, 0.90, 0.95, 1.00]


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
    out = []
    for row in condition_rows:
        if not row.get("event_formed"):
            continue
        named = v46.named_relative_components(row["report"])
        if named is not None:
            out.append(named)
    return out


def component_family(name: str) -> str:
    """細かなindex/サブ成分を捨て、構造familyだけを返す。"""
    if name.startswith("topology."):
        return "topology"
    if name.startswith("event."):
        return "event"
    base = name.split("[")[0]
    base = base.split(".")[0]
    return base


def relation_symbol(value: int) -> str:
    return "<" if value < 0 else (">" if value > 0 else "=")


def pair_motif(key: str, value: int) -> str:
    a, b = key.split("::")
    fa, fb = component_family(a), component_family(b)
    return f"PAIR:{fa}{relation_symbol(int(value))}{fb}"


def triple_motif(key: str, value: list[int]) -> str:
    a, b, c = key.split("::")
    fa, fb, fc = component_family(a), component_family(b), component_family(c)
    sig = ",".join(str(int(x)) for x in value)
    return f"TRIPLE:{fa}|{fb}|{fc}:order[{sig}]"


def exclusive_maps(rows: dict[str, list[dict[str, float]]]) -> dict:
    pair = {p: v47.invariant_pair_relations(rows[p]) for p in ("左", "中央")}
    triple = {p: v47.invariant_triple_relations(rows[p]) for p in ("左", "中央")}
    return {
        "pair": pair,
        "triple": triple,
        "left_pair_keys": sorted(set(pair["左"]) - set(pair["中央"])),
        "center_pair_keys": sorted(set(pair["中央"]) - set(pair["左"])),
        "left_triple_keys": sorted(set(triple["左"]) - set(triple["中央"])),
        "center_triple_keys": sorted(set(triple["中央"]) - set(triple["左"])),
    }


def motif_coverage_for_side(side: str, ex: dict) -> dict[str, set[str]]:
    pair_keys = ex["left_pair_keys"] if side == "左" else ex["center_pair_keys"]
    triple_keys = ex["left_triple_keys"] if side == "左" else ex["center_triple_keys"]
    pair_map = ex["pair"][side]
    triple_map = ex["triple"][side]

    coverage: dict[str, set[str]] = defaultdict(set)
    for key in pair_keys:
        motif = pair_motif(key, int(pair_map[key]))
        coverage[motif].add("P:" + key)
    for key in triple_keys:
        motif = triple_motif(key, list(triple_map[key]))
        coverage[motif].add("T:" + key)
    return dict(coverage)


def ranked_motif_cover(coverage: dict[str, set[str]]) -> dict:
    universe = set().union(*coverage.values()) if coverage else set()
    ranked = sorted(coverage.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    covered: set[str] = set()
    steps = []
    for motif, keys in ranked:
        new = keys - covered
        if not new:
            continue
        covered |= new
        steps.append({
            "motif": motif,
            "newly_covered": len(new),
            "motif_total_coverage": len(keys),
            "cumulative_covered": len(covered),
            "cumulative_fraction": 0.0 if not universe else len(covered) / len(universe),
            "covered_sample": sorted(new)[:20],
        })
    return {
        "universe_count": len(universe),
        "motif_count": len(coverage),
        "ranked_steps": steps,
        "coverage_fraction": 0.0 if not universe else len(covered) / len(universe),
    }


def count_to_target(steps: list[dict], target: float) -> int | None:
    for i, row in enumerate(steps, start=1):
        if float(row["cumulative_fraction"]) + 1e-12 >= target:
            return i
    return None


def motif_series(rows: list[dict[str, float]], motif: str) -> list[bool]:
    """Motifが各条件で少なくとも1つ実現しているか。"""
    out = []
    for row in rows:
        found = False
        names = sorted(row)
        if motif.startswith("PAIR:"):
            target = motif[len("PAIR:"):]
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    value = v47.relation(float(row[a]), float(row[b]))
                    if f"{component_family(a)}{relation_symbol(value)}{component_family(b)}" == target:
                        found = True
                        break
                if found:
                    break
        else:
            target = motif[len("TRIPLE:"):]
            # live安定性確認用。候補数が小さいので全組合せを監査する。
            import itertools
            for a, b, c in itertools.combinations(names, 3):
                sig = list(v47.triple_order_signature(float(row[a]), float(row[b]), float(row[c])))
                candidate = f"{component_family(a)}|{component_family(b)}|{component_family(c)}:order[{','.join(str(int(x)) for x in sig)}]"
                if candidate == target:
                    found = True
                    break
        out.append(found)
    return out


def side_analysis(side: str, coverage: dict[str, set[str]], rows: dict[str, list[dict[str, float]]]) -> dict:
    other = "中央" if side == "左" else "左"
    cover = ranked_motif_cover(coverage)
    thresholds = {
        str(int(t * 100)): count_to_target(cover["ranked_steps"], t)
        for t in TARGETS
    }
    top = [row["motif"] for row in cover["ranked_steps"][:20]]
    stability = []
    for motif in top:
        target_series = motif_series(rows[side], motif)
        other_series = motif_series(rows[other], motif)
        stability.append({
            "motif": motif,
            "target_series": target_series,
            "stable_on_target_side": bool(target_series) and all(target_series),
            "other_series": other_series,
            "present_all_other_side": bool(other_series) and all(other_series),
            "discriminative_presence": bool(target_series) and all(target_series) and not (bool(other_series) and all(other_series)),
        })
    return {
        "exclusive_higher_order_count": cover["universe_count"],
        "abstract_motif_count": cover["motif_count"],
        "compression_ratio": 0.0 if cover["motif_count"] == 0 else cover["universe_count"] / cover["motif_count"],
        "coverage": cover,
        "motifs_needed_for_coverage": thresholds,
        "top_motif_stability": stability,
        "top_motifs": top,
    }


def motif_set_comparison(left_cov: dict[str, set[str]], center_cov: dict[str, set[str]]) -> dict:
    left = set(left_cov)
    center = set(center_cov)
    return {
        "left_motif_count": len(left),
        "center_motif_count": len(center),
        "shared_motif_count": len(left & center),
        "left_only_motif_count": len(left - center),
        "center_only_motif_count": len(center - left),
        "left_only_motifs": sorted(left - center),
        "center_only_motifs": sorted(center - left),
        "motif_presence_separates": bool((left - center) or (center - left)),
    }


def estimate_state_bits(left: dict, center: dict) -> dict:
    l95 = left["motifs_needed_for_coverage"].get("95")
    c95 = center["motifs_needed_for_coverage"].get("95")
    total = None if l95 is None or c95 is None else l95 + c95
    # 最小のpresence stateなら1 motif = 1 bitという下限見積もり。
    return {
        "left_95pct_presence_bits_lower_bound": l95,
        "center_95pct_presence_bits_lower_bound": c95,
        "combined_95pct_presence_bits_lower_bound": total,
        "note": "Lower-bound estimate only: one binary presence bit per selected abstract motif; implementation metadata is not included.",
    }


def observe() -> dict:
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: named_rows(runs[position]) for position in POSITIONS}
    ex = exclusive_maps(rows)

    left_cov = motif_coverage_for_side("左", ex)
    center_cov = motif_coverage_for_side("中央", ex)
    left = side_analysis("左", left_cov, rows)
    center = side_analysis("中央", center_cov, rows)
    comparison = motif_set_comparison(left_cov, center_cov)
    bits = estimate_state_bits(left, center)

    left_complete = len(rows["左"]) == len(v44.CONDITIONS)
    center_complete = len(rows["中央"]) == len(v44.CONDITIONS)
    right_absent = all(not row["event_formed"] for row in runs["右"])

    l95 = left["motifs_needed_for_coverage"].get("95")
    c95 = center["motifs_needed_for_coverage"].get("95")
    compact = l95 is not None and c95 is not None and l95 <= 5 and c95 <= 5

    top_target_stable = all(
        row["stable_on_target_side"]
        for row in left["top_motif_stability"][: max(1, l95 or 1)]
    ) and all(
        row["stable_on_target_side"]
        for row in center["top_motif_stability"][: max(1, c95 or 1)]
    )

    if left_complete and center_complete and right_absent and compact and top_target_stable:
        verdict = "compact_abstract_relation_motifs_found"
        next_step = "validate_motifs_under_broader_live_perturbations_then_shadow_integrate_into_core"
        core_readiness = "shadow_candidate"
    elif left_complete and center_complete and right_absent and comparison["motif_presence_separates"]:
        verdict = "abstract_relation_motifs_reduce_complexity_but_are_not_compact_enough"
        next_step = "refine_motif_abstraction_or_select_stability_meta_motifs_before_core_integration"
        core_readiness = "not_yet"
    else:
        verdict = "abstract_relation_motifs_do_not_provide_reliable_separation"
        next_step = "revisit_relation_abstraction_before_core_integration"
        core_readiness = "not_yet"

    route = {p: v44.summarize_position(runs[p]) for p in ("左", "中央")}

    payload = {
        "experiment": "Core Growth Binding v50",
        "purpose": "Abstract v49 pairwise Relation Stability Signature elements into family-level relation motifs, then measure how many motifs are needed to explain 80/90/95% of left/center higher-order differences.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "human_selected_relations": False,
            "motif_abstraction": "component indices/subfeatures are removed; family names plus stable pair/triple order relation are retained",
        },
        "conditions": [
            {"name": name, "echo_scale": e, "position_scale": p}
            for name, e, p in v44.CONDITIONS
        ],
        "left_motifs": left,
        "center_motifs": center,
        "motif_comparison": comparison,
        "state_size_estimate": bits,
        "live_route_stability": {
            "左": route["左"]["minimum_route_jaccard_vs_baseline"],
            "中央": route["中央"]["minimum_route_jaccard_vs_baseline"],
        },
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent": right_absent,
            "left_exclusive_higher_order": left["exclusive_higher_order_count"],
            "center_exclusive_higher_order": center["exclusive_higher_order_count"],
            "left_abstract_motif_count": left["abstract_motif_count"],
            "center_abstract_motif_count": center["abstract_motif_count"],
            "left_compression_ratio": left["compression_ratio"],
            "center_compression_ratio": center["compression_ratio"],
            "left_motifs_for_80pct": left["motifs_needed_for_coverage"].get("80"),
            "left_motifs_for_90pct": left["motifs_needed_for_coverage"].get("90"),
            "left_motifs_for_95pct": l95,
            "center_motifs_for_80pct": center["motifs_needed_for_coverage"].get("80"),
            "center_motifs_for_90pct": center["motifs_needed_for_coverage"].get("90"),
            "center_motifs_for_95pct": c95,
            "motif_presence_separates_left_center": comparison["motif_presence_separates"],
            "compact_95pct_motifs": compact,
            "top_selected_motifs_live_stable": top_target_stable,
            "core_readiness": core_readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v50.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v50</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v50</h1><p class="lead">v49の個別Relationをfamily-level Motifへ抽象化し、indexや細かな成分位置を捨ててもRelation Stabilityの差を圧縮・保持できるかを検証する。</p><section class="panel"><div class="controls"><button id="run">Abstract Motifを抽出</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Motif生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function n(v){return v===null||v===undefined?'なし':v}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,cmp=d.motif_comparison,b=d.state_size_estimate;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event全条件<b>${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b>${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">左 Relation差<b>${s.left_exclusive_higher_order}</b></div><div class="metric">中央 Relation差<b>${s.center_exclusive_higher_order}</b></div><div class="metric">左 Abstract Motif数<b>${s.left_abstract_motif_count}</b></div><div class="metric">中央 Abstract Motif数<b>${s.center_abstract_motif_count}</b></div><div class="metric">左 圧縮率<b>${f(s.left_compression_ratio)}</b></div><div class="metric">中央 圧縮率<b>${f(s.center_compression_ratio)}</b></div><div class="metric">左 80% Motif数<b>${n(s.left_motifs_for_80pct)}</b></div><div class="metric">左 90% Motif数<b>${n(s.left_motifs_for_90pct)}</b></div><div class="metric">左 95% Motif数<b>${n(s.left_motifs_for_95pct)}</b></div><div class="metric">中央 80% Motif数<b>${n(s.center_motifs_for_80pct)}</b></div><div class="metric">中央 90% Motif数<b>${n(s.center_motifs_for_90pct)}</b></div><div class="metric">中央 95% Motif数<b>${n(s.center_motifs_for_95pct)}</b></div><div class="metric">Motif Presence分離<b class="${s.motif_presence_separates_left_center?'good':'warn'}">${yn(s.motif_presence_separates_left_center)}</b></div><div class="metric">95% Compact Motif<b class="${s.compact_95pct_motifs?'good':'warn'}">${yn(s.compact_95pct_motifs)}</b></div><div class="metric">選択Motif live安定<b>${yn(s.top_selected_motifs_live_stable)}</b></div><div class="metric">95% 状態bit下限<b>${n(b.combined_95pct_presence_bits_lower_bound)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'])}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'])}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v50: http://{HOST}:{PORT}")
    print("Abstract Relation Motif / live perturbation / no Core changes")
    serve(app, host=HOST, port=PORT)
