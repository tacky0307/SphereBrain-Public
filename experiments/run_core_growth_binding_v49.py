from __future__ import annotations

import itertools
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

HOST = "127.0.0.1"
START_PORT = 5095
OUT = ROOT / "data" / "core_growth_binding_v49" / "results"
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


def exclusive_higher_order(rows: dict[str, list[dict[str, float]]]) -> dict:
    pair = {p: v47.invariant_pair_relations(rows[p]) for p in ("左", "中央")}
    triple = {p: v47.invariant_triple_relations(rows[p]) for p in ("左", "中央")}
    return {
        "left_pairs": sorted(set(pair["左"]) - set(pair["中央"])),
        "center_pairs": sorted(set(pair["中央"]) - set(pair["左"])),
        "left_triples": sorted(set(triple["左"]) - set(triple["中央"])),
        "center_triples": sorted(set(triple["中央"]) - set(triple["左"])),
        "pair_maps": pair,
        "triple_maps": triple,
    }


def pair_keys_from_higher_order(key: str) -> list[str]:
    parts = key.split("::")
    if len(parts) == 2:
        return ["::".join(parts)]
    if len(parts) == 3:
        return ["::".join(pair) for pair in itertools.combinations(parts, 2)]
    return []


def relation_signature_candidates(exclusive_keys: list[str]) -> dict[str, set[str]]:
    coverage: dict[str, set[str]] = defaultdict(set)
    for key in exclusive_keys:
        for pair_key in pair_keys_from_higher_order(key):
            coverage[pair_key].add(key)
    return dict(coverage)


def greedy_cover(universe: set[str], candidates: dict[str, set[str]]) -> dict:
    remaining = set(universe)
    selected = []
    covered_total: set[str] = set()
    while remaining:
        best_name = None
        best_cover: set[str] = set()
        for name, covers in candidates.items():
            gain = remaining & covers
            if len(gain) > len(best_cover):
                best_name = name
                best_cover = gain
        if best_name is None or not best_cover:
            break
        covered_total |= best_cover
        remaining -= best_cover
        selected.append({
            "relation": best_name,
            "newly_covered": len(best_cover),
            "cumulative_covered": len(covered_total),
            "cumulative_fraction": 0.0 if not universe else len(covered_total) / len(universe),
            "covered_sample": sorted(best_cover)[:20],
        })
    return {
        "selected": selected,
        "selected_count": len(selected),
        "covered_count": len(covered_total),
        "coverage_fraction": 0.0 if not universe else len(covered_total) / len(universe),
        "uncovered_count": len(remaining),
        "uncovered_sample": sorted(remaining)[:30],
    }


def count_to_target(steps: list[dict], target: float) -> int | None:
    for i, row in enumerate(steps, start=1):
        if float(row["cumulative_fraction"]) + 1e-12 >= target:
            return i
    return None


def stability_series(rows: list[dict[str, float]], pair_key: str) -> list[int | None]:
    a, b = pair_key.split("::")
    out = []
    for row in rows:
        if a not in row or b not in row:
            out.append(None)
        else:
            out.append(v47.relation(float(row[a]), float(row[b])))
    return out


def signature_relation_record(pair_key: str, side_rows: list[dict[str, float]], other_rows: list[dict[str, float]]) -> dict:
    side = stability_series(side_rows, pair_key)
    other = stability_series(other_rows, pair_key)
    side_nonnull = [x for x in side if x is not None]
    other_nonnull = [x for x in other if x is not None]
    return {
        "relation": pair_key,
        "stable_side_series": side,
        "stable_side_unique_count": len(set(side_nonnull)),
        "other_side_series": other,
        "other_side_unique_count": len(set(other_nonnull)),
        "stable_on_target_side": bool(side_nonnull) and len(set(side_nonnull)) == 1,
        "unstable_or_different_on_other_side": (not other_nonnull) or len(set(other_nonnull)) != 1 or (set(other_nonnull) != set(side_nonnull)),
    }


def side_analysis(side: str, exclusive: dict, rows: dict[str, list[dict[str, float]]]) -> dict:
    other = "中央" if side == "左" else "左"
    pair_keys = exclusive["left_pairs"] if side == "左" else exclusive["center_pairs"]
    triple_keys = exclusive["left_triples"] if side == "左" else exclusive["center_triples"]
    universe = set(pair_keys + triple_keys)
    candidates = relation_signature_candidates(sorted(universe))
    cover = greedy_cover(universe, candidates)
    chosen = [row["relation"] for row in cover["selected"]]
    records = [signature_relation_record(key, rows[side], rows[other]) for key in chosen]
    thresholds = {str(int(t * 100)): count_to_target(cover["selected"], t) for t in TARGETS}
    return {
        "exclusive_pair_count": len(pair_keys),
        "exclusive_triple_count": len(triple_keys),
        "exclusive_total": len(universe),
        "candidate_relation_count": len(candidates),
        "greedy_cover": cover,
        "relations_needed_for_coverage": thresholds,
        "signature_relations": records,
        "top_signature_relations": chosen[:20],
    }


def combined_signature(left: dict, center: dict) -> dict:
    names = []
    for side, report in (("左", left), ("中央", center)):
        for row in report["greedy_cover"]["selected"]:
            names.append((side, row["relation"], row["newly_covered"]))
    names.sort(key=lambda x: (-x[2], x[0], x[1]))
    return {
        "relation_count": len(names),
        "ranked_relations": [
            {"side": side, "relation": relation, "newly_covered": covered}
            for side, relation, covered in names
        ],
    }


def observe() -> dict:
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: named_rows(runs[position]) for position in POSITIONS}
    exclusive = exclusive_higher_order(rows)

    left = side_analysis("左", exclusive, rows)
    center = side_analysis("中央", exclusive, rows)
    combined = combined_signature(left, center)

    left_complete = len(rows["左"]) == len(v44.CONDITIONS)
    center_complete = len(rows["中央"]) == len(v44.CONDITIONS)
    right_absent = all(not row["event_formed"] for row in runs["右"])

    left_95 = left["relations_needed_for_coverage"].get("95")
    center_95 = center["relations_needed_for_coverage"].get("95")
    compact_95 = (
        left_95 is not None and center_95 is not None and left_95 <= 5 and center_95 <= 5
    )

    if left_complete and center_complete and right_absent and compact_95:
        verdict = "compact_relation_stability_signature_found"
        next_step = "validate_compact_signature_on_broader_live_perturbations_then_consider_core_short_term_state"
        core_readiness = "candidate_after_broader_validation"
    elif left_complete and center_complete and right_absent:
        verdict = "relation_stability_signature_exists_but_is_not_compact_enough"
        next_step = "reduce_signature_or_seek_more_abstract_relation_motifs_before_core_integration"
        core_readiness = "not_yet"
    else:
        verdict = "minimal_relation_signature_inconclusive"
        next_step = "stabilize_event_or_feature_extraction_before_core_integration"
        core_readiness = "not_yet"

    route = {p: v44.summarize_position(runs[p]) for p in ("左", "中央")}

    payload = {
        "experiment": "Core Growth Binding v49",
        "purpose": "Compress v48 relation-stability differences into a small set of pairwise Relation Stability Signature elements that explain the left-only/center-only higher-order invariant patterns.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "optimization_note": "Greedy set cover is used as a compact-signature approximation; it is not claimed to be the mathematical minimum unless separately proven.",
            "human_selected_relations": False,
        },
        "conditions": [
            {"name": name, "echo_scale": e, "position_scale": p}
            for name, e, p in v44.CONDITIONS
        ],
        "left_signature": left,
        "center_signature": center,
        "combined_signature": combined,
        "live_route_stability": {
            "左": route["左"]["minimum_route_jaccard_vs_baseline"],
            "中央": route["中央"]["minimum_route_jaccard_vs_baseline"],
        },
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent": right_absent,
            "left_exclusive_higher_order": left["exclusive_total"],
            "center_exclusive_higher_order": center["exclusive_total"],
            "left_candidate_relations": left["candidate_relation_count"],
            "center_candidate_relations": center["candidate_relation_count"],
            "left_relations_for_80pct": left["relations_needed_for_coverage"].get("80"),
            "left_relations_for_90pct": left["relations_needed_for_coverage"].get("90"),
            "left_relations_for_95pct": left_95,
            "center_relations_for_80pct": center["relations_needed_for_coverage"].get("80"),
            "center_relations_for_90pct": center["relations_needed_for_coverage"].get("90"),
            "center_relations_for_95pct": center_95,
            "compact_95pct_signature": compact_95,
            "core_readiness": core_readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v49.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v49</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v49</h1><p class="lead">573件のHigher-Order差を、その起源となる少数のRelation Stability Signatureへgreedy set coverで圧縮する。Coreへ大量特徴を持ち込まず、80/90/95%を説明する最小候補数を測る。</p><section class="panel"><div class="controls"><button id="run">Minimal Signatureを抽出</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Signature生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function n(v){return v===null||v===undefined?'なし':v}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}document.getElementById('run').addEventListener('click',async()=>{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary;document.getElementById('metrics').innerHTML=`<div class="metric">左 Event全条件<b>${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b>${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">左 Higher-Order差<b>${s.left_exclusive_higher_order}</b></div><div class="metric">中央 Higher-Order差<b>${s.center_exclusive_higher_order}</b></div><div class="metric">左 80% Relation数<b>${n(s.left_relations_for_80pct)}</b></div><div class="metric">左 90% Relation数<b>${n(s.left_relations_for_90pct)}</b></div><div class="metric">左 95% Relation数<b>${n(s.left_relations_for_95pct)}</b></div><div class="metric">中央 80% Relation数<b>${n(s.center_relations_for_80pct)}</b></div><div class="metric">中央 90% Relation数<b>${n(s.center_relations_for_90pct)}</b></div><div class="metric">中央 95% Relation数<b>${n(s.center_relations_for_95pct)}</b></div><div class="metric">95% Compact Signature<b class="${s.compact_95pct_signature?'good':'warn'}">${yn(s.compact_95pct_signature)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'])}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'])}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)});
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
    print(f"Core Growth Binding v49: http://{HOST}:{PORT}")
    print("Minimal Relation Stability Signature / diagnostic only / no Core changes")
    serve(app, host=HOST, port=PORT)
