from __future__ import annotations

import itertools
import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v50 as v50
import run_core_growth_binding_v52 as v52

HOST = "127.0.0.1"
START_PORT = 5099
OUT = ROOT / "data" / "core_growth_binding_v53" / "results"
POSITIONS = ["左", "中央", "右"]
MAX_SIGNATURE = 5


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


def stability_class_scaled(count: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    fraction = count / total
    if fraction >= 1.0 - 1e-12:
        return "stable"
    if fraction >= 5 / 7:
        return "mostly"
    if fraction >= 2 / 7:
        return "unstable"
    return "absent"


def subset_profile(rows: list[dict[str, float]], names: list[str], motif: str) -> dict:
    series = v50.motif_series(rows, motif)
    by_name = {name: bool(value) for name, value in zip(names, series)}
    count = sum(1 for x in series if x)
    baseline = by_name.get("baseline")

    def resistant(group: str):
        if baseline is None or not baseline:
            return None
        available = [name for name in v52.GROUPS[group] if name in by_name]
        if not available:
            return None
        return all(bool(by_name[name]) == bool(baseline) for name in available)

    return {
        "motif": motif,
        "series": [bool(x) for x in series],
        "present_count": count,
        "condition_count": len(series),
        "stability_class": stability_class_scaled(count, len(series)),
        "baseline_present": baseline,
        "echo_resistant": resistant("echo"),
        "position_resistant": resistant("position"),
        "common_resistant": resistant("common"),
    }


def signature(profile: dict) -> tuple:
    return (
        profile["stability_class"],
        profile["baseline_present"],
        profile["echo_resistant"],
        profile["position_resistant"],
        profile["common_resistant"],
    )


def all_contexts(rows: dict[str, list[dict[str, float]]], condition_names: list[str]) -> list[dict]:
    contexts = [{
        "name": "full",
        "indices": list(range(len(condition_names))),
        "held_condition": None,
    }]
    for held in range(len(condition_names)):
        contexts.append({
            "name": f"leave_out:{condition_names[held]}",
            "indices": [i for i in range(len(condition_names)) if i != held],
            "held_condition": condition_names[held],
        })
    return contexts


def context_profiles(rows: dict[str, list[dict[str, float]]], conditions: list[str], motifs: list[str]) -> dict:
    result = {}
    for ctx in all_contexts(rows, conditions):
        idx = ctx["indices"]
        names = [conditions[i] for i in idx]
        profiles = {}
        for position in ("左", "中央"):
            subset = [rows[position][i] for i in idx]
            profiles[position] = {
                motif: subset_profile(subset, names, motif)
                for motif in motifs
            }
        result[ctx["name"]] = {
            "held_condition": ctx["held_condition"],
            "profiles": profiles,
        }
    return result


def difference_masks(contexts: dict, motifs: list[str]) -> tuple[list[int], int, list[str]]:
    context_names = list(contexts)
    full_mask = (1 << len(context_names)) - 1
    masks = []
    for motif in motifs:
        mask = 0
        for bit, name in enumerate(context_names):
            p = contexts[name]["profiles"]
            if signature(p["左"][motif]) != signature(p["中央"][motif]):
                mask |= 1 << bit
        masks.append(mask)
    return masks, full_mask, context_names


def reduce_candidates(motifs: list[str], masks: list[int]) -> tuple[list[str], list[int]]:
    best = {}
    for motif, mask in zip(motifs, masks):
        if mask == 0:
            continue
        best.setdefault(mask, motif)
        if motif < best[mask]:
            best[mask] = motif
    items = sorted(((motif, mask) for mask, motif in best.items()), key=lambda x: (-x[1].bit_count(), x[0]))
    kept = []
    for motif, mask in items:
        if any((mask | other) == other for _, other in kept):
            continue
        kept.append((motif, mask))
    return [x[0] for x in kept], [x[1] for x in kept]


def exact_cover(motifs: list[str], masks: list[int], full_mask: int) -> dict:
    motifs, masks = reduce_candidates(motifs, masks)
    if not motifs:
        return {"found": False, "size": None, "profiles": [], "candidate_count_after_reduction": 0}
    union = 0
    for m in masks:
        union |= m
    if union != full_mask:
        return {
            "found": False,
            "size": None,
            "profiles": [],
            "candidate_count_after_reduction": len(motifs),
            "uncovered_context_count": (full_mask ^ union).bit_count(),
        }

    for size in range(1, MAX_SIGNATURE + 1):
        print(f"v53: exact search <= {size} profile(s)...", flush=True)
        for combo in itertools.combinations(range(len(motifs)), size):
            cover = 0
            for i in combo:
                cover |= masks[i]
            if cover == full_mask:
                return {
                    "found": True,
                    "size": size,
                    "profiles": [motifs[i] for i in combo],
                    "candidate_count_after_reduction": len(motifs),
                }
    return {"found": False, "size": None, "profiles": [], "candidate_count_after_reduction": len(motifs)}


def selected_profile_audit(contexts: dict, selected: list[str]) -> dict:
    rows = []
    all_separate = True
    for name, ctx in contexts.items():
        p = ctx["profiles"]
        left = [list(signature(p["左"][motif])) for motif in selected]
        center = [list(signature(p["中央"][motif])) for motif in selected]
        separated = left != center
        all_separate = all_separate and separated
        rows.append({
            "context": name,
            "held_condition": ctx["held_condition"],
            "left_signature": left,
            "center_signature": center,
            "separated": separated,
        })
    return {"all_contexts_separated": all_separate, "contexts": rows}


def observe() -> dict:
    print("v53: rebuilding live runs and Stability Profiles...", flush=True)
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: v50.named_rows(runs[position]) for position in POSITIONS}
    conditions = [name for name, _, _ in v44.CONDITIONS]
    candidates = v52.motif_candidates(rows)
    contexts = context_profiles(rows, conditions, candidates)
    masks, full_mask, context_names = difference_masks(contexts, candidates)
    minimal = exact_cover(candidates, masks, full_mask)
    audit = selected_profile_audit(contexts, minimal["profiles"]) if minimal["found"] else {
        "all_contexts_separated": False,
        "contexts": [],
    }

    left_complete = len(rows["左"]) == len(conditions)
    center_complete = len(rows["中央"]) == len(conditions)
    right_absent = all(not row["event_formed"] for row in runs["右"])
    compact = bool(minimal["found"] and minimal["size"] is not None and minimal["size"] <= MAX_SIGNATURE)
    shadow_candidate = left_complete and center_complete and right_absent and compact and audit["all_contexts_separated"]

    if shadow_candidate:
        verdict = "minimal_discriminative_stability_profile_found"
        next_step = "shadow_integrate_stability_profile_state_into_core_without_affecting_route_or_learning"
        readiness = "shadow_candidate"
    elif minimal["found"]:
        verdict = "stability_profile_signature_found_but_not_robust_across_all_leave_one_out_contexts"
        next_step = "broaden_profile_context_or_refine_stability_abstraction_before_core_shadow"
        readiness = "not_yet"
    else:
        verdict = "no_1_to_5_stability_profile_signature_separates_full_and_leave_one_out_contexts"
        next_step = "seek_temporal_stability_meta_profile_before_core_integration"
        readiness = "not_yet"

    route = {p: v44.summarize_position(runs[p]) for p in ("左", "中央")}
    payload = {
        "experiment": "Core Growth Binding v53",
        "purpose": "Find the smallest set of coarse Motif Stability Profiles (up to five) whose left/center difference survives the full seven-condition profile and every leave-one-condition-out profile reconstruction.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "human_selected_profiles": False,
            "profile_fields": ["stability_class", "baseline_present", "echo_resistant", "position_resistant", "common_resistant"],
            "generalization_test": "Rebuild profiles from six conditions after hiding each condition in turn; selected profile set must still separate left from center in every reconstruction.",
        },
        "conditions": [{"name": n, "echo_scale": e, "position_scale": p} for n, e, p in v44.CONDITIONS],
        "candidate_profile_count": len(candidates),
        "context_names": context_names,
        "minimal_signature": minimal,
        "selected_profile_audit": audit,
        "right_control": {
            "event_absent_all_conditions": right_absent,
            "false_identity_event_count": sum(1 for row in runs["右"] if row["event_formed"]),
        },
        "live_route_stability": {
            "左": route["左"]["minimum_route_jaccard_vs_baseline"],
            "中央": route["中央"]["minimum_route_jaccard_vs_baseline"],
        },
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent": right_absent,
            "candidate_profile_count": len(candidates),
            "minimal_profile_signature_found": minimal["found"],
            "minimal_profile_count": minimal["size"],
            "full_plus_all_leave_one_out_separated": audit["all_contexts_separated"],
            "right_false_positive": not right_absent,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v53.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v53</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v53</h1><p class="lead">Motif PresenceではなくStability Profileを使い、full 7条件とleave-one-condition-out 7通りのすべてで左/中央を分ける最小1〜5 Profile Signatureを探索する。</p><section class="panel"><div class="controls"><button id="run">Minimal Stability Profileを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Profile Signature生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function n(v){return v===null||v===undefined?'なし':v}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,ms=d.minimal_signature;m.innerHTML=`<div class="metric">左 Event全条件<b>${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b>${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">候補Profile数<b>${s.candidate_profile_count}</b></div><div class="metric">最小Signature発見<b class="${s.minimal_profile_signature_found?'good':'warn'}">${yn(s.minimal_profile_signature_found)}</b></div><div class="metric">最小Profile数<b>${n(s.minimal_profile_count)}</b></div><div class="metric">Full+LOCO全分離<b class="${s.full_plus_all_leave_one_out_separated?'good':'warn'}">${yn(s.full_plus_all_leave_one_out_separated)}</b></div><div class="metric">右 誤検出<b class="${s.right_false_positive?'warn':'good'}">${yn(s.right_false_positive)}</b></div><div class="metric">削減後候補<b>${n(ms.candidate_count_after_reduction)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'])}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'])}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v53: http://{HOST}:{PORT}")
    print("Minimal Discriminative Stability Profile / full + LOCO reconstruction / no Core changes")
    serve(app, host=HOST, port=PORT)
