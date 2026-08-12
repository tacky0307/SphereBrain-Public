from __future__ import annotations

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

HOST = "127.0.0.1"
START_PORT = 5098
OUT = ROOT / "data" / "core_growth_binding_v52" / "results"
POSITIONS = ["左", "中央", "右"]

GROUPS = {
    "echo": ["echo_0.97", "echo_1.03"],
    "position": ["position_0.97", "position_1.03"],
    "common": ["common_0.97", "common_1.03"],
}


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
    return v50.named_rows(condition_rows)


def motif_candidates(rows: dict[str, list[dict[str, float]]]) -> list[str]:
    ex = v50.exclusive_maps(rows)
    left = v50.motif_coverage_for_side("左", ex)
    center = v50.motif_coverage_for_side("中央", ex)
    return sorted(set(left) | set(center))


def stability_class(count: int, total: int = 7) -> str:
    if count >= total:
        return "stable"
    if count >= 5:
        return "mostly"
    if count >= 2:
        return "unstable"
    return "absent"


def resistance(series_by_name: dict[str, bool], group: str) -> bool:
    baseline = bool(series_by_name.get("baseline", False))
    if not baseline:
        return False
    return all(bool(series_by_name.get(name, False)) == baseline for name in GROUPS[group])


def motif_profile(rows: list[dict[str, float]], condition_names: list[str], motif: str) -> dict:
    series = v50.motif_series(rows, motif)
    by_name = {name: bool(value) for name, value in zip(condition_names, series)}
    count = sum(1 for value in series if value)
    return {
        "motif": motif,
        "series": [bool(x) for x in series],
        "present_count": count,
        "present_fraction": 0.0 if not series else count / len(series),
        "stability_class": stability_class(count, len(series)),
        "baseline_present": bool(by_name.get("baseline", False)),
        "echo_resistant": resistance(by_name, "echo"),
        "position_resistant": resistance(by_name, "position"),
        "common_resistant": resistance(by_name, "common"),
    }


def coarse_signature(profile: dict) -> tuple:
    return (
        profile["stability_class"],
        bool(profile["baseline_present"]),
        bool(profile["echo_resistant"]),
        bool(profile["position_resistant"]),
        bool(profile["common_resistant"]),
    )


def compare_profiles(left: dict[str, dict], center: dict[str, dict]) -> dict:
    common = sorted(set(left) & set(center))
    different = {}
    same = []
    for motif in common:
        ls = coarse_signature(left[motif])
        cs = coarse_signature(center[motif])
        if ls == cs:
            same.append(motif)
        else:
            different[motif] = {
                "left": left[motif],
                "center": center[motif],
                "left_signature": list(ls),
                "center_signature": list(cs),
            }
    return {
        "common_motif_count": len(common),
        "same_profile_count": len(same),
        "different_profile_count": len(different),
        "different_profiles": different,
        "coarse_profile_separates": bool(different),
    }


def profile_family_counts(different: dict[str, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for motif in different:
        prefix = motif.split(":", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def compact_discriminators(comparison: dict) -> list[dict]:
    rows = []
    for motif, item in comparison["different_profiles"].items():
        lp = item["left"]
        cp = item["center"]
        score = 0
        if lp["stability_class"] != cp["stability_class"]:
            score += 4
        score += int(lp["baseline_present"] != cp["baseline_present"])
        score += int(lp["echo_resistant"] != cp["echo_resistant"])
        score += int(lp["position_resistant"] != cp["position_resistant"])
        score += int(lp["common_resistant"] != cp["common_resistant"])
        rows.append({"motif": motif, "difference_score": score, "left": lp, "center": cp})
    rows.sort(key=lambda r: (-r["difference_score"], r["motif"]))
    return rows


def observe() -> dict:
    print("v52: building live motif stability profiles...", flush=True)
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: named_rows(runs[position]) for position in POSITIONS}
    conditions = [name for name, _, _ in v44.CONDITIONS]
    candidates = motif_candidates(rows)

    profiles = {
        position: {
            motif: motif_profile(rows[position], conditions, motif)
            for motif in candidates
        }
        for position in ("左", "中央")
    }
    comparison = compare_profiles(profiles["左"], profiles["中央"])
    ranked = compact_discriminators(comparison)

    right_absent = all(not row["event_formed"] for row in runs["右"])
    left_complete = len(rows["左"]) == len(v44.CONDITIONS)
    center_complete = len(rows["中央"]) == len(v44.CONDITIONS)

    if left_complete and center_complete and right_absent and comparison["coarse_profile_separates"]:
        verdict = "motif_stability_profiles_separate_left_center"
        next_step = "seek_minimal_discriminative_stability_profile_then_shadow_integrate_if_generalization_passes"
        readiness = "profile_candidate"
    elif left_complete and center_complete and right_absent:
        verdict = "coarse_motif_stability_profiles_do_not_separate_left_center"
        next_step = "retain_condition_specific_temporal_profile_or_seek_path_level_stability motifs"
        readiness = "not_yet"
    else:
        verdict = "motif_stability_profile_inconclusive_due_to_event_instability"
        next_step = "stabilize_contact_event_before_profile_integration"
        readiness = "not_yet"

    route = {p: v44.summarize_position(runs[p]) for p in ("左", "中央")}
    payload = {
        "experiment": "Core Growth Binding v52",
        "purpose": "Represent each abstract motif by a coarse stability profile across seven live perturbation conditions rather than a single presence bit, and test whether stability shape separates left from center.",
        "contract": {
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "structural_assist_used": False,
            "core_file_modified": False,
            "live_propagation": True,
            "human_selected_motifs": False,
            "profile_fields": ["stability_class", "baseline_present", "echo_resistant", "position_resistant", "common_resistant"],
            "stability_classes": {"stable": "7/7", "mostly": "5-6/7", "unstable": "2-4/7", "absent": "0-1/7"},
        },
        "conditions": [{"name": n, "echo_scale": e, "position_scale": p} for n, e, p in v44.CONDITIONS],
        "candidate_motif_count": len(candidates),
        "profiles": profiles,
        "comparison": comparison,
        "ranked_discriminative_profiles": ranked,
        "profile_family_counts": profile_family_counts(comparison["different_profiles"]),
        "live_route_stability": {
            "左": route["左"]["minimum_route_jaccard_vs_baseline"],
            "中央": route["中央"]["minimum_route_jaccard_vs_baseline"],
        },
        "summary": {
            "left_event_all_conditions": left_complete,
            "center_event_all_conditions": center_complete,
            "right_event_absent": right_absent,
            "candidate_motif_count": len(candidates),
            "different_stability_profile_count": comparison["different_profile_count"],
            "coarse_profile_separates_left_center": comparison["coarse_profile_separates"],
            "top_discriminator_count": min(10, len(ranked)),
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v52.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v52</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v52</h1><p class="lead">Motifを単なるON/OFFではなく、7つのlive input変動に対する安定性プロフィールとして表す。粗い安定度と、E・位置・共通倍率への耐性の形で左/中央を比較する。</p><section class="panel"><div class="controls"><button id="run">Motif Stability Profileを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Profile生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});const d=await res.json(),s=d.summary,c=d.comparison;m.innerHTML=`<div class="metric">左 Event全条件<b>${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b>${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">候補Motif数<b>${s.candidate_motif_count}</b></div><div class="metric">異なるStability Profile<b class="${s.different_stability_profile_count>0?'good':'warn'}">${s.different_stability_profile_count}</b></div><div class="metric">Profile分離<b class="${s.coarse_profile_separates_left_center?'good':'warn'}">${yn(s.coarse_profile_separates_left_center)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'])}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'])}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v52: http://{HOST}:{PORT}")
    print("Motif Stability Profile / live perturbation / no Core changes")
    serve(app, host=HOST, port=PORT)
