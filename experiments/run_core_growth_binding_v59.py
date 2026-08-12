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
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v58 as v58

HOST = "127.0.0.1"
START_PORT = 5106
OUT = ROOT / "data" / "core_growth_binding_v59" / "results"
RECENT_WINDOW = 12
STALE_GAP = 10


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


def full_v53_signature(source: dict, position: str) -> list:
    return v58.full_v53_signature(source, position)


def false_region_indices(spec: dict, timeline: list[dict]) -> set[int]:
    """Indices where fast forgetting is not justified by a persistent boundary.

    Before the first persistent boundary is always a false-trigger scoring region.
    A scenario with no persistent boundary is entirely non-persistent.
    Temporary-reversion/outlier segments are also diagnostic non-persistent regions.
    """
    boundaries = sorted(int(x["index"]) for x in spec.get("boundaries", []))
    result: set[int] = set()
    if boundaries:
        result.update(range(1, boundaries[0]))
    else:
        result.update(range(1, len(timeline) + 1))

    if spec.get("outlier_window"):
        a, b = spec["outlier_window"]
        result.update(range(int(a), int(b) + 1))
    if spec.get("temporary_reversion"):
        a, b = spec["temporary_reversion"]
        result.update(range(int(a), int(b) + 1))
    return result


def recent_rows(timeline: list[dict], index: int, window: int = RECENT_WINDOW) -> list[dict]:
    start = max(0, index - window)
    return timeline[start:index]


def last_seen_gap(timeline: list[dict], index: int, condition: str) -> int | None:
    for j in range(index - 2, -1, -1):
        if timeline[j]["condition"] == condition:
            return (index - 1) - j
    return None


def classify_trigger(spec: dict, timeline: list[dict], row_index: int) -> tuple[str, dict]:
    row = timeline[row_index]
    idx = int(row["index"])
    condition = row["condition"]
    detector = row["detector"]
    prev = recent_rows(timeline, row_index)
    surprise_rows = [x for x in prev + [row] if int(x["detector"].get("surprise", 0)) == 1]
    recent_conditions = [x["condition"] for x in surprise_rows]
    distinct_surprise_conditions = len(set(recent_conditions))
    same_condition_surprises = sum(1 for c in recent_conditions if c == condition)
    gap = last_seen_gap(timeline, row_index + 1, condition)
    margin = detector.get("margin_before")
    probability = detector.get("probability_before")
    segment = row.get("evaluation_segment", "")

    facts = {
        "last_seen_gap": gap,
        "distinct_surprise_conditions_recent": distinct_surprise_conditions,
        "same_condition_surprises_recent": same_condition_surprises,
        "segment": segment,
        "margin_before": margin,
        "probability_before": probability,
    }

    if "outlier" in segment:
        return "clustered_outlier", facts
    if "temporary_reversion" in segment:
        return "temporary_reversion", facts
    if gap is not None and gap >= STALE_GAP:
        return "stale_evidence_reentry", facts
    if same_condition_surprises >= 2 and distinct_surprise_conditions <= 1:
        return "same_condition_repetition", facts
    if margin is not None and float(margin) < 0.60:
        return "low_confidence_misclassification", facts
    if distinct_surprise_conditions >= 2:
        return "multi_condition_conflict", facts
    if "biased" in segment or spec["name"] == "long_bias_then_shift":
        return "long_bias", facts
    return "other", facts


def audit_scenario(spec: dict, report: dict) -> dict:
    timeline = report["timeline"]
    false_region = false_region_indices(spec, timeline)
    false_steps = [
        r for r in timeline
        if int(r["index"]) in false_region and bool(r["detector"].get("fast_forgetting"))
    ]

    episodes = []
    in_episode = False
    current = None
    for i, row in enumerate(timeline):
        is_false_fast = int(row["index"]) in false_region and bool(row["detector"].get("fast_forgetting"))
        if is_false_fast and not in_episode:
            category, facts = classify_trigger(spec, timeline, i)
            current = {
                "start_index": int(row["index"]),
                "end_index": int(row["index"]),
                "start_condition": row["condition"],
                "start_segment": row.get("evaluation_segment"),
                "category": category,
                "facts": facts,
                "trigger_detector": row["detector"],
                "trigger_profile_confidence": float(row["profile"].get("confidence", 0.0)),
                "recent_surprise_sequence": [
                    int(x["detector"].get("surprise", 0)) for x in recent_rows(timeline, i) + [row]
                ],
                "recent_condition_sequence": [x["condition"] for x in recent_rows(timeline, i) + [row]],
            }
            in_episode = True
        elif is_false_fast and in_episode:
            current["end_index"] = int(row["index"])
        elif not is_false_fast and in_episode:
            current["duration_steps"] = current["end_index"] - current["start_index"] + 1
            episodes.append(current)
            current = None
            in_episode = False
    if in_episode and current is not None:
        current["duration_steps"] = current["end_index"] - current["start_index"] + 1
        episodes.append(current)

    return {
        "scenario": spec["name"],
        "false_trigger_step_count": len(false_steps),
        "false_trigger_episode_count": len(episodes),
        "episodes": episodes,
    }


def observe() -> dict:
    print("v59: reproducing v53 candidate and v58 streams...", flush=True)
    source = v53.observe()
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v59 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    targets = {"左": full_v53_signature(source, "左"), "中央": full_v53_signature(source, "中央")}

    specs = v58.build_scenarios(motif)
    audits = []
    v58_results = []
    for spec in specs:
        print(f"v59: auditing {spec['name']}", flush=True)
        report = v58.run_scenario(spec, motif, targets)
        v58_results.append(report)
        audits.append(audit_scenario(spec, report))

    all_episodes = [ep | {"scenario": a["scenario"]} for a in audits for ep in a["episodes"]]
    counts = Counter(ep["category"] for ep in all_episodes)
    total_steps = sum(a["false_trigger_step_count"] for a in audits)
    total_episodes = len(all_episodes)
    dominant = counts.most_common(1)[0][0] if counts else "none"
    dominant_count = counts.most_common(1)[0][1] if counts else 0
    dominant_fraction = 0.0 if total_episodes == 0 else dominant_count / total_episodes

    # Attribution only: detector behavior is unchanged from v58.
    if total_episodes == 0:
        verdict = "no_false_trigger_episode_reproduced"
        next_step = "recheck_v58_reproducibility_before_detector_change"
    elif dominant_fraction >= 0.60:
        verdict = "false_triggers_have_dominant_attributable_failure_mode"
        next_step = f"design_targeted_detector_fix_for_{dominant}_without_changing_adaptation_logic"
    else:
        verdict = "false_triggers_are_distributed_across_multiple_failure_modes"
        next_step = "design_multi_factor_drift_gate_before_behavioral_shadow_effect"

    payload = {
        "experiment": "Core Growth Binding v59",
        "purpose": "Attribute v58 false-trigger steps to underlying detector failure modes without changing the v57 detector. Separate fast-forgetting steps from independent false-trigger episodes and classify episode onsets.",
        "contract": {
            "detector_modified": False,
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "behavioral_shadow_effect": False,
            "position_label_stored_in_shadow": False,
            "false_trigger_episode_definition": "contiguous fast-forgetting steps inside non-persistent scoring regions",
        },
        "selected_motif": motif,
        "scenario_audits": audits,
        "category_counts": dict(counts),
        "summary": {
            "scenario_count": len(audits),
            "false_trigger_steps": total_steps,
            "false_trigger_episodes": total_episodes,
            "dominant_failure_mode": dominant,
            "dominant_failure_count": dominant_count,
            "dominant_failure_fraction": dominant_fraction,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v59.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v59</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v59</h1><p class="lead">v58のfalse triggerをStep数と独立Episode数へ分離し、各Episode開始時のsurprise履歴・margin・condition freshness・segmentを使って原因を帰属する。Detector本体は変更しない。</p><section class="panel"><div class="controls"><button id="run">False Triggerを解剖</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Attribution生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Scenario数<b>${s.scenario_count}</b></div><div class="metric">False Trigger Step<b class="warn">${s.false_trigger_steps}</b></div><div class="metric">False Trigger Episode<b>${s.false_trigger_episodes}</b></div><div class="metric">最多原因<b class="blue">${s.dominant_failure_mode}</b></div><div class="metric">最多原因件数<b>${s.dominant_failure_count}</b></div><div class="metric">最多原因比率<b>${Number(s.dominant_failure_fraction).toFixed(3)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v59: http://{HOST}:{PORT}")
    print("False Trigger Attribution / detector unchanged / no Core changes")
    serve(app, host=HOST, port=PORT)
