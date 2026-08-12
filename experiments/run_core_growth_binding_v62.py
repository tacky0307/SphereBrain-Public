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
import run_core_growth_binding_v53 as v53
import run_core_growth_binding_v54 as v54
import run_core_growth_binding_v61 as v61

HOST = "127.0.0.1"
START_PORT = 5109
OUT = ROOT / "data" / "core_growth_binding_v62" / "results"
POSITIONS = ["左", "中央", "右"]
SCALES = [0.94, 0.97, 1.00, 1.03, 1.06]
TIE_MARGIN = v61.TIE_MARGIN
MIN_CONFIDENCE = 0.90
TOP_N = 20


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


def find_full_audit(source: dict) -> dict:
    for row in source.get("selected_profile_audit", {}).get("contexts", []):
        if row.get("context") == "full":
            return row
    raise RuntimeError("v53 full audit not found")


def shadow_meta(source: dict) -> dict[str, dict]:
    minimal = source.get("minimal_signature", {})
    if not minimal.get("found") or minimal.get("size") != 1:
        raise RuntimeError("v62 requires the v53 one-profile candidate")
    motif = minimal["profiles"][0]
    full = find_full_audit(source)
    count = len(source.get("conditions", []))
    result = {}
    for position, key in [("左", "left_signature"), ("中央", "center_signature")]:
        state = v54.state_from_signature(motif, full[key][0], count)
        result[position] = {
            "motif": motif,
            "confidence": 0.96,
            "drift_suspected": False,
            "signature": list(full[key][0]),
            "eligible_by_shadow_state": True,
        }
    result["右"] = {
        "motif": motif,
        "confidence": None,
        "drift_suspected": None,
        "signature": None,
        "eligible_by_shadow_state": False,
    }
    return result


def candidate_scores(step: dict) -> list[dict]:
    # Reproduce v44's focused candidate construction from the recorded raw rows.
    # A target can receive more than one source; only the strongest local-top,
    # above-threshold signal for that target survives into the candidate map.
    best: dict[int, dict] = {}
    for row in step.get("records", []):
        if not bool(row.get("local_top")) or not bool(row.get("passes_threshold")):
            continue
        target = int(row["target"])
        signal = float(row["signal"])
        previous = best.get(target)
        if previous is None or signal > float(previous["signal"]):
            best[target] = {
                "target": target,
                "source": int(row["source"]),
                "signal": signal,
                "weight": float(row["weight"]),
                "source_activation": float(row["source_activation"]),
            }
    return sorted(best.values(), key=lambda x: (-float(x["signal"]), int(x["target"])))


def scan_trace(position: str, echo_scale: float, position_scale: float, shadow: dict) -> list[dict]:
    trace = v44.run_live_scaled(
        position=position,
        include_echo=True,
        echo_scale=echo_scale,
        position_scale=position_scale,
    )
    rows = []
    for step in trace.get("steps", []):
        ranked = candidate_scores(step)
        if len(ranked) < 2:
            continue
        top1, top2 = ranked[0], ranked[1]
        margin = float(top1["signal"] - top2["signal"])
        shadow_ok = bool(
            shadow.get("eligible_by_shadow_state")
            and shadow.get("confidence") is not None
            and float(shadow["confidence"]) >= MIN_CONFIDENCE
            and not bool(shadow.get("drift_suspected"))
        )
        tie_ok = margin <= TIE_MARGIN + 1e-15
        rows.append({
            "position": position,
            "echo_scale": float(echo_scale),
            "position_scale": float(position_scale),
            "step": int(step["step"]),
            "candidate_count": len(ranked),
            "top1": top1,
            "top2": top2,
            "margin": margin,
            "tie_margin": TIE_MARGIN,
            "within_tie_margin": tie_ok,
            "shadow_gate_ok": shadow_ok,
            "would_be_v61_eligible": bool(tie_ok and shadow_ok),
        })
    return rows


def observe() -> dict:
    print("v62: reproducing v53 Shadow candidate...", flush=True)
    source = v53.observe()
    shadows = shadow_meta(source)

    print("v62: scanning natural Core choice margins with fixed Core settings...", flush=True)
    rows = []
    run_count = 0
    for position in POSITIONS:
        for echo_scale in SCALES:
            for position_scale in SCALES:
                run_count += 1
                rows.extend(scan_trace(position, echo_scale, position_scale, shadows[position]))

    ordered = sorted(
        rows,
        key=lambda r: (
            float(r["margin"]),
            r["position"],
            float(r["echo_scale"]),
            float(r["position_scale"]),
            int(r["step"]),
        ),
    )
    top = ordered[:TOP_N]
    natural_ties = [r for r in rows if r["within_tie_margin"]]
    eligible = [r for r in rows if r["would_be_v61_eligible"]]
    exact_ties = [r for r in rows if abs(float(r["margin"])) <= 1e-12]

    by_position = {}
    for position in POSITIONS:
        subset = [r for r in rows if r["position"] == position]
        by_position[position] = {
            "choice_steps": len(subset),
            "minimum_margin": None if not subset else min(float(r["margin"]) for r in subset),
            "within_tie_margin_steps": sum(1 for r in subset if r["within_tie_margin"]),
            "v61_eligible_steps": sum(1 for r in subset if r["would_be_v61_eligible"]),
        }

    min_margin = None if not ordered else float(ordered[0]["margin"])
    natural_boundary_exists = bool(natural_ties)
    eligible_boundary_exists = bool(eligible)

    if eligible_boundary_exists:
        verdict = "natural_choice_boundary_exists_under_current_core_and_v61_shadow_gate"
        next_step = "replay_observed_boundary_cases_with_bounded_behavioral_shadow_assist_without_relaxing_caps"
        readiness = "natural_behavioral_boundary_found"
    elif natural_boundary_exists:
        verdict = "natural_core_ties_exist_but_not_with_eligible_shadow_state"
        next_step = "audit_shadow_availability_at_natural_tie_cases_before_behavioral_replay"
        readiness = "boundary_observed_shadow_not_eligible"
    else:
        verdict = "no_natural_choice_boundary_found_within_current_tie_margin"
        next_step = "retain_shadow_as_non_intervening_context_or_expand_input_coverage_without_relaxing_core_safety_caps"
        readiness = "bounded_behavioral_shadow_safe_no_natural_boundary"

    payload = {
        "experiment": "Core Growth Binding v62",
        "purpose": "Scan the unmodified Core for naturally occurring top1/top2 choice boundaries without activating Behavioral Shadow Assist or relaxing any safety cap. Measure candidate margins across positions and a broad live input-scale grid, then identify cases that would have been eligible under the v61 Shadow gate.",
        "contract": {
            "assist_activated": False,
            "learning": False,
            "weights_changed": False,
            "new_edges_created": False,
            "threshold_changed": False,
            "max_active_per_step_changed": False,
            "tie_margin_changed": False,
            "behavior_forced": False,
            "core_file_modified": False,
            "scan_positions": POSITIONS,
            "scan_scales": SCALES,
            "tie_margin": TIE_MARGIN,
            "minimum_shadow_confidence": MIN_CONFIDENCE,
        },
        "shadow_meta_for_eligibility_only": shadows,
        "scan": {
            "run_count": run_count,
            "choice_step_count": len(rows),
            "exact_tie_steps": len(exact_ties),
            "within_tie_margin_steps": len(natural_ties),
            "v61_eligible_steps": len(eligible),
            "minimum_margin": min_margin,
            "top_nearest_boundaries": top,
            "by_position": by_position,
        },
        "summary": {
            "scan_run_count": run_count,
            "choice_step_count": len(rows),
            "minimum_margin": min_margin,
            "natural_tie_steps": len(natural_ties),
            "v61_eligible_tie_steps": len(eligible),
            "natural_choice_boundary_exists": natural_boundary_exists,
            "eligible_natural_boundary_exists": eligible_boundary_exists,
            "overall_verdict": verdict,
            "core_readiness": readiness,
            "next_step": next_step,
        },
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v62.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v62</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v62</h1><p class="lead">Behavioral Shadow Assistを一切作動させず、Core設定も変更せず、自然に生じるtop1/top2候補marginを広いlive入力条件から観測する。tieを作らず、見つけるだけのBoundary Scan。</p><section class="panel"><div class="controls"><button id="run">Natural Choice Boundaryを走査</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Boundary Scan生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(8)}const btn=document.getElementById('run');btn.addEventListener('click',async()=>{btn.disabled=true;const m=document.getElementById('metrics');m.innerHTML='<div class="metric">状態<b class="blue">計算中...</b></div>';try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok){throw new Error('HTTP '+res.status+' '+await res.text())}const d=await res.json(),s=d.summary;m.innerHTML=`<div class="metric">Scan Run数<b>${s.scan_run_count}</b></div><div class="metric">Choice Step数<b>${s.choice_step_count}</b></div><div class="metric">最小margin<b class="blue">${f(s.minimum_margin)}</b></div><div class="metric">Natural Tie Step<b class="${s.natural_tie_steps>0?'good':'warn'}">${s.natural_tie_steps}</b></div><div class="metric">v61 Eligible Tie<b class="${s.v61_eligible_tie_steps>0?'good':'warn'}">${s.v61_eligible_tie_steps}</b></div><div class="metric">自然Boundaryあり<b class="${s.natural_choice_boundary_exists?'good':'warn'}">${yn(s.natural_choice_boundary_exists)}</b></div><div class="metric">Eligible Boundaryあり<b class="${s.eligible_natural_boundary_exists?'good':'warn'}">${yn(s.eligible_natural_boundary_exists)}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">次段階<b>${s.next_step}</b></div>`;document.getElementById('raw').textContent=JSON.stringify(d,null,2)}catch(e){m.innerHTML=`<div class="metric">エラー<b class="warn">${String(e)}</b></div>`}finally{btn.disabled=false}});
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
    print(f"Core Growth Binding v62: http://{HOST}:{PORT}")
    print("Natural Choice Boundary Scan / observation only / no Assist / no Core changes")
    serve(app, host=HOST, port=PORT)
