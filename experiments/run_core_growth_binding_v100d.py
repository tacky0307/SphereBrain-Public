from __future__ import annotations

import json
import math
import socket
import statistics
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_core_growth_binding_v100c as v100c

HOST = "127.0.0.1"
START_PORT = 5161
OUT = ROOT / "data" / "core_growth_binding_v100d" / "results"
TRIGGERS = list(v100c.TRIGGER_SETS.keys())


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


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(x[key]) for x in rows) if rows else 0.0


def corr(rows: list[dict], a: str, b: str) -> float:
    if len(rows) < 3:
        return 0.0
    xs = [float(x[a]) for x in rows]
    ys = [float(x[b]) for x in rows]
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def direction_consistency(rows: list[dict], feature: str, direction: int) -> dict:
    matches = 0
    comparable = 0
    by_trigger = []
    for trigger in TRIGGERS:
        p = [x for x in rows if x["trigger_id"] == trigger and x["persistent"]]
        v = [x for x in rows if x["trigger_id"] == trigger and not x["persistent"]]
        if not p or not v:
            continue
        gap = mean(p, feature) - mean(v, feature)
        comparable += 1
        ok = (gap > 0 and direction > 0) or (gap < 0 and direction < 0)
        if ok:
            matches += 1
        by_trigger.append({"trigger_id": trigger, "gap": gap, "direction_match": ok})
    return {"matches": matches, "comparable": comparable, "by_trigger": by_trigger}


def zscore(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    m = statistics.mean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (value - m) / sd if sd > 1e-12 else 0.0


def observe() -> dict:
    base = v100c.observe()
    rows = list(base.get("quorum_edges", []))
    persistent = [x for x in rows if x["persistent"]]
    vanished = [x for x in rows if not x["persistent"]]
    ranking = list(base.get("feature_ranking", []))
    robust = list(base.get("robust_candidates", []))

    profiles = []
    for item in robust:
        feature = item["feature"]
        effect = float(item["standardized_gap"])
        direction = 1 if effect > 0 else -1
        consistency = direction_consistency(rows, feature, direction)
        profiles.append({
            "feature": feature,
            "direction": "higher_in_persistent" if direction > 0 else "lower_in_persistent",
            "standardized_gap": effect,
            "persistent_mean": mean(persistent, feature),
            "vanished_mean": mean(vanished, feature),
            "direction_matches": consistency["matches"],
            "comparable_triggers": consistency["comparable"],
            "by_trigger": consistency["by_trigger"],
        })

    feature_names = [x["feature"] for x in profiles]
    correlations = []
    for i, a in enumerate(feature_names):
        for b in feature_names[i + 1:]:
            correlations.append({"a": a, "b": b, "r": corr(rows, a, b)})
    max_abs_corr = max((abs(x["r"]) for x in correlations), default=0.0)

    # Composite profile: each robust feature contributes in its observed direction.
    all_values = {f: [float(x[f]) for x in rows] for f in feature_names}
    scored = []
    for row in rows:
        score = 0.0
        for p in profiles:
            z = zscore(float(row[p["feature"]]), all_values[p["feature"]])
            score += z if p["direction"] == "higher_in_persistent" else -z
        scored.append({"persistent": bool(row["persistent"]), "trigger_id": row["trigger_id"], "score": score})

    p_scores = [x["score"] for x in scored if x["persistent"]]
    v_scores = [x["score"] for x in scored if not x["persistent"]]
    composite_gap = (statistics.mean(p_scores) - statistics.mean(v_scores)) if p_scores and v_scores else 0.0
    pooled = p_scores + v_scores
    composite_sd = statistics.pstdev(pooled) if len(pooled) > 1 else 0.0
    composite_effect = composite_gap / composite_sd if composite_sd > 1e-12 else composite_gap

    composite_trigger_matches = 0
    for trigger in TRIGGERS:
        p = [x["score"] for x in scored if x["trigger_id"] == trigger and x["persistent"]]
        v = [x["score"] for x in scored if x["trigger_id"] == trigger and not x["persistent"]]
        if p and v and statistics.mean(p) > statistics.mean(v):
            composite_trigger_matches += 1

    profile_signal = (
        len(profiles) >= 2
        and all(x["direction_matches"] >= 3 for x in profiles)
        and composite_effect >= 0.50
        and composite_trigger_matches >= 3
    )

    payload = {
        "experiment": "Core Growth Binding v100D — Persistent Quorum Assembly Profile",
        "contract": {
            "primary_core_modified": False,
            "v100c_data_reused": True,
            "persistence_used_only_as_posthoc_label": True,
            "feature_direction_preserved": True,
            "semantic_labels_used_for_profile": False,
        },
        "summary": {
            "quorum_edge_count": len(rows),
            "persistent_quorum_edges": len(persistent),
            "vanished_quorum_edges": len(vanished),
            "robust_candidate_count": len(profiles),
            "direction_consistent_candidates": sum(1 for x in profiles if x["direction_matches"] >= 3),
            "max_abs_candidate_correlation": max_abs_corr,
            "composite_effect": composite_effect,
            "composite_trigger_matches": composite_trigger_matches,
            "profile_signal": profile_signal,
            "measurement_pass": bool(base.get("summary", {}).get("measurement_pass")),
            "brain_file_unchanged": bool(base.get("summary", {}).get("brain_file_unchanged")),
            "core_readiness": (
                "persistent_quorum_assembly_profile_identified"
                if profile_signal else
                "persistent_quorum_assembly_profile_not_yet_stable"
            ),
        },
        "profiles": profiles,
        "candidate_correlations": correlations,
        "composite": {
            "persistent_mean": statistics.mean(p_scores) if p_scores else 0.0,
            "vanished_mean": statistics.mean(v_scores) if v_scores else 0.0,
            "standardized_gap": composite_effect,
            "trigger_matches": composite_trigger_matches,
        },
        "source_v100c_summary": base.get("summary", {}),
        "source_feature_ranking": ranking,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v100d.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v100D</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v100D：Persistent Quorum Assembly Profile</h1><p class="lead">v100Cで得たRobust candidateの方向・Trigger一貫性・相関・複合Profileをまとめ、Persistent quorumの形成状態を単独特徴ではなくProfileとして観察する。</p><section class="panel"><div class="controls"><button id="run">Profileを観察</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Candidate profile</h2><pre id="profiles" class="raw">未実行</pre></section><section class="panel"><h2>Correlations</h2><pre id="corr" class="raw">未実行</pre></section><section class="panel"><h2>Raw JSON</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Quorum Edge',s.quorum_edge_count),metric('Persistent',s.persistent_quorum_edges,'good'),metric('Vanished',s.vanished_quorum_edges),metric('Robust candidates',s.robust_candidate_count),metric('Direction consistent',`${s.direction_consistent_candidates}/${s.robust_candidate_count}`,s.direction_consistent_candidates===s.robust_candidate_count?'good':'warn'),metric('Max candidate corr',s.max_abs_candidate_correlation.toFixed(3)),metric('Composite effect',s.composite_effect.toFixed(3),s.composite_effect>=0.5?'good':'warn'),metric('Composite + triggers',`${s.composite_trigger_matches}/4`,s.composite_trigger_matches>=3?'good':'warn'),metric('Profile signal',yn(s.profile_signal),s.profile_signal?'good':'warn'),metric('観測PASS',yn(s.measurement_pass),s.measurement_pass?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('profiles').textContent=JSON.stringify(d.profiles,null,2);document.getElementById('corr').textContent=JSON.stringify(d.candidate_correlations,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"Core Growth Binding v100D: http://{HOST}:{PORT}")
    threading.Timer(0.8, open_browser).start()
    serve(app, host=HOST, port=PORT)
