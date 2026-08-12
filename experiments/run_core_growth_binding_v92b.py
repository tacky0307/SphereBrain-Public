from __future__ import annotations

import hashlib
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
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
from semantic_relative_selectivity import RelativeSelectivityConfig
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85
import run_core_growth_binding_v86 as v86
import run_core_growth_binding_v87 as v87
import run_core_growth_binding_v89 as v89

HOST = "127.0.0.1"
START_PORT = 5145
OUT = ROOT / "data" / "core_growth_binding_v92b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS
SEEDS = v89.SEEDS
DOMAINS = v89.DOMAINS
CFG = RelativeSelectivityConfig()


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


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def is_bridge(row: dict) -> bool:
    return bool(row["direct_success_now"])


class ShadowSelectivityDetector:
    """v92 detector logic without any weight intervention."""

    def __init__(self) -> None:
        self.cycle = 0
        self.consecutive: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.peak = 0.0
        self.max_drop = 0.0
        self.max_drop_when_mature = 0.0
        self.mature_cycles = 0
        self.candidate_cycles = 0
        self.consecutive_gate_cycles = 0
        self.context_gate_cycles = 0
        self.hub_gate_cycles = 0
        self.drop_gate_cycles = 0
        self.would_trigger_cycles: list[int] = []
        self.rows: list[dict] = []

    @staticmethod
    def key(edge) -> str:
        a, b = sorted(int(x) for x in edge)
        return f"{a}>{b}"

    def observe(self, brain, candidates: list[dict]) -> dict:
        self.cycle += 1
        current = {}
        for row in candidates:
            edge = tuple(sorted(int(x) for x in row["edge"]))
            current[self.key(edge)] = {
                "edge": edge,
                "context_score": float(row.get("context_score", 0.0)),
                "hub_score": float(row.get("hub_score", 1.0)),
            }
        if current:
            self.candidate_cycles += 1

        for key in set(self.consecutive) | set(current):
            if key not in current:
                self.consecutive[key] = 0
                continue
            prev = self.last_seen.get(key)
            self.consecutive[key] = int(self.consecutive.get(key, 0)) + 1 if prev == self.cycle - 1 else 1
            self.last_seen[key] = self.cycle

        has_consecutive = any(self.consecutive.get(k, 0) >= CFG.min_consecutive for k in current)
        has_context = any(r["context_score"] >= CFG.min_context_score for r in current.values())
        has_nonhub = any(r["hub_score"] <= CFG.max_hub_score for r in current.values())
        self.consecutive_gate_cycles += int(has_consecutive)
        self.context_gate_cycles += int(has_context)
        self.hub_gate_cycles += int(has_nonhub)

        mature = []
        for key, row in current.items():
            if (
                self.consecutive.get(key, 0) >= CFG.min_consecutive
                and row["context_score"] >= CFG.min_context_score
                and row["hub_score"] <= CFG.max_hub_score
            ):
                a, b = row["edge"]
                weight = float(brain.weights[a, b])
                intrinsic = weight * row["context_score"] * max(0.0, 1.0 - row["hub_score"])
                mature.append({**row, "intrinsic": intrinsic})

        score = sum(x["intrinsic"] for x in mature)
        if self.peak > 0:
            self.peak *= CFG.peak_decay
        self.peak = max(self.peak, score)
        drop = (self.peak - score) / self.peak if self.peak > 1e-12 else 0.0
        self.max_drop = max(self.max_drop, drop)
        if mature:
            self.mature_cycles += 1
            self.max_drop_when_mature = max(self.max_drop_when_mature, drop)
        drop_ok = bool(mature) and drop >= CFG.drop_trigger_ratio
        self.drop_gate_cycles += int(drop_ok)
        if drop_ok:
            self.would_trigger_cycles.append(self.cycle)

        row = {
            "cycle": self.cycle,
            "candidate_count": len(current),
            "has_consecutive": has_consecutive,
            "has_context": has_context,
            "has_nonhub": has_nonhub,
            "mature_count": len(mature),
            "score": score,
            "peak": self.peak,
            "drop_ratio": drop,
            "drop_ok": drop_ok,
        }
        self.rows.append(row)
        return row

    def failure_reason(self) -> str:
        if self.candidate_cycles == 0:
            return "no_candidate"
        if self.consecutive_gate_cycles == 0:
            return "consecutive_not_met"
        if self.context_gate_cycles == 0:
            return "context_score_not_met"
        if self.hub_gate_cycles == 0:
            return "hub_filter_blocked"
        if self.mature_cycles == 0:
            return "combined_maturity_not_met"
        if self.drop_gate_cycles == 0:
            return "drop_threshold_not_met"
        return "would_have_triggered"


def run_trial(domain, seed: int) -> dict:
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    exp_baseline = v89.make_baseline(experimental, domain)
    ctrl_baseline = v89.make_baseline(control, domain)
    exp_episodes = v89.episodes_for(domain, True)
    ctrl_episodes = v89.episodes_for(domain, False)
    detector = ShadowSelectivityDetector()
    records = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            records.append(v89.make_cycle_row(experimental, control, domain, exp_baseline, ctrl_baseline, cycle))
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)
            candidates = v87.selective_candidates(experimental, domain, exp_baseline["direct"], True)
            detector.observe(experimental, candidates)

    active_idx = [i for i, r in enumerate(records) if is_bridge(r)]
    if not active_idx:
        return {
            "domain": domain.name,
            "seed": int(seed),
            "ever_bridge": False,
            "stable": False,
            "detector_reason": detector.failure_reason(),
            "detector": detector,
            "experimental": experimental,
        }

    cls = v86.classify(records)
    first_i = active_idx[0]
    first_bridge_cycle = int(records[first_i]["cycle"])
    collapse_cycle = None
    for i in range(first_i + 1, len(records)):
        if not is_bridge(records[i]):
            collapse_cycle = int(records[i]["cycle"])
            break

    return {
        "domain": domain.name,
        "seed": int(seed),
        "ever_bridge": True,
        "stable": cls == "stable",
        "classification": cls,
        "first_bridge_cycle": first_bridge_cycle,
        "first_collapse_cycle": collapse_cycle,
        "detector_reason": detector.failure_reason(),
        "candidate_cycles": detector.candidate_cycles,
        "consecutive_gate_cycles": detector.consecutive_gate_cycles,
        "context_gate_cycles": detector.context_gate_cycles,
        "hub_gate_cycles": detector.hub_gate_cycles,
        "mature_cycles": detector.mature_cycles,
        "drop_gate_cycles": detector.drop_gate_cycles,
        "max_drop": detector.max_drop,
        "max_drop_when_mature": detector.max_drop_when_mature,
        "would_trigger_cycles": detector.would_trigger_cycles,
        "detector_rows": detector.rows,
        "detector": detector,
        "experimental": experimental,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    ever = [x for x in trials if x["ever_bridge"]]
    unstable = [x for x in ever if not x["stable"]]
    stable = [x for x in ever if x["stable"]]

    reason_counts = Counter(x["detector_reason"] for x in unstable)
    mature_unstable = [x for x in unstable if x.get("mature_cycles", 0) > 0]
    drop_missed = [x for x in mature_unstable if x.get("drop_gate_cycles", 0) == 0]
    would_trigger = [x for x in unstable if x.get("drop_gate_cycles", 0) > 0]

    threshold_near_misses = [
        x for x in drop_missed
        if x.get("max_drop_when_mature", 0.0) >= CFG.drop_trigger_ratio * 0.70
    ]

    gate_reach = {
        "candidate_seen": sum(1 for x in unstable if x.get("candidate_cycles", 0) > 0),
        "consecutive_met": sum(1 for x in unstable if x.get("consecutive_gate_cycles", 0) > 0),
        "context_met": sum(1 for x in unstable if x.get("context_gate_cycles", 0) > 0),
        "nonhub_met": sum(1 for x in unstable if x.get("hub_gate_cycles", 0) > 0),
        "mature_ensemble_seen": len(mature_unstable),
        "drop_threshold_met": len(would_trigger),
    }

    domain_rows = []
    for domain in DOMAINS:
        rows = [x for x in unstable if x["domain"] == domain.name]
        domain_rows.append({
            "domain": domain.name,
            "unstable": len(rows),
            "reasons": dict(Counter(x["detector_reason"] for x in rows)),
            "mature_seen": sum(1 for x in rows if x.get("mature_cycles", 0) > 0),
            "drop_met": sum(1 for x in rows if x.get("drop_gate_cycles", 0) > 0),
        })

    representative = ever[0] if ever else trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "selectivity_detection_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    dominant_reason = reason_counts.most_common(1)[0][0] if reason_counts else None
    attribution_pass = (
        len(unstable) > 0
        and sum(reason_counts.values()) == len(unstable)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if dominant_reason == "drop_threshold_not_met":
        readiness = "selectivity_detector_drop_gate_is_primary_bottleneck"
        verdict = "mature_local_ensembles_are_seen_but_v92_drop_threshold_rarely_matches_real_post_bridge_collapse"
        next_step = "test_detection_only_threshold_sweep_before_enabling_any_preservation_intervention"
    elif dominant_reason in {"consecutive_not_met", "combined_maturity_not_met"}:
        readiness = "selectivity_detector_maturity_gate_is_primary_bottleneck"
        verdict = "v92_fails_to_recognize_mature_bridge_ensembles_before_real_post_bridge_collapse"
        next_step = "redesign_maturity_detection_before_changing_preservation_strength"
    else:
        readiness = "selectivity_detector_failure_mode_attributed"
        verdict = "v92_zero_intervention_is_explained_by_specific_detector_gates"
        next_step = "validate_the_dominant_detector_gate_with_shadow_parameter_sweep_before_intervention"

    payload = {
        "experiment": "Core Growth Binding v92B — Selectivity Event Detection Microscope",
        "contract": {
            "primary_core_modified": False,
            "preserver_modified": False,
            "intervention_applied": False,
            "shadow_detector_only": True,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "trial_count": len(trials),
            "v92_config": CFG.__dict__,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(trials),
            "ever_bridge_trials": len(ever),
            "stable_trials": len(stable),
            "unstable_trials": len(unstable),
            "detector_reason_counts": dict(reason_counts),
            "dominant_detector_failure": dominant_reason,
            "gate_reach": gate_reach,
            "mature_unstable_trials": len(mature_unstable),
            "drop_missed_after_maturity": len(drop_missed),
            "drop_threshold_near_miss_trials": len(threshold_near_misses),
            "would_trigger_unstable_trials": len(would_trigger),
            "classification_exhaustive": sum(reason_counts.values()) == len(unstable),
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "domain_rows": domain_rows,
        "unstable_trials": [
            {k: v for k, v in x.items() if k not in {"detector", "experimental", "detector_rows"}}
            for x in unstable
        ],
    }
    (OUT / "latest_binding_v92b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v92B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v92B：Selectivity Event Detection Microscope</h1><p class="lead">v91Bで実際に崩壊したEver Bridge試行に対し、v92の検出条件だけをshadow実行する。weight介入は一切行わず、ゼロ介入の原因をゲート別に分解する。</p><section class="panel"><div class="controls"><button id="run">検出失敗を分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Gate到達</h2><pre id="gates" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Unstable Trial</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Ever Bridge',s.ever_bridge_trials),metric('Stable',s.stable_trials),metric('Unstable',s.unstable_trials,'warn'),metric('Dominant failure',s.dominant_detector_failure||'—'),metric('Mature到達',s.mature_unstable_trials),metric('Mature後drop未達',s.drop_missed_after_maturity),metric('Near miss',s.drop_threshold_near_miss_trials),metric('Shadow発火可能',s.would_trigger_unstable_trials),metric('分類済み',s.classification_exhaustive?`${s.unstable_trials}/${s.unstable_trials}`:'NO',s.classification_exhaustive?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('gates').textContent=JSON.stringify({reason_counts:s.detector_reason_counts,gate_reach:s.gate_reach},null,2);document.getElementById('groups').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('rows').textContent=JSON.stringify(d.unstable_trials,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post("/api/run")
def api_run():
    return jsonify(observe())


@app.get("/")
def index():
    return PAGE


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"v92B UI: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
