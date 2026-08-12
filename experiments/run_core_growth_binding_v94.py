from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v89 as v89
import run_core_growth_binding_v93b as v93b

HOST = "127.0.0.1"
START_PORT = 5148
OUT = ROOT / "data" / "core_growth_binding_v94" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v89.CHECKPOINTS

SEEDS = [
    42, 314, 2718, 8088, 12021,
    7, 73, 101, 509, 997,
    2027, 4099, 6029, 9011, 15013,
    22003, 30011, 41017, 52021, 65029,
    17, 131, 911, 1777, 3253,
    7001, 11003, 17011, 25013, 33013,
    43003, 57037, 71011, 83003, 97001,
    111119, 130003, 150001, 170003, 190027,
]


@dataclass(frozen=True)
class DomainSpec:
    name: str
    left_subject: str
    right_subject: str
    shared_context: str
    left_action: str
    right_action: str
    left_control_context: str
    right_control_context: str
    transfer_left_subject: str
    transfer_right_subject: str


DOMAINS = [
    DomainSpec("sky", "鳥", "飛行機", "空", "羽ばたく", "飛行する", "森", "空港", "蝶", "ドローン"),
    DomainSpec("sea", "魚", "船", "海", "泳ぐ", "進む", "川", "港", "イルカ", "潜水艦"),
    DomainSpec("road", "馬", "車", "道", "走る", "進む", "草原", "駐車場", "鹿", "バス"),
    DomainSpec("forest", "鹿", "キツツキ", "森", "走る", "つつく", "草原", "木立", "ウサギ", "クマゲラ"),
    DomainSpec("workshop", "ハンマー", "ドリル", "工房", "叩く", "穴を開ける", "倉庫", "工具箱", "木槌", "キリ"),
    DomainSpec("mountain", "ヤギ", "登山者", "山", "登る", "歩く", "牧場", "街", "カモシカ", "ハイカー"),
    DomainSpec("river", "カワウソ", "カヌー", "川", "泳ぐ", "進む", "池", "湖岸", "ビーバー", "ボート"),
    DomainSpec("kitchen", "包丁", "ミキサー", "台所", "切る", "混ぜる", "食卓", "収納庫", "ナイフ", "泡立て器"),
]


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


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def group_comparison(rows: list[dict], group_name: str, group_value) -> dict:
    subset = [x for x in rows if x[group_name] == group_value and x["ever_bridge"]]
    stable = [x for x in subset if x["stable"]]
    unstable = [x for x in subset if not x["stable"]]
    s = mean([float(x["consensus_retention_ratio"]) for x in stable])
    u = mean([float(x["consensus_retention_ratio"]) for x in unstable])
    ss = mean([float(x["support_retention_ratio"]) for x in stable])
    us = mean([float(x["support_retention_ratio"]) for x in unstable])
    q1 = mean([float(x["quorum_retention_ratio"]) for x in stable])
    q0 = mean([float(x["quorum_retention_ratio"]) for x in unstable])
    comparable = bool(stable and unstable)
    return {
        group_name: group_value,
        "ever_bridge": len(subset),
        "stable": len(stable),
        "unstable": len(unstable),
        "comparable": comparable,
        "stable_consensus_retention": s,
        "unstable_consensus_retention": u,
        "consensus_delta": None if s is None or u is None else s - u,
        "stable_support_retention": ss,
        "unstable_support_retention": us,
        "support_delta": None if ss is None or us is None else ss - us,
        "stable_quorum_retention": q1,
        "unstable_quorum_retention": q0,
        "quorum_delta": None if q1 is None or q0 is None else q1 - q0,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    all_trials = [v93b.run_trial(domain, seed) for domain in DOMAINS for seed in SEEDS]
    ever = [x for x in all_trials if x["ever_bridge"]]
    stable = [x for x in ever if x["stable"]]
    unstable = [x for x in ever if not x["stable"]]

    stable_retention = mean([float(x["consensus_retention_ratio"]) for x in stable])
    unstable_retention = mean([float(x["consensus_retention_ratio"]) for x in unstable])
    retention_delta = (
        stable_retention - unstable_retention
        if stable_retention is not None and unstable_retention is not None else 0.0
    )

    stable_support = mean([float(x["support_retention_ratio"]) for x in stable])
    unstable_support = mean([float(x["support_retention_ratio"]) for x in unstable])
    support_delta = (
        stable_support - unstable_support
        if stable_support is not None and unstable_support is not None else 0.0
    )

    stable_quorum = mean([float(x["quorum_retention_ratio"]) for x in stable])
    unstable_quorum = mean([float(x["quorum_retention_ratio"]) for x in unstable])
    quorum_delta = (
        stable_quorum - unstable_quorum
        if stable_quorum is not None and unstable_quorum is not None else 0.0
    )

    domain_rows = [group_comparison(ever, "domain", d.name) for d in DOMAINS]
    seed_rows = [group_comparison(ever, "seed", seed) for seed in SEEDS]
    comparable_domains = [x for x in domain_rows if x["comparable"]]
    positive_domains = [x for x in comparable_domains if float(x["consensus_delta"]) > 0]
    comparable_seeds = [x for x in seed_rows if x["comparable"]]
    positive_seeds = [x for x in comparable_seeds if float(x["consensus_delta"]) > 0]

    domain_positive_rate = len(positive_domains) / len(comparable_domains) if comparable_domains else 0.0
    seed_positive_rate = len(positive_seeds) / len(comparable_seeds) if comparable_seeds else 0.0

    unstable_collapses = [x for x in unstable if x.get("first_collapse_cycle") is not None]
    warning_count = sum(1 for x in unstable_collapses if x.get("warning_before_collapse"))
    warning_rate = warning_count / len(unstable_collapses) if unstable_collapses else 0.0

    representative = ever[0] if ever else all_trials[0]
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "consensus_persistence_robustness_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    enough_stable = len(stable) >= 10
    global_signal = retention_delta >= 0.08
    support_consistent = support_delta >= 0.04
    domain_robust = len(comparable_domains) >= 4 and domain_positive_rate >= 0.60
    seed_robust = len(comparable_seeds) >= 8 and seed_positive_rate >= 0.60

    robustness_pass = (
        enough_stable
        and global_signal
        and support_consistent
        and domain_robust
        and seed_robust
        and saveload_equal
        and production_unchanged
        and native_present
    )

    measurement_pass = (
        len(stable) > 0
        and len(unstable) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if robustness_pass:
        readiness = "consensus_persistence_robustness_validated"
        verdict = "consensus_persistence_separates_stable_from_unstable_semantic_bridges_across_expanded_seeds_and_domains"
        next_step = "test_consensus_persistence_as_a_label_free_stability_gate_before_any_primary_core_integration"
    elif global_signal:
        readiness = "consensus_persistence_global_signal_but_not_cross_group_robust"
        verdict = "consensus_persistence_difference_remains_globally_positive_but_is_not_consistent_enough_across_domains_or_seeds"
        next_step = "inspect_domain_and_seed_failure_groups_before_using_consensus_for_intervention"
    else:
        readiness = "consensus_persistence_robustness_not_validated"
        verdict = "v93b_consensus_persistence_advantage_does_not_reproduce_strongly_enough_under_expanded_validation"
        next_step = "deprioritize_structural_consensus_as_a_core_stability mechanism"

    payload = {
        "experiment": "Core Growth Binding v94 — Consensus Persistence Robustness Validation",
        "contract": {
            "primary_core_modified": False,
            "consolidation_used": False,
            "assist_used": False,
            "trial_count": len(all_trials),
            "seed_count": len(SEEDS),
            "domain_count": len(DOMAINS),
            "seeds": SEEDS,
            "domains": [x.name for x in DOMAINS],
            "consensus_formula_changed_from_v93": False,
            "retention_formula_changed_from_v93b": False,
            "future_stability_used_in_consensus_score": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "trial_count": len(all_trials),
            "ever_bridge_trials": len(ever),
            "stable_trials": len(stable),
            "unstable_trials": len(unstable),
            "stable_consensus_retention": stable_retention,
            "unstable_consensus_retention": unstable_retention,
            "consensus_retention_delta": retention_delta,
            "stable_support_retention": stable_support,
            "unstable_support_retention": unstable_support,
            "support_retention_delta": support_delta,
            "stable_quorum_retention": stable_quorum,
            "unstable_quorum_retention": unstable_quorum,
            "quorum_retention_delta": quorum_delta,
            "comparable_domains": len(comparable_domains),
            "positive_consensus_domains": len(positive_domains),
            "domain_positive_rate": domain_positive_rate,
            "comparable_seeds": len(comparable_seeds),
            "positive_consensus_seeds": len(positive_seeds),
            "seed_positive_rate": seed_positive_rate,
            "precollapse_warning_trials": warning_count,
            "precollapse_warning_rate": warning_rate,
            "enough_stable_samples": enough_stable,
            "global_persistence_signal": global_signal,
            "support_signal_consistent": support_consistent,
            "domain_robust": domain_robust,
            "seed_robust": seed_robust,
            "robustness_pass": robustness_pass,
            "measurement_pass": measurement_pass,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "domain_rows": domain_rows,
        "seed_rows": seed_rows,
        "ever_bridge_trials": [{k: v for k, v in x.items() if k != "experimental"} for x in ever],
    }
    (OUT / "latest_binding_v94.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v94</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v94：Consensus Persistence Robustness Validation</h1><p class="lead">v93/v93BのConsensus定義を変更せず、40 seed × 8 domain = 320 Trialへ拡張して、Stable BridgeのConsensus持続性が再現するかを検証する。</p><section class="panel"><div class="controls"><button id="run">320試行で再現性を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Seed別</h2><pre id="seeds" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function pct(v){return v==null?'—':(v*100).toFixed(1)+'%'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',s.trial_count),metric('Ever Bridge',s.ever_bridge_trials),metric('Stable',s.stable_trials,s.enough_stable_samples?'good':'warn'),metric('Unstable',s.unstable_trials),metric('Stable Consensus保持',pct(s.stable_consensus_retention)),metric('Unstable Consensus保持',pct(s.unstable_consensus_retention)),metric('保持率差',pct(s.consensus_retention_delta),s.global_persistence_signal?'good':'warn'),metric('Support保持率差',pct(s.support_retention_delta),s.support_signal_consistent?'good':'warn'),metric('Quorum保持率差',pct(s.quorum_retention_delta)),metric('Domain正方向',`${s.positive_consensus_domains}/${s.comparable_domains}`,s.domain_robust?'good':'warn'),metric('Seed正方向',`${s.positive_consensus_seeds}/${s.comparable_seeds}`,s.seed_robust?'good':'warn'),metric('崩壊前Warning率',pct(s.precollapse_warning_rate)),metric('Domain robust',yn(s.domain_robust),s.domain_robust?'good':'warn'),metric('Seed robust',yn(s.seed_robust),s.seed_robust?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('v94 PASS',yn(s.robustness_pass),s.robustness_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2);document.getElementById('seeds').textContent=JSON.stringify(d.seed_rows,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.post('/api/run')
def api_run():
    try:
        return jsonify(observe())
    except Exception as exc:
        import traceback
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


@app.get('/')
def index():
    return PAGE


def open_browser():
    import time
    time.sleep(.8)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == '__main__':
    print(f"v94: http://{HOST}:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()
    serve(app, host=HOST, port=PORT)
