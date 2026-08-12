from __future__ import annotations

import hashlib
import json
import socket
import statistics
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
import run_core_growth_binding_v91b as v91b
import run_core_growth_binding_v94 as v94
import run_core_growth_binding_v96 as v96

HOST = "127.0.0.1"
START_PORT = 5152
OUT = ROOT / "data" / "core_growth_binding_v96b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
DOMAINS = v94.DOMAINS


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


def domain_by_name(name: str):
    return next(x for x in DOMAINS if x.name == name)


def original_failure(domain, seed: int) -> dict:
    row = v91b.run_trial(domain, seed)
    return {
        "collapse_code": row.get("collapse_code"),
        "mechanism": row.get("mechanism"),
        "first_collapse_cycle": row.get("first_collapse_cycle"),
    }


def structural_growth_kind(row: dict) -> str:
    support_grew = float(row["support_source_retention"]) > 1.000001
    quorum_grew = float(row["quorum_retention"]) > 1.000001
    consensus_grew = float(row["consensus_retention"]) > 1.000001
    if support_grew or quorum_grew:
        return "structural_support_grew"
    if float(row["added_weight_total"]) > 0 and not support_grew and not quorum_grew:
        return "weight_only_support"
    if consensus_grew:
        return "consensus_score_only_grew"
    return "no_structural_growth"


def enrich(spec: dict) -> dict:
    row = v96.run_trial(spec, "consensus_support")
    domain = domain_by_name(spec["domain"])
    failure = original_failure(domain, int(spec["seed"]))
    row["rescued"] = bool(row["rescue_eligible"] and row["stable_after"])
    row["structural_growth_kind"] = structural_growth_kind(row)
    row["support_source_grew"] = bool(float(row["support_source_retention"]) > 1.000001)
    row["quorum_grew"] = bool(float(row["quorum_retention"]) > 1.000001)
    row["consensus_grew"] = bool(float(row["consensus_retention"]) > 1.000001)
    row.update(failure)
    return row


def mean(rows: list[dict], key: str) -> float | None:
    vals = [float(x[key]) for x in rows if x.get(key) is not None]
    return statistics.mean(vals) if vals else None


def rate(rows: list[dict], key: str) -> float:
    return sum(1 for x in rows if x.get(key)) / len(rows) if rows else 0.0


def compare(rescued: list[dict], failed: list[dict], key: str) -> dict:
    r = mean(rescued, key)
    f = mean(failed, key)
    return {
        "metric": key,
        "rescued_mean": r,
        "failed_mean": f,
        "difference_rescued_minus_failed": None if r is None or f is None else r - f,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    specs = v96.screen_unstable_trials()
    rows = [enrich(spec) for spec in specs]
    eligible = [x for x in rows if x["rescue_eligible"]]
    rescued = [x for x in eligible if x["rescued"]]
    failed = [x for x in eligible if not x["rescued"]]

    metrics = [
        "consensus_retention",
        "support_source_retention",
        "quorum_retention",
        "added_weight_total",
        "target_edge_count",
        "selected_edge_count",
        "final_consensus",
        "final_support_sources",
        "final_quorum_edges",
    ]
    comparisons = [compare(rescued, failed, key) for key in metrics]
    ranked = sorted(
        [x for x in comparisons if x["difference_rescued_minus_failed"] is not None],
        key=lambda x: abs(float(x["difference_rescued_minus_failed"])),
        reverse=True,
    )

    rescued_growth = Counter(x["structural_growth_kind"] for x in rescued)
    failed_growth = Counter(x["structural_growth_kind"] for x in failed)
    rescued_mechanisms = Counter(x.get("mechanism") or "unknown" for x in rescued)
    failed_mechanisms = Counter(x.get("mechanism") or "unknown" for x in failed)
    rescued_codes = Counter(x.get("collapse_code") or "unknown" for x in rescued)
    failed_codes = Counter(x.get("collapse_code") or "unknown" for x in failed)

    support_growth_rescue_rate = rate(rescued, "support_source_grew")
    support_growth_failure_rate = rate(failed, "support_source_grew")
    quorum_growth_rescue_rate = rate(rescued, "quorum_grew")
    quorum_growth_failure_rate = rate(failed, "quorum_grew")
    consensus_growth_rescue_rate = rate(rescued, "consensus_grew")
    consensus_growth_failure_rate = rate(failed, "consensus_grew")

    support_growth_signal = (
        len(rescued) >= 5
        and (
            support_growth_rescue_rate - support_growth_failure_rate >= 0.15
            or quorum_growth_rescue_rate - quorum_growth_failure_rate >= 0.15
        )
    )
    weight_only_dominant = (
        failed_growth.get("weight_only_support", 0) > failed_growth.get("structural_support_grew", 0)
    )

    domain_rows = []
    for domain in DOMAINS:
        drows = [x for x in eligible if x["domain"] == domain.name]
        if not drows:
            continue
        domain_rows.append({
            "domain": domain.name,
            "eligible": len(drows),
            "rescued": sum(1 for x in drows if x["rescued"]),
            "rescued_with_support_growth": sum(1 for x in drows if x["rescued"] and x["support_source_grew"]),
            "failed_with_support_growth": sum(1 for x in drows if not x["rescued"] and x["support_source_grew"]),
        })

    representative = rows[0] if rows else None
    OUT.mkdir(parents=True, exist_ok=True)
    saveload_equal = False
    if representative is not None:
        temp = OUT / "consensus_rescue_failure_attribution_roundtrip.json"
        before_weights = representative["experimental"].weights.tolist()
        representative["experimental"].save(temp)
        loaded = SphereBrain.load(temp)
        saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = representative is not None and hasattr(representative["experimental"], "learning_state")

    attribution_pass = (
        len(eligible) > 0
        and len(rescued) > 0
        and len(failed) > 0
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if support_growth_signal:
        readiness = "consensus_rescue_structural_support_growth_attributed"
        verdict = "successful_consensus_rescue_is_associated_with_growth_of_independent_structural_support_more_than_weight_only_strengthening"
        next_step = "test_experience_driven_creation_of_additional_independent_support_paths_without_direct_weight_support"
    elif weight_only_dominant:
        readiness = "consensus_rescue_weight_only_failure_attributed"
        verdict = "most_failed_rescue_trials_gain_weight_without_growing_independent_support_sources_or_quorum"
        next_step = "replace_direct_weight_support_with_experience_driven_multi_path_support_formation"
    else:
        readiness = "consensus_rescue_failure_partially_attributed"
        verdict = "rescue_success_and_failure_differ_but_independent_support_growth_is_not_yet_the_dominant_separator"
        next_step = "inspect_target_coverage_and_collapse_mechanism_specific_rescue_before_new_core_change"

    payload = {
        "experiment": "Core Growth Binding v96B — Consensus Rescue Failure Attribution",
        "contract": {
            "primary_core_modified": False,
            "v96_support_rule_modified": False,
            "cohort": "v94_ever_bridge_unstable",
            "mode_replayed": "consensus_support",
            "collapse_attribution": "v91B",
            "production_brain_json_saved": False,
        },
        "summary": {
            "unstable_cohort": len(rows),
            "rescue_eligible": len(eligible),
            "rescued": len(rescued),
            "not_rescued": len(failed),
            "rescue_rate": len(rescued) / len(eligible) if eligible else 0.0,
            "rescued_support_growth_rate": support_growth_rescue_rate,
            "failed_support_growth_rate": support_growth_failure_rate,
            "support_growth_rate_delta": support_growth_rescue_rate - support_growth_failure_rate,
            "rescued_quorum_growth_rate": quorum_growth_rescue_rate,
            "failed_quorum_growth_rate": quorum_growth_failure_rate,
            "quorum_growth_rate_delta": quorum_growth_rescue_rate - quorum_growth_failure_rate,
            "rescued_consensus_growth_rate": consensus_growth_rescue_rate,
            "failed_consensus_growth_rate": consensus_growth_failure_rate,
            "consensus_growth_rate_delta": consensus_growth_rescue_rate - consensus_growth_failure_rate,
            "support_growth_signal": support_growth_signal,
            "weight_only_failure_dominant": weight_only_dominant,
            "top_attribution_metric": ranked[0]["metric"] if ranked else None,
            "top_attribution_difference": ranked[0]["difference_rescued_minus_failed"] if ranked else None,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "attribution_pass": attribution_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "comparisons_ranked": ranked,
        "growth_kind": {
            "rescued": dict(rescued_growth),
            "failed": dict(failed_growth),
        },
        "collapse_mechanisms": {
            "rescued": dict(rescued_mechanisms),
            "failed": dict(failed_mechanisms),
        },
        "collapse_codes": {
            "rescued": dict(rescued_codes),
            "failed": dict(failed_codes),
        },
        "domain_rows": domain_rows,
        "eligible_trials": [{k: v for k, v in x.items() if k != "experimental"} for x in eligible],
    }
    (OUT / "latest_binding_v96b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v96B</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v96B：Consensus Rescue Failure Attribution</h1><p class="lead">v96のConsensus support条件をそのまま再現し、救済成功と失敗で独立Support source / Quorumが本当に増えたかを比較する。Core・補強ルールは変更しない。</p><section class="panel"><div class="controls"><button id="run">救済失敗を分解</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Rescued vs Failed</h2><pre id="compare" class="raw">未実行</pre></section><section class="panel"><h2>Growth分類 / Collapse</h2><pre id="groups" class="raw">未実行</pre></section><section class="panel"><h2>Domain別</h2><pre id="domains" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function pct(v){return `${(100*v).toFixed(1)}%`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Unstable cohort',s.unstable_cohort),metric('Rescue eligible',s.rescue_eligible),metric('Rescued',s.rescued,'good'),metric('Not rescued',s.not_rescued,'warn'),metric('Rescue率',pct(s.rescue_rate)),metric('Rescued Support↑',pct(s.rescued_support_growth_rate),'good'),metric('Failed Support↑',pct(s.failed_support_growth_rate)),metric('Support差',pct(s.support_growth_rate_delta),s.support_growth_rate_delta>0?'good':'warn'),metric('Rescued Quorum↑',pct(s.rescued_quorum_growth_rate),'good'),metric('Failed Quorum↑',pct(s.failed_quorum_growth_rate)),metric('Quorum差',pct(s.quorum_growth_rate_delta),s.quorum_growth_rate_delta>0?'good':'warn'),metric('Weight-only failure',yn(s.weight_only_failure_dominant),s.weight_only_failure_dominant?'good':'blue'),metric('Support growth signal',yn(s.support_growth_signal),s.support_growth_signal?'good':'warn'),metric('Top attribution',s.top_attribution_metric||'—'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Attribution PASS',yn(s.attribution_pass),s.attribution_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('compare').textContent=JSON.stringify(d.comparisons_ranked,null,2);document.getElementById('groups').textContent=JSON.stringify({growth_kind:d.growth_kind,collapse_mechanisms:d.collapse_mechanisms,collapse_codes:d.collapse_codes},null,2);document.getElementById('domains').textContent=JSON.stringify(d.domain_rows,null,2)}document.getElementById('run').onclick=run;</script></main></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    return jsonify(observe())


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"v96B server: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(app, host=HOST, port=PORT, threads=6)


if __name__ == "__main__":
    main()
