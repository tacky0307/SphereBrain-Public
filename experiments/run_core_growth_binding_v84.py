from __future__ import annotations

import hashlib
import json
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
import run_core_growth_binding_v82 as v82
import run_core_growth_binding_v82b as v82b
import run_core_growth_binding_v82c as v82c
import run_core_growth_binding_v83 as v83

HOST = "127.0.0.1"
START_PORT = 5133
OUT = ROOT / "data" / "core_growth_binding_v84" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = [0, 1, 3, 5, 10]
SEEDS = [42, 314, 2718, 8088, 12021]


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
    DomainSpec(
        "sky",
        "鳥", "飛行機", "空", "羽ばたく", "飛行する", "森", "空港", "蝶", "ドローン",
    ),
    DomainSpec(
        "sea",
        "魚", "船", "海", "泳ぐ", "進む", "川", "港", "イルカ", "潜水艦",
    ),
    DomainSpec(
        "road",
        "馬", "車", "道", "走る", "進む", "草原", "駐車場", "鹿", "バス",
    ),
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


def clean_primary_seed(seed: int) -> SphereBrain:
    source = SphereBrain.load(BRAIN_PATH)
    brain = SphereBrain(
        node_count=source.node_count,
        neighbors_per_node=source.neighbors_per_node,
        seed=int(seed),
        learning_rate=source.learning_rate,
        decay_rate=source.decay_rate,
        propagation_mode=source.propagation_mode,
        signal_decay=source.signal_decay,
        max_branches=source.max_branches,
        max_active_per_step=source.max_active_per_step,
        max_total_active_nodes=source.max_total_active_nodes,
        structural_assist_enabled=False,
        structural_gain=source.structural_gain,
        structural_tie_margin=source.structural_tie_margin,
        structural_near_zero_margin=source.structural_near_zero_margin,
        structural_relative_cap_ratio=source.structural_relative_cap_ratio,
        structural_absolute_cap=source.structural_absolute_cap,
    )
    brain.clear_experience_state()
    brain.clear_learning_state()
    return brain


def action_item(subject: str, action: str) -> v82.StructuredInput:
    return v82.StructuredInput(subject, "動作", action)


def context_item(subject: str, context: str) -> v82.StructuredInput:
    return v82.StructuredInput(subject, "場所", context)


def episodes_for(domain: DomainSpec, shared: bool) -> list[v83.EpisodeSpec]:
    if shared:
        return [
            v83.EpisodeSpec(domain.left_subject, domain.shared_context, domain.left_action),
            v83.EpisodeSpec(domain.right_subject, domain.shared_context, domain.right_action),
        ]
    return [
        v83.EpisodeSpec(domain.left_subject, domain.left_control_context, domain.left_action),
        v83.EpisodeSpec(domain.right_subject, domain.right_control_context, domain.right_action),
    ]


def make_baseline(brain: SphereBrain, domain: DomainSpec) -> dict[str, set[tuple[int, int]]]:
    left = action_item(domain.left_subject, domain.left_action)
    right = action_item(domain.right_subject, domain.right_action)
    transfer_left = action_item(domain.transfer_left_subject, domain.left_action)
    transfer_right = action_item(domain.transfer_right_subject, domain.right_action)
    return {
        "direct": v83.action_signature(brain, left)["edges"] & v83.action_signature(brain, right)["edges"],
        "transfer_left": v83.action_signature(brain, transfer_left)["edges"] & v83.action_signature(brain, right)["edges"],
        "transfer_right": v83.action_signature(brain, transfer_right)["edges"] & v83.action_signature(brain, left)["edges"],
    }


def domain_context_signature(brain: SphereBrain, domain: DomainSpec) -> dict:
    left = v83.action_signature(brain, context_item(domain.left_subject, domain.shared_context))
    right = v83.action_signature(brain, context_item(domain.right_subject, domain.shared_context))
    return {
        "nodes": left["nodes"] | right["nodes"],
        "edges": left["edges"] | right["edges"],
        "shared_nodes": left["nodes"] & right["nodes"],
        "shared_edges": left["edges"] & right["edges"],
    }


def control_context_signature(brain: SphereBrain, domain: DomainSpec) -> dict:
    left = v83.action_signature(brain, context_item(domain.left_subject, domain.left_control_context))
    right = v83.action_signature(brain, context_item(domain.right_subject, domain.right_control_context))
    return {"nodes": left["nodes"] | right["nodes"], "edges": left["edges"] | right["edges"]}


def pair_metrics(brain: SphereBrain, left, right, baseline: set[tuple[int, int]]) -> dict:
    ls = v83.action_signature(brain, left)
    rs = v83.action_signature(brain, right)
    shared = ls["edges"] & rs["edges"]
    return {
        "edge_similarity": v82.jaccard(ls["edges"], rs["edges"]),
        "activation_similarity": v82b.weighted_similarity(ls["activation"], rs["activation"]),
        "new_shared_edges": len(shared - baseline),
        "new_shared_edge_list": [list(x) for x in sorted(shared - baseline)],
    }


def measure(brain: SphereBrain, domain: DomainSpec, baseline: dict[str, set[tuple[int, int]]]) -> dict:
    left = action_item(domain.left_subject, domain.left_action)
    right = action_item(domain.right_subject, domain.right_action)
    transfer_left = action_item(domain.transfer_left_subject, domain.left_action)
    transfer_right = action_item(domain.transfer_right_subject, domain.right_action)
    direct = pair_metrics(brain, left, right, baseline["direct"])
    tl = pair_metrics(brain, transfer_left, right, baseline["transfer_left"])
    tr = pair_metrics(brain, transfer_right, left, baseline["transfer_right"])
    return {
        "direct": direct,
        "transfer": {
            "left_to_right": tl,
            "right_to_left": tr,
            "mean_edge_similarity": (tl["edge_similarity"] + tr["edge_similarity"]) / 2.0,
            "mean_activation_similarity": (tl["activation_similarity"] + tr["activation_similarity"]) / 2.0,
            "new_shared_edges": tl["new_shared_edges"] + tr["new_shared_edges"],
        },
    }


def context_candidates(
    experimental: SphereBrain,
    control: SphereBrain,
    domain: DomainSpec,
    exp_baseline: set[tuple[int, int]],
) -> list[dict]:
    left = action_item(domain.left_subject, domain.left_action)
    right = action_item(domain.right_subject, domain.right_action)
    exp_new = (
        v83.action_signature(experimental, left)["edges"]
        & v83.action_signature(experimental, right)["edges"]
    ) - exp_baseline
    ctrl_shared = (
        v83.action_signature(control, left)["edges"]
        & v83.action_signature(control, right)["edges"]
    )
    ctx = domain_context_signature(experimental, domain)
    ctrl_ctx = control_context_signature(control, domain)
    rows = []
    for edge in sorted(exp_new):
        row = v82c.annotate_edge(
            experimental,
            edge,
            first_cycle=None,
            sky=ctx,
            control_shared=ctrl_shared,
            control_context=ctrl_ctx,
        )
        if row["attribution"] == "context_linked_bridge_candidate":
            rows.append(row)
    return rows


def first_cycle(records: list[dict], predicate) -> int | None:
    for row in records:
        if predicate(row):
            return int(row["cycle"])
    return None


def run_trial(domain: DomainSpec, seed: int) -> dict:
    experimental = clean_primary_seed(seed)
    control = clean_primary_seed(seed)
    exp_baseline = make_baseline(experimental, domain)
    ctrl_baseline = make_baseline(control, domain)
    exp_episodes = episodes_for(domain, True)
    ctrl_episodes = episodes_for(domain, False)
    records = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            exp = measure(experimental, domain, exp_baseline)
            ctrl = measure(control, domain, ctrl_baseline)
            candidates = context_candidates(experimental, control, domain, exp_baseline["direct"])
            records.append({
                "cycle": cycle,
                "experimental": exp,
                "control": ctrl,
                "effects": {
                    "direct_edge_similarity": exp["direct"]["edge_similarity"] - ctrl["direct"]["edge_similarity"],
                    "direct_activation_similarity": exp["direct"]["activation_similarity"] - ctrl["direct"]["activation_similarity"],
                    "direct_new_shared_edges": exp["direct"]["new_shared_edges"] - ctrl["direct"]["new_shared_edges"],
                    "transfer_edge_similarity": exp["transfer"]["mean_edge_similarity"] - ctrl["transfer"]["mean_edge_similarity"],
                    "transfer_activation_similarity": exp["transfer"]["mean_activation_similarity"] - ctrl["transfer"]["mean_activation_similarity"],
                    "transfer_new_shared_edges": exp["transfer"]["new_shared_edges"] - ctrl["transfer"]["new_shared_edges"],
                },
                "context_linked_candidate_count": len(candidates),
            })
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)

    direct_cycle = first_cycle(records, lambda r: (
        r["experimental"]["direct"]["new_shared_edges"] > 0
        and r["effects"]["direct_new_shared_edges"] > 0
        and r["context_linked_candidate_count"] > 0
        and (
            r["effects"]["direct_edge_similarity"] > 0
            or r["effects"]["direct_activation_similarity"] > 0
        )
    ))
    transfer_cycle = first_cycle(records, lambda r: (
        r["experimental"]["transfer"]["new_shared_edges"] > 0
        and r["effects"]["transfer_new_shared_edges"] > 0
        and (
            r["effects"]["transfer_edge_similarity"] > 0
            or r["effects"]["transfer_activation_similarity"] > 0
        )
    ))
    final = records[-1]
    return {
        "domain": domain.name,
        "seed": seed,
        "direct_observed": direct_cycle is not None,
        "direct_first_cycle": direct_cycle,
        "transfer_observed": transfer_cycle is not None,
        "transfer_first_cycle": transfer_cycle,
        "final_context_linked_candidates": final["context_linked_candidate_count"],
        "final_direct_edge_effect": final["effects"]["direct_edge_similarity"],
        "final_direct_new_shared_edge_effect": final["effects"]["direct_new_shared_edges"],
        "final_transfer_edge_effect": final["effects"]["transfer_edge_similarity"],
        "final_transfer_new_shared_edge_effect": final["effects"]["transfer_new_shared_edges"],
        "records": records,
        "experimental": experimental,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = []
    for domain in DOMAINS:
        for seed in SEEDS:
            trials.append(run_trial(domain, seed))

    total = len(trials)
    direct_successes = sum(1 for x in trials if x["direct_observed"])
    transfer_successes = sum(1 for x in trials if x["transfer_observed"])
    direct_rate = direct_successes / total if total else 0.0
    transfer_rate = transfer_successes / total if total else 0.0

    domain_rows = []
    validated_domains = 0
    transfer_domains = 0
    for domain in DOMAINS:
        rows = [x for x in trials if x["domain"] == domain.name]
        d_direct = sum(1 for x in rows if x["direct_observed"])
        d_transfer = sum(1 for x in rows if x["transfer_observed"])
        d_direct_rate = d_direct / len(rows)
        d_transfer_rate = d_transfer / len(rows)
        if d_direct_rate >= 0.60:
            validated_domains += 1
        if d_transfer_rate >= 0.40:
            transfer_domains += 1
        domain_rows.append({
            "domain": domain.name,
            "trials": len(rows),
            "direct_successes": d_direct,
            "direct_rate": d_direct_rate,
            "transfer_successes": d_transfer,
            "transfer_rate": d_transfer_rate,
            "median_direct_first_cycle": statistics.median([x["direct_first_cycle"] for x in rows if x["direct_first_cycle"] is not None]) if d_direct else None,
            "mean_final_context_candidates": sum(x["final_context_linked_candidates"] for x in rows) / len(rows),
        })

    seed_rows = []
    robust_seeds = 0
    for seed in SEEDS:
        rows = [x for x in trials if x["seed"] == seed]
        count = sum(1 for x in rows if x["direct_observed"])
        if count >= 2:
            robust_seeds += 1
        seed_rows.append({
            "seed": seed,
            "domains_with_direct_bridge": count,
            "domains_with_transfer": sum(1 for x in rows if x["transfer_observed"]),
        })

    representative = next((x for x in trials if x["direct_observed"]), trials[0])
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_episode_multidomain_roundtrip.json"
    rep_brain = representative["experimental"]
    rep_before = rep_brain.snapshot_learning_state()
    rep_brain.save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = rep_before == loaded.snapshot_learning_state()
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)

    multi_domain_pass = (
        direct_rate >= 0.60
        and validated_domains >= 2
        and robust_seeds >= 3
        and transfer_domains >= 1
        and saveload_equal
        and native_present
        and production_unchanged
    )

    if multi_domain_pass:
        verdict = "semantic_episode_binding_reproduces_across_multiple_seeds_and_multiple_domains"
        readiness = "semantic_episode_binding_multidomain_candidate"
        next_step = "test_causal_value_of_context_linked_edges_and_then_integrate_episode_interface_cleanly"
    elif direct_rate >= 0.40:
        verdict = "semantic_episode_binding_is_partially_reproducible_but_still_sensitive_to_seed_or_domain"
        readiness = "semantic_episode_binding_partial_generalization"
        next_step = "attribute_failures_by_seed_topology_and_domain_before_core_changes"
    else:
        verdict = "v83_semantic_episode_effect_does_not_generalize_reliably_yet"
        readiness = "semantic_episode_binding_not_generalized"
        next_step = "inspect_why_v83_seed_domain_is_special_before_promoting_semantic_episode_design"

    payload = {
        "experiment": "Core Growth Binding v84 — Semantic Episode Multi-Seed & Multi-Domain Validation",
        "contract": {
            "primary_core_modified": False,
            "semantic_episode_algorithm_modified": False,
            "seeds": SEEDS,
            "domains": [d.name for d in DOMAINS],
            "trial_count": total,
            "checkpoints": CHECKPOINTS,
            "control_changes_only_context": True,
            "production_brain_json_saved": False,
        },
        "domain_summary": domain_rows,
        "seed_summary": seed_rows,
        "trials": [
            {k: v for k, v in row.items() if k != "experimental"}
            for row in trials
        ],
        "summary": {
            "trial_count": total,
            "direct_successes": direct_successes,
            "direct_success_rate": direct_rate,
            "transfer_successes": transfer_successes,
            "transfer_success_rate": transfer_rate,
            "validated_domains": validated_domains,
            "domain_count": len(DOMAINS),
            "transfer_domains": transfer_domains,
            "robust_seeds": robust_seeds,
            "seed_count": len(SEEDS),
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "semantic_multidomain_pass": multi_domain_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    (OUT / "latest_binding_v84.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v84</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1100px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v84：Semantic Episode Multi-Seed & Multi-Domain Validation</h1><p class="lead">v83のSemantic Episode Bindingを、空・海・道の3ドメイン × 5 seed = 15試行で再検証する。Primary CoreとEpisode方式は変更せず、再現性だけを見る。</p><section class="panel"><div class="controls"><button id="run">15試行を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>ドメイン別</h2><pre id="domains" class="raw">未実行</pre></section><section class="panel"><h2>Seed別</h2><pre id="seeds" class="raw">未実行</pre></section><section class="panel"><h2>全Trial</h2><pre id="trials" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function pct(v){return (100*v).toFixed(1)+'%'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Trial',`${s.trial_count}`),metric('直接Bridge',`${s.direct_successes}/${s.trial_count} (${pct(s.direct_success_rate)})`,s.direct_success_rate>=.6?'good':'warn'),metric('Transfer',`${s.transfer_successes}/${s.trial_count} (${pct(s.transfer_success_rate)})`,s.transfer_success_rate>=.4?'good':'warn'),metric('Validated Domain',`${s.validated_domains}/${s.domain_count}`,s.validated_domains>=2?'good':'warn'),metric('Transfer Domain',`${s.transfer_domains}/${s.domain_count}`,s.transfer_domains>=1?'good':'warn'),metric('Robust Seed',`${s.robust_seeds}/${s.seed_count}`,s.robust_seeds>=3?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('Primary Native Learning',yn(s.primary_native_learning_present),s.primary_native_learning_present?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Multi-Domain PASS',yn(s.semantic_multidomain_pass),s.semantic_multidomain_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('domains').textContent=JSON.stringify(d.domain_summary,null,2);document.getElementById('seeds').textContent=JSON.stringify(d.seed_summary,null,2);document.getElementById('trials').textContent=JSON.stringify(d.trials,null,2)}document.getElementById('run').onclick=run;
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/run")
def api_run():
    try:
        return jsonify(observe())
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v84: http://{HOST}:{PORT}")
    print("3 domains x 5 seeds / Semantic Episode generalization / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
