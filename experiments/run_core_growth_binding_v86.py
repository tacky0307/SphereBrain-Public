from __future__ import annotations

import hashlib
import json
import socket
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

from brain import SphereBrain
from semantic_bridge_homeostasis import SemanticBridgeHomeostasis
import run_core_growth_binding_v82c as v82c
import run_core_growth_binding_v83 as v83
import run_core_growth_binding_v84 as v84
import run_core_growth_binding_v85 as v85

HOST = "127.0.0.1"
START_PORT = 5136
OUT = ROOT / "data" / "core_growth_binding_v86" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = v85.CHECKPOINTS
SEEDS = v84.SEEDS
DOMAINS = v84.DOMAINS
STABILITY_WINDOW = v85.STABILITY_WINDOW
MODES = ["primary", "homeostatic", "homeostatic_assist"]


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


def action_shared_edges(brain: SphereBrain, domain) -> set[tuple[int, int]]:
    left = v84.action_item(domain.left_subject, domain.left_action)
    right = v84.action_item(domain.right_subject, domain.right_action)
    return (
        v83.action_signature(brain, left)["edges"]
        & v83.action_signature(brain, right)["edges"]
    )


def intrinsic_context_candidate_edges(
    brain: SphereBrain,
    domain,
    baseline: set[tuple[int, int]],
    *,
    shared_context: bool,
) -> list[tuple[int, int]]:
    new_shared = action_shared_edges(brain, domain) - baseline
    if shared_context:
        ctx = v84.domain_context_signature(brain, domain)
    else:
        ctx = v84.control_context_signature(brain, domain)
    context_nodes = set(int(x) for x in ctx["nodes"])
    out = []
    for edge in sorted(new_shared):
        endpoints = {int(edge[0]), int(edge[1])}
        hops = v82c.min_hops(brain, endpoints, context_nodes, cap=4)
        if endpoints & context_nodes or (hops is not None and hops <= 1):
            out.append(tuple(int(x) for x in edge))
    return out


def clone_pair(domain, seed: int, mode: str):
    experimental = v84.clean_primary_seed(seed)
    control = v84.clean_primary_seed(seed)
    assist = mode == "homeostatic_assist"
    experimental.set_structural_assist(assist)
    control.set_structural_assist(assist)
    return experimental, control


def classify(records: list[dict]) -> str:
    active = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    candidates = [int(r["cycle"]) for r in records if r["context_linked_candidate_count"] > 0]
    if not active:
        return "candidate_only" if candidates else "never"
    active_set = set(active)
    if all(c in active_set for c in STABILITY_WINDOW):
        return "stable"
    if active[0] <= 1 and active[-1] < 20:
        return "transient"
    if active[0] >= 3:
        return "late_unstable"
    return "intermittent"


def run_trial(domain, seed: int, mode: str) -> dict:
    experimental, control = clone_pair(domain, seed, mode)
    exp_baseline = v84.make_baseline(experimental, domain)
    ctrl_baseline = v84.make_baseline(control, domain)
    exp_episodes = v84.episodes_for(domain, True)
    ctrl_episodes = v84.episodes_for(domain, False)
    exp_homeostasis = SemanticBridgeHomeostasis()
    ctrl_homeostasis = SemanticBridgeHomeostasis()
    use_homeostasis = mode != "primary"
    records = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            row = v85.make_cycle_row(
                experimental, control, domain, exp_baseline, ctrl_baseline, cycle
            )
            row["exp_semantic_stability"] = exp_homeostasis.snapshot()
            row["ctrl_semantic_stability"] = ctrl_homeostasis.snapshot()
            records.append(row)
        if cycle < max(CHECKPOINTS):
            v83.train_episode_set(experimental, exp_episodes)
            v83.train_episode_set(control, ctrl_episodes)
            if use_homeostasis:
                exp_candidates = intrinsic_context_candidate_edges(
                    experimental,
                    domain,
                    exp_baseline["direct"],
                    shared_context=True,
                )
                ctrl_candidates = intrinsic_context_candidate_edges(
                    control,
                    domain,
                    ctrl_baseline["direct"],
                    shared_context=False,
                )
                exp_homeostasis.observe(experimental, exp_candidates)
                ctrl_homeostasis.observe(control, ctrl_candidates)

    cls = classify(records)
    active = [int(r["cycle"]) for r in records if r["direct_success_now"]]
    final = records[-1]
    direct_similarity = float(final["experimental"]["direct"]["edge_similarity"])
    return {
        "domain": domain.name,
        "seed": int(seed),
        "mode": mode,
        "classification": cls,
        "ever_bridge": bool(active),
        "first_bridge_cycle": active[0] if active else None,
        "stable": cls == "stable",
        "never": cls == "never",
        "final_direct_similarity": direct_similarity,
        "final_edge_effect": float(final["effects"]["direct_edge_similarity"]),
        "final_activation_effect": float(final["effects"]["direct_activation_similarity"]),
        "final_candidate_count": int(final["context_linked_candidate_count"]),
        "semantic_stability": exp_homeostasis.snapshot(),
        "assist_trace_count": len(getattr(experimental, "last_structural_assist_trace", [])),
        "records": records,
        "experimental": experimental,
    }


def summarize_mode(rows: list[dict], mode: str) -> dict:
    subset = [x for x in rows if x["mode"] == mode]
    classes: dict[str, int] = {}
    for row in subset:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
    return {
        "mode": mode,
        "trial_count": len(subset),
        "ever_bridge": sum(1 for x in subset if x["ever_bridge"]),
        "stable": sum(1 for x in subset if x["stable"]),
        "never": sum(1 for x in subset if x["never"]),
        "class_counts": classes,
        "mean_final_direct_similarity": sum(x["final_direct_similarity"] for x in subset) / len(subset),
        "mean_final_candidate_count": sum(x["final_candidate_count"] for x in subset) / len(subset),
    }


def paired_changes(rows: list[dict], mode: str) -> dict:
    baseline = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == "primary"}
    target = {(x["domain"], x["seed"]): x for x in rows if x["mode"] == mode}
    stable_gain = stable_loss = never_gain = never_loss = 0
    similarity_deltas = []
    for key, base in baseline.items():
        cur = target[key]
        stable_gain += int(cur["stable"] and not base["stable"])
        stable_loss += int(base["stable"] and not cur["stable"])
        never_gain += int(cur["never"] and not base["never"])
        never_loss += int(base["never"] and not cur["never"])
        similarity_deltas.append(cur["final_direct_similarity"] - base["final_direct_similarity"])
    return {
        "mode": mode,
        "stable_gain": stable_gain,
        "stable_loss": stable_loss,
        "never_gain": never_gain,
        "never_loss": never_loss,
        "mean_direct_similarity_delta_vs_primary": sum(similarity_deltas) / len(similarity_deltas),
        "max_direct_similarity_delta_vs_primary": max(similarity_deltas),
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    trials = [
        run_trial(domain, seed, mode)
        for mode in MODES
        for domain in DOMAINS
        for seed in SEEDS
    ]

    summaries = {mode: summarize_mode(trials, mode) for mode in MODES}
    homeo_pair = paired_changes(trials, "homeostatic")
    assist_pair = paired_changes(trials, "homeostatic_assist")
    primary = summaries["primary"]
    homeo = summaries["homeostatic"]
    assist = summaries["homeostatic_assist"]

    # Guard against solving the task by indiscriminately collapsing action routes.
    homeo_distinctness_safe = homeo_pair["max_direct_similarity_delta_vs_primary"] <= 0.12
    assist_distinctness_safe = assist_pair["max_direct_similarity_delta_vs_primary"] <= 0.12
    homeo_improves = homeo["stable"] > primary["stable"] and homeo["never"] <= primary["never"]
    assist_improves = assist["stable"] > primary["stable"] and assist["never"] <= primary["never"]

    representative = next(
        (x for x in trials if x["mode"] == "homeostatic" and x["stable"]),
        next(x for x in trials if x["mode"] == "homeostatic"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_bridge_homeostasis_roundtrip.json"
    before_weights = representative["experimental"].weights.tolist()
    representative["experimental"].save(temp)
    loaded = SphereBrain.load(temp)
    saveload_equal = before_weights == loaded.weights.tolist()
    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    pass_homeo = homeo_improves and homeo_distinctness_safe
    pass_assist = assist_improves and assist_distinctness_safe
    overall_pass = (
        (pass_homeo or pass_assist)
        and saveload_equal
        and production_unchanged
        and native_present
    )

    if pass_homeo and pass_assist:
        winner = "homeostatic_assist" if assist["stable"] > homeo["stable"] else "homeostatic"
        verdict = "semantic_homeostatic_consolidation_improves_bridge_stability_and_assist_remains_viable"
    elif pass_homeo:
        winner = "homeostatic"
        verdict = "semantic_homeostatic_consolidation_improves_bridge_stability_without_route_collapse"
    elif pass_assist:
        winner = "homeostatic_assist"
        verdict = "semantic_bridge_stability_improves_only_when_homeostasis_and_assist_are_combined"
    else:
        winner = "primary"
        verdict = "semantic_consolidation_candidate_does_not_yet_improve_stability_safely"

    payload = {
        "experiment": "Core Growth Binding v86 — Semantic Bridge Homeostatic Consolidation",
        "contract": {
            "primary_core_modified": False,
            "modes": MODES,
            "domains": [x.name for x in DOMAINS],
            "seeds": SEEDS,
            "trials_per_mode": len(DOMAINS) * len(SEEDS),
            "checkpoints": CHECKPOINTS,
            "semantic_answer_labels_used": False,
            "homeostasis_targets": "repeated_core_internal_context_linked_candidate_edges_only",
            "control_receives_same_homeostasis_rule": True,
            "assist_enabled_for_both_experimental_and_control_in_assist_mode": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "primary_stable": primary["stable"],
            "homeostatic_stable": homeo["stable"],
            "homeostatic_assist_stable": assist["stable"],
            "primary_never": primary["never"],
            "homeostatic_never": homeo["never"],
            "homeostatic_assist_never": assist["never"],
            "homeostatic_distinctness_safe": homeo_distinctness_safe,
            "assist_distinctness_safe": assist_distinctness_safe,
            "homeostatic_improves": homeo_improves,
            "assist_improves": assist_improves,
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "semantic_homeostasis_pass": overall_pass,
            "winner": winner,
            "overall_verdict": verdict,
            "next_step": (
                "validate_winning_semantic_consolidation_under_contradictory_episode_switching_before_primary_core_integration"
                if overall_pass
                else "inspect_why_candidate_edge_protection_failed_before_changing_primary_core"
            ),
        },
        "mode_summaries": summaries,
        "paired_changes": {
            "homeostatic": homeo_pair,
            "homeostatic_assist": assist_pair,
        },
        "trials": [{k: v for k, v in x.items() if k != "experimental"} for x in trials],
    }
    (OUT / "latest_binding_v86.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v86</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v86：Semantic Bridge Homeostatic Consolidation</h1><p class="lead">Current Primary / Semantic Homeostatic / Homeostatic + Assist を同じ3 domain × 5 seedで比較。意味の正解は教えず、Episode反復でCore内部に再出現するContext-linked候補Edgeだけを漸近的に保護する。</p><section class="panel"><div class="controls"><button id="run">Semantic Homeostasisを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Mode比較</h2><pre id="modes" class="raw">未実行</pre></section><section class="panel"><h2>Paired差</h2><pre id="pairs" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Primary Stable',s.primary_stable),metric('Homeostatic Stable',s.homeostatic_stable,s.homeostatic_improves?'good':'warn'),metric('Homeo+Assist Stable',s.homeostatic_assist_stable,s.assist_improves?'good':'warn'),metric('Primary Never',s.primary_never),metric('Homeostatic Never',s.homeostatic_never),metric('Homeo+Assist Never',s.homeostatic_assist_never),metric('Homeo Distinctness',yn(s.homeostatic_distinctness_safe),s.homeostatic_distinctness_safe?'good':'warn'),metric('Assist Distinctness',yn(s.assist_distinctness_safe),s.assist_distinctness_safe?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Semantic Homeostasis PASS',yn(s.semantic_homeostasis_pass),s.semantic_homeostasis_pass?'good':'warn'),metric('Winner',s.winner),metric('総合判定',s.overall_verdict)].join('');document.getElementById('modes').textContent=JSON.stringify(d.mode_summaries,null,2);document.getElementById('pairs').textContent=JSON.stringify(d.paired_changes,null,2)}document.getElementById('run').onclick=run;
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


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    print(f"Core Growth Binding v86: http://{HOST}:{PORT}")
    print("Primary vs Semantic Homeostasis vs Homeostasis+Assist / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
