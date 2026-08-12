from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import threading
import webbrowser
from itertools import combinations
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

HOST = "127.0.0.1"
START_PORT = 5116
OUT = ROOT / "data" / "core_growth_binding_v69" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
GRID = 3
TRAIN_REPEATS = 5
PROBE_STEPS = 8
THRESHOLD = 0.18

# Only numeric relative changes exist in the experimental contract.
DELTAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


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


def rc(node: int) -> tuple[int, int]:
    return divmod(node, GRID)


def node_at(r: int, c: int) -> int | None:
    if 0 <= r < GRID and 0 <= c < GRID:
        return r * GRID + c
    return None


def transitions_for_delta(delta: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = delta
    rows: list[tuple[int, int]] = []
    for source in range(GRID * GRID):
        r, c = rc(source)
        target = node_at(r + dr, c + dc)
        if target is not None:
            rows.append((source, target))
    return rows


def specific_token(source: int, target: int) -> str:
    return "s_" + hashlib.sha256(f"{source}>{target}".encode()).hexdigest()[:14]


def relative_token(delta: tuple[int, int]) -> str:
    dr, dc = delta
    return "r_" + hashlib.sha256(f"{dr},{dc}".encode()).hexdigest()[:14]


def unique_sources(*groups: list[int]) -> list[int]:
    out: list[int] = []
    for group in groups:
        for node in group:
            if node not in out:
                out.append(int(node))
    return out


def sources_for_transition(brain: SphereBrain, source: int, target: int, delta: tuple[int, int], *, include_specific: bool = True, include_relative: bool = True) -> list[int]:
    groups: list[list[int]] = []
    if include_specific:
        groups.append(brain.text_to_sources(specific_token(source, target), count=3))
    if include_relative:
        groups.append(brain.text_to_sources(relative_token(delta), count=3))
    return unique_sources(*groups)


def run_route(brain: SphereBrain, source: int, target: int, delta: tuple[int, int], *, learn: bool, include_specific: bool = True, include_relative: bool = True) -> dict:
    result = brain.propagate(
        sources_for_transition(brain, source, target, delta, include_specific=include_specific, include_relative=include_relative),
        steps=PROBE_STEPS,
        threshold=THRESHOLD,
        noise=0.0,
        learn=learn,
    )
    return {
        "nodes": list(result.activated_nodes),
        "edges": [list(x) for x in result.traversed_edges],
    }


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def route_similarity(a: dict, b: dict) -> float:
    nj = jaccard(a["nodes"], b["nodes"])
    ae = {tuple(x) for x in a["edges"]}
    be = {tuple(x) for x in b["edges"]}
    ej = jaccard(ae, be)
    return 0.35 * nj + 0.65 * ej


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def train_delta(brain: SphereBrain, delta: tuple[int, int], holdout: tuple[int, int]) -> None:
    for pair in transitions_for_delta(delta):
        if pair == holdout:
            continue
        for _ in range(TRAIN_REPEATS):
            run_route(brain, pair[0], pair[1], delta, learn=True)


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    brain = copy.deepcopy(base)

    # Leave one concrete transition out for every numeric delta.
    holdouts: dict[tuple[int, int], tuple[int, int]] = {}
    for delta in DELTAS:
        rows = transitions_for_delta(delta)
        holdouts[delta] = rows[len(rows) // 2]
        train_delta(brain, delta, holdouts[delta])

    # Probe every concrete transition after training.
    routes: dict[tuple[int, int], dict[tuple[int, int], dict]] = {}
    relative_only: dict[tuple[int, int], dict] = {}
    for delta in DELTAS:
        routes[delta] = {}
        for pair in transitions_for_delta(delta):
            routes[delta][pair] = run_route(brain, pair[0], pair[1], delta, learn=False)
        h = holdouts[delta]
        relative_only[delta] = run_route(brain, h[0], h[1], delta, learn=False, include_specific=False, include_relative=True)

    within_scores: list[float] = []
    cross_scores: list[float] = []
    holdout_to_same: list[float] = []
    holdout_to_cross: list[float] = []
    relative_component_scores: list[float] = []

    for delta in DELTAS:
        pairs = list(routes[delta])
        for a, b in combinations(pairs, 2):
            within_scores.append(route_similarity(routes[delta][a], routes[delta][b]))

        h = holdouts[delta]
        trained_same = [p for p in pairs if p != h]
        holdout_to_same.extend(route_similarity(routes[delta][h], routes[delta][p]) for p in trained_same)
        relative_component_scores.append(route_similarity(routes[delta][h], relative_only[delta]))

        for other in DELTAS:
            if other == delta:
                continue
            for p in routes[other]:
                cross_scores.append(route_similarity(routes[delta][h], routes[other][p]))
                holdout_to_cross.append(route_similarity(routes[delta][h], routes[other][p]))

    within_mean = mean(within_scores)
    cross_mean = mean(cross_scores)
    holdout_same_mean = mean(holdout_to_same)
    holdout_cross_mean = mean(holdout_to_cross)
    relative_component_mean = mean(relative_component_scores)

    separation = within_mean - cross_mean
    holdout_separation = holdout_same_mean - holdout_cross_mean

    # Native state records only opaque relative-condition evidence; no place/direction labels.
    expected = [relative_token(d) for d in DELTAS]
    for delta in DELTAS:
        brain.experience_state.observe(
            condition=relative_token(delta),
            present=True,
            motif="m69",
            expected_conditions=expected,
        )

    no_position_labels = not brain.experience_state.contains_forbidden_position_label()
    brain_file_unchanged = before_hash == file_hash(BRAIN_PATH)
    relative_structure_found = within_mean > cross_mean and separation > 0.02
    holdout_generalized = holdout_same_mean > holdout_cross_mean and holdout_separation > 0.02
    component_present = relative_component_mean > 0.20
    pass_all = relative_structure_found and holdout_generalized and component_present and no_position_labels and brain_file_unchanged

    if pass_all:
        verdict = "relative_transition_structure_generalizes_across_concrete_positions"
        readiness = "relative_transition_representation_ready"
        next_step = "use_relative_transition_credit_in_pe_puzzle_and_compare_with_specific_credit_and_assist"
    elif not relative_structure_found:
        verdict = "relative_component_does_not_dominate_over_specific_transition_routes"
        readiness = "relative_representation_not_separated"
        next_step = "audit_stimulus_binding_strength_between_specific_and_relative_components"
    elif not holdout_generalized:
        verdict = "same_delta_structure_exists_but_does_not_generalize_to_held_out_position"
        readiness = "relative_representation_not_generalized"
        next_step = "increase_shared_relative_experience_without_changing_behavioral_core"
    else:
        verdict = "relative_component_is_present_but_too_weak_for_credit_use"
        readiness = "relative_representation_weak"
        next_step = "audit_relative_component_route_share_before_puzzle_credit_assignment"

    rows = []
    for delta in DELTAS:
        rows.append({
            "delta": list(delta),
            "holdout": list(holdouts[delta]),
            "transition_count": len(transitions_for_delta(delta)),
            "holdout_relative_component_similarity": route_similarity(routes[delta][holdouts[delta]], relative_only[delta]),
        })

    payload = {
        "experiment": "Core Growth Binding v69 — Relative Transition Representation",
        "contract": {
            "direction_words_given_to_core": False,
            "position_labels_saved": False,
            "relative_input": "numeric_delta_only",
            "specific_and_relative_stimuli_mixed": True,
            "leave_one_concrete_transition_out_per_delta": True,
            "assist_used": False,
            "credit_assignment_used": False,
            "production_brain_json_saved": False,
        },
        "summary": {
            "within_delta_similarity": within_mean,
            "cross_delta_similarity": cross_mean,
            "separation_margin": separation,
            "holdout_same_delta_similarity": holdout_same_mean,
            "holdout_cross_delta_similarity": holdout_cross_mean,
            "holdout_separation_margin": holdout_separation,
            "relative_component_similarity": relative_component_mean,
            "relative_structure_found": relative_structure_found,
            "holdout_generalized": holdout_generalized,
            "relative_component_present": component_present,
            "position_labels_absent": no_position_labels,
            "brain_file_unchanged": brain_file_unchanged,
            "representation_pass": pass_all,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
        "delta_rows": rows,
        "experience_state": brain.snapshot_experience_state(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v69.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v69</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:19px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}
</style></head><body><main><h1>v69：Relative Transition Representation</h1><p class="lead">具体遷移刺激と数値Δ刺激を混合し、同じ相対変化が位置を跨いでCore内の共有構造になるかを検証する。方向語・Assist・Credit Assignmentは使わない。</p><section class="panel"><div class="controls"><button id="run">Relative Transitionを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Δ別 Holdout</h2><pre id="rows" class="raw">未実行</pre></section><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">未実行</pre></section><script>
const b=v=>v?'YES':'NO';const cls=v=>v?'good':'warn';function metric(k,v,c=''){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…','blue');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('同Δ similarity',s.within_delta_similarity.toFixed(6),'blue'),metric('異Δ similarity',s.cross_delta_similarity.toFixed(6),'blue'),metric('分離margin',s.separation_margin.toFixed(6),cls(s.relative_structure_found)),metric('Holdout 同Δ',s.holdout_same_delta_similarity.toFixed(6),'blue'),metric('Holdout 異Δ',s.holdout_cross_delta_similarity.toFixed(6),'blue'),metric('Holdout分離',s.holdout_separation_margin.toFixed(6),cls(s.holdout_generalized)),metric('Relative成分',s.relative_component_similarity.toFixed(6),cls(s.relative_component_present)),metric('位置ラベルなし',b(s.position_labels_absent),cls(s.position_labels_absent)),metric('brain.json',s.brain_file_unchanged?'不変':'変化',cls(s.brain_file_unchanged)),metric('Representation PASS',b(s.representation_pass),cls(s.representation_pass)),metric('Core readiness',s.core_readiness,'blue'),metric('総合判定',s.overall_verdict,'blue')].join('');document.getElementById('rows').textContent=JSON.stringify(d.delta_rows,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v69: http://{HOST}:{PORT}")
    print("Relative Transition Representation / no Assist / no Credit Assignment / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
