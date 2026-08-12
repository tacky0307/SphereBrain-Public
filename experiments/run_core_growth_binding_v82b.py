from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np
from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
import run_core_growth_binding_v82 as v82

HOST = "127.0.0.1"
START_PORT = 5130
OUT = ROOT / "data" / "core_growth_binding_v82b" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = [0, 1, 3, 5, 10, 20]

BIRD = v82.StructuredInput("鳥", "動作", "羽ばたく")
PLANE = v82.StructuredInput("飛行機", "動作", "飛行する")
BUTTERFLY = v82.StructuredInput("蝶", "動作", "羽ばたく")
DRONE = v82.StructuredInput("ドローン", "動作", "飛行する")

EXP_CURRICULUM = [
    BIRD,
    PLANE,
    v82.StructuredInput("鳥", "場所", "空"),
    v82.StructuredInput("飛行機", "場所", "空"),
]
CTRL_CURRICULUM = [
    BIRD,
    PLANE,
    v82.StructuredInput("鳥", "場所", "森"),
    v82.StructuredInput("飛行機", "場所", "空港"),
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


def signature(brain: SphereBrain, item: v82.StructuredInput) -> dict:
    exp = v82.experience(brain, item, learn=False)
    result = exp["content"]
    direct = v82.direct_nodes(brain, item)
    nodes = {int(v) for v in result.activated_nodes if int(v) not in direct}
    edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in result.traversed_edges
        if int(a) not in direct and int(b) not in direct
    }
    activation = np.asarray(result.final_activation, dtype=float).copy()
    for node in direct:
        activation[int(node)] = 0.0
    return {"nodes": nodes, "edges": edges, "activation": activation}


def weighted_similarity(left: np.ndarray, right: np.ndarray) -> float:
    indexes = set(np.flatnonzero(left > 0).tolist()) | set(np.flatnonzero(right > 0).tolist())
    if not indexes:
        return 1.0
    num = sum(min(float(left[i]), float(right[i])) for i in indexes)
    den = sum(max(float(left[i]), float(right[i])) for i in indexes)
    return num / den if den else 0.0


def pair_metrics(left: dict, right: dict, baseline_shared_edges: set[tuple[int, int]]) -> dict:
    shared_nodes = left["nodes"] & right["nodes"]
    shared_edges = left["edges"] & right["edges"]
    new_shared = shared_edges - baseline_shared_edges
    return {
        "node_similarity": v82.jaccard(left["nodes"], right["nodes"]),
        "edge_similarity": v82.jaccard(left["edges"], right["edges"]),
        "activation_similarity": weighted_similarity(left["activation"], right["activation"]),
        "shared_nodes": len(shared_nodes),
        "shared_edges": len(shared_edges),
        "new_shared_edges": len(new_shared),
        "shared_edge_list": [list(x) for x in sorted(shared_edges)],
        "new_shared_edge_list": [list(x) for x in sorted(new_shared)],
    }


def raw_pair(brain: SphereBrain, left_item, right_item) -> tuple[dict, dict, set[tuple[int, int]]]:
    left = signature(brain, left_item)
    right = signature(brain, right_item)
    return left, right, left["edges"] & right["edges"]


def measure(brain: SphereBrain, baseline: dict[str, set[tuple[int, int]]]) -> dict:
    bird = signature(brain, BIRD)
    plane = signature(brain, PLANE)
    butterfly = signature(brain, BUTTERFLY)
    drone = signature(brain, DRONE)

    target = pair_metrics(bird, plane, baseline["target"])
    butterfly_transfer = pair_metrics(butterfly, plane, baseline["butterfly_transfer"])
    drone_transfer = pair_metrics(drone, bird, baseline["drone_transfer"])

    return {
        "target": target,
        "transfer": {
            "butterfly_to_plane": butterfly_transfer,
            "drone_to_bird": drone_transfer,
            "mean_edge_similarity": (butterfly_transfer["edge_similarity"] + drone_transfer["edge_similarity"]) / 2.0,
            "mean_activation_similarity": (butterfly_transfer["activation_similarity"] + drone_transfer["activation_similarity"]) / 2.0,
            "new_shared_edges": butterfly_transfer["new_shared_edges"] + drone_transfer["new_shared_edges"],
        },
    }


def baselines(brain: SphereBrain) -> dict[str, set[tuple[int, int]]]:
    _, _, target = raw_pair(brain, BIRD, PLANE)
    _, _, butterfly = raw_pair(brain, BUTTERFLY, PLANE)
    _, _, drone = raw_pair(brain, DRONE, BIRD)
    return {"target": target, "butterfly_transfer": butterfly, "drone_transfer": drone}


def run_microscope() -> dict:
    experimental = v82.clean_primary()
    control = v82.clean_primary()
    exp_baseline = baselines(experimental)
    ctrl_baseline = baselines(control)
    records = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            exp = measure(experimental, exp_baseline)
            ctrl = measure(control, ctrl_baseline)
            records.append({
                "cycle": cycle,
                "experimental": exp,
                "control": ctrl,
                "effects": {
                    "edge_similarity": exp["target"]["edge_similarity"] - ctrl["target"]["edge_similarity"],
                    "activation_similarity": exp["target"]["activation_similarity"] - ctrl["target"]["activation_similarity"],
                    "shared_edges": exp["target"]["shared_edges"] - ctrl["target"]["shared_edges"],
                    "new_shared_edges": exp["target"]["new_shared_edges"] - ctrl["target"]["new_shared_edges"],
                    "transfer_edge_similarity": exp["transfer"]["mean_edge_similarity"] - ctrl["transfer"]["mean_edge_similarity"],
                    "transfer_activation_similarity": exp["transfer"]["mean_activation_similarity"] - ctrl["transfer"]["mean_activation_similarity"],
                    "transfer_new_shared_edges": exp["transfer"]["new_shared_edges"] - ctrl["transfer"]["new_shared_edges"],
                },
            })
        if cycle < max(CHECKPOINTS):
            v82.learn_items(experimental, EXP_CURRICULUM)
            v82.learn_items(control, CTRL_CURRICULUM)

    return {"records": records, "experimental": experimental}


def first_cycle(records: list[dict], predicate) -> int | None:
    for row in records:
        if predicate(row):
            return int(row["cycle"])
    return None


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    result = run_microscope()
    records = result["records"]

    # Structural bridge: target gains new shared internal edges beyond baseline,
    # beats the control, and at least one similarity signal also favors shared context.
    bridge_cycle = first_cycle(records, lambda r: (
        r["experimental"]["target"]["new_shared_edges"] > 0
        and r["effects"]["new_shared_edges"] > 0
        and (r["effects"]["edge_similarity"] > 0 or r["effects"]["activation_similarity"] > 0)
    ))
    transfer_cycle = first_cycle(records, lambda r: (
        r["experimental"]["transfer"]["new_shared_edges"] > 0
        and r["effects"]["transfer_new_shared_edges"] > 0
        and (r["effects"]["transfer_edge_similarity"] > 0 or r["effects"]["transfer_activation_similarity"] > 0)
    ))

    final = records[-1]
    bridge_observed = bridge_cycle is not None
    transfer_observed = transfer_cycle is not None

    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "bridge_microscope_roundtrip.json"
    learned = result["experimental"]
    before_measure = measure(learned, baselines(v82.clean_primary()))
    learned.save(temp)
    loaded = SphereBrain.load(temp)
    after_measure = measure(loaded, baselines(v82.clean_primary()))
    saveload_equal = (
        abs(before_measure["target"]["edge_similarity"] - after_measure["target"]["edge_similarity"]) < 1e-12
        and before_measure["target"]["shared_edge_list"] == after_measure["target"]["shared_edge_list"]
    )

    production_unchanged = before_hash == file_hash(BRAIN_PATH)
    primary_native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")

    if bridge_observed and transfer_observed:
        verdict = "shared_context_produces_new_internal_bridge_edges_and_transfer_structure"
        readiness = "semantic_bridge_mechanism_observed_on_primary_core"
        next_step = "repeat_bridge_microscope_across_multiple_seeds_and_semantic_domains"
    elif bridge_observed:
        verdict = "direct_shared_context_bridge_observed_but_transfer_generalization_not_yet_clear"
        readiness = "direct_semantic_bridge_candidate"
        next_step = "test_bridge_transfer_with_more_subjects_before_core_changes"
    else:
        verdict = "shared_context_did_not_create_a_distinct_internal_bridge_under_current_encoder_and_core"
        readiness = "semantic_bridge_mechanism_not_yet_observed"
        next_step = "inspect_stage_context_carryover_and_compare_old_semantic_core_before_changing_primary_learning"

    payload = {
        "experiment": "Core Growth Binding v82B — Shared Context Bridge Microscope",
        "contract": {
            "primary_core_modified": False,
            "semantic_encoder_modified": False,
            "checkpoints": CHECKPOINTS,
            "experimental_shared_context": [x.label for x in EXP_CURRICULUM],
            "control_separate_context": [x.label for x in CTRL_CURRICULUM],
            "direct_input_nodes_excluded": True,
            "baseline_shared_edges_subtracted": True,
            "production_brain_json_saved": False,
        },
        "records": records,
        "summary": {
            "direct_bridge_observed": bridge_observed,
            "direct_bridge_first_cycle": bridge_cycle,
            "transfer_bridge_observed": transfer_observed,
            "transfer_bridge_first_cycle": transfer_cycle,
            "final_context_edge_effect": final["effects"]["edge_similarity"],
            "final_context_activation_effect": final["effects"]["activation_similarity"],
            "final_new_shared_edge_effect": final["effects"]["new_shared_edges"],
            "final_transfer_edge_effect": final["effects"]["transfer_edge_similarity"],
            "final_transfer_new_shared_edge_effect": final["effects"]["transfer_new_shared_edges"],
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": primary_native_present,
            "brain_file_unchanged": production_unchanged,
            "bridge_microscope_pass": bridge_observed and saveload_equal and primary_native_present and production_unchanged,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    (OUT / "latest_binding_v82b.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v82B</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v82B：Shared Context Bridge Microscope</h1><p class="lead">v82で未確認だった共有文脈Bridgeだけを、0 / 1 / 3 / 5 / 10 / 20 cycleで観察する。入力Nodeを除外し、0cycle時点の偶然の共有Edgeも差し引いて、「経験後に新しく生まれたCore内部の橋」を追う。</p><section class="panel"><div class="controls"><button id="run">Bridgeを顕微鏡観測</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Cycle別詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function cyc(v){return v===null||v===undefined?'—':v}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('直接Bridge',yn(s.direct_bridge_observed),s.direct_bridge_observed?'good':'warn'),metric('Bridge最短cycle',cyc(s.direct_bridge_first_cycle),s.direct_bridge_observed?'good':'blue'),metric('Transfer Bridge',yn(s.transfer_bridge_observed),s.transfer_bridge_observed?'good':'warn'),metric('Transfer最短cycle',cyc(s.transfer_bridge_first_cycle),s.transfer_bridge_observed?'good':'blue'),metric('最終 Edge effect',Number(s.final_context_edge_effect).toFixed(6)),metric('最終 Activation effect',Number(s.final_context_activation_effect).toFixed(6)),metric('新規共有Edge effect',s.final_new_shared_edge_effect,s.final_new_shared_edge_effect>0?'good':'warn'),metric('Transfer新規Edge effect',s.final_transfer_new_shared_edge_effect,s.final_transfer_new_shared_edge_effect>0?'good':'warn'),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('Primary Native Learning',yn(s.primary_native_learning_present),s.primary_native_learning_present?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Microscope PASS',yn(s.bridge_microscope_pass),s.bridge_microscope_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('rows').textContent=JSON.stringify(d.records,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v82B: http://{HOST}:{PORT}")
    print("Shared Context Bridge microscope / Primary Core変更なし / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
