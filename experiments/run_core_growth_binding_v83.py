from __future__ import annotations

import hashlib
import json
import socket
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

HOST = "127.0.0.1"
START_PORT = 5132
OUT = ROOT / "data" / "core_growth_binding_v83" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = [0, 1, 3, 5, 10]


@dataclass(frozen=True)
class EpisodeSpec:
    subject: str
    context: str
    action: str

    @property
    def label(self) -> str:
        return f"{self.subject} → {self.context} → {self.action}"


EXP_EPISODES = [
    EpisodeSpec("鳥", "空", "羽ばたく"),
    EpisodeSpec("飛行機", "空", "飛行する"),
]
CTRL_EPISODES = [
    EpisodeSpec("鳥", "森", "羽ばたく"),
    EpisodeSpec("飛行機", "空港", "飛行する"),
]

BIRD_ACTION = v82.StructuredInput("鳥", "動作", "羽ばたく")
PLANE_ACTION = v82.StructuredInput("飛行機", "動作", "飛行する")
BUTTERFLY_ACTION = v82.StructuredInput("蝶", "動作", "羽ばたく")
DRONE_ACTION = v82.StructuredInput("ドローン", "動作", "飛行する")
BIRD_SKY = v82.StructuredInput("鳥", "場所", "空")
PLANE_SKY = v82.StructuredInput("飛行機", "場所", "空")


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


def episode_experience(brain: SphereBrain, spec: EpisodeSpec, *, learn: bool) -> dict:
    """One continuous semantic episode using the same semantic-v2 token namespaces.

    subject -> place relation -> context -> action relation -> action
    The previous stage tail is passed into the next stage as activity context.
    """
    noise = 0.004 if learn else 0.0

    subject_sources = (
        v82.component_nodes(brain, "role:subject", "subject", 2)
        + v82.component_nodes(brain, "entity", spec.subject, 3)
    )
    subject = brain.propagate(
        subject_sources, steps=8, threshold=0.18, noise=noise, learn=learn
    )

    place_relation_sources = (
        v82.component_nodes(brain, "role:relation", "relation", 2)
        + v82.component_nodes(brain, "relation", "場所", 3)
    )
    place_relation = brain.propagate(
        place_relation_sources,
        steps=8,
        threshold=0.18,
        noise=noise,
        learn=learn,
        context_nodes=v82.context_tail(subject),
    )

    context_sources = (
        v82.component_nodes(brain, "role:content", "content", 2)
        + v82.component_nodes(brain, "content", spec.context, 3)
    )
    context = brain.propagate(
        context_sources,
        steps=10,
        threshold=0.18,
        noise=noise,
        learn=learn,
        context_nodes=v82.context_tail(place_relation),
    )

    action_relation_sources = (
        v82.component_nodes(brain, "role:relation", "relation", 2)
        + v82.component_nodes(brain, "relation", "動作", 3)
    )
    action_relation = brain.propagate(
        action_relation_sources,
        steps=8,
        threshold=0.18,
        noise=noise,
        learn=learn,
        context_nodes=v82.context_tail(context),
    )

    action_sources = (
        v82.component_nodes(brain, "role:content", "content", 2)
        + v82.component_nodes(brain, "content", spec.action, 3)
    )
    action = brain.propagate(
        action_sources,
        steps=10,
        threshold=0.18,
        noise=noise,
        learn=learn,
        context_nodes=v82.context_tail(action_relation),
    )

    return {
        "subject": subject,
        "place_relation": place_relation,
        "context": context,
        "action_relation": action_relation,
        "action": action,
    }


def train_episode_set(brain: SphereBrain, episodes: list[EpisodeSpec]) -> None:
    for spec in episodes:
        episode_experience(brain, spec, learn=True)


def action_signature(brain: SphereBrain, item: v82.StructuredInput) -> dict:
    return v82b.signature(brain, item)


def shared_action_edges(brain: SphereBrain) -> set[tuple[int, int]]:
    return action_signature(brain, BIRD_ACTION)["edges"] & action_signature(brain, PLANE_ACTION)["edges"]


def sky_signature(brain: SphereBrain) -> dict:
    left = action_signature(brain, BIRD_SKY)
    right = action_signature(brain, PLANE_SKY)
    return {
        "nodes": left["nodes"] | right["nodes"],
        "edges": left["edges"] | right["edges"],
        "shared_nodes": left["nodes"] & right["nodes"],
        "shared_edges": left["edges"] & right["edges"],
    }


def pair_metrics(brain: SphereBrain, left_item, right_item, baseline_shared: set[tuple[int, int]]) -> dict:
    left = action_signature(brain, left_item)
    right = action_signature(brain, right_item)
    shared = left["edges"] & right["edges"]
    return {
        "edge_similarity": v82.jaccard(left["edges"], right["edges"]),
        "activation_similarity": v82b.weighted_similarity(left["activation"], right["activation"]),
        "shared_edges": len(shared),
        "new_shared_edges": len(shared - baseline_shared),
        "new_shared_edge_list": [list(x) for x in sorted(shared - baseline_shared)],
    }


def measure(brain: SphereBrain, baseline: dict[str, set[tuple[int, int]]]) -> dict:
    direct = pair_metrics(brain, BIRD_ACTION, PLANE_ACTION, baseline["direct"])
    butterfly = pair_metrics(brain, BUTTERFLY_ACTION, PLANE_ACTION, baseline["butterfly"])
    drone = pair_metrics(brain, DRONE_ACTION, BIRD_ACTION, baseline["drone"])
    return {
        "direct": direct,
        "transfer": {
            "butterfly_to_plane": butterfly,
            "drone_to_bird": drone,
            "mean_edge_similarity": (butterfly["edge_similarity"] + drone["edge_similarity"]) / 2.0,
            "mean_activation_similarity": (butterfly["activation_similarity"] + drone["activation_similarity"]) / 2.0,
            "new_shared_edges": butterfly["new_shared_edges"] + drone["new_shared_edges"],
        },
    }


def make_baseline(brain: SphereBrain) -> dict[str, set[tuple[int, int]]]:
    return {
        "direct": shared_action_edges(brain),
        "butterfly": action_signature(brain, BUTTERFLY_ACTION)["edges"] & action_signature(brain, PLANE_ACTION)["edges"],
        "drone": action_signature(brain, DRONE_ACTION)["edges"] & action_signature(brain, BIRD_ACTION)["edges"],
    }


def context_candidates(
    experimental: SphereBrain,
    control: SphereBrain,
    exp_baseline: set[tuple[int, int]],
) -> list[dict]:
    exp_new = shared_action_edges(experimental) - exp_baseline
    ctrl_shared = shared_action_edges(control)
    sky = sky_signature(experimental)
    ctrl_ctx = v82c.control_context_signature(control)
    out = []
    for edge in sorted(exp_new):
        row = v82c.annotate_edge(
            experimental,
            edge,
            first_cycle=None,
            sky=sky,
            control_shared=ctrl_shared,
            control_context=ctrl_ctx,
        )
        if row["attribution"] == "context_linked_bridge_candidate":
            out.append(row)
    return out


def first_cycle(records: list[dict], predicate) -> int | None:
    for row in records:
        if predicate(row):
            return int(row["cycle"])
    return None


def run_experiment() -> dict:
    experimental = v82.clean_primary()
    control = v82.clean_primary()
    exp_baseline = make_baseline(experimental)
    ctrl_baseline = make_baseline(control)
    records = []

    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            exp = measure(experimental, exp_baseline)
            ctrl = measure(control, ctrl_baseline)
            candidates = context_candidates(experimental, control, exp_baseline["direct"])
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
                "context_linked_candidates": candidates,
            })
        if cycle < max(CHECKPOINTS):
            train_episode_set(experimental, EXP_EPISODES)
            train_episode_set(control, CTRL_EPISODES)

    return {
        "experimental": experimental,
        "control": control,
        "records": records,
        "exp_baseline": exp_baseline,
    }


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    result = run_experiment()
    records = result["records"]

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
    final_candidates = final["context_linked_candidates"]
    direct_observed = direct_cycle is not None
    transfer_observed = transfer_cycle is not None

    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_episode_roundtrip.json"
    learned = result["experimental"]
    before_shared = sorted(shared_action_edges(learned))
    learned.save(temp)
    loaded = SphereBrain.load(temp)
    after_shared = sorted(shared_action_edges(loaded))
    saveload_equal = before_shared == after_shared
    native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)

    episode_pass = direct_observed and saveload_equal and native_present and production_unchanged

    if direct_observed and transfer_observed:
        verdict = "temporal_semantic_episode_binding_created_context_linked_internal_bridge_and_transfer"
        readiness = "semantic_episode_binding_observed_on_primary_core"
        next_step = "repeat_semantic_episode_binding_across_multiple_seeds_and_concept_domains"
    elif direct_observed:
        verdict = "temporal_semantic_episode_binding_created_a_direct_context_linked_bridge_but_transfer_is_not_yet_clear"
        readiness = "semantic_episode_direct_bridge_candidate"
        next_step = "test_direct_bridge_causality_by_selective_edge_ablation_before_core_changes"
    else:
        verdict = "continuous_semantic_episode_did_not_yet_create_a_distinct_context_linked_bridge"
        readiness = "semantic_episode_binding_not_yet_observed"
        next_step = "compare_continuous_episode_stage_carryover_against_separate_fact_encoding_before_modifying_primary_core"

    payload = {
        "experiment": "Core Growth Binding v83 — Semantic Episode Binding",
        "contract": {
            "primary_core_modified": False,
            "native_success_failure_learning_invoked": False,
            "reason_native_learning_not_invoked": "semantic episode has no externally supplied success/failure label",
            "experimental_episodes": [x.label for x in EXP_EPISODES],
            "control_episodes": [x.label for x in CTRL_EPISODES],
            "continuous_stage_order": ["subject", "place_relation", "context", "action_relation", "action"],
            "same_semantic_v2_token_namespaces": True,
            "checkpoints": CHECKPOINTS,
            "direct_input_nodes_excluded_in_measurement": True,
            "production_brain_json_saved": False,
        },
        "records": records,
        "summary": {
            "direct_episode_bridge_observed": direct_observed,
            "direct_episode_bridge_first_cycle": direct_cycle,
            "transfer_observed": transfer_observed,
            "transfer_first_cycle": transfer_cycle,
            "final_context_linked_candidate_count": len(final_candidates),
            "final_direct_edge_effect": final["effects"]["direct_edge_similarity"],
            "final_direct_activation_effect": final["effects"]["direct_activation_similarity"],
            "final_direct_new_shared_edge_effect": final["effects"]["direct_new_shared_edges"],
            "final_transfer_edge_effect": final["effects"]["transfer_edge_similarity"],
            "final_transfer_new_shared_edge_effect": final["effects"]["transfer_new_shared_edges"],
            "saveload_equal": saveload_equal,
            "primary_native_learning_present": native_present,
            "brain_file_unchanged": production_unchanged,
            "semantic_episode_pass": episode_pass,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    (OUT / "latest_binding_v83.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v83</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:1000px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v83：Semantic Episode Binding</h1><p class="lead">「鳥→空→羽ばたく」「飛行機→空→飛行する」を別々の事実ではなく、連続した一つの時間経験としてPrimary Coreへ流す。測定時には空を入力せず、Action内部経路に共有文脈の痕跡が残ったかを見る。</p><section class="panel"><div class="controls"><button id="run">Semantic Episodeを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Cycle別詳細</h2><pre id="rows" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}function cyc(v){return v===null||v===undefined?'—':v}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('直接Episode Bridge',yn(s.direct_episode_bridge_observed),s.direct_episode_bridge_observed?'good':'warn'),metric('Bridge最短cycle',cyc(s.direct_episode_bridge_first_cycle)),metric('Transfer',yn(s.transfer_observed),s.transfer_observed?'good':'warn'),metric('Transfer最短cycle',cyc(s.transfer_first_cycle)),metric('Context-linked候補',s.final_context_linked_candidate_count,s.final_context_linked_candidate_count>0?'good':'warn'),metric('最終 Edge effect',Number(s.final_direct_edge_effect).toFixed(6)),metric('新規共有Edge effect',s.final_direct_new_shared_edge_effect,s.final_direct_new_shared_edge_effect>0?'good':'blue'),metric('Transfer Edge effect',Number(s.final_transfer_edge_effect).toFixed(6)),metric('Save/Load',yn(s.saveload_equal),s.saveload_equal?'good':'warn'),metric('Primary Native Learning',yn(s.primary_native_learning_present),s.primary_native_learning_present?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Semantic Episode PASS',yn(s.semantic_episode_pass),s.semantic_episode_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('rows').textContent=JSON.stringify(d.records,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v83: http://{HOST}:{PORT}")
    print("Continuous semantic episode / Primary Core unchanged / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
