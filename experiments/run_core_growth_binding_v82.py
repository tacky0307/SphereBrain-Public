from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import threading
import webbrowser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

HOST = "127.0.0.1"
START_PORT = 5129
OUT = ROOT / "data" / "core_growth_binding_v82" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
CHECKPOINTS = [0, 1, 3, 5]


@dataclass(frozen=True)
class StructuredInput:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


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


def clean_primary() -> SphereBrain:
    source = SphereBrain.load(BRAIN_PATH)
    brain = SphereBrain(
        node_count=source.node_count,
        neighbors_per_node=source.neighbors_per_node,
        seed=source.seed,
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


def component_nodes(brain: SphereBrain, namespace: str, value: str, count: int = 3) -> list[int]:
    material = f"semantic-v2|{namespace}|{value.strip()}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    out: list[int] = []
    offset = 0
    while len(out) < count:
        if offset + 4 > len(digest):
            digest = hashlib.sha256(digest).digest()
            offset = 0
        node = int.from_bytes(digest[offset:offset + 4], "big") % brain.node_count
        if node not in out:
            out.append(node)
        offset += 4
    return out


def context_tail(result, limit: int = 18) -> list[int]:
    ordered: list[int] = []
    for step in reversed(result.activation_history):
        for node in step:
            if int(node) not in ordered:
                ordered.append(int(node))
            if len(ordered) >= limit:
                return ordered
    return ordered


def direct_nodes(brain: SphereBrain, item: StructuredInput) -> set[int]:
    return set(
        component_nodes(brain, "role:subject", "subject", 2)
        + component_nodes(brain, "entity", item.subject, 3)
        + component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", item.relation, 3)
        + component_nodes(brain, "role:content", "content", 2)
        + component_nodes(brain, "content", item.content, 3)
    )


def experience(brain: SphereBrain, item: StructuredInput, *, learn: bool) -> dict:
    noise = 0.004 if learn else 0.0
    ssrc = component_nodes(brain, "role:subject", "subject", 2) + component_nodes(brain, "entity", item.subject, 3)
    s = brain.propagate(ssrc, steps=8, threshold=0.18, noise=noise, learn=learn)
    rsrc = component_nodes(brain, "role:relation", "relation", 2) + component_nodes(brain, "relation", item.relation, 3)
    r = brain.propagate(rsrc, steps=8, threshold=0.18, noise=noise, learn=learn, context_nodes=context_tail(s))
    csrc = component_nodes(brain, "role:content", "content", 2) + component_nodes(brain, "content", item.content, 3)
    c = brain.propagate(csrc, steps=10, threshold=0.18, noise=noise, learn=learn, context_nodes=context_tail(r))
    return {"item": item, "subject": s, "relation": r, "content": c}


def filtered_signature(brain: SphereBrain, item: StructuredInput) -> dict:
    exp = experience(brain, item, learn=False)
    result = exp["content"]
    direct = direct_nodes(brain, item)
    nodes = {int(v) for v in result.activated_nodes if int(v) not in direct}
    edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in result.traversed_edges
        if int(a) not in direct and int(b) not in direct
    }
    return {"nodes": nodes, "edges": edges}


def jaccard(a, b) -> float:
    aa, bb = set(a), set(b)
    union = aa | bb
    return len(aa & bb) / len(union) if union else 1.0


def similarity(left: dict, right: dict) -> float:
    return 0.35 * jaccard(left["nodes"], right["nodes"]) + 0.65 * jaccard(left["edges"], right["edges"])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def measure_concepts(brain: SphereBrain) -> dict:
    animals = [
        StructuredInput("犬", "種類", "動物"),
        StructuredInput("猫", "種類", "動物"),
        StructuredInput("鳥", "種類", "動物"),
    ]
    artificials = [
        StructuredInput("車", "種類", "人工物"),
        StructuredInput("船", "種類", "人工物"),
    ]
    animal_sig = {x.label: filtered_signature(brain, x) for x in animals}
    artificial_sig = {x.label: filtered_signature(brain, x) for x in artificials}
    same = [similarity(animal_sig[a.label], animal_sig[b.label]) for a, b in combinations(animals, 2)]
    same += [similarity(artificial_sig[a.label], artificial_sig[b.label]) for a, b in combinations(artificials, 2)]
    cross = [similarity(animal_sig[a.label], artificial_sig[b.label]) for a in animals for b in artificials]
    return {
        "same_concept_similarity": mean(same),
        "different_concept_similarity": mean(cross),
        "separation": mean(same) - mean(cross),
    }


def learn_items(brain: SphereBrain, items: list[StructuredInput]) -> None:
    for item in items:
        experience(brain, item, learn=True)


def concept_experiment() -> dict:
    brain = clean_primary()
    curriculum = [
        StructuredInput("犬", "種類", "動物"),
        StructuredInput("猫", "種類", "動物"),
        StructuredInput("鳥", "種類", "動物"),
        StructuredInput("車", "種類", "人工物"),
        StructuredInput("船", "種類", "人工物"),
    ]
    records = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            records.append({"cycle": cycle, **measure_concepts(brain)})
        if cycle < max(CHECKPOINTS):
            learn_items(brain, curriculum)

    novel = StructuredInput("牛", "種類", "動物")
    before = filtered_signature(brain, novel)
    references = [filtered_signature(brain, x) for x in curriculum[:3]]
    artificial_refs = [filtered_signature(brain, x) for x in curriculum[3:]]
    before_animal = mean([similarity(before, x) for x in references])
    before_artificial = mean([similarity(before, x) for x in artificial_refs])
    experience(brain, novel, learn=True)
    after = filtered_signature(brain, novel)
    after_animal = mean([similarity(after, x) for x in references])
    after_artificial = mean([similarity(after, x) for x in artificial_refs])

    return {
        "records": records,
        "novel": {
            "label": novel.label,
            "before_animal_similarity": before_animal,
            "before_artificial_similarity": before_artificial,
            "after_one_experience_animal_similarity": after_animal,
            "after_one_experience_artificial_similarity": after_artificial,
            "one_shot_concept_margin": after_animal - after_artificial,
        },
        "brain": brain,
    }


def measure_bridge(brain: SphereBrain) -> dict:
    bird = StructuredInput("鳥", "動作", "羽ばたく")
    plane = StructuredInput("飛行機", "動作", "飛行する")
    butterfly = StructuredInput("蝶", "動作", "羽ばたく")
    drone = StructuredInput("ドローン", "動作", "飛行する")
    target = similarity(filtered_signature(brain, bird), filtered_signature(brain, plane))
    transfer = mean([
        similarity(filtered_signature(brain, butterfly), filtered_signature(brain, plane)),
        similarity(filtered_signature(brain, drone), filtered_signature(brain, bird)),
    ])
    return {"target_similarity": target, "transfer_similarity": transfer}


def bridge_experiment() -> dict:
    experimental = clean_primary()
    control = clean_primary()
    exp_curriculum = [
        StructuredInput("鳥", "動作", "羽ばたく"),
        StructuredInput("飛行機", "動作", "飛行する"),
        StructuredInput("鳥", "場所", "空"),
        StructuredInput("飛行機", "場所", "空"),
    ]
    ctrl_curriculum = [
        StructuredInput("鳥", "動作", "羽ばたく"),
        StructuredInput("飛行機", "動作", "飛行する"),
        StructuredInput("鳥", "場所", "森"),
        StructuredInput("飛行機", "場所", "空港"),
    ]
    records = []
    for cycle in range(max(CHECKPOINTS) + 1):
        if cycle in CHECKPOINTS:
            e = measure_bridge(experimental)
            c = measure_bridge(control)
            records.append({
                "cycle": cycle,
                "experimental": e,
                "control": c,
                "context_effect": e["target_similarity"] - c["target_similarity"],
                "transfer_effect": e["transfer_similarity"] - c["transfer_similarity"],
            })
        if cycle < max(CHECKPOINTS):
            learn_items(experimental, exp_curriculum)
            learn_items(control, ctrl_curriculum)
    return {"records": records, "experimental": experimental}


def earliest_cycle(records: list[dict], predicate) -> int | None:
    for row in records:
        if predicate(row):
            return int(row["cycle"])
    return None


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    concept = concept_experiment()
    bridge = bridge_experiment()

    concept_records = concept["records"]
    bridge_records = bridge["records"]
    final_concept = concept_records[-1]
    final_bridge = bridge_records[-1]

    concept_cycle = earliest_cycle(concept_records, lambda r: r["separation"] > 0.02)
    bridge_cycle = earliest_cycle(bridge_records, lambda r: r["context_effect"] > 0.01 and r["transfer_effect"] > 0.0)

    # Save/load revalidation on the learned concept Core only, without touching production brain.json.
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "semantic_primary_roundtrip.json"
    learned = concept["brain"]
    before_sig = measure_concepts(learned)
    learned.save(temp)
    loaded = SphereBrain.load(temp)
    after_sig = measure_concepts(loaded)
    saveload_equal = all(abs(float(before_sig[k]) - float(after_sig[k])) < 1e-12 for k in before_sig)

    primary_native_present = hasattr(loaded, "learning_state") and hasattr(loaded, "observe_learning_episode")
    production_unchanged = before_hash == file_hash(BRAIN_PATH)

    concept_pass = final_concept["separation"] > 0.02
    novel_pass = concept["novel"]["one_shot_concept_margin"] > 0.02
    bridge_pass = final_bridge["context_effect"] > 0.01 and final_bridge["transfer_effect"] > 0.0
    few_shot = (concept_cycle is not None and concept_cycle <= 3) or (bridge_cycle is not None and bridge_cycle <= 3)
    pass_all = concept_pass and novel_pass and bridge_pass and saveload_equal and primary_native_present and production_unchanged

    if pass_all:
        verdict = "primary_core_revalidates_semantic_structure_sharing_context_bridge_and_one_shot_integration"
        readiness = "semantic_experience_validated_on_primary_core"
        next_step = "expand_semantic_world_and_compare_learning_efficiency_against_old_semantic_core"
    elif concept_pass and bridge_pass:
        verdict = "primary_core_preserves_semantic_structure_but_one_shot_integration_is_still_weak"
        readiness = "semantic_structure_preserved_novel_integration_needs_work"
        next_step = "audit_novel_subject_binding_without_changing_primary_core_learning"
    else:
        verdict = "semantic_encoder_behavior_does_not_fully_reappear_on_primary_core_yet"
        readiness = "semantic_revalidation_incomplete"
        next_step = "inspect_subject_relation_content_stage_signatures_before_more_semantic_training"

    payload = {
        "experiment": "Core Growth Binding v82 — Semantic Experience Revalidation on Primary Core",
        "contract": {
            "source_design": "Semantic Encoder v2 subject-relation-content staged encoding",
            "primary_spherebrain_used": True,
            "direct_input_nodes_excluded_from_similarity": True,
            "checkpoints": CHECKPOINTS,
            "native_success_failure_learning_invoked": False,
            "reason_native_learning_not_invoked": "semantic facts have no success/failure episode label in this revalidation",
            "production_brain_json_saved": False,
        },
        "concept_sharing": {
            "records": concept_records,
            "earliest_separation_cycle": concept_cycle,
            "final_pass": concept_pass,
        },
        "novel_integration": concept["novel"],
        "shared_context_bridge": {
            "records": bridge_records,
            "earliest_bridge_cycle": bridge_cycle,
            "final_pass": bridge_pass,
        },
        "summary": {
            "concept_sharing_pass": concept_pass,
            "novel_one_shot_pass": novel_pass,
            "shared_context_bridge_pass": bridge_pass,
            "few_shot_structure_observed": few_shot,
            "saveload_semantic_signature_equal": saveload_equal,
            "primary_native_learning_present": primary_native_present,
            "brain_file_unchanged": production_unchanged,
            "semantic_revalidation_pass": pass_all,
            "core_readiness": readiness,
            "overall_verdict": verdict,
            "next_step": next_step,
        },
    }
    (OUT / "latest_binding_v82.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v82</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v82：Semantic Experience Revalidation on Primary Core</h1><p class="lead">昔のSemantic Encoder v2（主体→関係→内容）をv81 Primary SphereBrainで再検証する。直接入力Nodeを除外し、Core内部の共有経路・共有文脈・少数経験での新主体統合を見る。</p><section class="panel"><div class="controls"><button id="run">Semantic再検証を実行</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Concept Sharing</h2><pre id="concept" class="raw">未実行</pre></section><section class="panel"><h2>Shared Context Bridge</h2><pre id="bridge" class="raw">未実行</pre></section><section class="panel"><h2>生データ</h2><pre id="raw" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary,n=d.novel_integration,c=d.concept_sharing,b=d.shared_context_bridge;document.getElementById('metrics').innerHTML=[metric('Concept共有',yn(s.concept_sharing_pass),s.concept_sharing_pass?'good':'warn'),metric('新主体1回統合',yn(s.novel_one_shot_pass),s.novel_one_shot_pass?'good':'warn'),metric('共有文脈Bridge',yn(s.shared_context_bridge_pass),s.shared_context_bridge_pass?'good':'warn'),metric('Few-shot観測',yn(s.few_shot_structure_observed),s.few_shot_structure_observed?'good':'warn'),metric('Concept最短cycle',c.earliest_separation_cycle??'—'),metric('Bridge最短cycle',b.earliest_bridge_cycle??'—'),metric('牛→動物 margin',n.one_shot_concept_margin.toFixed(6)),metric('Save/Load',yn(s.saveload_semantic_signature_equal),s.saveload_semantic_signature_equal?'good':'warn'),metric('Primary Native Learning',yn(s.primary_native_learning_present),s.primary_native_learning_present?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Semantic PASS',yn(s.semantic_revalidation_pass),s.semantic_revalidation_pass?'good':'warn'),metric('Core readiness',s.core_readiness)].join('');document.getElementById('concept').textContent=JSON.stringify({records:c.records,novel:n},null,2);document.getElementById('bridge').textContent=JSON.stringify(b,null,2);document.getElementById('raw').textContent=JSON.stringify(d,null,2)}document.getElementById('run').onclick=run;
</script></body></html>'''

@app.get("/")
def index(): return PAGE

@app.post("/api/run")
def api_run():
    try: return jsonify(observe())
    except Exception as exc: return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def open_browser() -> None: webbrowser.open(f"http://{HOST}:{PORT}")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v82: http://{HOST}:{PORT}")
    print("Semantic Encoder v2 revalidation on Primary SphereBrain / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
