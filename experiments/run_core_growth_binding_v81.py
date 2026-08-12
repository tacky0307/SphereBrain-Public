from __future__ import annotations

import copy
import hashlib
import json
import random
import socket
import sys
import threading
import webbrowser
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from brain import SphereBrain
from native_learning_brain import NativeLearningSphereBrain
import run_core_growth_binding_v70 as v70
import run_core_growth_binding_v80 as v80

HOST = "127.0.0.1"
START_PORT = 5128
OUT = ROOT / "data" / "core_growth_binding_v81" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"
SEED = v80.SEED
EPISODES_PER_PHASE = v80.EPISODES_PER_PHASE
PHASES = v80.PHASES
SAVELOAD_AFTER = v80.SAVELOAD_AFTER
ALIGNED_MOTIF = "m70"


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


def observe() -> dict:
    before_hash = file_hash(BRAIN_PATH)
    base = SphereBrain.load(BRAIN_PATH)
    base.clear_experience_state()
    base.clear_learning_state()
    pretrained = v70.pretrain(base)

    # Reference remains the validated v79 runner path. Primary is now the actual
    # SphereBrain class, not the v80 candidate subclass.
    ref = copy.deepcopy(pretrained)
    primary = copy.deepcopy(pretrained)
    ref.clear_experience_state()
    ref.clear_learning_state()
    primary.clear_experience_state()
    primary.clear_learning_state()

    motif_counts: dict[str, int] = defaultdict(int)
    success_counts: dict[str, int] = defaultdict(int)
    stability: dict[str, float] = defaultdict(float)
    rng = random.Random(SEED)

    rows = []
    all_equal = True
    saveload_state_equal = False
    saveload_behavior_equal = False
    global_episode = 0

    for case in PHASES:
        for _ in range(EPISODES_PER_PHASE):
            global_episode += 1
            ep = v70.run_episode(
                ref,
                case,
                relative_mode=True,
                assist=False,
                rng=rng,
                explore=v70.EXPLORATION,
                max_steps=v70.TRAIN_MAX_STEPS,
            )
            desc = v80.descriptors(primary, ep)
            reference = v80.reference_update(
                ref, ep, motif_counts, success_counts, stability, global_episode
            )
            primary_result = primary.observe_learning_episode(
                desc,
                success=bool(ep["success"]),
                expected_conditions=v70.EXPECTED,
                motif=ALIGNED_MOTIF,
            )
            checks = v80.compare_state(
                ref,
                primary,
                motif_counts,
                success_counts,
                stability,
                reference["events"],
                primary_result["events"],
            )
            all_equal = all_equal and bool(checks["all"])
            rows.append({
                "episode": global_episode,
                "environment": case["name"],
                "success": bool(ep["success"]),
                "path": ep["path"],
                "checks": checks,
            })

            if global_episode == SAVELOAD_AFTER:
                OUT.mkdir(parents=True, exist_ok=True)
                temp_path = OUT / "primary_roundtrip.json"
                before_learning = primary.snapshot_learning_state()
                before_exp = primary.snapshot_experience_state()
                primary.save(temp_path)
                loaded = SphereBrain.load(temp_path)
                saveload_state_equal = (
                    v80.profile_equal(before_learning, loaded.snapshot_learning_state())
                    and v80.profile_equal(before_exp, loaded.snapshot_experience_state())
                )
                a = v70.run_episode(primary, case, relative_mode=True, assist=False)
                b = v70.run_episode(loaded, case, relative_mode=True, assist=False)
                saveload_behavior_equal = (
                    a["path"] == b["path"]
                    and v80.array_equal(primary.weights, loaded.weights)
                    and v80.array_equal(primary.usage, loaded.usage)
                )
                primary = loaded

    final_routes = []
    final_route_equal = True
    for case in PHASES:
        a = v70.run_episode(ref, case, relative_mode=True, assist=False)
        b = v70.run_episode(primary, case, relative_mode=True, assist=False)
        equal = (
            a["path"] == b["path"]
            and a["success"] == b["success"]
            and a["steps"] == b["steps"]
        )
        final_route_equal = final_route_equal and equal
        final_routes.append({
            "environment": case["name"],
            "reference": a,
            "primary": b,
            "equal": equal,
        })

    # Current production brain.json predates Primary Learning State; it must load
    # with an empty native state rather than failing or inventing learned history.
    old_loaded = SphereBrain.load(BRAIN_PATH)
    old_compatible = old_loaded.snapshot_learning_state() == {
        "schema_version": 1,
        "episode_counter": 0,
        "motif_counts": {},
        "success_counts": {},
        "stability": {},
    }

    # The former candidate name remains a compatibility wrapper around Primary Core.
    compat = NativeLearningSphereBrain.from_sphere_brain(primary)
    compat_state_equal = (
        v80.profile_equal(primary.snapshot_learning_state(), compat.snapshot_learning_state())
        and v80.profile_equal(primary.snapshot_experience_state(), compat.snapshot_experience_state())
        and v80.array_equal(primary.weights, compat.weights)
        and v80.array_equal(primary.usage, compat.usage)
    )
    compat_routes_equal = True
    for case in PHASES:
        a = v70.run_episode(primary, case, relative_mode=True, assist=False)
        b = v70.run_episode(compat, case, relative_mode=True, assist=False)
        compat_routes_equal = compat_routes_equal and (
            a["path"] == b["path"] and a["success"] == b["success"] and a["steps"] == b["steps"]
        )

    brain_unchanged = before_hash == file_hash(BRAIN_PATH)
    promotion_pass = (
        all_equal
        and saveload_state_equal
        and saveload_behavior_equal
        and final_route_equal
        and old_compatible
        and compat_state_equal
        and compat_routes_equal
        and brain_unchanged
    )

    payload = {
        "experiment": "Core Growth Binding v81 — Primary SphereBrain Core Promotion",
        "contract": {
            "new_algorithm_added": False,
            "primary_class": "brain.SphereBrain",
            "reference_algorithm": "v79_homeostatic_consolidation_plus_temporal_credit",
            "promoted_features": [
                "native_learning_state",
                "temporal_credit_assignment",
                "homeostatic_consolidation",
                "native_learning_save_load",
            ],
            "episode_count": global_episode,
            "phase_sequence": [x["name"] for x in PHASES],
            "every_episode_state_compared": True,
            "midrun_primary_save_load": True,
            "old_brain_json_backward_compatible": True,
            "candidate_class_reduced_to_compatibility_wrapper": True,
            "production_brain_json_saved": False,
        },
        "summary": {
            "episode_equivalence_count": sum(1 for x in rows if x["checks"]["all"]),
            "episode_count": len(rows),
            "all_episode_equivalent": all_equal,
            "primary_saveload_state_equal": saveload_state_equal,
            "primary_saveload_behavior_equal": saveload_behavior_equal,
            "final_route_equal": final_route_equal,
            "old_brain_json_compatible": old_compatible,
            "compatibility_wrapper_state_equal": compat_state_equal,
            "compatibility_wrapper_route_equal": compat_routes_equal,
            "brain_file_unchanged": brain_unchanged,
            "primary_core_promotion_pass": promotion_pass,
            "core_readiness": (
                "primary_spherebrain_native_learning_promoted"
                if promotion_pass
                else "primary_spherebrain_promotion_mismatch"
            ),
            "overall_verdict": (
                "validated_native_learning_is_now_part_of_primary_spherebrain_core"
                if promotion_pass
                else "primary_core_promotion_differs_from_validated_native_learning"
            ),
            "next_step": (
                "return_to_behavioral_world_tests_using_primary_spherebrain_only"
                if promotion_pass
                else "inspect_first_primary_core_mismatch_before_using_promoted_learning"
            ),
        },
        "episode_rows": rows,
        "final_routes": final_routes,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v81.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v81</title><style>
:root{--bg:#07111f;--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:30px 22px 70px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.lead{color:var(--muted);font-size:18px;line-height:1.65}.controls{display:flex;justify-content:flex-end}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:900px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>v81：Primary SphereBrain Core Promotion</h1><p class="lead">v80Bで完全一致したNative Learningを主SphereBrainへ昇格し、validated runnerとPrimary Coreを32 Episode完全比較する。新アルゴリズムは追加しない。</p><section class="panel"><div class="controls"><button id="run">Primary Core昇格を検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Episode別</h2><pre id="rows" class="raw">未実行</pre></section><section class="panel"><h2>最終Route</h2><pre id="routes" class="raw">未実行</pre></section><script>
function metric(k,v,c='blue'){return `<div class="metric"><span>${k}</span><b class="${c}">${v}</b></div>`}function yn(v){return v?'YES':'NO'}async function run(){document.getElementById('metrics').innerHTML=metric('状態','実行中…');const r=await fetch('/api/run',{method:'POST'});const d=await r.json();if(!r.ok){document.getElementById('metrics').innerHTML=metric('エラー',d.error||'失敗','warn');return}const s=d.summary;document.getElementById('metrics').innerHTML=[metric('Episode一致',`${s.episode_equivalence_count}/${s.episode_count}`,s.all_episode_equivalent?'good':'warn'),metric('全Episode一致',yn(s.all_episode_equivalent),s.all_episode_equivalent?'good':'warn'),metric('Primary Save/Load State',yn(s.primary_saveload_state_equal),s.primary_saveload_state_equal?'good':'warn'),metric('Primary Save/Load Behavior',yn(s.primary_saveload_behavior_equal),s.primary_saveload_behavior_equal?'good':'warn'),metric('最終Route一致',yn(s.final_route_equal),s.final_route_equal?'good':'warn'),metric('旧brain.json互換',yn(s.old_brain_json_compatible),s.old_brain_json_compatible?'good':'warn'),metric('互換Wrapper State',yn(s.compatibility_wrapper_state_equal),s.compatibility_wrapper_state_equal?'good':'warn'),metric('互換Wrapper Route',yn(s.compatibility_wrapper_route_equal),s.compatibility_wrapper_route_equal?'good':'warn'),metric('brain.json',s.brain_file_unchanged?'不変':'変化',s.brain_file_unchanged?'good':'warn'),metric('Primary昇格PASS',yn(s.primary_core_promotion_pass),s.primary_core_promotion_pass?'good':'warn'),metric('Core readiness',s.core_readiness),metric('総合判定',s.overall_verdict)].join('');document.getElementById('rows').textContent=JSON.stringify(d.episode_rows,null,2);document.getElementById('routes').textContent=JSON.stringify(d.final_routes,null,2)}document.getElementById('run').onclick=run;
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
    print(f"Core Growth Binding v81: http://{HOST}:{PORT}")
    print("Primary SphereBrain promotion / 32-episode equivalence / production brain.json saveなし")
    serve(app, host=HOST, port=PORT)
