from __future__ import annotations

import hashlib
import html
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_core_assist import StructuralAssistConfig, StructuralCoreAssist

OUT = ROOT / "data" / "structural_puzzle_lab_v1" / "results"
BRAIN_PATH = ROOT / "data" / "brain.json"


@dataclass
class MiniBrain:
    positions: np.ndarray
    adjacency: np.ndarray
    weights: np.ndarray
    usage: np.ndarray
    neighbors_per_node: int = 3


PUZZLES = [
    {
        "id": "merge",
        "title": "合流パズル",
        "description": "2本の流れが1本へ合流したあと、同点の2候補を構造がどう並べるか。",
        "history": [[0, 1], [2], [3]],
        "edges": [[(0, 2), (1, 2)], [(2, 3)]],
        "source": 3,
        "candidate_roles": ["高接続候補", "低接続候補"],
        "candidate_targets": [4, 5],
        "candidate_usage": [4, 4],
        "candidate_weights": [0.64, 0.64],
        "candidate_degrees": [3, 1],
    },
    {
        "id": "repetition",
        "title": "反復パズル",
        "description": "同じ局所経路を繰り返したあと、よく使われた候補と新しい候補を比較する。",
        "history": [[0], [1], [2], [0], [1], [2], [3]],
        "edges": [[(0, 1)], [(1, 2)], [(2, 0)], [(0, 1)], [(1, 2)], [(2, 3)]],
        "source": 3,
        "candidate_roles": ["反復に沿う候補", "未使用候補"],
        "candidate_targets": [4, 5],
        "candidate_usage": [18, 0],
        "candidate_weights": [0.62, 0.62],
        "candidate_degrees": [2, 2],
    },
    {
        "id": "missing_route",
        "title": "欠けた経路パズル",
        "description": "途中まで伸びた経路に対し、球体内で進行方向を保つ候補と戻る候補を比較する。",
        "history": [[0], [1], [2], [3]],
        "edges": [[(0, 1)], [(1, 2)], [(2, 3)]],
        "source": 3,
        "candidate_roles": ["前進候補", "後退候補"],
        "candidate_targets": [4, 5],
        "candidate_usage": [2, 2],
        "candidate_weights": [0.63, 0.63],
        "candidate_degrees": [2, 2],
    },
]


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def make_brain(puzzle: dict, offset: int = 0) -> MiniBrain:
    n = offset + 12
    positions = np.zeros((n, 3), dtype=float)
    # Relative geometry is copied exactly for the ID-shifted control.
    base_positions = {
        0: (-0.7, 0.3, 0.0), 1: (-0.7, -0.3, 0.0), 2: (-0.35, 0.0, 0.0),
        3: (0.0, 0.0, 0.0), 4: (0.55, 0.15, 0.0), 5: (-0.45, 0.15, 0.0),
        6: (0.65, 0.35, 0.0), 7: (0.65, -0.15, 0.0), 8: (-0.55, 0.35, 0.0),
    }
    for node, pos in base_positions.items():
        positions[offset + node] = pos

    adjacency = np.zeros((n, n), dtype=bool)
    weights = np.zeros((n, n), dtype=float)
    usage = np.zeros((n, n), dtype=int)

    def connect(a: int, b: int, weight: float = 0.5, use: int = 0) -> None:
        a += offset; b += offset
        adjacency[a, b] = adjacency[b, a] = True
        weights[a, b] = weights[b, a] = weight
        usage[a, b] = usage[b, a] = use

    for step_edges in puzzle["edges"]:
        for a, b in step_edges:
            connect(a, b, 0.58, 3)

    source = puzzle["source"]
    for idx, target in enumerate(puzzle["candidate_targets"]):
        connect(source, target, puzzle["candidate_weights"][idx], puzzle["candidate_usage"][idx])

    # Set target degree without changing candidate-local values.
    for idx, target in enumerate(puzzle["candidate_targets"]):
        extras = puzzle["candidate_degrees"][idx] - 1
        for extra in range(max(0, extras)):
            connect(target, 6 + idx * 2 + extra, 0.4, 0)

    return MiniBrain(positions, adjacency, weights, usage)


def shifted(values, offset: int):
    return [[v + offset for v in step] for step in values]


def shifted_edges(values, offset: int):
    return [[(a + offset, b + offset) for a, b in step] for step in values]


def run_puzzle(puzzle: dict, offset: int = 0) -> dict:
    brain = make_brain(puzzle, offset)
    source = puzzle["source"] + offset
    targets = [v + offset for v in puzzle["candidate_targets"]]
    # Exact baseline tie: the experiment asks whether structure resolves it.
    ranked = [(target, (0.500000, source)) for target in targets]
    history = shifted(puzzle["history"], offset)
    edges = shifted_edges(puzzle["edges"], offset)

    off = StructuralCoreAssist(StructuralAssistConfig(enabled=False))
    on = StructuralCoreAssist(StructuralAssistConfig(enabled=True))
    off_ranked, off_trace = off.reorder(brain, ranked, history, edges)
    on_ranked, on_trace = on.reorder(brain, ranked, history, edges)

    original_role_order = puzzle["candidate_roles"]
    on_target_order = [target for target, _ in on_ranked]
    on_role_order = [original_role_order[targets.index(target)] for target in on_target_order]
    return {
        "offset": offset,
        "baseline_scores": [0.5, 0.5],
        "off_role_order": original_role_order,
        "on_role_order": on_role_order,
        "off_target_order": [target for target, _ in off_ranked],
        "on_target_order": on_target_order,
        "trace": on_trace,
        "selected_role": on_role_order[0],
    }


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{html.escape(text)}</span>'


def make_html(payload: dict) -> str:
    cards = []
    for item in payload["puzzles"]:
        main = item["main"]
        ctrl = item["id_shift_control"]
        trace = main["trace"]
        resolved = bool(trace["tie_gate_active"] and trace["top_candidate_changed"])
        cards.append(f"""
<section class="card">
  <div class="head"><div><h2>{html.escape(item['title'])}</h2><p>{html.escape(item['description'])}</p></div>{badge('ID非依存' if item['id_invariant'] else 'ID依存', 'ok' if item['id_invariant'] else 'warn')}</div>
  <div class="flow">{html.escape(item['structure_text'])}</div>
  <div class="grid">
    <div><h3>構造OFF</h3><ol><li>{html.escape(main['off_role_order'][0])}</li><li>{html.escape(main['off_role_order'][1])}</li></ol></div>
    <div><h3>構造ON</h3><ol><li><b>{html.escape(main['on_role_order'][0])}</b></li><li>{html.escape(main['on_role_order'][1])}</li></ol></div>
  </div>
  <div class="metrics">
    <span>tie gate {badge('作動' if trace['tie_gate_active'] else '待機', 'active' if trace['tie_gate_active'] else 'idle')}</span>
    <span>同率解決 {badge('あり' if resolved else '順位維持', 'resolve' if resolved else 'idle')}</span>
    <span>最大変調 <b>{trace['absolute_modulation']:.8f}</b></span>
    <span>ID変更後の選択 <b>{html.escape(ctrl['selected_role'])}</b></span>
  </div>
</section>""")
    raw = html.escape(json.dumps(payload, ensure_ascii=False, indent=2))
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structural Puzzle Lab v1</title>
<style>
:root{{--bg:#0b1020;--panel:#151d2d;--panel2:#202a3d;--line:#34425e;--text:#eef4ff;--muted:#a9b6cd;--accent:#7dd3fc}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(150deg,#080d18,#111a2d);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1100px;margin:auto;padding:38px 20px 70px}}h1{{font-size:clamp(32px,5vw,56px);margin:0}}.lead{{color:var(--muted);font-size:18px;line-height:1.7}}.safe{{background:#13251d;border:1px solid #286344;padding:14px 18px;border-radius:14px;margin:22px 0}}.card{{background:rgba(21,29,45,.95);border:1px solid var(--line);border-radius:20px;padding:24px;margin:20px 0;box-shadow:0 20px 55px #0005}}.head{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}h2{{margin:0 0 8px}}p{{color:var(--muted)}}.flow{{font-family:ui-monospace,Consolas,monospace;background:#0c1322;border-radius:12px;padding:15px;color:var(--accent);margin:15px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.grid>div{{background:var(--panel2);border-radius:14px;padding:16px}}ol{{line-height:2}}.metrics{{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px;color:var(--muted)}}.metrics>span{{background:#0f1728;padding:9px 12px;border-radius:10px}}.badge{{display:inline-block;padding:4px 9px;border-radius:999px;font-weight:700;font-size:13px}}.ok{{background:#14532d;color:#86efac}}.warn{{background:#5f1723;color:#fda4af}}.active{{background:#5b4812;color:#fcd34d}}.resolve{{background:#3b286b;color:#d8b4fe}}.idle{{background:#29354d;color:#d3dbea}}details{{margin-top:28px}}pre{{white-space:pre-wrap;background:#070b13;padding:18px;border-radius:14px;overflow:auto}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}.head{{display:block}}}}
</style></head><body><main><h1>Structural Puzzle Lab v1</h1><p class="lead">意味や正解ラベルを与えず、合流・反復・方向という構造履歴が同点候補の並びへ作用するかを観察する。</p><div class="safe">学習OFF ／ noise OFF ／ brain.json保存なし ／ Node ID変更対照あり</div>{''.join(cards)}<details><summary>生データ</summary><pre>{raw}</pre></details></main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = file_hash(BRAIN_PATH)
    results = []
    structure_texts = {
        "merge": "A ─┐\n   ├→ C → D → [候補1 / 候補2]\nB ─┘",
        "repetition": "A → B → C → A → B → C → D → [候補1 / 候補2]",
        "missing_route": "A → B → C → D → [?]   （前進 / 後退）",
    }
    for puzzle in PUZZLES:
        main_run = run_puzzle(puzzle, 0)
        control = run_puzzle(puzzle, 20)
        results.append({
            "id": puzzle["id"],
            "title": puzzle["title"],
            "description": puzzle["description"],
            "structure_text": structure_texts[puzzle["id"]],
            "main": main_run,
            "id_shift_control": control,
            "id_invariant": main_run["selected_role"] == control["selected_role"],
        })
    after = file_hash(BRAIN_PATH)
    payload = {
        "experiment": "Structural Puzzle Lab v1",
        "language_free": True,
        "answer_labels": False,
        "learning": False,
        "noise": 0.0,
        "brain_file_unchanged": before == after,
        "puzzles": results,
        "checks": {
            "all_tie_gates_active": all(x["main"]["trace"]["tie_gate_active"] for x in results),
            "all_id_invariant": all(x["id_invariant"] for x in results),
            "no_brain_file_change": before == after,
            "all_modulation_bounded": all(x["main"]["trace"]["absolute_modulation"] <= 5e-5 + 1e-12 for x in results),
        },
    }
    json_path = OUT / "structural_puzzle_lab_v1.json"
    html_path = OUT / "structural_puzzle_lab_v1.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(make_html(payload), encoding="utf-8")
    print("Structural Puzzle Lab v1")
    print(f"checks: {payload['checks']}")
    print(f"HTML: {html_path}")
    print(f"JSON: {json_path}")
    webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
