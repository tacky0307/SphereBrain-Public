from __future__ import annotations

import hashlib
import html
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import SphereBrain

BRAIN_PATH = ROOT / "data" / "brain.json"
OUT = ROOT / "data" / "core_structural_assist_character_compare_v1" / "results"
DEFAULT_TEXTS = ["魔王", "勇者", "スライム"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_payload(result) -> dict:
    return {
        "source_nodes": [int(v) for v in result.source_nodes],
        "activated_nodes": [int(v) for v in result.activated_nodes],
        "traversed_edges": [[int(a), int(b)] for a, b in result.traversed_edges],
        "activation_history": [[int(v) for v in step] for step in result.activation_history],
        "final_active": [int(v) for v in np.flatnonzero(result.final_activation > 0)],
    }


def run_text(text: str) -> dict:
    before_hash = file_hash(BRAIN_PATH)
    off_brain = SphereBrain.load(BRAIN_PATH)
    on_brain = SphereBrain.load(BRAIN_PATH)
    sources = off_brain.text_to_sources(text)

    off_brain.set_structural_assist(False)
    off = off_brain.propagate(sources, steps=18, threshold=0.18, noise=0.0, learn=False)

    on_brain.set_structural_assist(True)
    on = on_brain.propagate(sources, steps=18, threshold=0.18, noise=0.0, learn=False)
    trace = list(on_brain.last_structural_assist_trace)
    after_hash = file_hash(BRAIN_PATH)

    off_data = result_payload(off)
    on_data = result_payload(on)
    return {
        "text": text,
        "source_nodes": sources,
        "off": off_data,
        "on": on_data,
        "trace": trace,
        "metrics": {
            "assist_activation_count": sum(bool(x.get("tie_gate_active")) for x in trace),
            "tie_resolution_count": sum(
                bool(x.get("tie_gate_active"))
                and bool(x.get("near_zero_tie"))
                and bool(x.get("top_candidate_changed"))
                for x in trace
            ),
            "routes_equal": off_data["traversed_edges"] == on_data["traversed_edges"],
            "history_equal": off_data["activation_history"] == on_data["activation_history"],
            "brain_file_unchanged": before_hash == after_hash,
            "activated_node_count": len(on_data["activated_nodes"]),
            "edge_count": len(on_data["traversed_edges"]),
            "step_count": max(0, len(on_data["activation_history"]) - 1),
        },
    }


def cell(value: str) -> str:
    return f"<td>{html.escape(value)}</td>"


def make_html(payload: dict) -> str:
    cases = payload["cases"]
    max_steps = max(len(c["on"]["activation_history"]) for c in cases)
    overview_cards = []
    for case in cases:
        m = case["metrics"]
        overview_cards.append(f"""
        <section class="character-card">
          <h2>{html.escape(case['text'])}</h2>
          <p class="nodes">入力Node: {', '.join(map(str, case['source_nodes']))}</p>
          <div class="stats">
            <div><small>構造補助</small><b>{m['assist_activation_count']}回</b></div>
            <div><small>同率解決</small><b>{m['tie_resolution_count']}回</b></div>
            <div><small>活動Node</small><b>{m['activated_node_count']}</b></div>
            <div><small>通過Edge</small><b>{m['edge_count']}</b></div>
          </div>
        </section>""")

    rows = []
    for step in range(max_steps):
        cells = [f"<th>Step {step}</th>"]
        for case in cases:
            history = case["on"]["activation_history"]
            nodes = history[step] if step < len(history) else []
            trace = case["trace"]
            item = trace[step - 1] if step > 0 and step - 1 < len(trace) else None
            if item is None:
                badge = "入力"
                detail = ""
            elif item.get("tie_gate_active") and item.get("near_zero_tie") and item.get("top_candidate_changed"):
                badge = "同率解決"
                detail = f"margin={item.get('baseline_margin')} / mod={item.get('absolute_modulation')}"
            elif item.get("tie_gate_active"):
                badge = "構造補助"
                detail = f"margin={item.get('baseline_margin')} / mod={item.get('absolute_modulation')}"
            else:
                badge = "待機"
                detail = f"margin={item.get('baseline_margin')}"
            body = f"<span class='badge'>{badge}</span><div class='node-list'>{', '.join(map(str, nodes)) or 'なし'}</div><small>{html.escape(detail)}</small>"
            cells.append(f"<td>{body}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    edge_blocks = []
    for case in cases:
        off_edges = " · ".join(f"{a}-{b}" for a, b in case["off"]["traversed_edges"]) or "なし"
        on_edges = " · ".join(f"{a}-{b}" for a, b in case["on"]["traversed_edges"]) or "なし"
        edge_blocks.append(f"""
        <section class="edge-card"><h3>{html.escape(case['text'])}</h3>
        <p><b>OFF</b><br>{html.escape(off_edges)}</p>
        <p><b>ON</b><br>{html.escape(on_edges)}</p></section>""")

    raw = html.escape(json.dumps(payload, ensure_ascii=False, indent=2))
    return f"""<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>魔王・勇者・スライム Core比較</title><style>
:root{{--bg:#0d1321;--panel:#151d2d;--panel2:#1c2639;--text:#edf3ff;--muted:#9eacc5;--line:#31405c;--accent:#7dd3fc}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(160deg,#0b1020,#111a2d);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1380px;margin:auto;padding:32px 20px 64px}}h1{{font-size:clamp(28px,4vw,48px);margin:0 0 8px}}.sub{{color:var(--muted)}}.overview{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}}.character-card,.edge-card,.table-wrap,details{{background:rgba(21,29,45,.95);border:1px solid var(--line);border-radius:18px;padding:20px}}.character-card h2{{margin-top:0;color:var(--accent)}}.nodes,.node-list{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}}.stats{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.stats div{{background:var(--panel2);padding:12px;border-radius:12px}}small{{color:var(--muted)}}b{{font-size:20px}}.table-wrap{{margin-top:24px;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:980px}}th,td{{border-bottom:1px solid var(--line);padding:14px;vertical-align:top;text-align:left}}th{{color:var(--accent)}}.badge{{display:inline-block;background:#334155;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:700;margin-bottom:8px}}.node-list{{font-weight:700;line-height:1.6}}.edges{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}}pre{{white-space:pre-wrap;background:#090e19;padding:16px;border-radius:12px;overflow:auto}}@media(max-width:900px){{.overview,.edges{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>魔王・勇者・スライム<br>Core Structural Assist 比較</h1><p class='sub'>同じbrain.jsonを、学習OFF・noise OFF・保存なしで比較。</p>
<div class='overview'>{''.join(overview_cards)}</div>
<div class='table-wrap'><h2>ステップ比較</h2><table><thead><tr><th>Step</th>{''.join(f'<th>{html.escape(c["text"])}</th>' for c in cases)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>通過Edge比較</h2><div class='edges'>{''.join(edge_blocks)}</div>
<details><summary>生データを表示</summary><pre>{raw}</pre></details></main></body></html>"""


def main() -> None:
    if not BRAIN_PATH.exists():
        raise FileNotFoundError(f"brain file not found: {BRAIN_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)
    print("魔王・勇者・スライム Core比較")
    print("学習OFF / noise OFF / brain.json保存なし")
    cases = [run_text(text) for text in DEFAULT_TEXTS]
    payload = {
        "experiment": "Core Structural Assist Character Compare v1",
        "texts": DEFAULT_TEXTS,
        "cases": cases,
        "all_brain_files_unchanged": all(c["metrics"]["brain_file_unchanged"] for c in cases),
    }
    json_path = OUT / "core_structural_assist_character_compare_v1.json"
    html_path = OUT / "core_structural_assist_character_compare_v1.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(make_html(payload), encoding="utf-8")
    print(f"HTML: {html_path}")
    print(f"JSON: {json_path}")
    webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
