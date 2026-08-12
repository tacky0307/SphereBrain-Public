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
OUT = ROOT / "data" / "core_structural_assist_live_observer_v1" / "results"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_payload(result) -> dict:
    return {
        "source_nodes": [int(v) for v in result.source_nodes],
        "activated_nodes": [int(v) for v in result.activated_nodes],
        "traversed_edges": [[int(a), int(b)] for a, b in result.traversed_edges],
        "activation_history": [[int(v) for v in step] for step in result.activation_history],
        "final_active": [int(v) for v in np.flatnonzero(result.final_activation > 0)],
        "final_values": {
            str(int(v)): float(result.final_activation[v])
            for v in np.flatnonzero(result.final_activation > 0)
        },
    }


def run_once(text: str) -> dict:
    before_hash = file_hash(BRAIN_PATH)
    off_brain = SphereBrain.load(BRAIN_PATH)
    on_brain = SphereBrain.load(BRAIN_PATH)
    sources = off_brain.text_to_sources(text)

    off_brain.set_structural_assist(False)
    off = off_brain.propagate(sources, steps=18, threshold=0.18, noise=0.0, learn=False)
    off_trace = list(off_brain.last_structural_assist_trace)

    on_brain.set_structural_assist(True)
    on = on_brain.propagate(sources, steps=18, threshold=0.18, noise=0.0, learn=False)
    on_trace = list(on_brain.last_structural_assist_trace)
    after_hash = file_hash(BRAIN_PATH)

    off_data = result_payload(off)
    on_data = result_payload(on)
    return {
        "experiment": "Core Structural Assist Live Observer v1",
        "input_text": text,
        "source_nodes": sources,
        "safety": {
            "learning": False,
            "noise": 0.0,
            "brain_saved": False,
            "brain_file_unchanged": before_hash == after_hash,
            "sha256_before": before_hash,
            "sha256_after": after_hash,
        },
        "off": off_data,
        "on": on_data,
        "off_trace": off_trace,
        "on_trace": on_trace,
        "comparison": {
            "routes_equal": off_data["traversed_edges"] == on_data["traversed_edges"],
            "activated_nodes_equal": off_data["activated_nodes"] == on_data["activated_nodes"],
            "activation_history_equal": off_data["activation_history"] == on_data["activation_history"],
            "final_values_equal": off_data["final_values"] == on_data["final_values"],
            "assist_activation_count": sum(bool(x.get("tie_gate_active")) for x in on_trace),
            "tie_resolution_count": sum(
                bool(x.get("tie_gate_active"))
                and bool(x.get("near_zero_tie"))
                and bool(x.get("top_candidate_changed"))
                for x in on_trace
            ),
        },
    }


def badge(value: bool, yes: str = "一致", no: str = "差あり") -> str:
    cls = "ok" if value else "warn"
    return f'<span class="badge {cls}">{yes if value else no}</span>'


def render_steps(payload: dict) -> str:
    off_history = payload["off"]["activation_history"]
    on_history = payload["on"]["activation_history"]
    trace = payload["on_trace"]
    total = max(len(off_history), len(on_history), len(trace) + 1)
    cards = []
    for step in range(total):
        off_nodes = off_history[step] if step < len(off_history) else []
        on_nodes = on_history[step] if step < len(on_history) else []
        item = trace[step - 1] if step > 0 and step - 1 < len(trace) else None
        if item is None:
            assist = '<span class="badge idle">入力</span>'
            margin = "—"
            modulation = "—"
        else:
            active = bool(item.get("tie_gate_active"))
            changed = bool(item.get("top_candidate_changed"))
            near = bool(item.get("near_zero_tie"))
            if active and changed and near:
                assist = '<span class="badge resolve">構造が同率を解決</span>'
            elif active:
                assist = '<span class="badge active">構造補助が作動</span>'
            else:
                assist = '<span class="badge idle">構造は待機</span>'
            margin_value = item.get("baseline_margin")
            margin = "—" if margin_value is None else f"{margin_value:.10g}"
            modulation = f"{float(item.get('absolute_modulation', 0.0)):.10g}"
        cards.append(f"""
        <section class="step-card">
          <div class="step-head"><h3>Step {step}</h3>{assist}</div>
          <div class="metrics"><span>通常マージン <b>{margin}</b></span><span>最大変調 <b>{modulation}</b></span></div>
          <div class="columns">
            <div><h4>OFF 活動Node</h4><div class="nodes">{html.escape(', '.join(map(str, off_nodes)) or 'なし')}</div></div>
            <div><h4>ON 活動Node</h4><div class="nodes">{html.escape(', '.join(map(str, on_nodes)) or 'なし')}</div></div>
          </div>
        </section>""")
    return "\n".join(cards)


def make_html(payload: dict) -> str:
    comp = payload["comparison"]
    text = html.escape(payload["input_text"])
    source_nodes = ", ".join(map(str, payload["source_nodes"]))
    edges_off = " · ".join(f"{a}–{b}" for a, b in payload["off"]["traversed_edges"])
    edges_on = " · ".join(f"{a}–{b}" for a, b in payload["on"]["traversed_edges"])
    raw = html.escape(json.dumps(payload, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Core Structural Assist Live Observer v1</title>
<style>
:root{{--bg:#0d1321;--panel:#151d2d;--panel2:#1c2639;--text:#edf3ff;--muted:#9eacc5;--line:#31405c;--accent:#7dd3fc;--ok:#86efac;--warn:#fda4af;--active:#fcd34d;--resolve:#c4b5fd}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(160deg,#0b1020,#111a2d);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 20px 64px}} h1{{font-size:clamp(26px,4vw,44px);margin:0 0 8px}} h2{{margin-top:36px}} .sub{{color:var(--muted)}}
.hero,.summary,.step-card,.route{{background:rgba(21,29,45,.94);border:1px solid var(--line);border-radius:18px;padding:22px;margin-top:18px;box-shadow:0 18px 50px #0005}}
.input{{font-size:22px;font-weight:700;color:var(--accent);margin:10px 0}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.stat{{background:var(--panel2);border-radius:13px;padding:14px}} .stat small{{display:block;color:var(--muted);margin-bottom:6px}} .stat b{{font-size:22px}}
.badge{{display:inline-block;border-radius:999px;padding:5px 10px;font-size:13px;font-weight:700}} .ok{{background:#14532d;color:var(--ok)}} .warn{{background:#5f1723;color:var(--warn)}} .idle{{background:#29354d;color:#cbd5e1}} .active{{background:#5b4812;color:var(--active)}} .resolve{{background:#3b286b;color:var(--resolve)}}
.step-head{{display:flex;align-items:center;justify-content:space-between;gap:12px}} .step-head h3{{margin:0}} .metrics{{display:flex;gap:22px;flex-wrap:wrap;color:var(--muted);font-size:14px;margin:12px 0}}
.columns{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .columns>div{{background:var(--panel2);border-radius:12px;padding:14px}} h4{{margin:0 0 8px;color:var(--muted)}} .nodes,.edges{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.7;word-break:break-word}}
details{{margin-top:28px}} pre{{white-space:pre-wrap;background:#090e19;padding:18px;border-radius:14px;overflow:auto;color:#cbd5e1}} @media(max-width:650px){{.columns{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="hero"><h1>Core Structural Assist<br>Live Observer v1</h1><p class="sub">Coreの通常動作と構造補助ONを、保存せず同時観察する。</p><div class="input">「{text}」</div><div>入力Node: <b>{source_nodes}</b></div></div>
<div class="summary"><h2>観察結果</h2><div class="grid">
<div class="stat"><small>構造補助の作動</small><b>{comp['assist_activation_count']} 回</b></div>
<div class="stat"><small>同率を構造が解決</small><b>{comp['tie_resolution_count']} 回</b></div>
<div class="stat"><small>経路</small>{badge(comp['routes_equal'])}</div>
<div class="stat"><small>活動履歴</small>{badge(comp['activation_history_equal'])}</div>
<div class="stat"><small>brain.json</small>{badge(payload['safety']['brain_file_unchanged'],'不変','変更あり')}</div>
</div></div>
<h2>ステップごとの動き</h2>{render_steps(payload)}
<div class="route"><h2>通過Edge</h2><h4>OFF</h4><div class="edges">{html.escape(edges_off or 'なし')}</div><h4 style="margin-top:18px">ON</h4><div class="edges">{html.escape(edges_on or 'なし')}</div></div>
<details><summary>生データを表示</summary><pre>{raw}</pre></details>
</main></body></html>"""


def main() -> None:
    if not BRAIN_PATH.exists():
        raise FileNotFoundError(f"brain file not found: {BRAIN_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)
    print("Core Structural Assist Live Observer v1")
    print("学習OFF / noise OFF / brain.json保存なし")
    text = input("観察する文章を入力してください: ").strip()
    if not text:
        text = "今日は晴れて気持ちいい"
        print(f"空入力のため例文を使用します: {text}")
    payload = run_once(text)
    json_path = OUT / "core_structural_assist_live_observer_v1.json"
    html_path = OUT / "core_structural_assist_live_observer_v1.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(make_html(payload), encoding="utf-8")
    print(f"構造補助作動: {payload['comparison']['assist_activation_count']} 回")
    print(f"同率解決: {payload['comparison']['tie_resolution_count']} 回")
    print(f"brain unchanged: {payload['safety']['brain_file_unchanged']}")
    print(f"HTML: {html_path}")
    print(f"JSON: {json_path}")
    webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
