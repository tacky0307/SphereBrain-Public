from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3
import subprocess
import webbrowser

from flask import Flask, redirect, render_template_string, url_for
from waitress import serve

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
app = Flask(__name__)


@dataclass(frozen=True)
class Tool:
    key: str
    title: str
    bat: str
    purpose: str
    category: str
    note: str = ""


TOOLS = [
    Tool("concept-cluster", "Concept Cluster Lab", "run_concept_cluster_lab.bat", "安定した数値経路の重なりから、概念のまとまりを観測します。", "概念・まとまり", "Reflectionの観測履歴が3回以上必要です。"),
    Tool("concept-observer", "Concept Observer", "run_concept_observer.bat", "入力ごとの活動経路と、似た活動パターンを観測します。", "概念・まとまり"),
    Tool("concept-lab", "Concept Lab", "run_concept_lab.bat", "概念候補の形成過程を実験します。", "概念・まとまり"),
    Tool("experience-bridge", "Experience Bridge Observer", "run_experience_bridge_observer.bat", "異なる経験群を結ぶ橋・中継経路を観測します。", "関係・橋"),
    Tool("branch-observer-v02", "Branch Observer v0.2", "run_branch_observer_v02.bat", "活動の分岐、収束、迂回の構造を観測します。", "分岐・経路"),
    Tool("branch-observer", "Branch Observer", "run_branch_observer.bat", "旧版の分岐観測ツールです。比較用に残します。", "分岐・経路", "通常はv0.2を優先してください。"),
    Tool("reflection", "Reflection Lab", "run_reflection_lab.bat", "TraceからReflectionが何を再入力し、何が残ったかを観測します。", "Reflection"),
    Tool("route-choice", "Route Choice Learning Lab", "run_route_choice_lab.bat", "候補選択におけるCoreとFeedbackの寄与を観測します。", "教育・選択"),
    Tool("probe", "Probe Lab", "run_probe_lab.bat", "固定刺激を流し、Core内部の活動変化を測定します。", "基礎観測"),
]


def table_count(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path, timeout=2) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0])
    except sqlite3.Error:
        return -1


def data_status() -> list[dict]:
    names = [
        "pattern_candidates.db",
        "memory.db",
        "spherebrain.db",
        "trace.db",
        "reflection.db",
    ]
    result = []
    for name in names:
        path = DATA_DIR / name
        count = table_count(path)
        result.append(
            {
                "name": name,
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "tables": count,
            }
        )
    return result


def tool_rows() -> list[dict]:
    rows = []
    for tool in TOOLS:
        path = BASE / tool.bat
        rows.append({
            "tool": tool,
            "exists": path.exists(),
            "path": str(path),
        })
    return rows


@app.get("/")
def index():
    rows = tool_rows()
    return render_template_string(
        PAGE,
        rows=rows,
        data=data_status(),
        available=sum(1 for row in rows if row["exists"]),
        total=len(rows),
    )


@app.post("/launch/<key>")
def launch(key: str):
    tool = next((item for item in TOOLS if item.key == key), None)
    if tool is None:
        return "Unknown tool", 404
    bat_path = BASE / tool.bat
    if not bat_path.exists():
        return f"見つかりません: {tool.bat}", 404

    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(
        ["cmd", "/c", str(bat_path)],
        cwd=str(BASE),
        creationflags=creationflags,
    )
    return redirect(url_for("index"))


PAGE = r"""
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Observation Hub</title>
<style>
:root{--bg:#07111f;--panel:#0e1d31;--panel2:#122844;--line:#284a70;--text:#edf5ff;--muted:#92abc7;--cyan:#66d8ff;--green:#77e49e;--orange:#ff9b52;--red:#ff7f89}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,rgba(58,124,184,.22),transparent 36%),var(--bg);color:var(--text);font-family:Inter,"Yu Gothic UI",system-ui,sans-serif}.wrap{max-width:1420px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line);background:rgba(7,17,31,.8);position:sticky;top:0;z-index:5;backdrop-filter:blur(12px)}h1{margin:0;font-size:30px}h2{margin:0 0 14px}.lead,.muted{color:var(--muted)}.summary{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:20px}.panel{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:18px;padding:20px}.value{font-size:34px;font-weight:850}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--cyan)}.tools{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.card{background:#091827;border:1px solid var(--line);border-radius:16px;padding:18px;display:flex;flex-direction:column;min-height:220px}.card h3{margin:5px 0 10px;font-size:21px}.category{color:var(--cyan);font-size:12px;letter-spacing:.1em}.purpose{color:#d5e5f8;line-height:1.7;flex:1}.note{font-size:13px;color:var(--muted);border-left:3px solid var(--orange);padding-left:10px;margin:8px 0 14px}.actions{display:flex;gap:10px;align-items:center}.launch{border:0;border-radius:10px;background:linear-gradient(135deg,#ed6b30,var(--orange));color:white;padding:11px 18px;font-weight:800;cursor:pointer}.launch:disabled{background:#33455a;color:#8da0b5;cursor:not-allowed}.status{font-size:13px}.ok{color:var(--green)}.ng{color:var(--red)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid rgba(40,74,112,.65)}th{color:var(--muted);font-size:12px}.guide{line-height:1.8;color:#d5e5f8}.guide strong{color:var(--cyan)}@media(max-width:900px){.summary,.tools{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="wrap"><h1>SphereBrain Observation Hub</h1><div class="lead">以前の観測ツールを、ここから目に見える形で開くための入口</div></div></header>
<main class="wrap">
<section class="summary">
<div class="panel"><div class="eyebrow">Available Tools</div><div class="value">{{available}} / {{total}}</div><div class="muted">緑色のツールは現在のフォルダから起動できます。</div></div>
<div class="panel"><div class="eyebrow">How to use</div><div class="guide"><strong>1.</strong> この画面を起動したままにする<br><strong>2.</strong> 見たい観測ツールの「起動」を押す<br><strong>3.</strong> 別画面で開いたツールから解析する</div></div>
</section>

<section class="panel" style="margin-top:18px"><h2>観測ツール</h2><div class="tools">
{% for row in rows %}
<article class="card">
<div class="category">{{row.tool.category}}</div>
<h3>{{row.tool.title}}</h3>
<div class="purpose">{{row.tool.purpose}}</div>
{% if row.tool.note %}<div class="note">{{row.tool.note}}</div>{% endif %}
<div class="actions">
<form method="post" action="{{url_for('launch', key=row.tool.key)}}"><button class="launch" {% if not row.exists %}disabled{% endif %}>起動</button></form>
<span class="status {{'ok' if row.exists else 'ng'}}">{{'利用可能' if row.exists else 'BATが見つかりません'}}</span>
</div>
</article>
{% endfor %}
</div></section>

<section class="panel" style="margin-top:18px"><h2>データファイル確認</h2>
<table><thead><tr><th>ファイル</th><th>状態</th><th>サイズ</th><th>テーブル数</th></tr></thead><tbody>
{% for item in data %}<tr><td>{{item.name}}</td><td class="{{'ok' if item.exists else 'ng'}}">{{'存在' if item.exists else 'なし'}}</td><td>{{item.size}}</td><td>{{item.tables if item.tables is not none else '-'}}</td></tr>{% endfor %}
</tbody></table>
<div class="note" style="margin-top:15px">データを初期化した直後は、ツール自体が起動できても「観測履歴が足りない」と表示されることがあります。その場合は、Probe・Reflectionなど必要な前段観測を数回行ってから再解析します。</div>
</section>
</main>
</body></html>
"""


if __name__ == "__main__":
    url = "http://127.0.0.1:5088"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    serve(app, host="127.0.0.1", port=5088, threads=8)
