from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, redirect, render_template_string, request, url_for
from waitress import serve

BASE = Path(__file__).resolve().parent
PATTERN_DB_FILE = BASE / "data" / "pattern_candidates.db"
app = Flask(__name__)


@dataclass(frozen=True)
class Thresholds:
    minimum_runs: int = 3
    minimum_experience_growth: int = 100
    minimum_score: float = 75.0
    minimum_selectivity: float = 20.0
    minimum_stability: float = 80.0


def initialize_db() -> None:
    PATTERN_DB_FILE.parent.mkdir(exist_ok=True)
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS observed_concepts (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                pattern_json TEXT NOT NULL,
                route_label TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'observed',
                first_run_id INTEGER NOT NULL,
                last_run_id INTEGER NOT NULL,
                supporting_runs INTEGER NOT NULL,
                experience_growth INTEGER NOT NULL,
                best_score REAL NOT NULL,
                best_selectivity REAL NOT NULL,
                best_stability REAL NOT NULL,
                evidence_texts TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS concept_observer_runs (
                observer_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_reflection_run_id INTEGER,
                parameters_json TEXT NOT NULL,
                qualified_count INTEGER NOT NULL,
                saved_count INTEGER NOT NULL
            );
            """
        )


def _reflection_tables_exist(conn: sqlite3.Connection) -> bool:
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('reflection_runs','reflection_pattern_snapshots')"
        )
    }
    return names == {"reflection_runs", "reflection_pattern_snapshots"}


def find_candidates(thresholds: Thresholds) -> tuple[list[dict], int | None]:
    initialize_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        if not _reflection_tables_exist(conn):
            return [], None

        latest_run = conn.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()[0]
        if latest_run is None:
            return [], None

        rows = conn.execute(
            """
            SELECT s.*, r.total_experiences
            FROM reflection_pattern_snapshots s
            JOIN reflection_runs r ON r.run_id = s.run_id
            WHERE s.classification = 'concept candidate'
              AND s.score >= ?
              AND s.selectivity >= ?
              AND s.stability >= ?
            ORDER BY s.pattern_id, s.run_id DESC
            """,
            (thresholds.minimum_score, thresholds.minimum_selectivity, thresholds.minimum_stability),
        ).fetchall()

    by_pattern: dict[str, list[dict]] = {}
    for row in rows:
        by_pattern.setdefault(row["pattern_id"], []).append(dict(row))

    qualified: list[dict] = []
    for pattern_id, snapshots in by_pattern.items():
        snapshots.sort(key=lambda item: item["run_id"], reverse=True)
        if snapshots[0]["run_id"] != latest_run:
            continue

        consecutive = [snapshots[0]]
        expected = latest_run - 1
        for snapshot in snapshots[1:]:
            if snapshot["run_id"] != expected:
                break
            consecutive.append(snapshot)
            expected -= 1

        if len(consecutive) < thresholds.minimum_runs:
            continue

        window = consecutive[: thresholds.minimum_runs]
        newest = window[0]
        oldest = window[-1]
        growth = newest["total_experiences"] - oldest["total_experiences"]
        if growth < thresholds.minimum_experience_growth:
            continue

        evidence: list[str] = []
        for snapshot in window:
            for text in json.loads(snapshot["target_texts"] or "[]"):
                if text not in evidence:
                    evidence.append(text)

        qualified.append(
            {
                "pattern_id": pattern_id,
                "kind": newest["kind"],
                "pattern_json": newest["pattern_json"],
                "route_label": newest["label"],
                "first_run_id": oldest["run_id"],
                "last_run_id": newest["run_id"],
                "supporting_runs": len(window),
                "experience_growth": growth,
                "best_score": max(item["score"] for item in window),
                "best_selectivity": max(item["selectivity"] for item in window),
                "best_stability": max(item["stability"] for item in window),
                "evidence_texts": evidence[:12],
            }
        )

    qualified.sort(
        key=lambda item: (
            -item["best_score"],
            -item["best_stability"],
            -item["best_selectivity"],
            item["pattern_id"],
        )
    )
    return qualified, latest_run


def save_candidates(candidates: list[dict], latest_run: int | None, thresholds: Thresholds) -> int:
    initialize_db()
    saved = 0
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        for item in candidates:
            before = conn.total_changes
            conn.execute(
                """
                INSERT INTO observed_concepts (
                    pattern_id, kind, pattern_json, route_label, first_run_id, last_run_id,
                    supporting_runs, experience_growth, best_score, best_selectivity,
                    best_stability, evidence_texts, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    kind=excluded.kind,
                    pattern_json=excluded.pattern_json,
                    route_label=excluded.route_label,
                    first_run_id=MIN(observed_concepts.first_run_id, excluded.first_run_id),
                    last_run_id=excluded.last_run_id,
                    supporting_runs=MAX(observed_concepts.supporting_runs, excluded.supporting_runs),
                    experience_growth=MAX(observed_concepts.experience_growth, excluded.experience_growth),
                    best_score=MAX(observed_concepts.best_score, excluded.best_score),
                    best_selectivity=MAX(observed_concepts.best_selectivity, excluded.best_selectivity),
                    best_stability=MAX(observed_concepts.best_stability, excluded.best_stability),
                    evidence_texts=excluded.evidence_texts,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    item["pattern_id"], item["kind"], item["pattern_json"], item["route_label"],
                    item["first_run_id"], item["last_run_id"], item["supporting_runs"],
                    item["experience_growth"], item["best_score"], item["best_selectivity"],
                    item["best_stability"], json.dumps(item["evidence_texts"], ensure_ascii=False),
                ),
            )
            if conn.total_changes > before:
                saved += 1

        conn.execute(
            """
            INSERT INTO concept_observer_runs(
                source_reflection_run_id, parameters_json, qualified_count, saved_count
            ) VALUES (?, ?, ?, ?)
            """,
            (
                latest_run,
                json.dumps(thresholds.__dict__, ensure_ascii=False),
                len(candidates),
                saved,
            ),
        )
    return saved


def load_concepts() -> list[dict]:
    initialize_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM observed_concepts ORDER BY concept_id ASC"
        ).fetchall()
    concepts = []
    for row in rows:
        item = dict(row)
        item["display_id"] = f"C-{item['concept_id']:04d}"
        item["evidence_texts"] = json.loads(item["evidence_texts"] or "[]")
        concepts.append(item)
    return concepts


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Concept Observer v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1420px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}h2{margin:8px 0 14px}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.controls{display:grid;grid-template-columns:repeat(5,1fr) auto;gap:12px;align-items:end}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat,.concept{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.concept{margin:10px 0}.head{display:flex;justify-content:space-between;gap:14px}.id{color:var(--green);font-weight:800}.metrics{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-top:9px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}@media(max-width:1050px){.controls,.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.controls,.stats{grid-template-columns:1fr}.head{display:block}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Concept Observer v0.1</h1><p>Reflectionで長期安定した活動を、言葉ではなく経路として概念登録する</p></div></header><main class="wrap">
<section class="card"><form method="get"><div class="controls">
<div><div class="eyebrow">Minimum Runs</div><input type="number" name="runs" min="2" value="{{t.minimum_runs}}"></div>
<div><div class="eyebrow">Experience Growth</div><input type="number" name="growth" min="1" value="{{t.minimum_experience_growth}}"></div>
<div><div class="eyebrow">Minimum Score</div><input type="number" name="score" min="0" max="100" value="{{t.minimum_score}}"></div>
<div><div class="eyebrow">Minimum Selectivity</div><input type="number" name="selectivity" min="0" max="100" value="{{t.minimum_selectivity}}"></div>
<div><div class="eyebrow">Minimum Stability</div><input type="number" name="stability" min="0" max="100" value="{{t.minimum_stability}}"></div>
<button>候補を再計算</button></div></form>
<p class="note">同じデータで画面を更新しただけでは概念になりません。複数Runに連続して現れ、その間に経験数が増えた活動だけを候補にします。Coreとmemory.dbには一切書き戻しません。</p></section>
<section class="stats card"><div class="stat"><div class="eyebrow">Latest Reflection Run</div><div class="value">#{{latest or '-'}}</div></div><div class="stat"><div class="eyebrow">Qualified Now</div><div class="value">{{candidates|length}}</div></div><div class="stat"><div class="eyebrow">Observed Concepts</div><div class="value">{{concepts|length}}</div></div><div class="stat"><div class="eyebrow">Mode</div><div class="value">A</div><div class="muted">観測のみ</div></div></section>
<section class="card"><div class="head"><div><div class="eyebrow">Qualified Candidates</div><h2>今回、概念として登録できる活動</h2></div>{% if candidates %}<form method="post" action="{{url_for('observe')}}"><input type="hidden" name="runs" value="{{t.minimum_runs}}"><input type="hidden" name="growth" value="{{t.minimum_experience_growth}}"><input type="hidden" name="score" value="{{t.minimum_score}}"><input type="hidden" name="selectivity" value="{{t.minimum_selectivity}}"><input type="hidden" name="stability" value="{{t.minimum_stability}}"><button>観測概念として保存</button></form>{% endif %}</div>
{% for item in candidates %}<div class="concept"><div class="head"><code>{{item.route_label}}</code><span class="id">candidate</span></div><div class="metrics"><span>{{item.kind}}</span><span>Run #{{item.first_run_id}}〜#{{item.last_run_id}}</span><span>{{item.supporting_runs}}連続Run</span><span>経験 +{{item.experience_growth}}</span><span>総合 {{item.best_score}}</span><span>選択差 {{item.best_selectivity}} pt</span><span>安定度 {{item.best_stability}}%</span></div><div>{% for text in item.evidence_texts %}<span class="pill">{{text}}</span>{% endfor %}</div></div>{% else %}<p class="muted">現在の条件を満たす活動はまだありません。</p>{% endfor %}</section>
<section class="card"><div class="eyebrow">Concept Registry</div><h2>観測済み概念</h2>
{% for item in concepts %}<div class="concept"><div class="head"><div><span class="id">{{item.display_id}}</span> <code>{{item.route_label}}</code></div><span class="muted">{{item.status}}</span></div><div class="metrics"><span>{{item.kind}}</span><span>Run #{{item.first_run_id}}〜#{{item.last_run_id}}</span><span>経験増加 {{item.experience_growth}}</span><span>総合 {{item.best_score}}</span><span>選択差 {{item.best_selectivity}} pt</span><span>安定度 {{item.best_stability}}%</span></div><div>{% for text in item.evidence_texts %}<span class="pill">{{text}}</span>{% endfor %}</div></div>{% else %}<p class="muted">まだ保存された概念はありません。</p>{% endfor %}</section>
</main></body></html>
"""


def read_thresholds(values) -> Thresholds:
    return Thresholds(
        minimum_runs=max(2, int(values.get("runs", 3))),
        minimum_experience_growth=max(1, int(values.get("growth", 100))),
        minimum_score=min(100.0, max(0.0, float(values.get("score", 75)))),
        minimum_selectivity=min(100.0, max(0.0, float(values.get("selectivity", 20)))),
        minimum_stability=min(100.0, max(0.0, float(values.get("stability", 80)))),
    )


@app.route("/")
def index():
    thresholds = read_thresholds(request.args)
    candidates, latest = find_candidates(thresholds)
    return render_template_string(
        PAGE,
        t=thresholds,
        candidates=candidates,
        concepts=load_concepts(),
        latest=latest,
    )


@app.post("/observe")
def observe():
    thresholds = read_thresholds(request.form)
    candidates, latest = find_candidates(thresholds)
    save_candidates(candidates, latest, thresholds)
    return redirect(
        url_for(
            "index",
            runs=thresholds.minimum_runs,
            growth=thresholds.minimum_experience_growth,
            score=thresholds.minimum_score,
            selectivity=thresholds.minimum_selectivity,
            stability=thresholds.minimum_stability,
        )
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5054")
    print("SphereBrain Concept Observer v0.1: http://127.0.0.1:5054")
    serve(app, host="127.0.0.1", port=5054, threads=4)
