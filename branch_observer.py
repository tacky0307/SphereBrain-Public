from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

BASE = Path(__file__).resolve().parent
SOURCE_DB = BASE / "data" / "pattern_candidates.db"
OBSERVER_DB = BASE / "data" / "branch_observer.db"
app = Flask(__name__)


@dataclass(frozen=True)
class Settings:
    runs: int = 3
    minimum_support: int = 6
    minimum_destination_support: int = 2
    minimum_destinations: int = 2
    minimum_selectivity: float = 0.15


def _json_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def initialize_observer_db() -> None:
    OBSERVER_DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(OBSERVER_DB, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS branch_observer_runs (
                observer_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_reflection_run_id INTEGER,
                settings_json TEXT NOT NULL,
                branch_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observed_branches (
                observer_run_id INTEGER NOT NULL,
                source_node INTEGER NOT NULL,
                support INTEGER NOT NULL,
                destination_count INTEGER NOT NULL,
                entropy REAL NOT NULL,
                dominant_ratio REAL NOT NULL,
                selectivity REAL NOT NULL,
                branch_score REAL NOT NULL,
                destinations_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                PRIMARY KEY(observer_run_id, source_node)
            );
            """
        )


def load_snapshots(settings: Settings) -> tuple[list[dict], list[int], str | None]:
    if not SOURCE_DB.exists():
        return [], [], f"source database not found: {SOURCE_DB}"

    with sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, "reflection_runs") or not _table_exists(
            conn, "reflection_pattern_snapshots"
        ):
            return [], [], "Reflection snapshots have not been created yet."

        run_ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT run_id FROM reflection_runs ORDER BY run_id DESC LIMIT ?",
                (settings.runs,),
            )
        ]
        if len(run_ids) < settings.runs:
            return [], run_ids, f"At least {settings.runs} Reflection runs are required."

        placeholders = ",".join("?" for _ in run_ids)
        rows = conn.execute(
            f"""
            SELECT *
            FROM reflection_pattern_snapshots
            WHERE run_id IN ({placeholders})
            ORDER BY run_id, pattern_id
            """,
            run_ids,
        ).fetchall()

    snapshots: list[dict] = []
    for row in rows:
        item = dict(row)
        route = [int(node) for node in _json_list(item.get("pattern_json"))]
        if len(route) < 2:
            continue
        texts = [str(text) for text in _json_list(item.get("target_texts")) if str(text)]
        snapshots.append(
            {
                "run_id": int(item["run_id"]),
                "pattern_id": str(item.get("pattern_id", "")),
                "label": str(item.get("label", " → ".join(map(str, route)))),
                "classification": str(item.get("classification", "")),
                "route": route,
                "texts": texts,
            }
        )
    return snapshots, sorted(run_ids), None


def _normalized_entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((count / total) * log2(count / total) for count in counts.values())
    return entropy / log2(len(counts))


def analyze_branches(settings: Settings) -> tuple[list[dict], list[int], str | None]:
    snapshots, run_ids, error = load_snapshots(settings)
    if error:
        return [], run_ids, error

    # Duplicate overlapping Reflection fragments can contain the same transition.
    # Count each (run, pattern, source, destination, evidence) only once.
    seen_edges: set[tuple] = set()
    by_source: dict[int, dict] = defaultdict(
        lambda: {
            "destinations": Counter(),
            "runs": defaultdict(Counter),
            "text_destinations": defaultdict(Counter),
            "examples": defaultdict(list),
            "patterns": set(),
        }
    )

    for snapshot in snapshots:
        route = snapshot["route"]
        evidence_values = snapshot["texts"] or ["(unlabelled experience)"]
        for source, destination in zip(route, route[1:]):
            base_key = (
                snapshot["run_id"],
                snapshot["pattern_id"],
                source,
                destination,
            )
            if base_key not in seen_edges:
                seen_edges.add(base_key)
                record = by_source[source]
                record["destinations"][destination] += 1
                record["runs"][snapshot["run_id"]][destination] += 1
                record["patterns"].add(snapshot["pattern_id"])
                if len(record["examples"][destination]) < 5:
                    record["examples"][destination].append(snapshot["label"])

            for text in evidence_values:
                evidence_key = (*base_key, text)
                if evidence_key in seen_edges:
                    continue
                seen_edges.add(evidence_key)
                by_source[source]["text_destinations"][text][destination] += 1

    branches: list[dict] = []
    required_runs = set(run_ids)

    for source, record in by_source.items():
        destination_counts: Counter[int] = record["destinations"]
        eligible = Counter(
            {
                destination: count
                for destination, count in destination_counts.items()
                if count >= settings.minimum_destination_support
            }
        )
        if len(eligible) < settings.minimum_destinations:
            continue
        if sum(eligible.values()) < settings.minimum_support:
            continue

        stable_runs = {
            run_id
            for run_id, counts in record["runs"].items()
            if len(
                [
                    destination
                    for destination, count in counts.items()
                    if count >= 1 and destination in eligible
                ]
            )
            >= settings.minimum_destinations
        }
        if stable_runs != required_runs:
            continue

        total = sum(eligible.values())
        dominant_ratio = max(eligible.values()) / total
        entropy = _normalized_entropy(eligible)

        evidence_rows: list[dict] = []
        best_selectivity = 0.0
        all_text_counts: dict[str, Counter[int]] = record["text_destinations"]
        for text, text_counts in all_text_counts.items():
            text_total = sum(text_counts[d] for d in eligible)
            if text_total == 0:
                continue
            other_counts = Counter()
            for other_text, counts in all_text_counts.items():
                if other_text != text:
                    other_counts.update(counts)
            other_total = sum(other_counts[d] for d in eligible)
            for destination in eligible:
                p_text = text_counts[destination] / text_total
                p_other = (
                    other_counts[destination] / other_total if other_total else 0.0
                )
                selectivity = p_text - p_other
                best_selectivity = max(best_selectivity, selectivity)
                if selectivity >= settings.minimum_selectivity:
                    evidence_rows.append(
                        {
                            "text": text,
                            "destination": destination,
                            "probability": round(p_text * 100, 1),
                            "contrast": round(selectivity * 100, 1),
                        }
                    )

        support_factor = min(1.0, total / max(settings.minimum_support * 3, 1))
        branch_score = round(
            100 * (0.35 * entropy + 0.40 * best_selectivity + 0.25 * support_factor),
            1,
        )

        destinations = []
        for destination, count in eligible.most_common():
            destinations.append(
                {
                    "node": destination,
                    "count": count,
                    "ratio": round(count / total * 100, 1),
                    "examples": record["examples"][destination],
                    "runs": {
                        run_id: record["runs"][run_id][destination] for run_id in run_ids
                    },
                }
            )

        evidence_rows.sort(
            key=lambda row: (-row["contrast"], -row["probability"], row["text"])
        )
        branches.append(
            {
                "source": source,
                "support": total,
                "pattern_count": len(record["patterns"]),
                "destination_count": len(eligible),
                "entropy": round(entropy * 100, 1),
                "dominant_ratio": round(dominant_ratio * 100, 1),
                "selectivity": round(best_selectivity * 100, 1),
                "branch_score": branch_score,
                "destinations": destinations,
                "evidence": evidence_rows[:12],
            }
        )

    branches.sort(
        key=lambda item: (
            -item["branch_score"],
            -item["selectivity"],
            -item["support"],
            item["source"],
        )
    )
    return branches, run_ids, None


def save_observation(branches: list[dict], run_ids: list[int], settings: Settings) -> int:
    initialize_observer_db()
    source_run = max(run_ids) if run_ids else None
    with sqlite3.connect(OBSERVER_DB, timeout=30) as conn:
        cursor = conn.execute(
            """
            INSERT INTO branch_observer_runs(
                source_reflection_run_id, settings_json, branch_count
            ) VALUES (?, ?, ?)
            """,
            (
                source_run,
                json.dumps(settings.__dict__, ensure_ascii=False),
                len(branches),
            ),
        )
        observer_run_id = int(cursor.lastrowid)
        for branch in branches:
            conn.execute(
                """
                INSERT INTO observed_branches(
                    observer_run_id, source_node, support, destination_count,
                    entropy, dominant_ratio, selectivity, branch_score,
                    destinations_json, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observer_run_id,
                    branch["source"],
                    branch["support"],
                    branch["destination_count"],
                    branch["entropy"],
                    branch["dominant_ratio"],
                    branch["selectivity"],
                    branch["branch_score"],
                    json.dumps(branch["destinations"], ensure_ascii=False),
                    json.dumps(branch["evidence"], ensure_ascii=False),
                ),
            )
    return observer_run_id


PAGE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Branch Observer v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--orange:#ff9d52;--green:#69e09a;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}
.wrap{max-width:1420px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}h2{margin:7px 0 14px}p,.muted{color:var(--muted)}
.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}
.controls{display:grid;grid-template-columns:repeat(5,1fr) auto;gap:12px;align-items:end}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.stat,.branch{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}
input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}
button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:#fff;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}
.branch{margin:12px 0}.head{display:flex;justify-content:space-between;gap:12px}.metrics{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}
.dest{display:grid;grid-template-columns:110px 1fr 90px;gap:12px;align-items:center;padding:9px 0;border-top:1px solid rgba(36,70,109,.55)}
.bar{height:9px;background:#142942;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.evidence{color:var(--yellow)}
.note{border-left:3px solid var(--cyan);padding-left:12px}.error{border-color:#a64c4c;color:#ffc4c4}
@media(max-width:1050px){.controls{grid-template-columns:repeat(2,1fr)}.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.controls,.stats{grid-template-columns:1fr}.head{display:block}.dest{grid-template-columns:80px 1fr 60px}}
</style>
</head>
<body>
<header><div class="wrap"><h1>SphereBrain Branch Observer v0.1</h1><p>活動経路がどこで分かれ、経験によって行き先が変わるかを観測する</p></div></header>
<main class="wrap">
<section class="card">
<form method="get"><div class="controls">
<div><div class="eyebrow">Reflection Runs</div><input name="runs" type="number" min="2" value="{{s.runs}}"></div>
<div><div class="eyebrow">Minimum Support</div><input name="support" type="number" min="2" value="{{s.minimum_support}}"></div>
<div><div class="eyebrow">Destination Support</div><input name="dest_support" type="number" min="1" value="{{s.minimum_destination_support}}"></div>
<div><div class="eyebrow">Minimum Destinations</div><input name="destinations" type="number" min="2" value="{{s.minimum_destinations}}"></div>
<div><div class="eyebrow">Evidence Selectivity %</div><input name="selectivity" type="number" min="0" max="100" value="{{(s.minimum_selectivity*100)|int}}"></div>
<button>分岐を解析</button>
</div></form>
<p class="note">言葉は分岐を作る条件には使いません。数値経路から分岐を検出し、入力文は「どの経験で行き先が変わったか」を人間が確認する証拠としてだけ表示します。</p>
</section>

{% if error %}<section class="card error">{{error}}</section>{% endif %}

<section class="stats card">
<div class="stat"><div class="eyebrow">Latest Reflection Run</div><div class="value">#{{run_ids|max if run_ids else '-'}}</div></div>
<div class="stat"><div class="eyebrow">Observed Runs</div><div class="value">{{run_ids|length}}</div></div>
<div class="stat"><div class="eyebrow">Stable Branches</div><div class="value">{{branches|length}}</div></div>
<div class="stat"><div class="eyebrow">Best Branch Score</div><div class="value">{{branches[0].branch_score if branches else 0}}</div></div>
</section>

<section class="card">
<div class="head"><div><div class="eyebrow">Hypothesis B-001</div><h2>知性は、経路を通ることだけでなく、経験に応じて経路を選択することで現れる</h2></div>
{% if branches %}<form method="post"><input type="hidden" name="settings" value='{{settings_json}}'><button>今回の観測を保存</button></form>{% endif %}</div>
<p>重要なのは分岐の数だけではありません。異なる経験で異なる行き先が選ばれる「選択性」があるかを確認します。</p>
</section>

<section class="card">
<div class="eyebrow">Branch Points</div>
{% for branch in branches %}
<article class="branch">
<div class="head"><div><h2>Node {{branch.source}}</h2><div class="metrics">
<span>Score {{branch.branch_score}}</span><span>Support {{branch.support}}</span><span>Patterns {{branch.pattern_count}}</span>
<span>Destinations {{branch.destination_count}}</span><span>Entropy {{branch.entropy}}%</span>
<span>Dominant {{branch.dominant_ratio}}%</span><span>Selectivity {{branch.selectivity}}%</span>
</div></div><div class="value">{{branch.source}} → ?</div></div>

<h3>行き先</h3>
{% for d in branch.destinations %}
<div class="dest"><strong>→ {{d.node}}</strong><div><div class="bar"><div class="fill" style="width:{{d.ratio}}%"></div></div><small class="muted">Runs: {% for run,count in d.runs.items() %}#{{run}}={{count}} {% endfor %}</small></div><strong>{{d.ratio}}%</strong></div>
{% endfor %}

{% if branch.evidence %}
<h3>経験による選択の証拠</h3>
{% for e in branch.evidence %}
<span class="pill evidence">{{e.text}} → {{e.destination}}：{{e.probability}}%（差 {{e.contrast}}pt）</span>
{% endfor %}
{% else %}
<p class="muted">分岐はありますが、現在の証拠では経験ごとの行き先の差を確認できません。</p>
{% endif %}
</article>
{% else %}
<p class="muted">現在の条件で、複数Runにわたり安定した分岐は見つかりませんでした。これは「識別がない」と即断する結果ではなく、観測単位や閾値を再検討する材料です。</p>
{% endfor %}
</section>
</main>
</body></html>
"""


def settings_from_request() -> Settings:
    def integer(name: str, default: int, minimum: int) -> int:
        try:
            return max(minimum, int(request.values.get(name, default)))
        except (TypeError, ValueError):
            return default

    try:
        selectivity = float(request.values.get("selectivity", 15)) / 100
    except (TypeError, ValueError):
        selectivity = 0.15
    return Settings(
        runs=integer("runs", 3, 2),
        minimum_support=integer("support", 6, 2),
        minimum_destination_support=integer("dest_support", 2, 1),
        minimum_destinations=integer("destinations", 2, 2),
        minimum_selectivity=min(1.0, max(0.0, selectivity)),
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            raw = json.loads(request.form.get("settings", "{}"))
            settings = Settings(**raw)
        except (TypeError, ValueError):
            settings = Settings()
        branches, run_ids, error = analyze_branches(settings)
        observer_run_id = save_observation(branches, run_ids, settings)
        return render_template_string(
            PAGE,
            s=settings,
            branches=branches,
            run_ids=run_ids,
            error=f"Observation #{observer_run_id} saved to branch_observer.db.",
            settings_json=json.dumps(settings.__dict__),
        )

    settings = settings_from_request()
    branches, run_ids, error = analyze_branches(settings)
    return render_template_string(
        PAGE,
        s=settings,
        branches=branches,
        run_ids=run_ids,
        error=error,
        settings_json=json.dumps(settings.__dict__),
    )


if __name__ == "__main__":
    initialize_observer_db()
    webbrowser.open("http://127.0.0.1:5056")
    serve(app, host="127.0.0.1", port=5056)
