from __future__ import annotations

from collections import Counter, defaultdict, deque
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
    node_overlap: float = 0.50
    experience_overlap: float = 0.60
    minimum_patterns: int = 2


def initialize_db() -> None:
    PATTERN_DB_FILE.parent.mkdir(exist_ok=True)
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS concept_cluster_runs (
                cluster_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_reflection_run_id INTEGER,
                parameters_json TEXT NOT NULL,
                cluster_count INTEGER NOT NULL,
                pattern_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_clusters (
                cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_key TEXT NOT NULL UNIQUE,
                source_reflection_run_id INTEGER NOT NULL,
                member_count INTEGER NOT NULL,
                core_nodes_json TEXT NOT NULL,
                all_nodes_json TEXT NOT NULL,
                evidence_texts_json TEXT NOT NULL,
                average_score REAL NOT NULL,
                average_selectivity REAL NOT NULL,
                average_stability REAL NOT NULL,
                supporting_runs INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS concept_cluster_members (
                cluster_key TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                route_label TEXT NOT NULL,
                kind TEXT NOT NULL,
                PRIMARY KEY (cluster_key, pattern_id)
            );
            """
        )


def _tables_exist(conn: sqlite3.Connection) -> bool:
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('reflection_runs','reflection_pattern_snapshots')"
        )
    }
    return names == {"reflection_runs", "reflection_pattern_snapshots"}


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _nodes(snapshot: dict) -> set[int]:
    return {int(value) for value in json.loads(snapshot["pattern_json"] or "[]")}


def _texts(snapshot: dict) -> set[str]:
    return {str(value) for value in json.loads(snapshot["target_texts"] or "[]")}


def load_stable_patterns(thresholds: Thresholds) -> tuple[dict[str, dict], int | None]:
    initialize_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        if not _tables_exist(conn):
            return {}, None
        latest_run = conn.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()[0]
        if latest_run is None:
            return {}, None
        run_ids = [
            row[0]
            for row in conn.execute(
                "SELECT run_id FROM reflection_runs WHERE run_id <= ? ORDER BY run_id DESC LIMIT ?",
                (latest_run, thresholds.minimum_runs),
            )
        ]
        if len(run_ids) < thresholds.minimum_runs:
            return {}, latest_run
        placeholders = ",".join("?" for _ in run_ids)
        rows = conn.execute(
            f"""
            SELECT * FROM reflection_pattern_snapshots
            WHERE run_id IN ({placeholders}) AND classification='concept candidate'
            ORDER BY pattern_id, run_id DESC
            """,
            run_ids,
        ).fetchall()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["pattern_id"]].append(dict(row))

    stable: dict[str, dict] = {}
    required = set(run_ids)
    for pattern_id, snapshots in grouped.items():
        if {item["run_id"] for item in snapshots} != required:
            continue
        latest = max(snapshots, key=lambda item: item["run_id"])
        latest["nodes"] = _nodes(latest)
        latest["texts"] = _texts(latest)
        latest["supporting_runs"] = len(snapshots)
        stable[pattern_id] = latest
    return stable, latest_run


def relation_is_stable(left_id: str, right_id: str, run_ids: list[int], thresholds: Thresholds, conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        f"""
        SELECT * FROM reflection_pattern_snapshots
        WHERE pattern_id IN (?, ?) AND run_id IN ({','.join('?' for _ in run_ids)})
          AND classification='concept candidate'
        ORDER BY run_id
        """,
        [left_id, right_id, *run_ids],
    ).fetchall()
    by_run: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        item = dict(row)
        by_run[item["run_id"]][item["pattern_id"]] = item
    for run_id in run_ids:
        pair = by_run.get(run_id, {})
        if left_id not in pair or right_id not in pair:
            return False
        left, right = pair[left_id], pair[right_id]
        node_similarity = _jaccard(_nodes(left), _nodes(right))
        experience_similarity = _jaccard(_texts(left), _texts(right))
        if node_similarity < thresholds.node_overlap and experience_similarity < thresholds.experience_overlap:
            return False
    return True


def build_clusters(thresholds: Thresholds) -> tuple[list[dict], int | None]:
    patterns, latest_run = load_stable_patterns(thresholds)
    if latest_run is None or not patterns:
        return [], latest_run

    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        run_ids = [
            row[0]
            for row in conn.execute(
                "SELECT run_id FROM reflection_runs WHERE run_id <= ? ORDER BY run_id DESC LIMIT ?",
                (latest_run, thresholds.minimum_runs),
            )
        ]
        adjacency: dict[str, set[str]] = {pattern_id: set() for pattern_id in patterns}
        ids = sorted(patterns)
        for index, left_id in enumerate(ids):
            left = patterns[left_id]
            for right_id in ids[index + 1:]:
                right = patterns[right_id]
                node_similarity = _jaccard(left["nodes"], right["nodes"])
                experience_similarity = _jaccard(left["texts"], right["texts"])
                if node_similarity < thresholds.node_overlap and experience_similarity < thresholds.experience_overlap:
                    continue
                if relation_is_stable(left_id, right_id, run_ids, thresholds, conn):
                    adjacency[left_id].add(right_id)
                    adjacency[right_id].add(left_id)

    visited: set[str] = set()
    clusters: list[dict] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        members: list[str] = []
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(members) < thresholds.minimum_patterns:
            continue

        snapshots = [patterns[pattern_id] for pattern_id in members]
        node_counts: Counter[int] = Counter()
        evidence_counts: Counter[str] = Counter()
        for snapshot in snapshots:
            node_counts.update(snapshot["nodes"])
            evidence_counts.update(snapshot["texts"])
        core_threshold = max(2, round(len(snapshots) * 0.60))
        core_nodes = sorted(node for node, count in node_counts.items() if count >= core_threshold)
        all_nodes = sorted(node_counts)
        evidence = [text for text, _ in evidence_counts.most_common(12)]
        cluster_key = "|".join(sorted(members))
        clusters.append(
            {
                "cluster_key": cluster_key,
                "members": sorted(snapshots, key=lambda item: (-item["score"], item["pattern_id"])),
                "member_count": len(snapshots),
                "core_nodes": core_nodes,
                "all_nodes": all_nodes,
                "evidence_texts": evidence,
                "average_score": round(sum(item["score"] for item in snapshots) / len(snapshots), 1),
                "average_selectivity": round(sum(item["selectivity"] for item in snapshots) / len(snapshots), 1),
                "average_stability": round(sum(item["stability"] for item in snapshots) / len(snapshots), 1),
                "supporting_runs": thresholds.minimum_runs,
            }
        )

    clusters.sort(key=lambda item: (-item["member_count"], -item["average_score"], item["cluster_key"]))
    return clusters, latest_run


def save_clusters(clusters: list[dict], latest_run: int | None, thresholds: Thresholds) -> int:
    initialize_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        for cluster in clusters:
            conn.execute(
                """
                INSERT INTO concept_clusters(
                    cluster_key, source_reflection_run_id, member_count, core_nodes_json,
                    all_nodes_json, evidence_texts_json, average_score, average_selectivity,
                    average_stability, supporting_runs, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(cluster_key) DO UPDATE SET
                    source_reflection_run_id=excluded.source_reflection_run_id,
                    member_count=excluded.member_count,
                    core_nodes_json=excluded.core_nodes_json,
                    all_nodes_json=excluded.all_nodes_json,
                    evidence_texts_json=excluded.evidence_texts_json,
                    average_score=excluded.average_score,
                    average_selectivity=excluded.average_selectivity,
                    average_stability=excluded.average_stability,
                    supporting_runs=excluded.supporting_runs,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    cluster["cluster_key"], latest_run, cluster["member_count"],
                    json.dumps(cluster["core_nodes"]), json.dumps(cluster["all_nodes"]),
                    json.dumps(cluster["evidence_texts"], ensure_ascii=False),
                    cluster["average_score"], cluster["average_selectivity"],
                    cluster["average_stability"], cluster["supporting_runs"],
                ),
            )
            conn.execute("DELETE FROM concept_cluster_members WHERE cluster_key=?", (cluster["cluster_key"],))
            for member in cluster["members"]:
                conn.execute(
                    "INSERT INTO concept_cluster_members(cluster_key, pattern_id, route_label, kind) VALUES(?,?,?,?)",
                    (cluster["cluster_key"], member["pattern_id"], member["label"], member["kind"]),
                )
        conn.execute(
            "INSERT INTO concept_cluster_runs(source_reflection_run_id, parameters_json, cluster_count, pattern_count) VALUES(?,?,?,?)",
            (
                latest_run,
                json.dumps(thresholds.__dict__, ensure_ascii=False),
                len(clusters),
                sum(cluster["member_count"] for cluster in clusters),
            ),
        )
    return len(clusters)


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Concept Cluster Lab v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1420px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}h2{margin:8px 0 14px}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.controls{display:grid;grid-template-columns:repeat(4,1fr) auto;gap:12px;align-items:end}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat,.cluster{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.cluster{margin:12px 0}.head{display:flex;justify-content:space-between;gap:12px}.metrics{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.member{padding:8px 10px;margin:6px 0;border-radius:10px;border:1px solid rgba(36,70,109,.7)}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}@media(max-width:1000px){.controls,.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:650px){.controls,.stats{grid-template-columns:1fr}.head{display:block}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Concept Cluster Lab v0.1</h1><p>安定した活動パターンを束ね、概念がネットワークかどうかを観測する</p></div></header><main class="wrap">
<section class="card"><form method="get"><div class="controls">
<div><div class="eyebrow">Minimum Runs</div><input type="number" name="runs" min="2" value="{{t.minimum_runs}}"></div>
<div><div class="eyebrow">Node Overlap %</div><input type="number" name="nodes" min="0" max="100" value="{{(t.node_overlap*100)|int}}"></div>
<div><div class="eyebrow">Experience Overlap %</div><input type="number" name="texts" min="0" max="100" value="{{(t.experience_overlap*100)|int}}"></div>
<div><div class="eyebrow">Minimum Patterns</div><input type="number" name="patterns" min="2" value="{{t.minimum_patterns}}"></div>
<button>クラスターを解析</button></div></form>
<p class="note">文章の単語一致は使いません。Reflectionが活動分布から付けた経験群IDと、数値経路の重なりだけを使います。Coreとmemory.dbは変更しません。</p></section>
<section class="stats card"><div class="stat"><div class="eyebrow">Reflection Run</div><div class="value">#{{latest or '-'}}</div></div><div class="stat"><div class="eyebrow">Clusters</div><div class="value">{{clusters|length}}</div></div><div class="stat"><div class="eyebrow">Bundled Patterns</div><div class="value">{{pattern_total}}</div></div><div class="stat"><div class="eyebrow">Largest Cluster</div><div class="value">{{largest}}</div></div></section>
<section class="card"><form method="post" action="{{url_for('save')}}"><input type="hidden" name="runs" value="{{t.minimum_runs}}"><input type="hidden" name="nodes" value="{{(t.node_overlap*100)|int}}"><input type="hidden" name="texts" value="{{(t.experience_overlap*100)|int}}"><input type="hidden" name="patterns" value="{{t.minimum_patterns}}"><button>現在のクラスターを保存</button></form></section>
<section class="card"><div class="eyebrow">Concept Network Candidates</div><h2>概念クラスター</h2>
{% for c in clusters %}<div class="cluster"><div class="head"><div><b class="eyebrow">Cluster {{loop.index}}</b><div class="metrics"><span>{{c.member_count}} patterns</span><span>score {{c.average_score}}</span><span>selectivity {{c.average_selectivity}}pt</span><span>stability {{c.average_stability}}%</span><span>{{c.supporting_runs}} runs</span></div></div><b>{{c.core_nodes|length}} core nodes</b></div>
<h3>中心ノード</h3>{% for n in c.core_nodes %}<span class="pill">{{n}}</span>{% else %}<span class="muted">共通核はまだ弱い</span>{% endfor %}
<h3>証拠として現れた経験</h3>{% for text in c.evidence_texts %}<span class="pill">{{text}}</span>{% endfor %}
<h3>構成Pattern</h3>{% for m in c.members[:20] %}<div class="member"><code>{{m.label}}</code> <span class="muted">{{m.kind}} / score {{m.score}}</span></div>{% endfor %}</div>{% else %}<p class="muted">現在の条件で安定したクラスターはありません。</p>{% endfor %}</section>
</main></body></html>
"""


def thresholds_from_request() -> Thresholds:
    return Thresholds(
        minimum_runs=max(2, int(request.values.get("runs", 3))),
        node_overlap=min(1.0, max(0.0, float(request.values.get("nodes", 50)) / 100.0)),
        experience_overlap=min(1.0, max(0.0, float(request.values.get("texts", 60)) / 100.0)),
        minimum_patterns=max(2, int(request.values.get("patterns", 2))),
    )


@app.route("/")
def index():
    thresholds = thresholds_from_request()
    clusters, latest = build_clusters(thresholds)
    return render_template_string(
        PAGE,
        t=thresholds,
        clusters=clusters,
        latest=latest,
        pattern_total=sum(cluster["member_count"] for cluster in clusters),
        largest=max((cluster["member_count"] for cluster in clusters), default=0),
    )


@app.post("/save")
def save():
    thresholds = thresholds_from_request()
    clusters, latest = build_clusters(thresholds)
    save_clusters(clusters, latest, thresholds)
    return redirect(url_for("index", runs=thresholds.minimum_runs, nodes=int(thresholds.node_overlap * 100), texts=int(thresholds.experience_overlap * 100), patterns=thresholds.minimum_patterns))


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5055")
    print("SphereBrain Concept Cluster Lab v0.1: http://127.0.0.1:5055")
    serve(app, host="127.0.0.1", port=5055, threads=4)
