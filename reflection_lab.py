from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, redirect, render_template_string, request, url_for
from waitress import serve

BASE = Path(__file__).resolve().parent
DB_FILE = BASE / "data" / "memory.db"
PATTERN_DB_FILE = BASE / "data" / "pattern_candidates.db"
app = Flask(__name__)


@dataclass(frozen=True)
class PatternKey:
    kind: str
    values: tuple[int, ...]

    @property
    def label(self) -> str:
        if self.kind in {"edge", "path3"}:
            return " → ".join(str(v) for v in self.values)
        return "{" + ", ".join(str(v) for v in self.values) + "}"


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def load_memories(limit: int = 8000) -> list[dict]:
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, kind, input_text, activated_nodes, traversed_edges
            FROM memories
            WHERE COALESCE(input_text, '') <> '' AND kind IN ('trainer', 'input')
            ORDER BY id ASC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    result = []
    for row in rows:
        item = dict(row)
        item["input_text"] = str(item.get("input_text") or "").strip()
        item["activated_nodes"] = [int(v) for v in json.loads(item["activated_nodes"])]
        item["traversed_edges"] = [normalize_edge(v) for v in json.loads(item["traversed_edges"])]
        result.append(item)
    return result


def extract_patterns(memory: dict) -> set[PatternKey]:
    patterns: set[PatternKey] = set()
    edges = list(dict.fromkeys(memory["traversed_edges"]))
    nodes = list(dict.fromkeys(memory["activated_nodes"]))
    for edge in edges:
        patterns.add(PatternKey("edge", edge))
    for index in range(len(nodes) - 2):
        patterns.add(PatternKey("path3", tuple(nodes[index:index + 3])))
    bucketed = sorted({node % 64 for node in nodes})
    for index in range(len(bucketed) - 2):
        patterns.add(PatternKey("nodes3", tuple(bucketed[index:index + 3])))
    return patterns


def pattern_id(pattern: PatternKey) -> str:
    return pattern.kind + ":" + ",".join(str(v) for v in pattern.values)


def initialize_pattern_db() -> None:
    PATTERN_DB_FILE.parent.mkdir(exist_ok=True)
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reflection_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                total_experiences INTEGER NOT NULL,
                distinct_texts INTEGER NOT NULL,
                parameters_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reflection_pattern_snapshots (
                run_id INTEGER NOT NULL,
                pattern_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                pattern_json TEXT NOT NULL,
                label TEXT NOT NULL,
                classification TEXT NOT NULL,
                experience_count INTEGER NOT NULL,
                global_rate REAL NOT NULL,
                target_rate REAL NOT NULL,
                other_rate REAL NOT NULL,
                selectivity REAL NOT NULL,
                stability REAL NOT NULL,
                score REAL NOT NULL,
                target_texts TEXT NOT NULL,
                other_texts TEXT NOT NULL,
                PRIMARY KEY (run_id, pattern_id)
            );
            CREATE TABLE IF NOT EXISTS reflection_pattern_changes (
                from_run_id INTEGER,
                to_run_id INTEGER NOT NULL,
                pattern_id TEXT NOT NULL,
                change_type TEXT NOT NULL,
                previous_classification TEXT,
                current_classification TEXT,
                score_delta REAL NOT NULL,
                selectivity_delta REAL NOT NULL,
                stability_delta REAL NOT NULL,
                PRIMARY KEY (to_run_id, pattern_id, change_type)
            );
            CREATE TABLE IF NOT EXISTS reflection_pins (
                pattern_id TEXT PRIMARY KEY,
                pinned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def _safe_rate(value: int, total: int) -> float:
    return value / total if total else 0.0


def analyze(memories: list[dict], min_experiences: int = 2, min_texts: int = 2,
            common_threshold: float = 0.80, selectivity_threshold: float = 0.20) -> dict:
    total = len(memories)
    text_totals: Counter[str] = Counter(m["input_text"] for m in memories)
    text_count = len(text_totals)
    pattern_count: Counter[PatternKey] = Counter()
    pattern_by_text: dict[PatternKey, Counter[str]] = defaultdict(Counter)
    patterns_per_memory: list[set[PatternKey]] = []

    for memory in memories:
        patterns = extract_patterns(memory)
        patterns_per_memory.append(patterns)
        text = memory["input_text"]
        for pattern in patterns:
            pattern_count[pattern] += 1
            pattern_by_text[pattern][text] += 1

    text_seen: Counter[str] = Counter()
    first_hits: dict[PatternKey, Counter[str]] = defaultdict(Counter)
    second_hits: dict[PatternKey, Counter[str]] = defaultdict(Counter)
    first_sizes: Counter[str] = Counter()
    second_sizes: Counter[str] = Counter()
    for memory, patterns in zip(memories, patterns_per_memory):
        text = memory["input_text"]
        midpoint = max(1, text_totals[text] // 2)
        if text_seen[text] < midpoint:
            first_sizes[text] += 1
            target = first_hits
        else:
            second_sizes[text] += 1
            target = second_hits
        for pattern in patterns:
            target[pattern][text] += 1
        text_seen[text] += 1

    groups = {name: [] for name in ["common skeleton", "concept candidate", "individual memory", "selective pattern", "unclassified"]}
    all_items: list[dict] = []
    for pattern, count in pattern_count.items():
        if count < min_experiences:
            continue
        rates = {text: _safe_rate(pattern_by_text[pattern][text], occurrences) for text, occurrences in text_totals.items()}
        present_texts = [text for text, rate in rates.items() if rate > 0]
        distinct_text_count = len(present_texts)
        global_rate = _safe_rate(count, total)
        target_cutoff = max(0.50, global_rate + 0.15)
        target_texts = [text for text, rate in rates.items() if rate >= target_cutoff]
        if not target_texts and rates:
            peak = max(rates.values())
            target_texts = [text for text, rate in rates.items() if rate >= peak - 0.05 and rate >= 0.35]
        other_texts_all = [text for text in text_totals if text not in target_texts]
        target_experiences = sum(text_totals[t] for t in target_texts)
        target_hits = sum(pattern_by_text[pattern][t] for t in target_texts)
        other_experiences = sum(text_totals[t] for t in other_texts_all)
        other_hits = sum(pattern_by_text[pattern][t] for t in other_texts_all)
        target_rate = _safe_rate(target_hits, target_experiences)
        other_rate = _safe_rate(other_hits, other_experiences)
        selectivity = target_rate - other_rate
        stability_values = []
        for text in target_texts:
            first_rate = _safe_rate(first_hits[pattern][text], first_sizes[text])
            second_rate = _safe_rate(second_hits[pattern][text], second_sizes[text])
            stability_values.append(max(0.0, 1.0 - abs(first_rate - second_rate)))
        stability = sum(stability_values) / len(stability_values) if stability_values else 0.0

        classification = "unclassified"
        if global_rate >= common_threshold and distinct_text_count >= max(2, int(text_count * 0.8)):
            classification = "common skeleton"
        elif distinct_text_count == 1:
            classification = "individual memory"
        elif len(target_texts) >= min_texts and target_rate >= 0.60 and selectivity >= selectivity_threshold and stability >= 0.60 and global_rate < common_threshold:
            classification = "concept candidate"
        elif len(target_texts) >= min_texts and selectivity >= 0.10:
            classification = "selective pattern"

        diversity = min(1.0, len(target_texts) / max(2, min_texts * 2))
        score = max(0.0, selectivity) * 0.45 + target_rate * 0.25 + stability * 0.20 + diversity * 0.10
        if classification == "common skeleton":
            score = global_rate * 0.70 + stability * 0.30
        elif classification == "individual memory":
            score = max(rates.values(), default=0.0) * 0.70 + stability * 0.30

        item = {
            "id": pattern_id(pattern), "label": pattern.label, "kind": pattern.kind,
            "values": pattern.values, "classification": classification, "count": count,
            "global_rate": round(global_rate * 100, 1), "target_rate": round(target_rate * 100, 1),
            "other_rate": round(other_rate * 100, 1), "selectivity": round(selectivity * 100, 1),
            "stability": round(stability * 100, 1), "score": round(score * 100, 1),
            "target_texts": sorted(target_texts, key=lambda t: (-rates[t], t))[:8],
            "other_texts": sorted(other_texts_all, key=lambda t: (rates[t], t))[:8],
        }
        groups[classification].append(item)
        all_items.append(item)

    for items in groups.values():
        items.sort(key=lambda i: (-i["score"], -i["selectivity"], -i["count"], i["id"]))
    return {
        "total": total, "text_count": text_count, "all_items": all_items,
        "common_count": len(groups["common skeleton"]), "concept_count": len(groups["concept candidate"]),
        "individual_count": len(groups["individual memory"]), "selective_count": len(groups["selective pattern"]),
        "groups": {name: items[:80] for name, items in groups.items()},
    }


def save_run(result: dict, parameters: dict) -> tuple[int, int | None]:
    initialize_pattern_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        previous = conn.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()[0]
        cur = conn.execute(
            "INSERT INTO reflection_runs(total_experiences, distinct_texts, parameters_json) VALUES(?,?,?)",
            (result["total"], result["text_count"], json.dumps(parameters, ensure_ascii=False)),
        )
        run_id = int(cur.lastrowid)
        for item in result["all_items"]:
            conn.execute(
                """INSERT INTO reflection_pattern_snapshots(
                run_id,pattern_id,kind,pattern_json,label,classification,experience_count,global_rate,target_rate,
                other_rate,selectivity,stability,score,target_texts,other_texts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, item["id"], item["kind"], json.dumps(item["values"]), item["label"], item["classification"],
                 item["count"], item["global_rate"], item["target_rate"], item["other_rate"], item["selectivity"],
                 item["stability"], item["score"], json.dumps(item["target_texts"], ensure_ascii=False),
                 json.dumps(item["other_texts"], ensure_ascii=False)),
            )
        if previous:
            old_rows = conn.execute("SELECT * FROM reflection_pattern_snapshots WHERE run_id=?", (previous,)).fetchall()
            old = {row[1]: row for row in old_rows}
            current = {item["id"]: item for item in result["all_items"]}
            for pid in sorted(set(old) | set(current)):
                before = old.get(pid)
                after = current.get(pid)
                if before is None:
                    change = "new"
                    prev_cls, cur_cls = None, after["classification"]
                    score_delta, sel_delta, stab_delta = after["score"], after["selectivity"], after["stability"]
                elif after is None:
                    change = "disappeared"
                    prev_cls, cur_cls = before[5], None
                    score_delta, sel_delta, stab_delta = -before[12], -before[10], -before[11]
                else:
                    prev_cls, cur_cls = before[5], after["classification"]
                    score_delta = after["score"] - before[12]
                    sel_delta = after["selectivity"] - before[10]
                    stab_delta = after["stability"] - before[11]
                    if prev_cls != cur_cls:
                        change = "reclassified"
                    elif score_delta >= 5 or sel_delta >= 5 or stab_delta >= 5:
                        change = "strengthened"
                    elif score_delta <= -5 or sel_delta <= -5 or stab_delta <= -5:
                        change = "weakened"
                    else:
                        continue
                conn.execute(
                    """INSERT OR REPLACE INTO reflection_pattern_changes
                    (from_run_id,to_run_id,pattern_id,change_type,previous_classification,current_classification,
                    score_delta,selectivity_delta,stability_delta) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (previous, run_id, pid, change, prev_cls, cur_cls, round(score_delta, 1), round(sel_delta, 1), round(stab_delta, 1)),
                )
        return run_id, previous


def load_history(run_id: int) -> dict:
    initialize_pattern_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        runs = [dict(r) for r in conn.execute("SELECT * FROM reflection_runs ORDER BY run_id DESC LIMIT 20")]
        changes = [dict(r) for r in conn.execute(
            "SELECT * FROM reflection_pattern_changes WHERE to_run_id=? ORDER BY ABS(score_delta)+ABS(selectivity_delta)+ABS(stability_delta) DESC LIMIT 80",
            (run_id,),
        )]
        pins = {r[0] for r in conn.execute("SELECT pattern_id FROM reflection_pins")}
        snapshots = {
            r["pattern_id"]: dict(r) for r in conn.execute(
                "SELECT * FROM reflection_pattern_snapshots WHERE run_id=?", (run_id,)
            )
        }
    for change in changes:
        snap = snapshots.get(change["pattern_id"])
        change["label"] = snap["label"] if snap else change["pattern_id"]
        change["pinned"] = change["pattern_id"] in pins
    summary = Counter(c["change_type"] for c in changes)
    return {"runs": runs, "changes": changes, "summary": summary, "pins": pins}


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Reflection Lab v0.3</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff;--yellow:#ffd166;--red:#ff8b8b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1420px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}h2{margin:8px 0 14px}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.controls{display:grid;grid-template-columns:repeat(4,1fr) auto;gap:12px;align-items:end}input,select{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button,.button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer;text-decoration:none}.pattern{padding:14px;background:#071522;border:1px solid rgba(36,70,109,.7);border-radius:12px;margin:10px 0}.head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}.metrics{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-top:9px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.pill.other{color:var(--muted)}.status{font-size:12px;border-radius:999px;padding:5px 9px;border:1px solid var(--line);white-space:nowrap}.concept{color:var(--green)}.skeleton{color:var(--cyan)}.individual{color:var(--yellow)}.selective{color:var(--purple)}.new{color:var(--green)}.strengthened{color:var(--cyan)}.weakened{color:var(--yellow)}.disappeared{color:var(--red)}.reclassified{color:var(--purple)}.split{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}.history{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.pin{background:none;border:1px solid var(--line);padding:5px 9px}.section-note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}@media(max-width:1000px){.stats,.history{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:1fr 1fr}.split{grid-template-columns:1fr}}@media(max-width:650px){.stats,.history,.controls{grid-template-columns:1fr}.head{display:block}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Reflection Lab v0.3</h1><p>活動分類に時間軸を加え、知性の形成過程を観測する</p></div></header><main class="wrap">
<section class="card"><form method="get"><div class="controls">
<div><div class="eyebrow">Minimum Experiences</div><input type="number" name="min_exp" value="{{min_exp}}"></div>
<div><div class="eyebrow">Minimum Target Texts</div><input type="number" name="min_texts" value="{{min_texts}}"></div>
<div><div class="eyebrow">Common Skeleton %</div><input type="number" name="common" value="{{common}}"></div>
<div><div class="eyebrow">Minimum Selectivity pt</div><input type="number" name="selectivity" value="{{selectivity}}"></div>
<button>解析スナップショットを保存</button></div></form><p class="muted">各解析をRunとして保存し、前回との差だけを観測します。Coreとmemory.dbは書き換えません。</p></section>
<section class="stats card"><div class="stat"><div class="eyebrow">Run</div><div class="value">#{{run_id}}</div><div class="muted">前回 #{{previous or '-'}}</div></div><div class="stat"><div class="eyebrow">Experiences</div><div class="value">{{x.total}}</div></div><div class="stat"><div class="eyebrow">Common Skeleton</div><div class="value">{{x.common_count}}</div></div><div class="stat"><div class="eyebrow">Concept Candidates</div><div class="value">{{x.concept_count}}</div></div><div class="stat"><div class="eyebrow">Individual / Selective</div><div class="value">{{x.individual_count}} / {{x.selective_count}}</div></div></section>
<section class="card"><div class="eyebrow">Growth Summary</div><h2>今回の変化</h2><div class="history">{% for key,label in [('new','新生'),('strengthened','強化'),('weakened','弱化'),('disappeared','消失'),('reclassified','分類変更')] %}<div class="stat"><div class="eyebrow {{key}}">{{label}}</div><div class="value">{{history.summary.get(key,0)}}</div></div>{% endfor %}</div>
{% for c in history.changes %}<div class="pattern"><div class="head"><div><code>{{c.label}}</code><div class="metrics"><span class="{{c.change_type}}">{{c.change_type}}</span><span>{{c.previous_classification or '-'}} → {{c.current_classification or '-'}}</span><span>総合 {{'%+.1f'|format(c.score_delta)}} </span><span>選択差 {{'%+.1f'|format(c.selectivity_delta)}} pt</span><span>安定度 {{'%+.1f'|format(c.stability_delta)}} pt</span></div></div><form method="post" action="{{url_for('toggle_pin')}}"><input type="hidden" name="pattern_id" value="{{c.pattern_id}}"><button class="pin">{{'★' if c.pinned else '☆'}}</button></form></div></div>{% else %}<p class="muted">初回Run、または閾値を超える変化はありません。</p>{% endfor %}</section>
{% set sections=[('concept candidate','概念候補','concept'),('common skeleton','共通骨格','skeleton'),('individual memory','個別記憶','individual'),('selective pattern','選択的活動','selective')] %}{% for key,title,css in sections %}<section class="card"><div class="eyebrow">Current Classification</div><h2>{{title}} <span class="muted">{{x.groups[key]|length}}件表示</span></h2>{% for item in x.groups[key] %}<div class="pattern"><div class="head"><div><code>{{item.label}}</code><div class="metrics"><span>{{item.kind}}</span><span>{{item.count}}経験</span><span>全体 {{item.global_rate}}%</span><span>対象群 {{item.target_rate}}%</span><span>その他 {{item.other_rate}}%</span><span>差分 {{'%+.1f'|format(item.selectivity)}} pt</span><span>安定度 {{item.stability}}%</span></div></div><span class="status {{css}}">{{item.classification}}</span></div><div class="split"><div>{% for t in item.target_texts %}<span class="pill">{{t}}</span>{% endfor %}</div><div>{% for t in item.other_texts %}<span class="pill other">{{t}}</span>{% endfor %}</div></div></div>{% else %}<p class="muted">候補なし</p>{% endfor %}</section>{% endfor %}
</main></body></html>
"""


@app.route("/")
def index():
    min_exp = max(2, int(request.args.get("min_exp", 2)))
    min_texts = max(2, int(request.args.get("min_texts", 2)))
    common = min(100, max(50, int(request.args.get("common", 80))))
    selectivity = min(90, max(5, int(request.args.get("selectivity", 20))))
    memories = load_memories()
    result = analyze(memories, min_exp, min_texts, common / 100.0, selectivity / 100.0)
    params = {"min_exp": min_exp, "min_texts": min_texts, "common": common, "selectivity": selectivity}
    run_id, previous = save_run(result, params)
    history = load_history(run_id)
    return render_template_string(PAGE, x=result, run_id=run_id, previous=previous, history=history,
                                  min_exp=min_exp, min_texts=min_texts, common=common, selectivity=selectivity)


@app.post("/pin")
def toggle_pin():
    pattern = request.form.get("pattern_id", "").strip()
    if pattern:
        initialize_pattern_db()
        with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
            exists = conn.execute("SELECT 1 FROM reflection_pins WHERE pattern_id=?", (pattern,)).fetchone()
            if exists:
                conn.execute("DELETE FROM reflection_pins WHERE pattern_id=?", (pattern,))
            else:
                conn.execute("INSERT INTO reflection_pins(pattern_id) VALUES(?)", (pattern,))
    return redirect(url_for("index"))


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5053")
    print("SphereBrain Reflection Lab v0.3: http://127.0.0.1:5053")
    serve(app, host="127.0.0.1", port=5053, threads=4)
