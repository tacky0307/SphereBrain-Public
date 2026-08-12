from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import json
import math
import random
import sqlite3
import webbrowser

import numpy as np
from flask import Flask, render_template_string, request
from waitress import serve

from brain import SphereBrain

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"
FEEDBACK_FILE = DATA / "route_choice_feedback.db"
app = Flask(__name__)


@dataclass
class RouteCandidate:
    key: str
    label: str
    nodes: list[int]
    edges: list[tuple[int, int]]
    source_text: str
    decoy: bool
    score: float = 0.0
    percent: float = 0.0


def norm_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def route_key(edges: list[tuple[int, int]]) -> str:
    raw = ";".join(f"{a}-{b}" for a, b in edges)
    return sha256(raw.encode("utf-8")).hexdigest()[:20]


def init_feedback_db() -> None:
    DATA.mkdir(exist_ok=True)
    with sqlite3.connect(FEEDBACK_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS route_feedback (
            prefix_signature TEXT NOT NULL,
            route_key TEXT NOT NULL,
            positive INTEGER NOT NULL DEFAULT 0,
            negative INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(prefix_signature, route_key)
        )
        """)


def load_routes(limit: int = 500) -> list[RouteCandidate]:
    if not DB_FILE.exists():
        return []
    with sqlite3.connect(f"file:{DB_FILE.as_posix()}?mode=ro", uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT input_text,activated_nodes,traversed_edges FROM memories "
            "WHERE kind='input' ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    routes, seen = [], set()
    for row in rows:
        try:
            edges = [norm_edge(int(e[0]), int(e[1])) for e in json.loads(row["traversed_edges"] or "[]")]
            nodes = [int(n) for n in json.loads(row["activated_nodes"] or "[]")]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if len(edges) < 2:
            continue
        key = route_key(edges)
        if key in seen:
            continue
        seen.add(key)
        routes.append(RouteCandidate(key, "", nodes, edges, row["input_text"] or "(名称なし)", False))
    return routes


def prefix_signature(brain: SphereBrain, text: str) -> tuple[str, list[int]]:
    sources = [int(n) for n in brain.text_to_sources(text)]
    signature = sha256(",".join(map(str, sorted(sources))).encode("utf-8")).hexdigest()[:20]
    return signature, sources


def feedback_bias(signature: str, key: str) -> float:
    init_feedback_db()
    with sqlite3.connect(FEEDBACK_FILE) as conn:
        row = conn.execute(
            "SELECT positive,negative FROM route_feedback WHERE prefix_signature=? AND route_key=?",
            (signature, key),
        ).fetchone()
    if row is None:
        return 0.0
    positive, negative = int(row[0]), int(row[1])
    return (positive - negative) / (positive + negative + 3.0)


def make_decoys(real_routes: list[RouteCandidate], count: int, seed_text: str) -> list[RouteCandidate]:
    rng = random.Random(int.from_bytes(sha256(seed_text.encode("utf-8")).digest()[:8], "big"))
    if len(real_routes) < 2:
        return []
    decoys, used = [], {r.key for r in real_routes}
    attempts = 0
    while len(decoys) < count and attempts < count * 20:
        attempts += 1
        left, right = rng.sample(real_routes, 2)
        cut_l = max(1, len(left.edges) // 2)
        cut_r = max(1, len(right.edges) // 2)
        edges = left.edges[:cut_l] + right.edges[-cut_r:]
        rng.shuffle(edges)
        key = route_key(edges)
        if key in used or len(edges) < 2:
            continue
        used.add(key)
        nodes = sorted({n for edge in edges for n in edge})
        decoys.append(RouteCandidate(key, "", nodes, edges, "偽経路（複数経験を組み替え）", True))
    return decoys


def score_candidate(brain: SphereBrain, sources: list[int], signature: str, candidate: RouteCandidate) -> float:
    weights, usages = [], []
    for a, b in candidate.edges:
        if 0 <= a < brain.node_count and 0 <= b < brain.node_count and brain.adjacency[a, b]:
            weights.append(float(brain.weights[a, b]))
            usages.append(int(brain.usage[a, b]))
    strength = float(np.mean(weights)) if weights else 0.0
    familiarity = float(np.mean([u / (u + 5.0) for u in usages])) if usages else 0.0

    route_nodes = set(candidate.nodes)
    adjacency_hits = 0
    spatial = []
    for source in sources:
        neighbors = set(np.flatnonzero(brain.adjacency[source]).tolist())
        if neighbors & route_nodes:
            adjacency_hits += 1
        if route_nodes:
            distances = np.linalg.norm(brain.positions[list(route_nodes)] - brain.positions[source], axis=1)
            spatial.append(1.0 / (1.0 + float(np.min(distances))))
    entry_affinity = 0.6 * (adjacency_hits / max(1, len(sources))) + 0.4 * (float(np.mean(spatial)) if spatial else 0.0)
    learned = feedback_bias(signature, candidate.key)
    decoy_penalty = 0.05 if candidate.decoy else 0.0
    return 0.42 * strength + 0.28 * familiarity + 0.20 * entry_affinity + 0.15 * learned - decoy_penalty


def build_candidates(text: str, candidate_count: int, decoy_count: int) -> dict:
    if not BRAIN_FILE.exists():
        raise FileNotFoundError("data/brain.json がありません。")
    brain = SphereBrain.load(BRAIN_FILE)
    signature, sources = prefix_signature(brain, text)
    all_real = load_routes()
    if not all_real:
        raise RuntimeError("候補に使える過去経路がありません。先に通常入力で経験を蓄積してください。")

    for route in all_real:
        route.score = score_candidate(brain, sources, signature, route)
    all_real.sort(key=lambda r: (-r.score, r.key))
    real = all_real[:max(2, candidate_count - decoy_count)]
    candidates = real + make_decoys(all_real[:40], decoy_count, text)
    for route in candidates:
        route.score = score_candidate(brain, sources, signature, route)
    candidates.sort(key=lambda r: (-r.score, r.key))

    exps = [math.exp((c.score - max(x.score for x in candidates)) * 7.0) for c in candidates]
    total = sum(exps) or 1.0
    for index, (candidate, value) in enumerate(zip(candidates, exps)):
        candidate.label = chr(ord("A") + index)
        candidate.percent = value / total * 100.0

    return {"signature": signature, "sources": sources, "candidates": candidates}


def apply_feedback(text: str, payload: list[dict], correct_key: str) -> str:
    brain = SphereBrain.load(BRAIN_FILE)
    signature, sources = prefix_signature(brain, text)
    init_feedback_db()

    with sqlite3.connect(FEEDBACK_FILE) as conn:
        for item in payload:
            key = str(item["key"])
            positive = 1 if key == correct_key else 0
            negative = 0 if key == correct_key else 1
            conn.execute("""
                INSERT INTO route_feedback(prefix_signature,route_key,positive,negative)
                VALUES(?,?,?,?)
                ON CONFLICT(prefix_signature,route_key) DO UPDATE SET
                  positive=positive+excluded.positive,
                  negative=negative+excluded.negative,
                  updated_at=CURRENT_TIMESTAMP
            """, (signature, key, positive, negative))

    for item in payload:
        is_correct = str(item["key"]) == correct_key
        for a, b in item.get("edges", []):
            a, b = int(a), int(b)
            if not (0 <= a < brain.node_count and 0 <= b < brain.node_count and brain.adjacency[a, b]):
                continue
            if is_correct:
                brain.weights[a, b] = brain.weights[b, a] = min(1.0, brain.weights[a, b] + 0.04 * (1.0 - brain.weights[a, b]))
                brain.usage[a, b] += 1
                brain.usage[b, a] += 1
                brain.node_usage[a] += 1
                brain.node_usage[b] += 1
            else:
                brain.weights[a, b] = brain.weights[b, a] = max(0.05, brain.weights[a, b] * 0.985)

    correct = next((item for item in payload if str(item["key"]) == correct_key), None)
    if correct:
        route_nodes = {int(n) for edge in correct.get("edges", []) for n in edge}
        for source in sources:
            neighbors = [int(n) for n in np.flatnonzero(brain.adjacency[source]) if int(n) in route_nodes]
            for target in neighbors:
                brain.weights[source, target] = brain.weights[target, source] = min(1.0, brain.weights[source, target] + 0.06 * (1.0 - brain.weights[source, target]))
                brain.usage[source, target] += 1
                brain.usage[target, source] += 1

    brain.save(BRAIN_FILE)
    return "正解経路を強化し、その他の候補経路を弱くしました。"


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Route Choice Learning Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--orange:#ff9d52;--green:#8ce3a9;--red:#ff8585}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 180px 180px;gap:12px}input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:12px;font-size:16px}button{background:linear-gradient(135deg,#ec6f35,#ff9d52);border:0;color:#fff;border-radius:10px;padding:12px 18px;font-weight:800;cursor:pointer}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em}.muted{color:var(--muted)}.note{border-left:3px solid var(--cyan);padding-left:12px}.candidate{display:grid;grid-template-columns:70px 120px 1fr 150px;gap:14px;align-items:center;padding:15px 0;border-bottom:1px solid var(--line)}.letter{font-size:32px;font-weight:900;color:var(--cyan)}.percent{font-size:28px;font-weight:900}.route{font-family:ui-monospace,monospace;color:#bfeeff;word-break:break-all}.decoy{color:var(--red)}.real{color:var(--green)}.feedback{display:flex;gap:9px;flex-wrap:wrap}.choice{background:#071522;border:1px solid var(--line);padding:9px 13px;border-radius:9px}.success{border-color:var(--green);color:var(--green)}@media(max-width:760px){.grid,.candidate{grid-template-columns:1fr}.candidate{gap:5px}}
</style></head><body><main class="wrap">
<div class="card"><div class="eyebrow">ROUTE CHOICE LEARNING LAB v0.1</div><h1>SphereBrainに経路を提示し、選択を教える</h1><p class="muted">Coreへ渡す候補は言葉ではなく経路です。表示上だけA・B・Cへ翻訳します。</p></div>
<div class="card"><form method="post"><input type="hidden" name="action" value="evaluate"><div class="grid"><div><div class="eyebrow">PARTIAL INPUT</div><h2>途中入力</h2><input name="text" value="{{text}}" required></div><div><div class="eyebrow">CANDIDATES</div><h2>候補数</h2><input name="count" type="number" min="3" max="8" value="{{count}}"></div><div><div class="eyebrow">DECOYS</div><h2>偽経路数</h2><input name="decoys" type="number" min="0" max="4" value="{{decoys}}"></div></div><p><button type="submit">経路候補を評価する</button></p></form><p class="note muted">評価時は読み取りのみ。下の○を教える操作をした時だけbrain.jsonを更新します。</p></div>
{% if message %}<div class="card success">{{message}}</div>{% endif %}{% if error %}<div class="card decoy">{{error}}</div>{% endif %}
{% if result %}<div class="card"><div class="eyebrow">RESONANCE</div><h2>「{{text}}」に対する経路候補</h2>
{% for c in result.candidates %}<div class="candidate"><div class="letter">{{c.label}}</div><div class="percent">{{'%.1f'|format(c.percent)}}%</div><div><div class="route">{% for e in c.edges[:10] %}{{e[0]}}→{{e[1]}}{% if not loop.last %} / {% endif %}{% endfor %}</div><div class="{% if c.decoy %}decoy{% else %}real{% endif %}">{% if c.decoy %}偽経路{% else %}実経験経路{% endif %} — {{c.source_text}}</div></div><form method="post"><input type="hidden" name="action" value="feedback"><input type="hidden" name="text" value="{{text}}"><input type="hidden" name="payload" value='{{payload}}'><input type="hidden" name="correct_key" value="{{c.key}}"><button type="submit">{{c.label}} が○</button></form></div>{% endfor %}
<p class="note muted">選ばなかった候補はすべて×として学習します。偽経路を○にすることもできますが、実験意図と異なるため通常は選ばないでください。</p></div>{% endif %}
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text, count, decoys = "犬は", 5, 2
    result = None
    payload = ""
    message = error = ""
    if request.method == "POST":
        action = request.form.get("action", "evaluate")
        text = request.form.get("text", "").strip()
        try:
            if action == "feedback":
                raw = json.loads(request.form.get("payload", "[]"))
                message = apply_feedback(text, raw, request.form.get("correct_key", ""))
                result = build_candidates(text, count, decoys)
            else:
                count = max(3, min(8, int(request.form.get("count", "5"))))
                decoys = max(0, min(min(4, count - 1), int(request.form.get("decoys", "2"))))
                result = build_candidates(text, count, decoys)
            if result:
                payload = json.dumps([{"key": c.key, "edges": c.edges} for c in result["candidates"]], separators=(",", ":"))
        except Exception as exc:
            error = str(exc)
    return render_template_string(PAGE, text=text, count=count, decoys=decoys, result=result, payload=payload, message=message, error=error)


def main() -> None:
    url = "http://127.0.0.1:5077"
    webbrowser.open(url)
    serve(app, host="127.0.0.1", port=5077, threads=4)


if __name__ == "__main__":
    main()
