from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

BASE = Path(__file__).resolve().parent
DB_FILE = BASE / "data" / "memory.db"
app = Flask(__name__)

DEFAULT_A = "空"
DEFAULT_B = "青,青い,青く"


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def edge_set(memory: dict) -> set[tuple[int, int]]:
    return {normalize_edge(edge) for edge in memory["traversed_edges"]}


def load_memories(limit: int = 3000) -> list[dict]:
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
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        item = dict(row)
        item["activated_nodes"] = json.loads(item["activated_nodes"])
        item["traversed_edges"] = json.loads(item["traversed_edges"])
        result.append(item)
    return result


def parse_terms(raw: str) -> list[str]:
    return [term.strip() for term in raw.replace("\n", ",").split(",") if term.strip()]


def matches(memory: dict, terms: list[str]) -> bool:
    text = memory.get("input_text") or ""
    return bool(terms) and any(term in text for term in terms)


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def frequencies(group: list[dict]) -> Counter[tuple[int, int]]:
    freq: Counter[tuple[int, int]] = Counter()
    for memory in group:
        freq.update(edge_set(memory))
    return freq


def rate(freq: Counter, edge: tuple[int, int], size: int) -> float:
    return freq.get(edge, 0) / size if size else 0.0


def summarize(group: list[dict]) -> dict:
    sets = [edge_set(m) for m in group]
    pair_scores = [jaccard(a, b) for a, b in combinations(sets, 2)]
    freq = frequencies(group)
    return {
        "count": len(group),
        "avg_similarity": round(sum(pair_scores) / len(pair_scores) * 100, 1) if pair_scores else 0.0,
        "freq": freq,
        "examples": [m["input_text"] for m in group[:8]],
    }


def difference_analysis(memories: list[dict], terms_a: list[str], terms_b: list[str]) -> dict:
    group_a = [m for m in memories if matches(m, terms_a)]
    group_b = [m for m in memories if matches(m, terms_b)]
    background = [m for m in memories if not matches(m, terms_a) and not matches(m, terms_b)]

    sa, sb, bg = summarize(group_a), summarize(group_b), summarize(background)
    all_edges = set(sa["freq"]) | set(sb["freq"]) | set(bg["freq"])

    rows = []
    for edge in all_edges:
        ra = rate(sa["freq"], edge, sa["count"])
        rb = rate(sb["freq"], edge, sb["count"])
        rbg = rate(bg["freq"], edge, bg["count"])
        lift_a, lift_b = ra - rbg, rb - rbg
        rows.append({
            "a": edge[0], "b": edge[1],
            "ra": round(ra * 100, 1), "rb": round(rb * 100, 1), "rbg": round(rbg * 100, 1),
            "lift_a": round(lift_a * 100, 1), "lift_b": round(lift_b * 100, 1),
            "raw_lift_a": lift_a, "raw_lift_b": lift_b,
        })

    skeleton = [r for r in rows if r["rbg"] >= 70 and abs(r["raw_lift_a"]) < 0.15 and abs(r["raw_lift_b"]) < 0.15]
    a_only = [r for r in rows if r["ra"] >= 30 and r["raw_lift_a"] >= 0.15 and r["raw_lift_a"] >= r["raw_lift_b"] + 0.08]
    b_only = [r for r in rows if r["rb"] >= 30 and r["raw_lift_b"] >= 0.15 and r["raw_lift_b"] >= r["raw_lift_a"] + 0.08]
    bridge = [r for r in rows if r["ra"] >= 30 and r["rb"] >= 30 and r["raw_lift_a"] >= 0.12 and r["raw_lift_b"] >= 0.12 and abs(r["raw_lift_a"] - r["raw_lift_b"]) < 0.12]

    a_only.sort(key=lambda r: (-r["raw_lift_a"], -r["ra"]))
    b_only.sort(key=lambda r: (-r["raw_lift_b"], -r["rb"]))
    bridge.sort(key=lambda r: (-(r["raw_lift_a"] + r["raw_lift_b"]), -min(r["ra"], r["rb"])))
    skeleton.sort(key=lambda r: -r["rbg"])

    return {
        "sa": sa, "sb": sb, "bg": bg,
        "a_only": a_only[:30], "b_only": b_only[:30], "bridge": bridge[:30], "skeleton": skeleton[:30],
        "a_count": len(a_only), "b_count": len(b_only), "bridge_count": len(bridge), "skeleton_count": len(skeleton),
    }


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Concept Lab v0.2</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1380px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.formgrid{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end}.edge{padding:10px 12px;background:#071522;border:1px solid rgba(36,70,109,.7);border-radius:10px;margin:7px 0}.edgehead{display:flex;justify-content:space-between;gap:12px}.rates{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:7px;color:var(--muted);font-size:12px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.warning{color:#ffd0a9;background:rgba(255,157,82,.09);border:1px solid rgba(255,157,82,.35);padding:12px;border-radius:12px}.legend{display:flex;gap:10px;flex-wrap:wrap}.tag{padding:7px 10px;border-radius:999px;border:1px solid var(--line)}.a{color:var(--green)}.b{color:var(--yellow)}.bridge{color:var(--purple)}.skeleton{color:var(--muted)}@media(max-width:900px){.grid,.stats,.formgrid,.rates{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Concept Lab v0.2</h1><p>全経験の共通骨格を差し引き、概念候補の「平均との差分」を観測する</p></div></header>
<main class="wrap">
<div class="warning">v0.2は、候補A・候補Bに含まれない経験を背景群として使い、各経路の「候補内出現率 − 背景出現率」を計算します。表示は概念の断定ではなく、概念固有・橋渡し・共通骨格の候補です。</div>
<section class="card"><form method="get"><div class="formgrid"><div><div class="eyebrow">Candidate A</div><input name="a" value="{{ raw_a }}"></div><div><div class="eyebrow">Candidate B</div><input name="b" value="{{ raw_b }}"></div><button>差分を解析する</button></div></form></section>
<section class="stats card"><div class="stat"><div class="eyebrow">A Experiences</div><div class="value">{{ x.sa.count }}</div><div class="muted">群内一致 {{ x.sa.avg_similarity }}%</div></div><div class="stat"><div class="eyebrow">B Experiences</div><div class="value">{{ x.sb.count }}</div><div class="muted">群内一致 {{ x.sb.avg_similarity }}%</div></div><div class="stat"><div class="eyebrow">Background</div><div class="value">{{ x.bg.count }}</div><div class="muted">平均との差の基準</div></div><div class="stat"><div class="eyebrow">Skeleton</div><div class="value">{{ x.skeleton_count }}</div><div class="muted">全群共通の骨格候補</div></div></section>
<section class="card"><div class="eyebrow">Concept Difference Map</div><h2>概念差分マップ</h2><div class="legend"><span class="tag a">A固有 {{x.a_count}}</span><span class="tag b">B固有 {{x.b_count}}</span><span class="tag bridge">橋渡し {{x.bridge_count}}</span><span class="tag skeleton">共通骨格 {{x.skeleton_count}}</span></div><p class="muted">固有経路は背景より15ポイント以上高く、もう一方の候補より8ポイント以上優勢な経路。橋渡しはA・Bの両方で背景より12ポイント以上高い経路です。</p></section>
<section class="grid">
{% for title,items,cls in [('候補Aに偏る経路',x.a_only,'a'),('候補Bに偏る経路',x.b_only,'b'),('AとBを橋渡しする経路',x.bridge,'bridge'),('全経験の共通骨格',x.skeleton,'skeleton')] %}
<article class="card"><div class="eyebrow {{cls}}">Difference Routes</div><h2>{{title}}</h2>{% for e in items %}<div class="edge"><div class="edgehead"><code>{{e.a}} → {{e.b}}</code><b class="{{cls}}">A差 {{e.lift_a}} / B差 {{e.lift_b}} pt</b></div><div class="rates"><span>A {{e.ra}}%</span><span>B {{e.rb}}%</span><span>背景 {{e.rbg}}%</span></div></div>{% else %}<p class="muted">現在の条件では候補が見つかりません。</p>{% endfor %}</article>
{% endfor %}
</section>
<section class="grid">{% for label,s in [('A',x.sa),('B',x.sb)] %}<article class="card"><div class="eyebrow">Representative Experiences {{label}}</div><h2>{{ terms_a|join(' / ') if label=='A' else terms_b|join(' / ') }}</h2>{% for t in s.examples %}<span class="pill">{{t}}</span>{% endfor %}</article>{% endfor %}</section>
</main></body></html>
"""


@app.route("/")
def index():
    raw_a = request.args.get("a", DEFAULT_A)
    raw_b = request.args.get("b", DEFAULT_B)
    memories = load_memories()
    terms_a, terms_b = parse_terms(raw_a), parse_terms(raw_b)
    x = difference_analysis(memories, terms_a, terms_b)
    return render_template_string(PAGE, raw_a=raw_a, raw_b=raw_b, terms_a=terms_a, terms_b=terms_b, x=x)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5052")
    print("SphereBrain Concept Lab v0.2: http://127.0.0.1:5052")
    serve(app, host="127.0.0.1", port=5052, threads=4)
