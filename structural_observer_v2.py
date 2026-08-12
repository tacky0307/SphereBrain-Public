from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

BASE = Path(__file__).resolve().parent
MEMORY_DB = BASE / "data" / "memory.db"
app = Flask(__name__)


def norm_edge(edge):
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def load_memories(limit=12000):
    if not MEMORY_DB.exists():
        return []
    with sqlite3.connect(f"file:{MEMORY_DB.as_posix()}?mode=ro", uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, kind, input_text, activated_nodes, traversed_edges "
            "FROM memories WHERE COALESCE(input_text,'')<>'' AND kind IN ('input','trainer') "
            "ORDER BY id ASC LIMIT ?", (int(limit),)
        ).fetchall()
    out = []
    for row in rows:
        try:
            edges = [norm_edge(v) for v in json.loads(row['traversed_edges'] or '[]')]
        except Exception:
            edges = []
        try:
            nodes = [int(v) for v in json.loads(row['activated_nodes'] or '[]')]
        except Exception:
            nodes = []
        out.append({
            'id': row['id'], 'created_at': row['created_at'], 'kind': row['kind'],
            'text': str(row['input_text'] or '').strip(), 'edges': list(dict.fromkeys(edges)),
            'nodes': list(dict.fromkeys(nodes)),
        })
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def analyze(memories, common_threshold=0.55, cluster_threshold=0.35):
    total = max(1, len(memories))
    by_text = defaultdict(list)
    edge_count = Counter()
    edge_texts = defaultdict(set)
    node_count = Counter()
    for m in memories:
        by_text[m['text']].append(m)
        for e in m['edges']:
            edge_count[e] += 1
            edge_texts[e].add(m['text'])
        node_count.update(m['nodes'])

    distinct_texts = max(1, len(by_text))
    common_edges = {
        e for e in edge_count
        if edge_count[e] / total >= common_threshold
        or len(edge_texts[e]) / distinct_texts >= 0.70
    }

    signatures = {}
    text_stats = []
    for text, rows in by_text.items():
        counts = Counter(e for r in rows for e in r['edges'] if e not in common_edges)
        stable = {e for e, c in counts.items() if c / len(rows) >= 0.50}
        signatures[text] = stable
        text_stats.append({
            'text': text, 'runs': len(rows), 'signature_edges': len(stable),
            'raw_edges': len(set(e for r in rows for e in r['edges'])),
            'stability': round(100 * (sum(counts.values()) / max(1, len(rows) * max(1, len(counts)))), 1),
        })

    texts = sorted(signatures)
    adjacency = {t: set() for t in texts}
    similarities = []
    for i, left in enumerate(texts):
        for right in texts[i+1:]:
            score = jaccard(signatures[left], signatures[right])
            if score >= cluster_threshold and signatures[left] and signatures[right]:
                adjacency[left].add(right)
                adjacency[right].add(left)
                similarities.append((score, left, right))

    clusters = []
    seen = set()
    for start in texts:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(members) >= 2:
            edge_membership = Counter(e for t in members for e in signatures[t])
            core = [e for e, c in edge_membership.items() if c / len(members) >= 0.60]
            clusters.append({'members': sorted(members), 'core_edges': core, 'edge_count': len(edge_membership)})

    text_cluster = {}
    for idx, c in enumerate(clusters):
        for t in c['members']:
            text_cluster[t] = idx

    bridge_candidates = []
    for edge, texts_for_edge in edge_texts.items():
        cluster_ids = {text_cluster[t] for t in texts_for_edge if t in text_cluster}
        if len(cluster_ids) >= 2:
            bridge_candidates.append({
                'edge': edge, 'clusters': len(cluster_ids), 'texts': len(texts_for_edge),
                'uses': edge_count[edge]
            })
    bridge_candidates.sort(key=lambda x: (-x['clusters'], -x['texts'], -x['uses'], x['edge']))

    highways = []
    for edge, uses in edge_count.items():
        diversity = len(edge_texts[edge])
        bridge = next((b['clusters'] for b in bridge_candidates if b['edge'] == edge), 0)
        score = uses * 0.45 + diversity * 4.0 + bridge * 8.0
        highways.append({'edge': edge, 'uses': uses, 'texts': diversity, 'clusters': bridge, 'score': round(score, 1)})
    highways.sort(key=lambda x: (-x['score'], -x['uses']))

    return {
        'total': len(memories), 'distinct_texts': len(by_text), 'common_edges': sorted(common_edges),
        'signatures': sorted(text_stats, key=lambda x: (-x['runs'], x['text'])),
        'clusters': clusters, 'bridges': bridge_candidates[:30], 'highways': highways[:30],
        'similarities': sorted(similarities, reverse=True)[:30],
    }


PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SphereBrain Structural Observer v2</title><style>
:root{--bg:#07111f;--panel:#11223a;--line:#2a4a70;--text:#edf4ff;--muted:#9cb1ca;--cyan:#6edcff;--orange:#ef914f;--green:#78e6a4}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:22px}h1{margin:0 0 8px}.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#132641,#0d1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin:16px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#091522;border:1px solid var(--line);border-radius:14px;padding:14px}.value{font-size:30px;font-weight:800}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #233d5c;text-align:left}th{color:var(--cyan)}input{background:#081522;color:var(--text);border:1px solid var(--line);padding:9px;border-radius:9px}button{background:var(--orange);color:white;border:0;padding:10px 16px;border-radius:9px;font-weight:700}.note{border-left:3px solid var(--cyan);padding-left:12px}.good{color:var(--green)}@media(max-width:900px){.stats,.grid{grid-template-columns:1fr}}</style></head><body><div class="wrap">
<h1>SphereBrain Structural Observer v2</h1><p class="muted">現在の memory.db を直接読み、言葉の正解ではなく、経路のまとまり・共通骨格・橋・高速道路を観測します。CoreやDBは変更しません。</p>
<form class="card"><label>共通骨格しきい値 <input name="common" type="number" min="10" max="95" value="{{common}}">%</label>　<label>クラスタ類似度 <input name="cluster" type="number" min="5" max="95" value="{{cluster}}">%</label>　<button>再解析</button></form>
<div class="stats"><div class="stat"><div class="muted">Experiences</div><div class="value">{{r.total}}</div></div><div class="stat"><div class="muted">Distinct Inputs</div><div class="value">{{r.distinct_texts}}</div></div><div class="stat"><div class="muted">Common Edges</div><div class="value">{{r.common_edges|length}}</div></div><div class="stat"><div class="muted">Clusters</div><div class="value">{{r.clusters|length}}</div></div></div>
<div class="card"><h2>Common Skeleton</h2><p class="muted">多くの経験に共通する処理骨格。多すぎる場合は画一化、少なすぎる場合は共有構造不足の可能性があります。</p>{% for e in r.common_edges[:80] %}<span class="pill">{{e[0]}}–{{e[1]}}</span>{% else %}<p>現在の条件では共通骨格なし。</p>{% endfor %}</div>
<div class="grid"><div class="card"><h2>Experience Signatures</h2><table><tr><th>入力</th><th>回数</th><th>固有経路</th><th>全経路</th></tr>{% for x in r.signatures %}<tr><td>{{x.text}}</td><td>{{x.runs}}</td><td>{{x.signature_edges}}</td><td>{{x.raw_edges}}</td></tr>{% endfor %}</table></div>
<div class="card"><h2>Concept / Route Clusters</h2>{% for c in r.clusters %}<div class="card"><b>Cluster {{loop.index}}</b><div>{% for t in c.members %}<span class="pill">{{t}}</span>{% endfor %}</div><p class="muted">共有核 {{c.core_edges|length}} edge / 全体 {{c.edge_count}} edge</p></div>{% else %}<p>現在のしきい値ではクラスタなし。類似度を下げるか、同じ経験を複数回追加して再確認してください。</p>{% endfor %}</div></div>
<div class="grid"><div class="card"><h2>Bridge Candidates</h2><table><tr><th>Edge</th><th>Cluster</th><th>入力種類</th><th>使用</th></tr>{% for b in r.bridges %}<tr><td>{{b.edge[0]}}–{{b.edge[1]}}</td><td>{{b.clusters}}</td><td>{{b.texts}}</td><td>{{b.uses}}</td></tr>{% else %}<tr><td colspan="4">現在の条件ではBridge Candidateなし。</td></tr>{% endfor %}</table></div>
<div class="card"><h2>Highway Candidates</h2><table><tr><th>Edge</th><th>使用</th><th>入力種類</th><th>Cluster</th><th>Score</th></tr>{% for h in r.highways %}<tr><td>{{h.edge[0]}}–{{h.edge[1]}}</td><td>{{h.uses}}</td><td>{{h.texts}}</td><td>{{h.clusters}}</td><td>{{h.score}}</td></tr>{% endfor %}</table></div></div>
<div class="card note"><b>読み方</b><p>高速道路は「太いだけ」ではなく、複数の入力・複数クラスタで使われる経路を上位にします。クラスタは文章の単語一致ではなく、共通骨格を除いた経路集合の重なりだけで作ります。</p></div>
</div></body></html>'''


@app.route('/')
def index():
    common = max(10, min(95, int(request.args.get('common', 55))))
    cluster = max(5, min(95, int(request.args.get('cluster', 35))))
    memories = load_memories()
    result = analyze(memories, common/100.0, cluster/100.0)
    return render_template_string(PAGE, r=result, common=common, cluster=cluster)


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5090')
    serve(app, host='127.0.0.1', port=5090)
