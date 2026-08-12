from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

BASE = Path(__file__).resolve().parent
SOURCE_DB = BASE / "data" / "pattern_candidates.db"
app = Flask(__name__)


@dataclass(frozen=True)
class Settings:
    runs: int = 3
    min_chain_length: int = 3
    skeleton_ratio: float = 0.60
    min_bridge_score: float = 0.25
    max_experiences: int = 30


def _json(value, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def _lcs(a: list[int], b: list[int]) -> list[int]:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x == y else max(table[i - 1][j], table[i][j - 1])
    out, i, j = [], len(a), len(b)
    while i and j:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1]); i -= 1; j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(out))


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _merge_fragments(fragments: list[list[int]]) -> list[int]:
    unique, seen = [], set()
    for frag in fragments:
        key = tuple(frag)
        if key not in seen:
            seen.add(key); unique.append(frag)
    unique.sort(key=lambda x: (-len(x), x))
    chain = unique[0][:] if unique else []
    remaining = unique[1:]
    changed = True
    while remaining and changed:
        changed = False
        for frag in remaining[:]:
            for size in range(min(len(chain), len(frag)), 0, -1):
                if chain[-size:] == frag[:size]:
                    chain.extend(frag[size:]); remaining.remove(frag); changed = True; break
                if frag[-size:] == chain[:size]:
                    chain = frag[:-size] + chain; remaining.remove(frag); changed = True; break
            if changed:
                break
    return chain


def load_experiences(settings: Settings):
    if not SOURCE_DB.exists():
        return {}, [], None
    uri = f"file:{SOURCE_DB.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        latest = conn.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()[0]
        if latest is None:
            return {}, [], None
        run_ids = [r[0] for r in conn.execute(
            "SELECT run_id FROM reflection_runs WHERE run_id<=? ORDER BY run_id DESC LIMIT ?",
            (latest, settings.runs),
        )]
        if len(run_ids) < settings.runs:
            return {}, run_ids, latest
        placeholders = ",".join("?" for _ in run_ids)
        rows = conn.execute(
            f"SELECT * FROM reflection_pattern_snapshots WHERE run_id IN ({placeholders}) "
            "AND classification='concept candidate' ORDER BY run_id, pattern_id",
            run_ids,
        ).fetchall()

    by_text_run = defaultdict(list)
    for row in rows:
        route = [int(x) for x in _json(row["pattern_json"], [])]
        if len(route) < 2:
            continue
        texts = [str(x) for x in _json(row["target_texts"], [])] or ["(unlabelled experience)"]
        for text in texts:
            by_text_run[(text, int(row["run_id"]))].append(route)

    raw = defaultdict(dict)
    for (text, run_id), fragments in by_text_run.items():
        chain = _merge_fragments(fragments)
        if len(chain) >= settings.min_chain_length:
            raw[text][run_id] = chain

    experiences = {}
    for text, runs in raw.items():
        if not all(run_id in runs for run_id in run_ids):
            continue
        ordered = [runs[run_id] for run_id in run_ids]
        stable = ordered[0]
        for chain in ordered[1:]:
            stable = _lcs(stable, chain)
        if len(stable) < settings.min_chain_length:
            continue
        experiences[text] = {
            "text": text,
            "chain": stable,
            "nodes": set(stable),
            "edges": set(zip(stable, stable[1:])),
            "runs": ordered,
        }
    limited = dict(sorted(experiences.items(), key=lambda kv: (-len(kv[1]["chain"]), kv[0]))[:settings.max_experiences])
    return limited, run_ids, latest


def analyze_bridges(experiences: dict[str, dict], settings: Settings):
    if not experiences:
        return [], [], set()
    node_counts = Counter()
    for item in experiences.values():
        node_counts.update(item["nodes"])
    skeleton_threshold = max(2, round(len(experiences) * settings.skeleton_ratio))
    skeleton = {n for n, count in node_counts.items() if count >= skeleton_threshold}

    signatures = {}
    for name, item in experiences.items():
        residual_chain = [n for n in item["chain"] if n not in skeleton]
        residual_nodes = set(residual_chain)
        residual_edges = set(zip(residual_chain, residual_chain[1:]))
        signatures[name] = {
            **item,
            "residual_chain": residual_chain,
            "residual_nodes": residual_nodes,
            "residual_edges": residual_edges,
        }

    bridges = []
    names = sorted(signatures)
    for i, left_name in enumerate(names):
        left = signatures[left_name]
        for right_name in names[i + 1:]:
            right = signatures[right_name]
            raw_nodes = _jaccard(left["nodes"], right["nodes"])
            residual_nodes = _jaccard(left["residual_nodes"], right["residual_nodes"])
            residual_edges = _jaccard(left["residual_edges"], right["residual_edges"])
            ordered = _lcs(left["residual_chain"], right["residual_chain"])
            ordered_ratio = len(ordered) / max(len(left["residual_chain"]), len(right["residual_chain"]), 1)
            # Common skeleton is evidence of shared processing, not enough for a bridge.
            score = 0.45 * residual_nodes + 0.35 * residual_edges + 0.20 * ordered_ratio
            if score < settings.min_bridge_score:
                continue
            bridges.append({
                "left": left_name,
                "right": right_name,
                "score": round(score * 100, 1),
                "raw": round(raw_nodes * 100, 1),
                "residual": round(residual_nodes * 100, 1),
                "edge": round(residual_edges * 100, 1),
                "shared": ordered,
                "left_only": [n for n in left["residual_chain"] if n not in set(ordered)],
                "right_only": [n for n in right["residual_chain"] if n not in set(ordered)],
            })
    bridges.sort(key=lambda x: (-x["score"], -len(x["shared"]), x["left"], x["right"]))
    return bridges, list(signatures.values()), skeleton


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Experience Bridge Observer v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--orange:#ff9d52;--green:#8ce3a9;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(80,145,210,.17),transparent 35%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1450px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.controls,.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.stats{grid-template-columns:repeat(4,1fr)}.stat,.experience,.bridge{background:#071522;border:1px solid var(--line);border-radius:14px;padding:16px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.value{font-size:30px;font-weight:800;margin-top:5px}input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:10px;font-size:15px}button{background:linear-gradient(135deg,#ec6f35,#ff9d52);border:0;color:white;border-radius:10px;padding:12px 18px;font-weight:800}.chain{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:10px 0}.node{border:1px solid var(--line);background:#0d2035;border-radius:999px;padding:6px 10px;color:var(--cyan);font-weight:700}.muted{color:var(--muted)}.good{color:var(--green)}.difference{color:var(--yellow)}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}@media(max-width:900px){.controls,.stats,.grid{grid-template-columns:1fr 1fr}}@media(max-width:600px){.controls,.stats,.grid{grid-template-columns:1fr}}
</style></head><body><main class="wrap">
<div class="card"><div class="eyebrow">Experience Bridge Observer v0.1</div><h1>経験が、別の経験を呼び起こす橋になり得るか</h1><p class="muted">共通骨格を除いた活動差分・経路順序・エッジ共有からBridge Candidateを観測します。</p></div>
<div class="card"><form method="get"><div class="controls">
<div><div class="eyebrow">Runs</div><input name="runs" type="number" min="2" value="{{s.runs}}"></div>
<div><div class="eyebrow">Min Chain Length</div><input name="length" type="number" min="2" value="{{s.min_chain_length}}"></div>
<div><div class="eyebrow">Skeleton Ratio %</div><input name="skeleton" type="number" min="10" max="100" value="{{(s.skeleton_ratio*100)|int}}"></div>
<div><div class="eyebrow">Min Bridge Score %</div><input name="score" type="number" min="0" max="100" value="{{(s.min_bridge_score*100)|int}}"></div>
<div><button>Bridgeを解析</button></div></div></form>
<p class="note">文章や単語一致はBridge判定に使いません。これは実際の再活性化ではなく、将来Weak Reactivationを試す前の観測専用候補です。Coreとmemory.dbは変更しません。</p></div>
<div class="card stats"><div class="stat"><div class="eyebrow">Latest Run</div><div class="value">#{{latest or '-'}}</div></div><div class="stat"><div class="eyebrow">Stable Experiences</div><div class="value">{{experiences|length}}</div></div><div class="stat"><div class="eyebrow">Skeleton Nodes</div><div class="value">{{skeleton|length}}</div></div><div class="stat"><div class="eyebrow">Bridge Candidates</div><div class="value">{{bridges|length}}</div></div></div>
<div class="card"><div class="eyebrow">Hypothesis EB-001</div><h2>言葉は、既存の経験構造を再接続し、別の経験を弱く呼び起こす触媒になり得る</h2></div>
<div class="card"><div class="eyebrow">Common Skeleton</div><h2>全経験に共通しやすい処理骨格</h2><div class="chain">{% for n in skeleton|sort %}<span class="node">{{n}}</span>{% endfor %}</div></div>
<div class="card"><div class="eyebrow">Experience Signatures</div><h2>共通骨格を除いた経験差分</h2><div class="grid">{% for e in experiences %}<div class="experience"><h3>{{e.text}}</h3><div class="muted">Full {{e.chain|length}} nodes / Residual {{e.residual_chain|length}} nodes</div><div class="chain">{% for n in e.residual_chain %}<span class="node">{{n}}</span>{% else %}<span class="muted">固有差分なし</span>{% endfor %}</div></div>{% endfor %}</div></div>
<div class="card"><div class="eyebrow">Bridge Candidates</div><h2>一方の経験が他方を呼び起こし得る候補</h2><div class="grid">{% for b in bridges %}<div class="bridge"><h3>{{b.left}} ↔ {{b.right}}</h3><div class="value">{{b.score}}%</div><div class="muted">Bridge Score</div><p>Raw overlap {{b.raw}}% / Residual nodes {{b.residual}}% / Residual edges {{b.edge}}%</p><div class="good">共有差分経路</div><div class="chain">{% for n in b.shared %}<span class="node">{{n}}</span>{% else %}<span class="muted">順序共有なし</span>{% endfor %}</div><div class="difference">{{b.left}}のみ: {{b.left_only or '-'}}<br>{{b.right}}のみ: {{b.right_only or '-'}}</div></div>{% else %}<p class="muted">現在の閾値ではBridge Candidateなし。これは重要な観測結果です。</p>{% endfor %}</div></div>
</main></body></html>
"""


@app.route("/")
def index():
    s = Settings(
        runs=max(2, int(request.args.get("runs", 3))),
        min_chain_length=max(2, int(request.args.get("length", 3))),
        skeleton_ratio=min(1.0, max(0.1, float(request.args.get("skeleton", 60)) / 100)),
        min_bridge_score=min(1.0, max(0.0, float(request.args.get("score", 25)) / 100)),
    )
    raw, run_ids, latest = load_experiences(s)
    bridges, experiences, skeleton = analyze_bridges(raw, s)
    return render_template_string(PAGE, s=s, bridges=bridges, experiences=experiences, skeleton=skeleton, latest=latest, run_ids=run_ids)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5058")
    serve(app, host="127.0.0.1", port=5058)
