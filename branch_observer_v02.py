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
    min_chain_support: int = 2
    family_similarity: float = 0.45
    max_experiences: int = 20


def _load_json(value: object, default):
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def _tables_exist(conn: sqlite3.Connection) -> bool:
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('reflection_runs','reflection_pattern_snapshots')"
        )
    }
    return names == {"reflection_runs", "reflection_pattern_snapshots"}


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _lcs(left: list[int], right: list[int]) -> list[int]:
    if not left or not right:
        return []
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
    out: list[int] = []
    i, j = len(left), len(right)
    while i and j:
        if left[i - 1] == right[j - 1]:
            out.append(left[i - 1]); i -= 1; j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(out))


def load_chains(settings: Settings) -> tuple[dict[str, dict], list[int], int | None]:
    if not SOURCE_DB.exists():
        return {}, [], None
    uri = f"file:{SOURCE_DB.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        if not _tables_exist(conn):
            return {}, [], None
        latest = conn.execute("SELECT MAX(run_id) FROM reflection_runs").fetchone()[0]
        if latest is None:
            return {}, [], None
        run_ids = [row[0] for row in conn.execute(
            "SELECT run_id FROM reflection_runs WHERE run_id<=? ORDER BY run_id DESC LIMIT ?",
            (latest, settings.runs),
        )]
        placeholders = ",".join("?" for _ in run_ids)
        rows = conn.execute(
            f"SELECT * FROM reflection_pattern_snapshots WHERE run_id IN ({placeholders}) "
            "AND classification='concept candidate' ORDER BY run_id, pattern_id",
            run_ids,
        ).fetchall()

    # A text may be attached to several overlapping route fragments. Merge fragments into
    # one observable chain by ordering/overlapping them rather than treating each fragment
    # as an independent decision.
    by_text_run: dict[tuple[str, int], list[list[int]]] = defaultdict(list)
    for row in rows:
        route = [int(x) for x in _load_json(row["pattern_json"], [])]
        if len(route) < 2:
            continue
        texts = [str(x) for x in _load_json(row["target_texts"], [])] or ["(unlabelled experience)"]
        for text in texts:
            by_text_run[(text, int(row["run_id"]))].append(route)

    def merge_fragments(fragments: list[list[int]]) -> list[int]:
        unique = []
        seen = set()
        for frag in fragments:
            key = tuple(frag)
            if key not in seen:
                unique.append(frag); seen.add(key)
        unique.sort(key=lambda x: (-len(x), x))
        chain = unique[0][:] if unique else []
        remaining = unique[1:]
        changed = True
        while remaining and changed:
            changed = False
            for frag in remaining[:]:
                best = 0
                for size in range(min(len(chain), len(frag)), 0, -1):
                    if chain[-size:] == frag[:size]:
                        chain.extend(frag[size:]); best = size; break
                    if frag[-size:] == chain[:size]:
                        chain = frag[:-size] + chain; best = size; break
                if best:
                    remaining.remove(frag); changed = True
        return chain

    raw: dict[str, dict] = defaultdict(lambda: {"runs": {}, "chains": [], "nodes": set()})
    for (text, run_id), fragments in by_text_run.items():
        chain = merge_fragments(fragments)
        if len(chain) < settings.min_chain_length:
            continue
        raw[text]["runs"][run_id] = chain
        raw[text]["chains"].append(chain)
        raw[text]["nodes"].update(chain)

    experiences: dict[str, dict] = {}
    for text, item in raw.items():
        if len(item["runs"]) < settings.runs:
            continue
        ordered = [item["runs"][run_id] for run_id in run_ids if run_id in item["runs"]]
        common = ordered[0]
        for chain in ordered[1:]:
            common = _lcs(common, chain)
        if len(common) < settings.min_chain_length:
            continue
        edge_counts = Counter()
        for chain in ordered:
            edge_counts.update(zip(chain, chain[1:]))
        stable_edges = [edge for edge, count in edge_counts.items() if count >= settings.min_chain_support]
        experiences[text] = {
            "text": text,
            "chain": common,
            "length": len(common),
            "stable_edges": stable_edges,
            "node_set": set(common),
            "run_chains": ordered,
        }
    return experiences, run_ids, latest


def compare_experiences(experiences: dict[str, dict]) -> list[dict]:
    names = sorted(experiences)
    comparisons: list[dict] = []
    for i, left_name in enumerate(names):
        left = experiences[left_name]
        for right_name in names[i + 1:]:
            right = experiences[right_name]
            shared = _lcs(left["chain"], right["chain"])
            denominator = max(len(left["chain"]), len(right["chain"]), 1)
            similarity = len(shared) / denominator
            if not shared:
                continue
            left_only = [n for n in left["chain"] if n not in set(shared)]
            right_only = [n for n in right["chain"] if n not in set(shared)]
            comparisons.append({
                "left": left_name,
                "right": right_name,
                "similarity": round(similarity * 100, 1),
                "shared": shared,
                "left_only": left_only,
                "right_only": right_only,
            })
    comparisons.sort(key=lambda x: (-x["similarity"], -len(x["shared"]), x["left"], x["right"]))
    return comparisons


def build_families(experiences: dict[str, dict], threshold: float) -> list[dict]:
    names = sorted(experiences)
    adjacency = {name: set() for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            score = _jaccard(experiences[left]["node_set"], experiences[right]["node_set"])
            if score >= threshold:
                adjacency[left].add(right); adjacency[right].add(left)
    visited = set(); families = []
    for start in names:
        if start in visited:
            continue
        stack = [start]; visited.add(start); members = []
        while stack:
            current = stack.pop(); members.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor); stack.append(neighbor)
        if len(members) < 2:
            continue
        counts = Counter()
        for name in members:
            counts.update(experiences[name]["chain"])
        core_threshold = max(2, round(len(members) * 0.6))
        core_nodes = sorted(node for node, count in counts.items() if count >= core_threshold)
        families.append({"members": sorted(members), "core_nodes": core_nodes, "size": len(members)})
    families.sort(key=lambda x: (-x["size"], -len(x["core_nodes"]), x["members"]))
    return families


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Branch Observer v0.2</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--orange:#ff9d52;--green:#8ce3a9;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(80,145,210,.17),transparent 35%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1450px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.controls,.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.stats{grid-template-columns:repeat(4,1fr)}.stat,.experience,.comparison,.family{background:#071522;border:1px solid var(--line);border-radius:14px;padding:16px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.value{font-size:30px;font-weight:800;margin-top:5px}input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:10px;font-size:15px}button{background:linear-gradient(135deg,#ec6f35,#ff9d52);border:0;color:white;border-radius:10px;padding:12px 18px;font-weight:800}.chain{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:12px 0}.node{border:1px solid var(--line);background:#0d2035;border-radius:999px;padding:7px 11px;color:var(--cyan);font-weight:700}.arrow{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.muted{color:var(--muted)}.shared{color:var(--green)}.difference{color:var(--yellow)}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}@media(max-width:900px){.controls,.stats,.grid{grid-template-columns:1fr 1fr}}@media(max-width:600px){.controls,.stats,.grid{grid-template-columns:1fr}}
</style></head><body><main class="wrap">
<div class="card"><div class="eyebrow">Branch Observer v0.2</div><h1>分岐点から、分岐の連鎖と地形へ</h1><p class="muted">単独のBranchではなく、同じ経験で繰り返されるBranch Chain、経験間の共有経路、Branch Familyを観測します。</p></div>
<div class="card"><form method="get"><div class="controls">
<div><div class="eyebrow">Runs</div><input name="runs" type="number" min="2" value="{{s.runs}}"></div>
<div><div class="eyebrow">Min Chain Length</div><input name="length" type="number" min="2" value="{{s.min_chain_length}}"></div>
<div><div class="eyebrow">Edge Support</div><input name="support" type="number" min="1" value="{{s.min_chain_support}}"></div>
<div><div class="eyebrow">Family Similarity %</div><input name="family" type="number" min="0" max="100" value="{{(s.family_similarity*100)|int}}"></div>
<div><button>Branch Chainを解析</button></div></div></form>
<p class="note">入力文はChainを作る条件ではありません。数値経路を先に統合し、文章は実世界の経験との対応を確認するラベルとしてのみ表示します。</p></div>
<div class="card stats"><div class="stat"><div class="eyebrow">Latest Run</div><div class="value">#{{latest or '-'}}</div></div><div class="stat"><div class="eyebrow">Stable Experiences</div><div class="value">{{experiences|length}}</div></div><div class="stat"><div class="eyebrow">Comparisons</div><div class="value">{{comparisons|length}}</div></div><div class="stat"><div class="eyebrow">Branch Families</div><div class="value">{{families|length}}</div></div></div>
<div class="card"><div class="eyebrow">Hypothesis B-002</div><h2>知性は単発の選択ではなく、経験によって安定した選択の連鎖として現れる</h2></div>
<div class="card"><div class="eyebrow">Branch Chains</div><h2>経験ごとの安定経路</h2><div class="grid">{% for e in experiences %}<div class="experience"><h3>{{e.text}}</h3><div class="muted">Stable length {{e.length}} / {{e.run_chains|length}} runs</div><div class="chain">{% for n in e.chain %}<span class="node">{{n}}</span>{% if not loop.last %}<span class="arrow">→</span>{% endif %}{% endfor %}</div></div>{% endfor %}</div></div>
<div class="card"><div class="eyebrow">Shared / Diverged</div><h2>経験間の共通経路と分岐</h2><div class="grid">{% for c in comparisons[:20] %}<div class="comparison"><h3>{{c.left}} × {{c.right}}</h3><div class="value">{{c.similarity}}%</div><div class="muted">共有経路</div><div class="chain shared">{% for n in c.shared %}<span class="node">{{n}}</span>{% if not loop.last %}<span>→</span>{% endif %}{% endfor %}</div><div class="difference">{{c.left}}のみ: {{c.left_only or '-'}}<br>{{c.right}}のみ: {{c.right_only or '-'}}</div></div>{% endfor %}</div></div>
<div class="card"><div class="eyebrow">Branch Families</div><h2>繰り返される分岐地形</h2><div class="grid">{% for f in families %}<div class="family"><h3>Family {{loop.index}}</h3><div class="muted">{{f.size}} experiences</div><div>{% for m in f.members %}<span class="pill">{{m}}</span>{% endfor %}</div><p>Core nodes: {{f.core_nodes or 'まだ共通核なし'}}</p></div>{% endfor %}</div></div>
</main></body></html>
"""


@app.get("/")
def index():
    settings = Settings(
        runs=max(2, int(request.args.get("runs", 3))),
        min_chain_length=max(2, int(request.args.get("length", 3))),
        min_chain_support=max(1, int(request.args.get("support", 2))),
        family_similarity=max(0.0, min(1.0, float(request.args.get("family", 45)) / 100.0)),
    )
    experiences, run_ids, latest = load_chains(settings)
    visible = sorted(experiences.values(), key=lambda x: (-x["length"], x["text"]))[:settings.max_experiences]
    visible_map = {item["text"]: item for item in visible}
    comparisons = compare_experiences(visible_map)
    families = build_families(visible_map, settings.family_similarity)
    return render_template_string(PAGE, s=settings, latest=latest, run_ids=run_ids,
                                  experiences=visible, comparisons=comparisons, families=families)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5057")
    serve(app, host="127.0.0.1", port=5057)
