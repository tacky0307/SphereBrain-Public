from __future__ import annotations

import json
import random
import sys
import threading
from collections import deque
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline

SIZE = 15
SEED = 20260805
TRAIN_REPEATS = 3
PORT = 5084
APP_DATA = ROOT / "data" / "llm_core_pe_maze"
CORE_DATA = APP_DATA / "core"
CACHE_FILE = APP_DATA / "embedding_cache.json"
POLICY_FILE = APP_DATA / "policy.json"
LOCK = threading.RLock()
STATUS = {"ready": False, "running": False, "message": "未学習", "error": ""}

DIRS = {
    "上": (-1, 0),
    "下": (1, 0),
    "左": (0, -1),
    "右": (0, 1),
}


class CachedAdapter(pipeline.OpenAIAdapter):
    def __init__(self) -> None:
        super().__init__()
        APP_DATA.mkdir(parents=True, exist_ok=True)
        try:
            self.cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
        except Exception:
            self.cache = {}

    def embed(self, text: str) -> list[float]:
        key = text.strip()
        if key in self.cache:
            return list(self.cache[key])
        value = super().embed(key)
        self.cache[key] = value
        CACHE_FILE.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return value


def configure_core() -> None:
    pipeline.DATA = CORE_DATA
    pipeline.BRAIN_FILE = CORE_DATA / "brain.json"
    pipeline.DB_FILE = CORE_DATA / "experiences.db"
    pipeline.PROJECTION_FILE = CORE_DATA / "projection.npy"
    pipeline.PROJECTION_SEED = 20260804


def make_maze() -> list[list[int]]:
    grid = [[1 for _ in range(SIZE)] for _ in range(SIZE)]
    rng = random.Random(SEED)
    start = (1, 1)
    grid[1][1] = 0
    stack = [start]
    while stack:
        r, c = stack[-1]
        candidates = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < SIZE - 1 and 1 <= nc < SIZE - 1 and grid[nr][nc] == 1:
                candidates.append((nr, nc, dr // 2, dc // 2))
        if not candidates:
            stack.pop()
            continue
        nr, nc, wr, wc = rng.choice(candidates)
        grid[r + wr][c + wc] = 0
        grid[nr][nc] = 0
        stack.append((nr, nc))
    return grid


MAZE = make_maze()
START = (1, 1)
GOAL = (SIZE - 2, SIZE - 2)
MAZE[GOAL[0]][GOAL[1]] = 0


def shortest_policy() -> dict[tuple[int, int], str]:
    q = deque([GOAL])
    distance = {GOAL: 0}
    while q:
        r, c = q.popleft()
        for _, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < SIZE and 0 <= nc < SIZE and MAZE[nr][nc] == 0 and (nr, nc) not in distance:
                distance[(nr, nc)] = distance[(r, c)] + 1
                q.append((nr, nc))

    policy = {}
    for pos, d in distance.items():
        if pos == GOAL:
            policy[pos] = "停止"
            continue
        r, c = pos
        choices = []
        for name, (dr, dc) in DIRS.items():
            nxt = (r + dr, c + dc)
            if nxt in distance and distance[nxt] < d:
                choices.append((distance[nxt], name))
        if choices:
            policy[pos] = min(choices)[1]
    return policy


POLICY = shortest_policy()


def available_moves(pos: tuple[int, int]) -> list[str]:
    r, c = pos
    result = []
    for name, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE and MAZE[nr][nc] == 0:
            result.append(name)
    return result


def state_text(pos: tuple[int, int]) -> str:
    r, c = pos
    gr, gc = GOAL
    vertical = "同じ高さ" if r == gr else ("上" if r < gr else "下")
    horizontal = "同じ列" if c == gc else ("左" if c < gc else "右")
    moves = "・".join(available_moves(pos)) or "なし"
    return (
        f"15行15列の迷路。Pは{r + 1}行{c + 1}列、Eは{gr + 1}行{gc + 1}列。"
        f"PはEより{vertical}、{horizontal}にある。移動可能方向は{moves}。"
    )


def observe(text: str, adapter: CachedAdapter) -> dict:
    _, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(sources, steps=14, threshold=0.18, noise=0.0, learn=False)
    return {
        "nodes": list(result.activated_nodes),
        "edges": [list(edge) for edge in result.traversed_edges],
    }


def overlap(a: dict, b: dict) -> float:
    an, bn = set(a["nodes"]), set(b["nodes"])
    ae, be = {tuple(x) for x in a["edges"]}, {tuple(x) for x in b["edges"]}
    nj = len(an & bn) / len(an | bn) if an | bn else 0.0
    ej = len(ae & be) / len(ae | be) if ae | be else 0.0
    return 0.35 * nj + 0.65 * ej


def train_core() -> None:
    with LOCK:
        if STATUS["running"]:
            return
        STATUS.update({"running": True, "ready": False, "message": "学習開始", "error": ""})
    try:
        configure_core()
        adapter = CachedAdapter()
        pipeline.reset_experiment()
        records = []
        positions = sorted(POLICY)
        for index, pos in enumerate(positions, start=1):
            text = state_text(pos)
            action = POLICY[pos]
            with LOCK:
                STATUS["message"] = f"学習中 {index}/{len(positions)}：{pos} → {action}"
            pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)
            records.append({
                "position": list(pos),
                "text": text,
                "action": action,
                "route": observe(text, adapter),
            })
        APP_DATA.mkdir(parents=True, exist_ok=True)
        POLICY_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        with LOCK:
            STATUS.update({"running": False, "ready": True, "message": f"学習完了：{len(records)}状態"})
    except Exception as exc:
        with LOCK:
            STATUS.update({"running": False, "ready": False, "message": "学習失敗", "error": str(exc)})


def choose_action(pos: tuple[int, int]) -> tuple[str, float, tuple[int, int]]:
    if not POLICY_FILE.exists():
        raise RuntimeError("先にLLM→Core学習を実行してください。")
    configure_core()
    adapter = CachedAdapter()
    current = observe(state_text(pos), adapter)
    records = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    ranked = []
    for record in records:
        score = overlap(current, record["route"])
        ranked.append((score, record["action"], tuple(record["position"])))
    ranked.sort(key=lambda x: x[0], reverse=True)
    valid = set(available_moves(pos)) | {"停止"}
    for score, action, learned_pos in ranked:
        if action in valid:
            return action, score, learned_pos
    return "停止", 0.0, pos


def move(pos: tuple[int, int], action: str) -> tuple[int, int]:
    if action not in DIRS:
        return pos
    dr, dc = DIRS[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    return nxt if MAZE[nxt[0]][nxt[1]] == 0 else pos


PAGE = r"""
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain P → E Maze</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--card:#14243d;--line:#35577f;--accent:#ef9858;--text:#f6f8fc;--muted:#9fc1e8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,"Yu Gothic UI",sans-serif}main{max-width:1180px;margin:auto;padding:24px}.panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;margin-bottom:18px}.layout{display:grid;grid-template-columns:minmax(420px,720px) 1fr;gap:20px}.maze{display:grid;grid-template-columns:repeat(15,1fr);aspect-ratio:1;border:2px solid #6d8fb5}.cell{display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,255,255,.05);font-weight:800;font-size:clamp(12px,2.2vw,24px)}.wall{background:#020914}.floor{background:#17314e}.path{background:#224c68}.player{background:#ef9858;color:#07111f}.goal{background:#72d39b;color:#07111f}button{border:0;border-radius:11px;padding:12px 17px;margin:5px;font-weight:750;font-size:15px;cursor:pointer;background:var(--accent);color:white}.secondary{background:#31567d}.status{color:var(--muted);line-height:1.7}.big{font-size:22px}.metrics{display:grid;gap:10px}.metric{background:#091827;border:1px solid #31577e;border-radius:11px;padding:12px}.metric b{display:block;font-size:20px}@media(max-width:850px){.layout{grid-template-columns:1fr}}
</style></head><body><main>
<h1>SphereBrain P → E パズル</h1><p class="status">15×15の迷路。LLMが盤面状態を数値刺激へ変え、Coreが経験経路から次の一手を選びます。</p>
<div class="panel"><button id="train" class="secondary">LLM → Coreに最短経路を学習</button><span id="status" class="status">状態確認中…</span></div>
<div class="layout"><section class="panel"><div id="maze" class="maze"></div></section><aside class="panel">
<div class="metrics"><div class="metric"><span>現在位置</span><b id="pos">-</b></div><div class="metric"><span>手数</span><b id="steps">0</b></div><div class="metric"><span>Coreの選択</span><b id="action">未実行</b></div><div class="metric"><span>経路重なり</span><b id="score">-</b></div><div class="metric"><span>参照した学習位置</span><b id="ref">-</b></div></div>
<div style="margin-top:16px"><button id="one">Coreで一手</button><button id="auto">自動でEへ</button><button id="reset" class="secondary">リセット</button></div>
<p id="message" class="status"></p></aside></div>
<script>
let state={position:[1,1],steps:0,trail:[]};
async function status(){const r=await fetch('/api/status');const s=await r.json();document.getElementById('status').textContent=s.message+(s.error?'：'+s.error:'');document.getElementById('one').disabled=!s.ready||s.running;document.getElementById('auto').disabled=!s.ready||s.running;if(s.running)setTimeout(status,1200)}
function draw(data){const maze=document.getElementById('maze');maze.innerHTML='';const trail=new Set(data.trail.map(x=>x.join(',')));for(let r=0;r<data.maze.length;r++)for(let c=0;c<data.maze[r].length;c++){const d=document.createElement('div');let cls=data.maze[r][c]?'wall':'floor';if(trail.has(`${r},${c}`))cls+=' path';if(r===data.goal[0]&&c===data.goal[1]){cls='goal';d.textContent='E'}if(r===data.position[0]&&c===data.position[1]){cls='player';d.textContent='P'}d.className='cell '+cls;maze.appendChild(d)}document.getElementById('pos').textContent=`${data.position[0]+1}行 ${data.position[1]+1}列`;document.getElementById('steps').textContent=data.steps}
async function reset(){const r=await fetch('/api/reset',{method:'POST'});state=await r.json();draw(state);document.getElementById('action').textContent='未実行';document.getElementById('score').textContent='-';document.getElementById('ref').textContent='-';document.getElementById('message').textContent=''}
async function step(){const r=await fetch('/api/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({position:state.position,steps:state.steps,trail:state.trail})});const d=await r.json();if(!r.ok)throw new Error(d.error||'失敗');state=d;draw(state);document.getElementById('action').textContent=d.action;document.getElementById('score').textContent=(d.score*100).toFixed(1)+'%';document.getElementById('ref').textContent=`${d.learned_position[0]+1}行 ${d.learned_position[1]+1}列`;document.getElementById('message').textContent=d.finished?'Eに到着しました！':'Coreが次の一手を選びました。';return d.finished}
document.getElementById('train').onclick=async()=>{await fetch('/api/train',{method:'POST'});status()};document.getElementById('one').onclick=()=>step().catch(e=>document.getElementById('message').textContent=e.message);document.getElementById('reset').onclick=reset;document.getElementById('auto').onclick=async()=>{for(let i=0;i<300;i++){if(await step())break;await new Promise(r=>setTimeout(r,180))}};reset();status();
</script></body></html>
"""

app = Flask(__name__)

@app.get("/")
def index():
    return render_template_string(PAGE)

@app.get("/api/status")
def api_status():
    if POLICY_FILE.exists() and not STATUS["running"]:
        STATUS["ready"] = True
        if STATUS["message"] == "未学習":
            STATUS["message"] = "学習済み"
    return jsonify(STATUS)

@app.post("/api/train")
def api_train():
    if not STATUS["running"]:
        threading.Thread(target=train_core, daemon=True).start()
    return jsonify({"started": True})

@app.post("/api/reset")
def api_reset():
    return jsonify({"maze": MAZE, "position": list(START), "goal": list(GOAL), "steps": 0, "trail": [list(START)]})

@app.post("/api/step")
def api_step():
    try:
        payload = request.get_json(force=True)
        pos = tuple(payload.get("position", START))
        steps = int(payload.get("steps", 0))
        trail = [list(x) for x in payload.get("trail", [list(START)])]
        action, score, learned_pos = choose_action(pos)
        nxt = move(pos, action)
        if list(nxt) not in trail:
            trail.append(list(nxt))
        return jsonify({
            "maze": MAZE,
            "position": list(nxt),
            "goal": list(GOAL),
            "steps": steps + 1,
            "trail": trail,
            "action": action,
            "score": score,
            "learned_position": list(learned_pos),
            "finished": nxt == GOAL,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def main() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    print("SphereBrain P-to-E maze")
    print(f"http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
