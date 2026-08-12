from __future__ import annotations

import hashlib
import json
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_core_assist import StructuralAssistConfig, StructuralCoreAssist

HOST = "127.0.0.1"
PORT = 5033
BRAIN_PATH = ROOT / "data" / "brain.json"
OUT = ROOT / "data" / "structural_grid_puzzle_v1" / "results"

GRID = 3
START = 0
GOAL = 8
BLOCKED = {3, 4}
ROLE = {0: "P", 8: "G"}
DIRECTIONS = {"上": (-1, 0), "下": (1, 0), "左": (0, -1), "右": (0, 1)}


@dataclass
class MiniBrain:
    positions: np.ndarray
    adjacency: np.ndarray
    weights: np.ndarray
    usage: np.ndarray
    neighbors_per_node: int = 4


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def rc(node: int) -> tuple[int, int]:
    return divmod(node, GRID)


def node_at(row: int, col: int) -> int | None:
    if 0 <= row < GRID and 0 <= col < GRID:
        return row * GRID + col
    return None


def build_brain() -> MiniBrain:
    n = GRID * GRID
    positions = np.zeros((n, 3), dtype=float)
    adjacency = np.zeros((n, n), dtype=bool)
    weights = np.zeros((n, n), dtype=float)
    usage = np.zeros((n, n), dtype=int)

    for node in range(n):
        row, col = rc(node)
        positions[node] = ((col - 1) * 0.55, (1 - row) * 0.55, 0.0)

    for node in range(n):
        if node in BLOCKED:
            continue
        row, col = rc(node)
        for dr, dc in DIRECTIONS.values():
            target = node_at(row + dr, col + dc)
            if target is None or target in BLOCKED:
                continue
            adjacency[node, target] = True
            weights[node, target] = 0.62

    return MiniBrain(positions, adjacency, weights, usage)


class PuzzleSession:
    def __init__(self) -> None:
        self.before_hash = file_hash(BRAIN_PATH)
        self.reset()

    def reset(self) -> None:
        self.brain = build_brain()
        self.current = START
        self.path = [START]
        self.history = [[START]]
        self.edges_by_step: list[list[tuple[int, int]]] = []
        self.turn = 0
        self.speech = ["準備ができました。"]
        self.last = self.observe()

    def candidates(self) -> list[dict]:
        row, col = rc(self.current)
        found = []
        for label, (dr, dc) in DIRECTIONS.items():
            target = node_at(row + dr, col + dc)
            if target is None or target in BLOCKED:
                continue
            if not self.brain.adjacency[self.current, target]:
                continue
            found.append({"label": label, "target": target})
        return found

    def rank(self, enabled: bool) -> tuple[list[dict], dict]:
        candidates = self.candidates()
        ranked = [(item["target"], (0.5, self.current)) for item in candidates]
        assist = StructuralCoreAssist(StructuralAssistConfig(enabled=enabled))
        reordered, trace = assist.reorder(
            self.brain,
            ranked,
            self.history,
            self.edges_by_step,
        )
        by_target = {item["target"]: item for item in candidates}
        result = []
        for target, _payload in reordered:
            item = dict(by_target[target])
            item["visited"] = target in self.path
            item["degree"] = int(np.count_nonzero(self.brain.adjacency[target]))
            item["usage"] = int(self.brain.usage[self.current, target])
            result.append(item)
        return result, trace

    def observe(self) -> dict:
        off, off_trace = self.rank(False)
        on, on_trace = self.rank(True)
        return {
            "off": off,
            "on": on,
            "off_trace": off_trace,
            "on_trace": on_trace,
            "top_changed": bool(on_trace.get("top_candidate_changed")),
        }

    def move(self, target: int) -> dict:
        legal = {item["target"] for item in self.candidates()}
        if target not in legal:
            raise ValueError("そのマスへは移動できません。")
        source = self.current
        self.current = target
        self.turn += 1
        self.path.append(target)
        self.edges_by_step.append([(source, target)])
        self.history.append([target])
        self.brain.usage[source, target] += 1
        self.brain.usage[target, source] += 1
        if target == GOAL:
            self.speech.append(f"{self.turn}手でゴールに到達しました。")
        else:
            self.speech.append(f"{self.turn}手目：Node {source} から Node {target} へ移動しました。")
        self.last = self.observe()
        self.save_result()
        return self.state()

    def auto_step(self, mode: str) -> dict:
        ranking = self.last["on" if mode == "on" else "off"]
        if not ranking:
            raise ValueError("移動候補がありません。")
        unvisited = [item for item in ranking if not item["visited"]]
        chosen = (unvisited or ranking)[0]
        return self.move(int(chosen["target"]))

    def save_result(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        payload = self.state()
        payload["brain_file_unchanged"] = self.before_hash == file_hash(BRAIN_PATH)
        (OUT / "structural_grid_puzzle_v1.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def state(self) -> dict:
        cells = []
        for node in range(GRID * GRID):
            cells.append({
                "node": node,
                "blocked": node in BLOCKED,
                "player": node == self.current,
                "goal": node == GOAL,
                "visited": node in self.path,
                "label": "P" if node == self.current else ROLE.get(node, ""),
            })
        return {
            "experiment": "Structural Grid Puzzle v1",
            "turn": self.turn,
            "current": self.current,
            "goal": GOAL,
            "finished": self.current == GOAL,
            "cells": cells,
            "path": self.path,
            "history": self.history,
            "edges_by_step": self.edges_by_step,
            "comparison": self.last,
            "speech": self.speech,
            "brain_file_unchanged": self.before_hash == file_hash(BRAIN_PATH),
        }


app = Flask(__name__)
session = PuzzleSession()

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SphereWorld Puzzle — Structural Assist</title><style>
:root{--bg:#09111e;--panel:#17253c;--panel2:#0e1929;--line:#385273;--text:#f2f6ff;--muted:#a8b7ce;--blue:#8ed8ff;--green:#8cf0ae;--orange:#ffb35c}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#111f35);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1320px;margin:auto;padding:28px}.title{font-size:clamp(30px,4vw,52px);margin:0}.lead{color:var(--muted)}.layout{display:grid;grid-template-columns:minmax(420px,1fr) minmax(460px,1.15fr);gap:22px}.panel{background:rgba(23,37,60,.95);border:1px solid var(--line);border-radius:20px;padding:24px}.board{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.cell{aspect-ratio:1;border:1px solid #466386;border-radius:18px;background:linear-gradient(145deg,#243a58,#1a2c47);display:flex;align-items:center;justify-content:center;font-size:52px;font-weight:800;position:relative}.cell.blocked{background:repeating-linear-gradient(45deg,#101a2b,#101a2b 12px,#223149 12px,#223149 24px)}.cell.visited:after{content:"";position:absolute;inset:9px;border:2px dashed #6ca9cf55;border-radius:13px}.cell.player{outline:3px solid var(--blue)}.cell.goal{color:var(--green)}.say{background:#0c1726;border-left:5px solid var(--orange);padding:20px;border-radius:16px;margin-bottom:18px}.say small{color:var(--blue);letter-spacing:.14em;font-weight:800}.say strong{display:block;color:var(--green);font-size:30px;margin-top:8px}.chips{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.chip{border:1px solid #3a5a80;border-radius:99px;padding:7px 11px;color:#bcd0ea}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.rank{background:var(--panel2);border-radius:14px;padding:15px}.rank h3{margin-top:0}.candidate{display:flex;justify-content:space-between;border-top:1px solid #263b57;padding:10px 0}.changed{color:#e9c4ff}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}button{border:0;border-radius:12px;padding:12px 16px;font-weight:800;cursor:pointer;background:#8ed8ff;color:#07101d}button.secondary{background:#2a405e;color:#eef4ff}button.on{background:#b897ff}button:disabled{opacity:.35;cursor:not-allowed}.log{max-height:190px;overflow:auto;background:#091321;border-radius:12px;padding:12px;color:#b9c9dd}.safe{color:var(--green)}@media(max-width:900px){.layout{grid-template-columns:1fr}}
</style></head><body><main><h1 class="title">SphereWorld Puzzle</h1><p class="lead">3×3世界を、現在のStructural Assist OFF / ONで同時観測する。</p><div class="layout"><section class="panel"><h2>パズル世界</h2><div id="board" class="board"></div><div class="controls"><button class="secondary" onclick="resetGame()">リセット</button><button onclick="autoStep('off')">構造OFFで1手</button><button class="on" onclick="autoStep('on')">構造ONで1手</button></div></section><section class="panel"><div class="say"><small>SPHEREBRAIN SAYS</small><strong id="message">準備ができました。</strong></div><div id="chips" class="chips"></div><h2>行動候補</h2><div class="compare"><div class="rank"><h3>構造OFF</h3><div id="off"></div></div><div class="rank"><h3>構造ON</h3><div id="on"></div></div></div><h2>Structural Assist Trace</h2><pre id="trace" class="log"></pre><h2>発話履歴</h2><div id="speech" class="log"></div></section></div></main><script>
let state=null;async function api(url,opts={}){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opts});const j=await r.json();if(!r.ok)throw new Error(j.error||'error');return j}function renderRank(id,items){document.getElementById(id).innerHTML=items.length?items.map((x,i)=>`<div class="candidate"><span>${i+1}. ${x.label} → Node ${x.target}${x.visited?'（既通過）':''}</span><button onclick="move(${x.target})">選択</button></div>`).join(''):'候補なし'}function render(s){state=s;document.getElementById('board').innerHTML=s.cells.map(c=>`<div class="cell ${c.blocked?'blocked':''} ${c.visited?'visited':''} ${c.player?'player':''} ${c.goal?'goal':''}">${c.blocked?'':c.label}</div>`).join('');document.getElementById('message').textContent=s.finished?`${s.turn}手でゴールしました。`:(s.speech.at(-1)||'準備ができました。');document.getElementById('chips').innerHTML=`<span class="chip">${s.turn}手</span><span class="chip">Node ${s.current}</span><span class="chip ${s.brain_file_unchanged?'safe':''}">brain.json ${s.brain_file_unchanged?'不変':'変化'}</span><span class="chip ${s.comparison.top_changed?'changed':''}">順位${s.comparison.top_changed?'変化':'維持'}</span>`;renderRank('off',s.comparison.off);renderRank('on',s.comparison.on);document.getElementById('trace').textContent=JSON.stringify(s.comparison.on_trace,null,2);document.getElementById('speech').innerHTML=s.speech.map(x=>`<div>${x}</div>`).join('')}async function load(){render(await api('/api/state'))}async function move(target){try{render(await api('/api/move',{method:'POST',body:JSON.stringify({target})}))}catch(e){alert(e.message)}}async function autoStep(mode){try{render(await api('/api/auto',{method:'POST',body:JSON.stringify({mode})}))}catch(e){alert(e.message)}}async function resetGame(){render(await api('/api/reset',{method:'POST'}))}load();
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.get("/api/state")
def api_state():
    return jsonify(session.state())


@app.post("/api/reset")
def api_reset():
    session.reset()
    return jsonify(session.state())


@app.post("/api/move")
def api_move():
    try:
        target = int((request.get_json(silent=True) or {}).get("target"))
        return jsonify(session.move(target))
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/auto")
def api_auto():
    try:
        mode = str((request.get_json(silent=True) or {}).get("mode", "on"))
        if mode not in {"on", "off"}:
            raise ValueError("modeはonまたはoffです。")
        return jsonify(session.auto_step(mode))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    print(f"Structural Grid Puzzle v1: http://{HOST}:{PORT}")
    print("learning OFF / noise OFF / brain.json saveなし")
    serve(app, host=HOST, port=PORT)
