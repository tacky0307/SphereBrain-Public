from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock, Thread
import csv
import io
import random
import time
import webbrowser

from flask import Flask, redirect, render_template_string, request, url_for
from waitress import serve

from brain import SphereBrain
from memory_store import MemoryStore


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"


PRESETS: dict[str, list[str]] = {
    "空": [
        "空は青い",
        "青い空が見える",
        "空を見上げる",
        "空に雲がある",
        "夕焼け空がきれい",
        "夜空に星が見える",
        "曇り空が広がっている",
        "空は広い",
        "鳥が空を飛ぶ",
        "飛行機が空を飛ぶ",
        "空が明るくなった",
        "空が暗くなった",
    ],
    "青": [
        "海は青い",
        "青い花が咲いている",
        "青い車が止まっている",
        "青い服を着る",
        "青いボールが転がる",
        "青信号になった",
        "青い絵の具を使う",
        "青い鳥が飛んでいる",
        "青い屋根の家がある",
        "青い箱を開ける",
        "水面が青く見える",
        "遠くの山が青く見える",
    ],
    "雨": [
        "今日は雨です",
        "雨が降っています",
        "雨音が聞こえる",
        "雨で道がぬれている",
        "傘をさして歩く",
        "雨雲が空を覆っている",
        "小雨が降り始めた",
        "強い雨が降っている",
        "雨上がりに虹が出た",
        "窓に雨粒がついている",
        "雨の日は空が暗い",
        "雨がやんだ",
    ],
    "感情": [
        "私はうれしい",
        "今日は楽しい",
        "笑顔になった",
        "心が落ち着いている",
        "少し悲しい",
        "不安を感じている",
        "安心した",
        "驚いている",
        "寂しい気持ちになった",
        "希望を感じる",
        "怒りを感じた",
        "穏やかな気持ちだ",
    ],
}


@dataclass
class TrainerState:
    running: bool = False
    stop_requested: bool = False
    completed: int = 0
    total: int = 0
    current_text: str = ""
    started_at: str = ""
    finished_at: str = ""
    message: str = "待機中"
    error: str = ""
    dataset_size: int = 0
    cycles: int = 1


state = TrainerState()
state_lock = Lock()
brain_lock = Lock()
brain = SphereBrain.load(BRAIN_FILE) if BRAIN_FILE.exists() else SphereBrain(
    node_count=600,
    neighbors_per_node=8,
)
memory = MemoryStore(DB_FILE)
app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{% if state.running %}<meta http-equiv="refresh" content="2">{% endif %}
<title>SphereBrain Experience Trainer v0.5</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--red:#ff7777}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}
header{border-bottom:1px solid var(--line);background:rgba(7,17,31,.94)} .wrap{max-width:1180px;margin:auto;padding:22px}
h1{margin:0;font-size:27px} header p{margin:6px 0 0;color:var(--muted)}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}.card{background:linear-gradient(180deg,rgba(17,39,66,.97),rgba(12,27,47,.97));border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 14px 40px rgba(0,0,0,.18)}
h2{margin:0 0 15px;font-size:20px}.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.12em;font-size:12px;margin-bottom:7px}
textarea{width:100%;min-height:330px;border:1px solid #31567f;border-radius:13px;background:#071522;color:var(--text);padding:14px;font-size:15px;line-height:1.65;resize:vertical}
label{display:block;color:var(--muted);font-size:13px;margin-bottom:6px}input,select{width:100%;border:1px solid #31567f;border-radius:10px;background:#071522;color:var(--text);padding:11px;font-size:14px}
.options{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:14px 0}.check{display:flex;align-items:center;gap:9px;color:var(--text)}.check input{width:auto}
button{border:0;border-radius:10px;padding:11px 17px;font-weight:700;cursor:pointer;background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:#fff}button.secondary{background:#294766}button.stop{background:#7d2a32}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.progress{height:18px;border:1px solid #31567f;background:#071522;border-radius:999px;overflow:hidden}.bar{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .25s}.big{font-size:34px;font-weight:800;margin:8px 0}.muted{color:var(--muted)}
.status{display:grid;gap:12px}.row{display:flex;justify-content:space-between;gap:15px;padding-bottom:10px;border-bottom:1px solid rgba(36,70,109,.65)}.row span:first-child{color:var(--muted)}
.notice{padding:13px;border-radius:12px;border:1px solid rgba(255,157,82,.35);background:rgba(255,157,82,.09);color:#ffd0a9;margin-bottom:16px}.ok{color:var(--green)}.error{color:var(--red)}
.preset-buttons{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.preset-buttons button{padding:8px 12px;background:#294766}
@media(max-width:850px){.grid{grid-template-columns:1fr}.options{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="wrap"><h1>SphereBrain Experience Trainer v0.5</h1><p>多様な経験を、順序と回数を管理しながらCoreへ与える実験装置</p></div></header>
<main class="wrap">
<div class="notice"><strong>重要：</strong>Observatoryの <code>app.py</code> は停止してからTrainerを起動してください。同じbrain.jsonを二つのプログラムから同時に保存しないためです。</div>
<div class="grid">
<section class="card">
<div class="eyebrow">Dataset</div><h2>経験データセット</h2>
<div class="preset-buttons">
{% for name in presets %}<form method="post" action="/preset"><input type="hidden" name="name" value="{{ name }}"><button type="submit">{{ name }}</button></form>{% endfor %}
<form method="post" action="/preset"><input type="hidden" name="name" value="全部"><button type="submit">4概念すべて</button></form>
</div>
<form method="post" action="/start" enctype="multipart/form-data">
<textarea name="texts" placeholder="1行に1つの経験を入力してください">{{ draft }}</textarea>
<div class="options">
<div><label>周回数</label><input name="cycles" type="number" min="1" max="100" value="{{ default_cycles }}"></div>
<div><label>入力間隔（秒）</label><input name="interval" type="number" min="0" max="10" step="0.05" value="0.10"></div>
<div><label>CSV / TXT読み込み</label><input name="dataset_file" type="file" accept=".csv,.txt"></div>
<div><label>実行方法</label><label class="check"><input type="checkbox" name="shuffle" checked>各周回でランダム順</label></div>
</div>
<div class="actions"><button type="submit" {% if state.running %}disabled{% endif %}>学習を開始する</button></div>
</form>
<p class="muted">CSVは <code>category,text</code> または <code>text</code> 列に対応します。列がない場合は各行を文章として読み込みます。</p>
</section>
<section class="card">
<div class="eyebrow">Training State</div><h2>進行状況</h2>
<div class="status">
<div><div class="big">{{ state.completed }} / {{ state.total }}</div><div class="progress"><div class="bar" style="width:{{ progress }}%"></div></div></div>
<div class="row"><span>状態</span><strong class="{% if state.error %}error{% elif state.running %}ok{% endif %}">{{ state.message }}</strong></div>
<div class="row"><span>現在の経験</span><strong>{{ state.current_text or "まだありません" }}</strong></div>
<div class="row"><span>文章数</span><strong>{{ state.dataset_size }}</strong></div>
<div class="row"><span>周回数</span><strong>{{ state.cycles }}</strong></div>
<div class="row"><span>開始</span><strong>{{ state.started_at or "-" }}</strong></div>
<div class="row"><span>終了</span><strong>{{ state.finished_at or "-" }}</strong></div>
{% if state.error %}<p class="error">{{ state.error }}</p>{% endif %}
<div class="actions">
{% if state.running %}<form method="post" action="/stop"><button class="stop" type="submit">安全に停止する</button></form>{% endif %}
<form method="post" action="/save"><button class="secondary" type="submit">現在のCoreを保存</button></form>
</div>
</div>
</section>
</div>
</main>
</body>
</html>
"""


def parse_dataset(text: str, uploaded: bytes | None = None) -> list[str]:
    lines: list[str] = []
    if uploaded:
        decoded = uploaded.decode("utf-8-sig", errors="replace")
        try:
            rows = list(csv.DictReader(io.StringIO(decoded)))
            if rows and rows[0]:
                for row in rows:
                    value = (row.get("text") or row.get("文章") or "").strip()
                    if value:
                        lines.append(value)
            else:
                lines.extend(decoded.splitlines())
        except csv.Error:
            lines.extend(decoded.splitlines())
    lines.extend(text.splitlines())

    result: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        value = " ".join(raw.strip().split())
        if not value or value.startswith("#") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def save_core() -> None:
    with brain_lock:
        brain.save(BRAIN_FILE)


def ingest_one(text: str) -> None:
    sources = brain.text_to_sources(text, count=4)
    context = memory.recent_context_nodes(memory_limit=7, node_limit=18)
    with brain_lock:
        result = brain.propagate(
            source_nodes=sources,
            context_nodes=context,
            steps=20,
            learn=True,
        )
    memory.add_memory(
        kind="trainer",
        input_text=text,
        source_nodes=result.source_nodes,
        activated_nodes=result.activated_nodes,
        traversed_edges=result.traversed_edges,
        importance=1.0,
    )


def training_worker(texts: list[str], cycles: int, shuffle: bool, interval: float) -> None:
    try:
        completed = 0
        for cycle in range(cycles):
            ordered = texts.copy()
            if shuffle:
                random.shuffle(ordered)
            for text in ordered:
                with state_lock:
                    if state.stop_requested:
                        state.running = False
                        state.message = "停止しました"
                        state.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
                        save_core()
                        return
                    state.current_text = text
                    state.message = f"学習中（{cycle + 1}/{cycles}周）"
                ingest_one(text)
                completed += 1
                with state_lock:
                    state.completed = completed
                if completed % 10 == 0:
                    save_core()
                if interval > 0:
                    time.sleep(interval)
        save_core()
        with state_lock:
            state.running = False
            state.message = "学習完了"
            state.current_text = ""
            state.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        with state_lock:
            state.running = False
            state.message = "エラーで停止"
            state.error = str(exc)
            state.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")


@app.route("/")
def index():
    with state_lock:
        snapshot = TrainerState(**asdict(state))
    progress = 0.0 if snapshot.total <= 0 else round(snapshot.completed / snapshot.total * 100, 1)
    draft = memory.get_value("trainer_draft", "")
    return render_template_string(
        PAGE,
        state=snapshot,
        progress=progress,
        presets=PRESETS.keys(),
        draft=draft,
        default_cycles=5,
    )


@app.post("/preset")
def preset():
    name = request.form.get("name", "")
    if name == "全部":
        texts = [text for values in PRESETS.values() for text in values]
    else:
        texts = PRESETS.get(name, [])
    memory.set_value("trainer_draft", "\n".join(texts))
    return redirect(url_for("index"))


@app.post("/start")
def start():
    with state_lock:
        if state.running:
            return redirect(url_for("index"))

    raw_text = request.form.get("texts", "")
    uploaded_file = request.files.get("dataset_file")
    uploaded = uploaded_file.read() if uploaded_file and uploaded_file.filename else None
    texts = parse_dataset(raw_text, uploaded)
    cycles = max(1, min(100, int(request.form.get("cycles", "1"))))
    interval = max(0.0, min(10.0, float(request.form.get("interval", "0.1"))))
    shuffle = request.form.get("shuffle") == "on"
    memory.set_value("trainer_draft", "\n".join(texts))

    if not texts:
        with state_lock:
            state.error = "学習する文章がありません。"
            state.message = "入力待ち"
        return redirect(url_for("index"))

    with state_lock:
        state.running = True
        state.stop_requested = False
        state.completed = 0
        state.total = len(texts) * cycles
        state.current_text = ""
        state.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        state.finished_at = ""
        state.message = "開始準備中"
        state.error = ""
        state.dataset_size = len(texts)
        state.cycles = cycles

    Thread(
        target=training_worker,
        args=(texts, cycles, shuffle, interval),
        daemon=True,
    ).start()
    return redirect(url_for("index"))


@app.post("/stop")
def stop():
    with state_lock:
        state.stop_requested = True
        state.message = "停止要求を受け付けました"
    return redirect(url_for("index"))


@app.post("/save")
def save():
    save_core()
    with state_lock:
        if not state.running:
            state.message = "Coreを保存しました"
    return redirect(url_for("index"))


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5051")
    print("SphereBrain Experience Trainer v0.5: http://127.0.0.1:5051")
    serve(app, host="127.0.0.1", port=5051, threads=6)
