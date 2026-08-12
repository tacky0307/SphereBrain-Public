from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from datetime import datetime
import json
import shutil
import time
import webbrowser

from flask import Flask, request, redirect, url_for, render_template_string, send_file
from waitress import serve

from brain import SphereBrain
from memory_store import MemoryStore
from visualization import build_html
from audio_listener import SystemAudioListener


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BACKUPS = DATA / "backups"
DATA.mkdir(exist_ok=True)
BACKUPS.mkdir(exist_ok=True)

BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"
VIEW_FILE = DATA / "brain_view.html"
CONFIG_FILE = DATA / "config.json"

brain = SphereBrain.load(BRAIN_FILE) if BRAIN_FILE.exists() else SphereBrain(
    node_count=600,
    neighbors_per_node=8,
)
memory = MemoryStore(DB_FILE)
brain_lock = Lock()

last_result = None
last_input = ""
last_activity = datetime.now().isoformat(timespec="seconds")
running = True

app = Flask(__name__)


def load_config() -> dict:
    default = {
        "audio_model": "small",
        "audio_chunk_seconds": 18,
        "auto_start_audio": False,
        "idle_cycle_seconds": 45,
        "backup_hours": 6,
    }
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(
            json.dumps(default, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return default
    try:
        saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        default.update(saved)
    except Exception:
        pass
    return default


config = load_config()


def activity_summary() -> dict:
    if last_result is None:
        return {
            "active_nodes": 0,
            "active_edges": 0,
            "activation_rate": 0.0,
            "route_preview": "まだ活動はありません",
        }

    nodes = list(last_result.activated_nodes)
    edges = list(last_result.traversed_edges)
    rate = (len(nodes) / max(1, brain.node_count)) * 100
    route_preview = " → ".join(str(node) for node in nodes[:8])
    if len(nodes) > 8:
        route_preview += " → …"

    return {
        "active_nodes": len(nodes),
        "active_edges": len(edges),
        "activation_rate": round(rate, 1),
        "route_preview": route_preview or "活動経路を記録中",
    }


PAGE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Sphere Brain Observatory v0.4</title>
<style>
:root {
  --bg:#07111f;
  --panel:#0d1b2f;
  --panel-2:#11243e;
  --line:#203b5f;
  --text:#e6edf7;
  --muted:#93a7bf;
  --cyan:#68d8ff;
  --green:#69e09a;
  --orange:#ff9d52;
  --red:#ff7b7b;
  --purple:#b89cff;
}
* { box-sizing:border-box; }
body {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin:0;
  background:radial-gradient(circle at top right, rgba(52,113,168,.18), transparent 34%),var(--bg);
  color:var(--text);
}
header { border-bottom:1px solid var(--line); background:rgba(7,17,31,.92); backdrop-filter:blur(10px); position:sticky; top:0; z-index:10; }
.header-inner { max-width:1380px; margin:0 auto; padding:18px 24px; display:flex; align-items:center; justify-content:space-between; gap:20px; }
.brand h1 { margin:0; font-size:26px; letter-spacing:.02em; }
.brand p { margin:5px 0 0; color:var(--muted); font-size:14px; }
.live { display:inline-flex; align-items:center; gap:8px; color:var(--green); font-size:14px; }
.live::before { content:""; width:9px; height:9px; border-radius:999px; background:var(--green); box-shadow:0 0 16px rgba(105,224,154,.9); }
main { max-width:1380px; margin:0 auto; padding:24px; }
.hero-grid { display:grid; grid-template-columns:1.15fr .85fr; gap:18px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:18px; }
.card { background:linear-gradient(180deg,rgba(17,36,62,.96),rgba(13,27,47,.96)); border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:0 14px 40px rgba(0,0,0,.18); }
.card h2 { margin:0 0 16px; font-size:19px; }
.eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.12em; font-size:12px; }
.metric { font-size:30px; font-weight:700; margin-top:8px; }
.metric-label { color:var(--muted); font-size:13px; }
textarea { width:100%; min-height:134px; resize:vertical; border:1px solid #2a4c73; border-radius:13px; background:#081522; color:var(--text); padding:14px; font-size:16px; outline:none; }
textarea:focus { border-color:var(--cyan); box-shadow:0 0 0 3px rgba(104,216,255,.12); }
button { background:linear-gradient(135deg,#ee6b2f,#ff9d52); color:white; border:0; padding:11px 17px; border-radius:10px; font-size:14px; font-weight:650; cursor:pointer; }
button.secondary { background:#233b59; }
button.stop { background:#7c2930; }
button:hover { filter:brightness(1.08); }
.actions { display:flex; gap:10px; flex-wrap:wrap; }
.status-line { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.badge { display:inline-flex; align-items:center; gap:7px; background:rgba(105,224,154,.12); color:var(--green); border:1px solid rgba(105,224,154,.25); padding:5px 9px; border-radius:999px; font-size:13px; }
.badge.off { background:rgba(147,167,191,.12); color:var(--muted); border-color:rgba(147,167,191,.2); }
.badge.error { background:rgba(255,123,123,.12); color:var(--red); border-color:rgba(255,123,123,.25); }
.detail-list { display:grid; gap:10px; margin-top:14px; }
.detail-row { display:flex; justify-content:space-between; gap:16px; padding-bottom:10px; border-bottom:1px solid rgba(32,59,95,.65); }
.detail-row:last-child { border-bottom:0; padding-bottom:0; }
.detail-row span:first-child { color:var(--muted); }
.route-box { margin-top:16px; padding:14px; border-radius:12px; background:#081522; border:1px solid #203b5f; color:var(--cyan); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-word; }
.compare-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:16px; }
.compare-box { background:#081522; border:1px solid #203b5f; border-radius:14px; padding:16px; }
.compare-value { font-size:28px; font-weight:750; margin-top:6px; }
.compare-label { color:var(--muted); font-size:12px; }
.fingerprint { display:grid; grid-template-columns:repeat(24,1fr); gap:4px; margin-top:14px; align-items:end; height:90px; }
.fingerprint span { display:block; min-height:6px; border-radius:5px 5px 2px 2px; background:linear-gradient(180deg,var(--purple),var(--cyan)); opacity:.85; }
.route-list { display:grid; gap:8px; margin-top:12px; }
.route-item { display:flex; justify-content:space-between; gap:12px; padding:10px 12px; border-radius:10px; background:#081522; border:1px solid rgba(32,59,95,.75); }
.route-item code { color:var(--orange); }
iframe { width:100%; height:720px; border:1px solid var(--line); border-radius:14px; background:white; }
small,.muted { color:var(--muted); }
code { background:#081522; color:#b9d6f3; padding:3px 6px; border-radius:6px; }
table { width:100%; border-collapse:collapse; }
td,th { padding:11px 9px; border-bottom:1px solid rgba(32,59,95,.7); text-align:left; vertical-align:top; font-size:14px; }
th { color:var(--muted); font-weight:600; }
.section-gap { margin-top:18px; }
@media(max-width:1050px){ .hero-grid,.grid{grid-template-columns:1fr;} .stats{grid-template-columns:repeat(2,1fr);} .compare-grid{grid-template-columns:1fr;} }
@media(max-width:650px){ main{padding:16px;} .header-inner{padding:15px 16px;align-items:flex-start;} .stats{grid-template-columns:1fr;} iframe{height:520px;} .detail-row{flex-direction:column;gap:4px;} .fingerprint{gap:2px;} }
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">
      <h1>Sphere Brain Observatory v0.4</h1>
      <p>経験によって変化し続ける認識構造を観測する研究室</p>
    </div>
    <div class="live">CORE ONLINE</div>
  </div>
</header>
<main>
  <section class="stats">
    <div class="card"><div class="eyebrow">Experience</div><div class="metric">{{ memory_count }}</div><div class="metric-label">蓄積された経験</div></div>
    <div class="card"><div class="eyebrow">Active Nodes</div><div class="metric">{{ activity.active_nodes }}</div><div class="metric-label">直近活動ノード</div></div>
    <div class="card"><div class="eyebrow">Active Routes</div><div class="metric">{{ activity.active_edges }}</div><div class="metric-label">直近通過経路</div></div>
    <div class="card"><div class="eyebrow">Activation</div><div class="metric">{{ activity.activation_rate }}%</div><div class="metric-label">Core活動率</div></div>
  </section>

  <section class="hero-grid">
    <div class="card">
      <div class="eyebrow">Stimulus Input</div>
      <h2>経験を与える</h2>
      <form method="post" action="/input">
        <textarea name="text" placeholder="SphereBrainへ与える言葉や文章を入力してください"></textarea>
        <p><button type="submit">Coreへ入力する</button></p>
      </form>
      <small>入力は刺激へ変換され、直近の文脈とともにCoreを通過します。使われた経路は強化されます。</small>
    </div>

    <div class="card">
      <div class="status-line">
        <div><div class="eyebrow">Current State</div><h2>現在の状態</h2></div>
        <span class="badge">稼働中</span>
      </div>
      <div class="detail-list">
        <div class="detail-row"><span>最後の経験</span><strong>{{ last_input or route.current_text or "まだありません" }}</strong></div>
        <div class="detail-row"><span>最終活動</span><strong>{{ last_activity }}</strong></div>
        <div class="detail-row"><span>Coreノード数</span><strong>{{ brain.node_count }}</strong></div>
      </div>
      <div class="route-box">{{ activity.route_preview }}</div>
    </div>
  </section>

  <section class="card section-gap">
    <div class="eyebrow">Route Observatory</div>
    <h2>経路の一致・差分・指紋</h2>
    {% if route.available %}
    <div class="compare-grid">
      <div class="compare-box"><div class="compare-label">前回入力との経路一致率</div><div class="compare-value">{{ route.previous_similarity }}%</div><div class="muted">共通 {{ route.previous_common_routes }} 経路</div></div>
      <div class="compare-box"><div class="compare-label">過去入力との最高一致率</div><div class="compare-value">{{ route.best_match_similarity }}%</div><div class="muted">{{ route.best_match_text or "比較対象なし" }}</div></div>
      <div class="compare-box"><div class="compare-label">今回だけの経路</div><div class="compare-value">{{ route.new_routes }}</div><div class="muted">過去{{ comparison_limit }}件との比較</div></div>
    </div>
    <div class="grid section-gap">
      <div>
        <div class="eyebrow">Core Fingerprint</div>
        <h2>{{ route.current_text }}</h2>
        <div class="fingerprint">
          {% for value in route.fingerprint %}<span style="height:{{ 8 + value * 82 }}px" title="{{ value }}"></span>{% endfor %}
        </div>
        <p class="muted">活動ノードを24区画へ投影した簡易指紋。文章間の違いを形として比較します。</p>
      </div>
      <div>
        <div class="eyebrow">Frequent Routes</div>
        <h2>直近入力で繰り返された経路</h2>
        <div class="route-list">
          {% for item in route.top_routes %}
          <div class="route-item"><code>{{ item.a }} → {{ item.b }}</code><strong>{{ item.count }}回</strong></div>
          {% endfor %}
        </div>
      </div>
    </div>
    {% else %}
    <p class="muted">入力経験が蓄積されると、経路の一致率と指紋がここに表示されます。</p>
    {% endif %}
  </section>

  <section class="grid section-gap">
    <div class="card">
      <div class="eyebrow">Audio Experience</div>
      <h2>PC音声の聴取</h2>
      {% if audio.running %}<p><span class="badge">聴取中</span> {{ audio.state }}</p>
      {% elif audio.last_error %}<p><span class="badge error">エラー</span> {{ audio.last_error }}</p>
      {% else %}<p><span class="badge off">停止中</span></p>{% endif %}
      <div class="detail-list">
        <div class="detail-row"><span>文字化した区間</span><strong>{{ audio.chunks_processed }}</strong></div>
        <div class="detail-row"><span>無音として省略</span><strong>{{ audio.chunks_skipped }}</strong></div>
        <div class="detail-row"><span>最新の文字</span><strong>{{ audio.last_text or "まだありません" }}</strong></div>
      </div>
      <div class="actions" style="margin-top:16px">
        <form method="post" action="/audio/start"><button type="submit">音声聴取を開始</button></form>
        <form method="post" action="/audio/stop"><button class="stop" type="submit">音声聴取を停止</button></form>
      </div>
      <p class="muted">既定スピーカーの音を約{{ chunk_seconds }}秒ごとにローカルで文字化します。生の音声は保存しません。</p>
    </div>

    <div class="card">
      <div class="eyebrow">Persistence</div>
      <h2>保存とバックアップ</h2>
      <div class="detail-list">
        <div class="detail-row"><span>Core</span><code>data/brain.json</code></div>
        <div class="detail-row"><span>Experience Store</span><code>data/memory.db</code></div>
        <div class="detail-row"><span>自動バックアップ</span><strong>{{ backup_hours }}時間ごと</strong></div>
      </div>
      <div class="actions" style="margin-top:16px">
        <form method="post" action="/save"><button class="secondary" type="submit">今すぐ保存</button></form>
        <form method="post" action="/backup"><button class="secondary" type="submit">今すぐバックアップ</button></form>
      </div>
    </div>
  </section>

  <section class="card section-gap"><div class="eyebrow">Core Visualization</div><h2>球体内部の活動</h2><iframe src="/brain-view?ts={{ timestamp }}"></iframe></section>

  <section class="card section-gap">
    <div class="eyebrow">Trace Log</div><h2>最近の経験と活動記録</h2>
    <table>
      <tr><th>時刻</th><th>種類</th><th>経験</th><th>活動</th></tr>
      {% for item in memories %}
      <tr><td>{{ item.created_at }}</td><td>{{ item.kind }}</td><td>{{ (item.input_text or "内部活動")[:180] }}</td><td>{{ item.activated_nodes|length }}ノード / {{ item.traversed_edges|length }}経路</td></tr>
      {% endfor %}
    </table>
  </section>
</main>
</body>
</html>
"""


def ingest_text(text: str, kind: str = "input", importance: float = 1.0) -> None:
    global last_result, last_input, last_activity

    text = " ".join(text.strip().split())
    if not text:
        return

    chunks = [text[i:i+600] for i in range(0, len(text), 600)]

    for chunk in chunks:
        sources = brain.text_to_sources(chunk, count=4)
        context = memory.recent_context_nodes(memory_limit=7, node_limit=18)

        with brain_lock:
            result = brain.propagate(source_nodes=sources, context_nodes=context, steps=20, learn=True)

        memory.add_memory(
            kind=kind,
            input_text=chunk,
            source_nodes=result.source_nodes,
            activated_nodes=result.activated_nodes,
            traversed_edges=result.traversed_edges,
            importance=importance,
        )

        last_result = result
        last_input = chunk
        last_activity = datetime.now().isoformat(timespec="seconds")

    save_all()


audio_listener = SystemAudioListener(
    on_text=lambda text: ingest_text(text, kind="audio", importance=0.72),
    model_size=str(config["audio_model"]),
    chunk_seconds=int(config["audio_chunk_seconds"]),
)


def save_all() -> None:
    global last_result
    with brain_lock:
        brain.save(BRAIN_FILE)
        if last_result is None:
            strong = brain.strongest_edges(55)
            edges = [(e["a"], e["b"]) for e in strong]
            nodes = sorted({n for edge in edges for n in edge})
            build_html(brain, VIEW_FILE, edges, nodes, "Sphere Brain：強い経路")
        else:
            build_html(brain, VIEW_FILE, last_result.traversed_edges, last_result.activated_nodes, f"Sphere Brain：{last_input or '内部活動'}")


def make_backup() -> None:
    save_all()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUPS / stamp
    target.mkdir(exist_ok=True)
    for path in (BRAIN_FILE, DB_FILE, CONFIG_FILE):
        if path.exists():
            shutil.copy2(path, target / path.name)

    backups = sorted([p for p in BACKUPS.iterdir() if p.is_dir()])
    for old in backups[:-28]:
        shutil.rmtree(old, ignore_errors=True)


def background_loop() -> None:
    global last_result, last_activity
    idle_seconds = max(20, int(config["idle_cycle_seconds"]))
    backup_seconds = max(3600, int(config["backup_hours"]) * 3600)
    last_backup = time.monotonic()

    while running:
        time.sleep(idle_seconds)
        try:
            context = memory.recent_context_nodes(memory_limit=10, node_limit=30)
            if context:
                with brain_lock:
                    result = brain.idle_cycle(context)
                if result:
                    memory.add_memory(
                        kind="idle",
                        input_text="",
                        source_nodes=result.source_nodes,
                        activated_nodes=result.activated_nodes,
                        traversed_edges=result.traversed_edges,
                        importance=0.18,
                    )
                    last_result = result
                    last_activity = datetime.now().isoformat(timespec="seconds")
            save_all()

            if time.monotonic() - last_backup >= backup_seconds:
                make_backup()
                last_backup = time.monotonic()
        except Exception as exc:
            print("background error:", exc)


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        memory_count=memory.count(),
        memories=memory.recent(20),
        last_input=last_input,
        last_activity=last_activity,
        brain=brain,
        activity=activity_summary(),
        route=memory.route_analytics(comparison_limit=80),
        comparison_limit=80,
        audio=audio_listener.status,
        timestamp=int(time.time()),
        chunk_seconds=config["audio_chunk_seconds"],
        backup_hours=config["backup_hours"],
    )


@app.post("/input")
def input_text():
    text = request.form.get("text", "")
    ingest_text(text, kind="input", importance=1.0)
    return redirect(url_for("index"))


@app.post("/audio/start")
def audio_start():
    audio_listener.start()
    return redirect(url_for("index"))


@app.post("/audio/stop")
def audio_stop():
    audio_listener.stop()
    return redirect(url_for("index"))


@app.post("/save")
def save_now():
    save_all()
    return redirect(url_for("index"))


@app.post("/backup")
def backup_now():
    make_backup()
    return redirect(url_for("index"))


@app.route("/brain-view")
def brain_view():
    if not VIEW_FILE.exists():
        save_all()
    return send_file(VIEW_FILE)


if __name__ == "__main__":
    Thread(target=background_loop, daemon=True).start()
    save_all()

    if bool(config.get("auto_start_audio")):
        audio_listener.start()

    webbrowser.open("http://127.0.0.1:5050")
    print("Sphere Brain Observatory v0.4: http://127.0.0.1:5050")
    serve(app, host="127.0.0.1", port=5050, threads=6)
