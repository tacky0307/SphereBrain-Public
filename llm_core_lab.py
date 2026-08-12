from __future__ import annotations

from flask import Flask, render_template_string, request

import llm_core_metrics as metrics
import llm_core_pipeline as pipeline

app = Flask(__name__)

PAGE = r"""
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain LLM → Core → LLM Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#294d76;--text:#eef5ff;--muted:#9bb1ca;--cyan:#67dcff;--orange:#f28b4b;--green:#67e59a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1300px;margin:auto;padding:24px}header{background:linear-gradient(135deg,#112743,#091524);border-bottom:1px solid var(--line)}h1{margin:0 0 8px}.muted,p{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:linear-gradient(180deg,#132944,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:14px}.value{font-size:28px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}label{display:block;color:var(--cyan);margin:12px 0 6px}textarea,input{width:100%;padding:12px;background:#071522;color:var(--text);border:1px solid #355d88;border-radius:10px;font-size:16px}button{margin-top:16px;padding:12px 18px;border:0;border-radius:10px;background:linear-gradient(135deg,#e86f36,#f7a05f);color:white;font-weight:800;cursor:pointer}.danger{background:#6d2630}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px}.score{font-size:20px;font-weight:800}.answer{font-size:22px;line-height:1.7;color:var(--text);border-left:4px solid var(--green);padding-left:16px}.note{border-left:3px solid var(--cyan);padding-left:12px}.error{color:#ff9b9b}.safe{color:var(--green);font-weight:700}.compare{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}.compare .stat{padding:12px}.small{font-size:13px;color:var(--muted)}@media(max-width:900px){.grid,.stats,.compare{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="wrap"><h1>SphereBrain LLM → Core → LLM Lab</h1><p>LLMを感覚器官と表現器官として使い、経験による変化は専用Coreだけで起こす分離実験。</p></div></header>
<main class="wrap">
<section class="card">
<div class="eyebrow">Safety Boundary</div>
<p class="safe">既存の Semantic Encoder v2、data/brain.json、data/semantic_v2.db は変更しません。</p>
<p>この実験のCore・DB・射影行列は <code>data/llm_core_v1/</code> のみに保存されます。従来研究へ戻るときは従来ランチャーを起動するだけです。</p>
</section>

<section class="stats">
<div class="stat"><div class="eyebrow">Experiences</div><div class="value">{{stats.total_experiences}}</div></div>
<div class="stat"><div class="eyebrow">Distinct</div><div class="value">{{stats.distinct_texts}}</div></div>
<div class="stat"><div class="eyebrow">Embedding</div><div>{{stats.embedding_model}}</div></div>
<div class="stat"><div class="eyebrow">Decoder</div><div>{{stats.decoder_model}}</div></div>
</section>

<div class="grid">
<section class="card">
<div class="eyebrow">Experience</div><h2>文章を経験させる</h2>
<form method="post"><input type="hidden" name="action" value="train">
<label>入力文</label><textarea name="train_text" rows="4" required>{{train_text}}</textarea>
<label>反復回数</label><input type="number" name="repeats" min="1" max="100" value="{{repeats}}">
<button>LLM → Coreへ経験させる</button></form>
{% if trained %}<p><span class="pill">入口 {{trained.source_nodes|length}}</span><span class="pill">活動 {{trained.result.activated_nodes|length}} nodes</span><span class="pill">通過 {{trained.result.traversed_edges|length}} edges</span></p>{% endif %}
</section>

<section class="card">
<div class="eyebrow">Probe</div><h2>EmbeddingとCoreを分離して観測</h2>
<form method="post"><input type="hidden" name="action" value="probe">
<label>検証文</label><textarea name="probe_text" rows="4" required>{{probe_text}}</textarea>
<button>学習せず比較観測する</button></form>
{% if probed %}
<p><span class="pill">活動 {{probed.activated_nodes|length}} nodes</span><span class="pill">通過 {{probed.traversed_edges|length}} edges</span><span class="pill">比較経験 {{probed.comparison.experience_count}}</span></p>
<div class="compare">
<div class="stat"><div class="eyebrow">Embedding 最大</div><div class="score">{{'%.1f'|format(probed.comparison.embedding_max*100)}}%</div></div>
<div class="stat"><div class="eyebrow">Embedding 平均</div><div class="score">{{'%.1f'|format(probed.comparison.embedding_average*100)}}%</div></div>
<div class="stat"><div class="eyebrow">Core 最大</div><div class="score">{{'%.1f'|format(probed.comparison.core_max*100)}}%</div></div>
<div class="stat"><div class="eyebrow">Core 平均</div><div class="score">{{'%.1f'|format(probed.comparison.core_average*100)}}%</div></div>
</div>
<p class="note">EmbeddingはLLM入力側の固定基準。Core重なりは経験によって変化する観測値です。</p>
<table><tr><th>経験</th><th>Embedding</th><th>Core</th><th>Node</th><th>Edge</th></tr>{% for item in probed.matches %}<tr><td>{{item.text}}<div class="small">#{{item.experience_id}}</div></td><td>{{'%.1f'|format(item.embedding_similarity*100)}}%</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td>{{'%.1f'|format(item.node_overlap*100)}}%</td><td>{{'%.1f'|format(item.edge_overlap*100)}}%</td></tr>{% endfor %}</table>
{% endif %}
</section>
</div>

<section class="card">
<div class="eyebrow">Decode</div><h2>Coreの結果をLLMで言葉へ戻す</h2>
<form method="post"><input type="hidden" name="action" value="ask">
<label>入力文</label><textarea name="ask_text" rows="4" required>{{ask_text}}</textarea>
<button>LLM → Core → LLM</button></form>
{% if asked %}<p class="answer">{{asked.answer}}</p>
<table><tr><th>Decoderへ渡したCore選択候補</th><th>重なり</th></tr>{% for item in asked.observation.matches %}<tr><td>{{item.text}}</td><td>{{'%.1f'|format(item.score*100)}}%</td></tr>{% endfor %}</table>{% endif %}
</section>

<section class="card">
<div class="eyebrow">Reset</div><h2>この実験だけ初期化</h2>
<form method="post" onsubmit="return confirm('LLM-Core-LLM v1専用データだけを初期化します。よろしいですか？')"><input type="hidden" name="action" value="reset"><button class="danger">LLM-Core-LLM v1だけ初期化</button></form>
</section>
{% if error %}<section class="card"><p class="error">{{error}}</p></section>{% endif %}
</main>
</body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    train_text = "今日は晴れて気持ちいい"
    probe_text = "晴れた日は心地よい"
    ask_text = "今日は晴れている"
    repeats = 1
    trained = probed = asked = None
    error = ""

    if request.method == "POST":
        action = request.form.get("action", "")
        try:
            if action == "train":
                train_text = request.form.get("train_text", "")
                repeats = int(request.form.get("repeats", "1"))
                trained = pipeline.experience(train_text, repeats=repeats)
            elif action == "probe":
                probe_text = request.form.get("probe_text", "")
                probed = metrics.probe_with_metrics(probe_text)
            elif action == "ask":
                ask_text = request.form.get("ask_text", "")
                asked = pipeline.ask(ask_text)
            elif action == "reset":
                pipeline.reset_experiment()
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        PAGE,
        stats=pipeline.stats(),
        train_text=train_text,
        probe_text=probe_text,
        ask_text=ask_text,
        repeats=repeats,
        trained=trained,
        probed=probed,
        asked=asked,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5078, debug=False)
