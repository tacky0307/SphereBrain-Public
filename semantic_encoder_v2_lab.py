from __future__ import annotations

import webbrowser
from flask import Flask, render_template_string, request
from waitress import serve

import semantic_encoder_v2 as semantic

app = Flask(__name__)

PAGE = r"""
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Semantic Encoder v2 Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#294d76;--text:#eef5ff;--muted:#9bb1ca;--cyan:#67dcff;--orange:#f28b4b;--green:#67e59a;--yellow:#ffd76a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1450px;margin:auto;padding:24px}header{background:linear-gradient(135deg,#112743,#091524);border-bottom:1px solid var(--line)}h1{margin:0 0 8px}.muted,p{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:linear-gradient(180deg,#132944,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:14px}.value{font-size:30px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}label{display:block;color:var(--cyan);margin:12px 0 6px}input{width:100%;padding:12px;background:#071522;color:var(--text);border:1px solid #355d88;border-radius:10px;font-size:16px}button{margin-top:16px;padding:12px 18px;border:0;border-radius:10px;background:linear-gradient(135deg,#e86f36,#f7a05f);color:white;font-weight:800;cursor:pointer}.danger{background:#6d2630}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px}.score{font-size:22px;font-weight:800}.note{border-left:3px solid var(--cyan);padding-left:12px}.ok{color:var(--green)}.same{color:var(--green);font-weight:800}.different{color:var(--muted)}.delta-positive{color:var(--green)}.delta-negative{color:#ff8c8c}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.summary>div{background:#071522;border:1px solid var(--line);border-radius:12px;padding:12px}.summary strong{font-size:24px;display:block;margin-top:5px}@media(max-width:950px){.grid,.stats,.summary{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Semantic Encoder v2 Lab</h1><p>主体・関係・内容を再利用可能な刺激として順番にCoreへ渡し、部分刺激からの再活動を観測します。</p></div></header>
<main class="wrap">
<section class="stats">
<div class="stat"><div class="eyebrow">Experiences</div><div class="value">{{stats.total}}</div></div>
<div class="stat"><div class="eyebrow">Subjects</div><div class="value">{{stats.subjects}}</div></div>
<div class="stat"><div class="eyebrow">Relations</div><div class="value">{{stats.relations}}</div></div>
<div class="stat"><div class="eyebrow">Contents</div><div class="value">{{stats.contents}}</div></div>
</section>
<div class="grid">
<section class="card"><div class="eyebrow">Structured Experience</div><h2>構造を持つ経験を入力</h2>
<form method="post"><input type="hidden" name="action" value="train">
<label>主体</label><input name="subject" value="{{subject}}" required>
<label>関係</label><input name="relation" value="{{relation}}" required>
<label>内容</label><input name="content" value="{{content}}" required>
<label>反復回数</label><input type="number" min="1" max="100" name="repeats" value="{{repeats}}">
<button>Coreへ経験を流す</button></form>
<p class="note">主体 → 関係 → 内容の順で流し、各成分は別経験でも同じ入口刺激を再利用します。</p>
{% if trained %}<p class="ok">保存しました：{{trained.input.label}}／活動ノード {{trained.all_nodes|length}}／通過Edge {{trained.all_edges|length}}</p>{% endif %}
</section>
<section class="card"><div class="eyebrow">Partial Cue Recall</div><h2>同じ主体・関係の過去経験と比較</h2>
<form method="post"><input type="hidden" name="action" value="probe">
<label>主体</label><input name="probe_subject" value="{{probe_subject}}" required>
<label>関係</label><input name="probe_relation" value="{{probe_relation}}" required>
<button>候補を注入せずCoreを動かす</button></form>
<p class="note">従来どおり、同じ主体・同じ関係の保存経験だけと比較します。</p>
{% if probe %}<div><span class="pill">入口 {{probe.source_nodes}}</span><span class="pill">活動 {{probe.activated_nodes|length}} nodes</span><span class="pill">通過 {{probe.traversed_edges|length}} edges</span></div>
{% if probe.matches %}<table><tr><th>過去の内容</th><th>経路重なり</th><th>経験数</th></tr>{% for item in probe.matches %}<tr><td>{{item.content}}</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td>{{item.experiences}}</td></tr>{% endfor %}</table>{% else %}<p>対応する過去経験はありません。</p>{% endif %}{% endif %}
</section></div>

<div class="grid">
<section class="card"><div class="eyebrow">Cross-subject Comparison</div><h2>主体を越えて同じ関係を比較</h2>
<form method="post"><input type="hidden" name="action" value="cross">
<label>刺激する主体</label><input name="cross_subject" value="{{cross_subject}}" required>
<label>関係</label><input name="cross_relation" value="{{cross_relation}}" required>
<button>全主体の経験と比較する</button></form>
<p class="note">主体＋関係までの活動を、同じ関係に属する全主体の経験と比較します。</p>
{% if cross %}<div><span class="pill">入口 {{cross.source_nodes}}</span><span class="pill">活動 {{cross.activated_nodes|length}} nodes</span><span class="pill">通過 {{cross.traversed_edges|length}} edges</span></div>
{% if cross.matches %}<table><tr><th>主体</th><th>内容</th><th>重なり</th><th>経験数</th></tr>{% for item in cross.matches %}<tr><td>{{item.subject}}</td><td>{{item.content}}</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td>{{item.experiences}}</td></tr>{% endfor %}</table>{% endif %}{% endif %}
</section>
<section class="card"><div class="eyebrow">Subject-only Recall</div><h2>主体だけから全関係方向を観測</h2>
<form method="post"><input type="hidden" name="action" value="subject_only">
<label>主体</label><input name="only_subject" value="{{only_subject}}" required>
<button>主体だけでCoreを動かす</button></form>
<p class="note">関係を指定せず、主体刺激だけが複数の関係枝のどこへ近づくかを比較します。</p>
{% if subject_only %}<div><span class="pill">入口 {{subject_only.source_nodes}}</span><span class="pill">活動 {{subject_only.activated_nodes|length}} nodes</span><span class="pill">通過 {{subject_only.traversed_edges|length}} edges</span></div>
{% if subject_only.matches %}<table><tr><th>関係</th><th>内容</th><th>重なり</th><th>経験数</th></tr>{% for item in subject_only.matches %}<tr><td>{{item.relation}}</td><td>{{item.content}}</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td>{{item.experiences}}</td></tr>{% endfor %}</table>{% endif %}{% endif %}
</section></div>

<section class="card"><div class="eyebrow">Content-stage Separation</div><h2>内容段階だけを主体横断で比較</h2>
<form method="post"><input type="hidden" name="action" value="content_phase">
<label>調べる関係</label><input name="content_relation" value="{{content_relation}}" required>
<button>内容段階だけを解析する</button></form>
<p class="note">主体段階と関係段階を比較から外し、保存済み経験の「内容を入れた段階」のノード・Edgeだけを比較します。観測のみで、CoreとDBは変更しません。</p>
{% if content_phase %}
<div class="summary">
<div><span class="eyebrow">同じ内容・別主体</span><strong>{{'%.1f'|format(content_phase.same_average*100)}}%</strong><span>{{content_phase.same_pair_count}}組</span></div>
<div><span class="eyebrow">異なる内容</span><strong>{{'%.1f'|format(content_phase.different_average*100)}}%</strong><span>{{content_phase.different_pair_count}}組</span></div>
<div><span class="eyebrow">分離差</span><strong class="{{'delta-positive' if content_phase.separation > 0 else 'delta-negative'}}">{{'%+.1f'|format(content_phase.separation*100)}}pt</strong><span>同内容 − 異内容</span></div>
</div>
<table><tr><th>比較A</th><th>比較B</th><th>内容段階の重なり</th><th>判定</th></tr>
{% for item in content_phase.pairs %}<tr><td>{{item.left_subject}}｜{{item.left_content}}</td><td>{{item.right_subject}}｜{{item.right_content}}</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td class="{{'same' if item.same_content else 'different'}}">{{'同じ内容' if item.same_content else '異なる内容'}}</td></tr>{% endfor %}</table>
{% endif %}
</section>

<section class="card"><div class="eyebrow">Design Boundary</div><h2>今回追加した観測</h2>
<p><b>内容段階のみ：</b> 主体・関係までの共通経路を直接の比較対象から外し、内容入力後の活動だけで同じ内容が主体を越えて近づいたかを調べます。</p>
<p><b>分離差がプラス：</b> 同じ内容同士の方が、異なる内容同士より近い傾向。ゼロ付近なら未分化、マイナスなら内容以外の影響がまだ強い可能性があります。</p>
<form method="post" onsubmit="return confirm('Semantic v2専用のCoreとDBを初期化します。旧データは変更しません。よろしいですか？')"><input type="hidden" name="action" value="reset"><button class="danger">Semantic v2実験だけ初期化</button></form>
</section>
{% if error %}<section class="card"><p>{{error}}</p></section>{% endif %}
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    subject, relation, content, repeats = "犬", "動作", "歩く", 1
    probe_subject, probe_relation = "犬", "動作"
    cross_subject, cross_relation = "犬", "性質"
    only_subject = "犬"
    content_relation = "性質"
    trained = probe = cross = subject_only = content_phase = None
    error = ""

    if request.method == "POST":
        action = request.form.get("action", "")
        try:
            if action == "train":
                subject = request.form.get("subject", "")
                relation = request.form.get("relation", "")
                content = request.form.get("content", "")
                repeats = max(1, min(100, int(request.form.get("repeats", "1"))))
                trained = semantic.train(subject, relation, content, repeats)
            elif action == "probe":
                probe_subject = request.form.get("probe_subject", "")
                probe_relation = request.form.get("probe_relation", "")
                probe = semantic.recall_probe(probe_subject, probe_relation)
            elif action == "cross":
                cross_subject = request.form.get("cross_subject", "")
                cross_relation = request.form.get("cross_relation", "")
                cross = semantic.cross_subject_probe(cross_subject, cross_relation)
            elif action == "subject_only":
                only_subject = request.form.get("only_subject", "")
                subject_only = semantic.subject_only_probe(only_subject)
            elif action == "content_phase":
                content_relation = request.form.get("content_relation", "")
                content_phase = semantic.content_phase_analysis(content_relation)
            elif action == "reset":
                semantic.reset_experiment()
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        PAGE, stats=semantic.stats(), subject=subject, relation=relation, content=content,
        repeats=repeats, trained=trained, probe=probe, probe_subject=probe_subject,
        probe_relation=probe_relation, cross=cross, cross_subject=cross_subject,
        cross_relation=cross_relation, subject_only=subject_only, only_subject=only_subject,
        content_phase=content_phase, content_relation=content_relation, error=error,
    )


if __name__ == "__main__":
    semantic.initialize_db()
    url = "http://127.0.0.1:5092"
    webbrowser.open(url)
    serve(app, host="127.0.0.1", port=5092, threads=6)
