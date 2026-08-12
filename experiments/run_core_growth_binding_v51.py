from __future__ import annotations

import json
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify
from waitress import serve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_core_growth_binding_v3 as v3
import run_core_growth_binding_v44 as v44
import run_core_growth_binding_v50 as v50

HOST = "127.0.0.1"
START_PORT = 5097
OUT = ROOT / "data" / "core_growth_binding_v51" / "results"
POSITIONS = ["左", "中央", "右"]
MAX_SIGNATURE = 5


def choose_port(start: int) -> int:
    for port in range(start, start + 50):
        if port in {5060, 5061}:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("利用可能なローカルポートが見つかりません。")


PORT = choose_port(START_PORT)


def named_rows(condition_rows: list[dict]) -> list[dict[str, float]]:
    return v50.named_rows(condition_rows)


def motif_candidates(rows: dict[str, list[dict[str, float]]]) -> list[str]:
    ex = v50.exclusive_maps(rows)
    left_cov = v50.motif_coverage_for_side("左", ex)
    center_cov = v50.motif_coverage_for_side("中央", ex)
    return sorted(set(left_cov) | set(center_cov))


def motif_vector(row: dict[str, float], motifs: list[str]) -> tuple[int, ...]:
    return tuple(1 if v50.motif_series([row], motif)[0] else 0 for motif in motifs)


def motif_pair_masks(rows: dict[str, list[dict[str, float]]], motifs: list[str]) -> tuple[list[int], int]:
    left, center = rows["左"], rows["中央"]
    pair_count = len(left) * len(center)
    full_mask = (1 << pair_count) - 1
    masks = []
    for motif in motifs:
        left_p = v50.motif_series(left, motif)
        center_p = v50.motif_series(center, motif)
        mask = 0
        bit = 0
        for lv in left_p:
            for cv in center_p:
                if bool(lv) != bool(cv):
                    mask |= 1 << bit
                bit += 1
        masks.append(mask)
    return masks, full_mask


def reduce_candidates(motifs: list[str], masks: list[int]) -> tuple[list[str], list[int]]:
    best_for_mask: dict[int, str] = {}
    for motif, mask in zip(motifs, masks):
        if mask == 0:
            continue
        if mask not in best_for_mask or motif < best_for_mask[mask]:
            best_for_mask[mask] = motif
    items = sorted(((motif, mask) for mask, motif in best_for_mask.items()), key=lambda x: (-x[1].bit_count(), x[0]))
    kept: list[tuple[str, int]] = []
    for motif, mask in items:
        if any((mask | other_mask) == other_mask for _, other_mask in kept):
            continue
        kept.append((motif, mask))
    return [x[0] for x in kept], [x[1] for x in kept]


def exact_cover_up_to(motifs: list[str], masks: list[int], full_mask: int, max_size: int = MAX_SIGNATURE) -> dict:
    motifs, masks = reduce_candidates(motifs, masks)
    candidate_count = len(motifs)
    if full_mask == 0:
        return {"found": True, "size": 0, "motifs": [], "candidate_count_after_reduction": candidate_count}
    union = 0
    for mask in masks:
        union |= mask
    if union != full_mask:
        return {"found": False, "size": None, "motifs": [], "candidate_count_after_reduction": candidate_count, "uncoverable_pairs": (full_mask & ~union).bit_count()}

    pair_to_candidates: dict[int, list[int]] = {}
    for bit in range(full_mask.bit_length()):
        if (full_mask >> bit) & 1:
            pair_to_candidates[bit] = [i for i, mask in enumerate(masks) if (mask >> bit) & 1]

    def search(limit: int) -> list[int] | None:
        failed: set[tuple[int, int]] = set()

        def dfs(covered: int, chosen: tuple[int, ...]) -> tuple[int, ...] | None:
            if covered == full_mask:
                return chosen
            remaining_depth = limit - len(chosen)
            if remaining_depth <= 0:
                return None
            state = (covered, remaining_depth)
            if state in failed:
                return None
            uncovered = full_mask & ~covered
            gains = sorted(((mask & uncovered).bit_count() for mask in masks), reverse=True)
            if sum(gains[:remaining_depth]) < uncovered.bit_count():
                failed.add(state)
                return None
            uncovered_bits = [bit for bit in pair_to_candidates if (uncovered >> bit) & 1]
            pivot = min(uncovered_bits, key=lambda b: len(pair_to_candidates[b]))
            candidates = sorted(pair_to_candidates[pivot], key=lambda i: -((masks[i] & uncovered).bit_count()))
            tried_masks: set[int] = set()
            for i in candidates:
                gain_mask = masks[i] & uncovered
                if not gain_mask or masks[i] in tried_masks:
                    continue
                tried_masks.add(masks[i])
                result = dfs(covered | masks[i], chosen + (i,))
                if result is not None:
                    return result
            failed.add(state)
            return None

        result = dfs(0, tuple())
        return None if result is None else list(result)

    for size in range(1, max_size + 1):
        print(f"  exact search: <= {size} motif(s), reduced candidates={candidate_count}", flush=True)
        result = search(size)
        if result is not None:
            return {"found": True, "size": len(result), "motifs": [motifs[i] for i in result], "candidate_count_after_reduction": candidate_count}
    return {"found": False, "size": None, "motifs": [], "candidate_count_after_reduction": candidate_count}


def signature_separates(rows: dict[str, list[dict[str, float]]], motifs: list[str]) -> bool:
    left = {motif_vector(row, motifs) for row in rows["左"]}
    center = {motif_vector(row, motifs) for row in rows["中央"]}
    return left.isdisjoint(center)


def nearest_classify(train: dict[str, list[dict[str, float]]], held_row: dict[str, float], motifs: list[str]) -> dict:
    x = motif_vector(held_row, motifs)
    distances = {}
    for position in ("左", "中央"):
        vectors = [motif_vector(row, motifs) for row in train[position]]
        distances[position] = min(sum(a != b for a, b in zip(x, v)) for v in vectors) if vectors else 10**9
    prediction = None
    if distances["左"] < distances["中央"]:
        prediction = "左"
    elif distances["中央"] < distances["左"]:
        prediction = "中央"
    return {"vector": list(x), "distances": distances, "prediction": prediction}


def loco_validation(rows: dict[str, list[dict[str, float]]], condition_names: list[str]) -> dict:
    folds = []
    all_pass = True
    for held_index, condition in enumerate(condition_names):
        print(f"  LOCO {held_index + 1}/{len(condition_names)}: hold out {condition}", flush=True)
        train = {
            "左": [row for i, row in enumerate(rows["左"]) if i != held_index],
            "中央": [row for i, row in enumerate(rows["中央"]) if i != held_index],
        }
        candidates = motif_candidates(train)
        masks, full_mask = motif_pair_masks(train, candidates)
        signature = exact_cover_up_to(candidates, masks, full_mask)
        left_test = nearest_classify(train, rows["左"][held_index], signature["motifs"]) if signature["found"] else None
        center_test = nearest_classify(train, rows["中央"][held_index], signature["motifs"]) if signature["found"] else None
        fold_pass = bool(signature["found"] and left_test is not None and center_test is not None and left_test["prediction"] == "左" and center_test["prediction"] == "中央")
        all_pass = all_pass and fold_pass
        folds.append({"held_condition": condition, "signature": signature, "left_test": left_test, "center_test": center_test, "pass": fold_pass})
    return {"all_pass": all_pass, "folds": folds}


def observe() -> dict:
    print("v51: rebuilding live perturbation runs...", flush=True)
    runs = {position: v44.condition_runs(position) for position in POSITIONS}
    rows = {position: named_rows(runs[position]) for position in POSITIONS}
    print("v51: building abstract motif candidates...", flush=True)
    candidates = motif_candidates(rows)
    masks, full_mask = motif_pair_masks(rows, candidates)
    print(f"v51: raw motif candidates={len(candidates)}", flush=True)
    print("v51: searching minimal full-live signature...", flush=True)
    minimal = exact_cover_up_to(candidates, masks, full_mask)
    full_separation = minimal["found"] and signature_separates(rows, minimal["motifs"])
    condition_names = [name for name, _, _ in v44.CONDITIONS]
    print("v51: starting leave-one-condition-out validation...", flush=True)
    loco = loco_validation(rows, condition_names)
    print("v51: LOCO complete.", flush=True)

    right_absent = all(not row["event_formed"] for row in runs["右"])
    left_complete = len(rows["左"]) == len(v44.CONDITIONS)
    center_complete = len(rows["中央"]) == len(v44.CONDITIONS)
    compact = bool(minimal["found"] and minimal["size"] is not None and minimal["size"] <= MAX_SIGNATURE)
    shadow_candidate = left_complete and center_complete and right_absent and compact and full_separation and loco["all_pass"]

    if shadow_candidate:
        verdict = "minimal_discriminative_motif_signature_found"
        next_step = "shadow_integrate_signature_into_core_without_affecting_route_or_learning"
        core_readiness = "shadow_candidate"
    elif compact and full_separation:
        verdict = "compact_signature_fits_live_conditions_but_fails_leave_one_condition_out"
        next_step = "broaden_live_conditions_or_refine_motif_generalization_before_core_shadow"
        core_readiness = "not_yet"
    elif not minimal["found"]:
        verdict = "no_1_to_5_motif_signature_separates_all_live_conditions"
        next_step = "seek_more_abstract_or_temporal_discriminative_motifs"
        core_readiness = "not_yet"
    else:
        verdict = "minimal_discriminative_signature_inconclusive"
        next_step = "audit_signature_behavior_before_core_shadow"
        core_readiness = "not_yet"

    route = {p: v44.summarize_position(runs[p]) for p in ("左", "中央")}
    payload = {
        "experiment": "Core Growth Binding v51",
        "purpose": "Find the smallest abstract motif presence signature (up to five motifs) that separates all seven left live conditions from all seven center live conditions, then test leave-one-condition-out generalization.",
        "contract": {"learning": False, "weights_changed": False, "new_edges_created": False, "threshold_changed": False, "structural_assist_used": False, "core_file_modified": False, "live_propagation": True, "human_selected_motifs": False, "objective": "discrimination rather than explaining a fixed percentage of all higher-order differences", "search": "exact memoized set-cover search up to five motifs over all left-center sample pairs after duplicate/dominated candidate reduction"},
        "conditions": [{"name": n, "echo_scale": e, "position_scale": p} for n, e, p in v44.CONDITIONS],
        "candidate_motif_count": len(candidates),
        "minimal_signature": minimal,
        "full_live_separation": full_separation,
        "leave_one_condition_out": loco,
        "right_control": {"event_absent_all_conditions": right_absent, "false_identity_event_count": sum(1 for row in runs["右"] if row["event_formed"])},
        "live_route_stability": {"左": route["左"]["minimum_route_jaccard_vs_baseline"], "中央": route["中央"]["minimum_route_jaccard_vs_baseline"]},
        "summary": {"left_event_all_conditions": left_complete, "center_event_all_conditions": center_complete, "right_event_absent": right_absent, "minimal_signature_found": minimal["found"], "minimal_motif_count": minimal["size"], "full_live_separation": full_separation, "loco_all_pass": loco["all_pass"], "right_false_positive": not right_absent, "core_readiness": core_readiness, "overall_verdict": verdict, "next_step": next_step},
        "brain_file_unchanged": v3.base.BEFORE_HASH == v3.base.sha(v3.base.BRAIN_PATH),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest_binding_v51.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("v51: complete.", flush=True)
    return payload


app = Flask(__name__)

PAGE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Core Growth Binding v51</title><style>
:root{--panel:#17253c;--panel2:#0c1727;--line:#385273;--text:#f3f7ff;--muted:#aebbd0;--orange:#ffad67;--green:#91efb0;--red:#ff9fa7;--blue:#8ed8ff}*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#07101d,#12213a);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1550px;margin:auto;padding:32px 22px 70px}h1{font-size:clamp(34px,5vw,60px);margin:0}.lead{color:var(--muted);font-size:18px;line-height:1.6}.panel{background:#17253c;border:1px solid var(--line);border-radius:22px;padding:24px;margin-top:20px}.controls{display:flex;justify-content:flex-end;align-items:center;gap:16px}button{padding:14px 20px;border-radius:12px;border:1px solid #466486;background:var(--orange);color:#101722;font-size:16px;font-weight:900;cursor:pointer}button:disabled{opacity:.6;cursor:wait}.status{color:var(--muted);font-weight:700}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric{background:var(--panel2);padding:16px;border-radius:14px;min-width:0}.metric b{display:block;font-size:18px;margin-top:6px;overflow-wrap:anywhere}.good{color:var(--green)}.warn{color:var(--red)}.blue{color:var(--blue)}.raw{white-space:pre-wrap;max-height:900px;overflow:auto;background:#07111d;padding:17px;border-radius:14px;color:#c7d5e9}@media(max-width:950px){.metrics{grid-template-columns:1fr}}</style></head><body><main><h1>Core Growth Binding v51</h1><p class="lead">左7条件と中央7条件を区別するために必要な最小Abstract Motif Signatureを探索し、leave-one-condition-outで未見条件への一般化を確認する。</p><section class="panel"><div class="controls"><span id="status" class="status">待機中</span><button id="run">Minimal Discriminative Signatureを検証</button></div></section><section class="panel"><h2>主要結果</h2><div id="metrics" class="metrics"></div></section><section class="panel"><h2>Signature生データ</h2><pre id="raw" class="raw">まだ診断していません。</pre></section></main><script>
function yn(v){return v?'YES':'NO'}function n(v){return v===null||v===undefined?'なし':v}function f(v){return v===undefined||v===null?'なし':Number(v).toFixed(6)}
const btn=document.getElementById('run'),status=document.getElementById('status'),metrics=document.getElementById('metrics'),raw=document.getElementById('raw');
btn.addEventListener('click',async()=>{btn.disabled=true;status.textContent='計算中… ターミナルにも進捗を表示しています';raw.textContent='v51を計算しています。LOCOを含むため少し時間がかかる場合があります。';await new Promise(r=>setTimeout(r,30));try{const res=await fetch('/api/observe',{method:'POST'});if(!res.ok)throw new Error('HTTP '+res.status);const d=await res.json(),s=d.summary,m=d.minimal_signature;metrics.innerHTML=`<div class="metric">左 Event全条件<b>${yn(s.left_event_all_conditions)}</b></div><div class="metric">中央 Event全条件<b>${yn(s.center_event_all_conditions)}</b></div><div class="metric">右 Eventなし<b>${yn(s.right_event_absent)}</b></div><div class="metric">最小Signature発見<b class="${s.minimal_signature_found?'good':'warn'}">${yn(s.minimal_signature_found)}</b></div><div class="metric">最小Motif数<b class="blue">${n(s.minimal_motif_count)}</b></div><div class="metric">全live分離<b>${yn(s.full_live_separation)}</b></div><div class="metric">LOCO全PASS<b class="${s.loco_all_pass?'good':'warn'}">${yn(s.loco_all_pass)}</b></div><div class="metric">右 誤検出<b class="${s.right_false_positive?'warn':'good'}">${yn(s.right_false_positive)}</b></div><div class="metric">候補Motif数<b>${d.candidate_motif_count}</b></div><div class="metric">削減後候補<b>${n(m.candidate_count_after_reduction)}</b></div><div class="metric">Core readiness<b class="blue">${s.core_readiness}</b></div><div class="metric">総合判定<b class="blue">${s.overall_verdict}</b></div><div class="metric">左 最小route Jaccard<b>${f(d.live_route_stability['左'])}</b></div><div class="metric">中央 最小route Jaccard<b>${f(d.live_route_stability['中央'])}</b></div><div class="metric">brain.json<b class="good">${d.brain_file_unchanged?'不変':'変化'}</b></div>`;raw.textContent=JSON.stringify(d,null,2);status.textContent='完了';}catch(e){status.textContent='エラー';raw.textContent='v51実行エラー: '+e;console.error(e);}finally{btn.disabled=false;}});
</script></body></html>'''


@app.get("/")
def index():
    return PAGE


@app.post("/api/observe")
def api_observe():
    return jsonify(observe())


def open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    print(f"Core Growth Binding v51: http://{HOST}:{PORT}")
    print("Minimal Discriminative Motif Signature / memoized exact <=5 set cover / LOCO / no Core changes")
    serve(app, host=HOST, port=PORT)
