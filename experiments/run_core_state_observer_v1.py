from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_core_pipeline as pipeline
from core_state_observer import CoreStateObserver

DATA = ROOT / "data" / "core_state_observer_v1"
RESULTS = DATA / "results"
TRAIN_REPEATS = 10
TRAINING = {
    "sunny": ["今日は晴れて気持ちいい", "青い空が広がって爽やかだ", "暖かな日差しが心地よい"],
    "rainy": ["今日は雨で肌寒い", "暗い空から雨が降っている", "冷たい雨で気分が沈む"],
    "ambiguous": ["雨上がりの空に日が差している", "曇っているが少し暖かい", "晴れているのに冷たい風が吹く"],
}
PROBES = {
    "sunny_close": "今日の天気は最高だ",
    "rainy_close": "冷たい雨が降り続いている",
    "ambiguous_close": "雲の切れ間から少し光が差している",
    "unrelated": "犬は公園を走っている",
}


def configure_core() -> None:
    pipeline.DATA = DATA / "core"
    pipeline.BRAIN_FILE = pipeline.DATA / "brain.json"
    pipeline.DB_FILE = pipeline.DATA / "experiences.db"
    pipeline.PROJECTION_FILE = DATA / "projection.npy"
    pipeline.PROJECTION_SEED = 20260806


def cosine(left, right) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def jaccard(left, right) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def observe(text: str, adapter, observer: CoreStateObserver) -> dict:
    embedding, stimulus = pipeline.encode_text(text, adapter)
    brain = pipeline.load_brain()
    sources = pipeline.stimulus_to_sources(brain, stimulus)
    result = brain.propagate(sources, steps=14, threshold=0.18, noise=0.0, learn=False)
    return {
        "text": text,
        "embedding": embedding,
        "nodes": list(result.activated_nodes),
        "edges": [list(edge) for edge in result.traversed_edges],
        "observer": observer.observe(brain, result),
    }


def route_similarity(left: dict, right: dict) -> float:
    nodes = jaccard(left["nodes"], right["nodes"])
    left_edges = {tuple(edge) for edge in left["edges"]}
    right_edges = {tuple(edge) for edge in right["edges"]}
    return 0.35 * nodes + 0.65 * jaccard(left_edges, right_edges)


def main() -> None:
    configure_core()
    RESULTS.mkdir(parents=True, exist_ok=True)
    adapter = pipeline.OpenAIAdapter()
    observer = CoreStateObserver()
    pipeline.reset_experiment()

    for texts in TRAINING.values():
        for text in texts:
            pipeline.experience(text, repeats=TRAIN_REPEATS, adapter=adapter)

    references = {group: [observe(text, adapter, observer) for text in texts] for group, texts in TRAINING.items()}
    probes = {name: observe(text, adapter, observer) for name, text in PROBES.items()}
    rows = []
    for probe_name, probe in probes.items():
        for group, items in references.items():
            for index, reference in enumerate(items, start=1):
                rows.append({
                    "probe": probe_name,
                    "reference_group": group,
                    "reference_index": index,
                    "embedding_similarity": cosine(probe["embedding"], reference["embedding"]),
                    "route_similarity": route_similarity(probe, reference),
                    "observer_similarity": cosine(probe["observer"]["state_vector"], reference["observer"]["state_vector"]),
                })

    payload = {
        "experiment": "Core State Observer v1",
        "read_only": True,
        "train_repeats": TRAIN_REPEATS,
        "training": TRAINING,
        "probes": probes,
        "references": references,
        "comparisons": rows,
    }
    json_path = RESULTS / "core_state_observer_v1.json"
    csv_path = RESULTS / "core_state_observer_v1.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()
