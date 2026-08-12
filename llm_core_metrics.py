from __future__ import annotations

import json
import sqlite3
from typing import Iterable

import numpy as np

import llm_core_pipeline as pipeline


class CapturingAdapter:
    """OpenAIAdapter wrapper that records the embedding used by probe()."""

    def __init__(self) -> None:
        self.inner = pipeline.OpenAIAdapter()
        self.last_embedding: list[float] | None = None

    def embed(self, text: str) -> list[float]:
        embedding = self.inner.embed(text)
        self.last_embedding = embedding
        return embedding

    def decode(self, observation: dict) -> str:
        return self.inner.decode(observation)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def probe_with_metrics(text: str, display_limit: int = 10) -> dict:
    """Probe Core and compare LLM embedding similarity separately.

    Core overlap is allowed to change with experience. Embedding cosine similarity
    is calculated directly from the fixed input vectors and acts as the baseline.
    """
    pipeline.initialize_db()
    with sqlite3.connect(pipeline.DB_FILE, timeout=30) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0])

    adapter = CapturingAdapter()
    observation = pipeline.probe(text, limit=max(1, total), adapter=adapter)
    current_embedding = adapter.last_embedding or []

    with sqlite3.connect(pipeline.DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, embedding FROM experiences ORDER BY id DESC"
        ).fetchall()

    stored_embeddings = {
        int(row["id"]): json.loads(row["embedding"])
        for row in rows
    }

    all_matches: list[dict] = []
    for item in observation["matches"]:
        enriched = dict(item)
        stored = stored_embeddings.get(int(item["experience_id"]), [])
        enriched["embedding_similarity"] = round(
            cosine_similarity(current_embedding, stored), 6
        )
        all_matches.append(enriched)

    core_scores = [float(item["score"]) for item in all_matches]
    embedding_scores = [float(item["embedding_similarity"]) for item in all_matches]
    node_scores = [float(item["node_overlap"]) for item in all_matches]
    edge_scores = [float(item["edge_overlap"]) for item in all_matches]

    observation["matches"] = all_matches[: max(1, int(display_limit))]
    observation["comparison"] = {
        "experience_count": len(all_matches),
        "core_max": max(core_scores, default=0.0),
        "core_average": _mean(core_scores),
        "embedding_max": max(embedding_scores, default=0.0),
        "embedding_average": _mean(embedding_scores),
        "node_overlap_max": max(node_scores, default=0.0),
        "node_overlap_average": _mean(node_scores),
        "edge_overlap_max": max(edge_scores, default=0.0),
        "edge_overlap_average": _mean(edge_scores),
    }
    return observation
