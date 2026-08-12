from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import os
import sqlite3

import numpy as np

from brain import SignalResult, SphereBrain

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "llm_core_v1"
BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "experiences.db"
SOURCE_BRAIN_FILE = BASE / "data" / "brain.json"
PROJECTION_FILE = DATA / "projection.npy"

EMBEDDING_MODEL = os.getenv("SPHERE_EMBEDDING_MODEL", "text-embedding-3-small")
DECODER_MODEL = os.getenv("SPHERE_DECODER_MODEL", "gpt-5-mini")
STIMULUS_DIM = int(os.getenv("SPHERE_STIMULUS_DIM", "128"))
SOURCE_COUNT = int(os.getenv("SPHERE_SOURCE_COUNT", "10"))
PROJECTION_SEED = int(os.getenv("SPHERE_PROJECTION_SEED", "20260804"))


@dataclass
class EncodedExperience:
    text: str
    embedding: list[float]
    stimulus: list[float]
    source_nodes: list[int]
    result: SignalResult


class OpenAIAdapter:
    """LLM is used only as an input/output interface around SphereBrain Core."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai パッケージがありません。pip install -r requirements.txt を実行してください。"
            ) from exc
        self.client = OpenAI()

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return list(response.data[0].embedding)

    def decode(self, observation: dict) -> str:
        instructions = (
            "あなたはSphereBrainのDecoderです。"
            "外部知識で答えを補わず、与えられたCore観測と想起候補だけを自然な日本語へ翻訳してください。"
            "Coreの確信が弱い場合は、断定せず曖昧さを明示してください。"
            "内部IDや数値をそのまま羅列せず、1〜3文で簡潔に表現してください。"
        )
        response = self.client.responses.create(
            model=DECODER_MODEL,
            instructions=instructions,
            input=json.dumps(observation, ensure_ascii=False),
        )
        return response.output_text.strip()


def initialize_db() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                raw_text TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding TEXT NOT NULL,
                stimulus TEXT NOT NULL,
                source_nodes TEXT NOT NULL,
                activated_nodes TEXT NOT NULL,
                traversed_edges TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_llm_core_created_at ON experiences(created_at);
            """
        )


def create_clean_brain() -> SphereBrain:
    DATA.mkdir(parents=True, exist_ok=True)
    if SOURCE_BRAIN_FILE.exists():
        source = SphereBrain.load(SOURCE_BRAIN_FILE)
        brain = SphereBrain(
            node_count=source.node_count,
            neighbors_per_node=source.neighbors_per_node,
            seed=source.seed,
            learning_rate=source.learning_rate,
            decay_rate=source.decay_rate,
            propagation_mode=source.propagation_mode,
            signal_decay=source.signal_decay,
            max_branches=source.max_branches,
            max_active_per_step=source.max_active_per_step,
            max_total_active_nodes=source.max_total_active_nodes,
        )
    else:
        brain = SphereBrain(node_count=600)
    brain.save(BRAIN_FILE)
    return brain


def load_brain() -> SphereBrain:
    return SphereBrain.load(BRAIN_FILE) if BRAIN_FILE.exists() else create_clean_brain()


def reset_experiment() -> None:
    """Reset only this experiment without deleting open files on Windows.

    SQLite/OneDrive/antivirus can temporarily keep the DB file open. Clearing
    table contents and overwriting the Core avoids WinError 32 while preserving
    the same experimental boundary.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("DELETE FROM experiences")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='experiences'")
        conn.commit()

    # Recreate and overwrite the isolated Core instead of unlinking its file.
    create_clean_brain()

    # The projection is deterministic from PROJECTION_SEED. Keeping the existing
    # matrix preserves identical input conditions between repeated experiments.


def _projection(input_dim: int) -> np.ndarray:
    DATA.mkdir(parents=True, exist_ok=True)
    if PROJECTION_FILE.exists():
        matrix = np.load(PROJECTION_FILE)
        if matrix.shape == (STIMULUS_DIM, input_dim):
            return matrix

    rng = np.random.default_rng(PROJECTION_SEED)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(STIMULUS_DIM), size=(STIMULUS_DIM, input_dim))
    np.save(PROJECTION_FILE, matrix)
    return matrix


def project_embedding(embedding: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(embedding), dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embeddingが空、または一次元ではありません。")
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    projected = _projection(vector.size) @ vector
    projected_norm = np.linalg.norm(projected)
    if projected_norm > 0:
        projected = projected / projected_norm
    return projected


def stimulus_to_sources(brain: SphereBrain, stimulus: np.ndarray) -> list[int]:
    """Map strongest signed stimulus dimensions to stable Core entry nodes."""
    count = min(SOURCE_COUNT, stimulus.size)
    strongest = np.argsort(np.abs(stimulus))[-count:][::-1]
    nodes: list[int] = []
    for dimension in strongest:
        sign = "positive" if stimulus[dimension] >= 0 else "negative"
        material = f"llm-core-v1|{dimension}|{sign}".encode("utf-8")
        node = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % brain.node_count
        if node not in nodes:
            nodes.append(node)
    return nodes


def encode_text(text: str, adapter: OpenAIAdapter | None = None) -> tuple[list[float], np.ndarray]:
    clean = text.strip()
    if not clean:
        raise ValueError("入力が空です。")
    adapter = adapter or OpenAIAdapter()
    embedding = adapter.embed(clean)
    return embedding, project_embedding(embedding)


def experience(text: str, repeats: int = 1, adapter: OpenAIAdapter | None = None) -> EncodedExperience:
    adapter = adapter or OpenAIAdapter()
    embedding, stimulus = encode_text(text, adapter)
    brain = load_brain()
    sources = stimulus_to_sources(brain, stimulus)
    latest: SignalResult | None = None

    for _ in range(max(1, int(repeats))):
        latest = brain.propagate(
            sources,
            steps=14,
            threshold=0.18,
            noise=0.004,
            learn=True,
        )
        save_experience(text.strip(), embedding, stimulus, sources, latest)

    brain.save(BRAIN_FILE)
    assert latest is not None
    return EncodedExperience(text.strip(), embedding, stimulus.tolist(), sources, latest)


def save_experience(
    text: str,
    embedding: list[float],
    stimulus: np.ndarray,
    source_nodes: list[int],
    result: SignalResult,
) -> None:
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            """
            INSERT INTO experiences(
                raw_text, embedding_model, embedding, stimulus,
                source_nodes, activated_nodes, traversed_edges
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                text,
                EMBEDDING_MODEL,
                json.dumps(embedding),
                json.dumps(stimulus.tolist()),
                json.dumps(source_nodes),
                json.dumps(result.activated_nodes),
                json.dumps(result.traversed_edges),
            ),
        )


def _jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def probe(text: str, limit: int = 5, adapter: OpenAIAdapter | None = None) -> dict:
    adapter = adapter or OpenAIAdapter()
    embedding, stimulus = encode_text(text, adapter)
    brain = load_brain()
    sources = stimulus_to_sources(brain, stimulus)
    result = brain.propagate(
        sources,
        steps=14,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )

    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM experiences ORDER BY id DESC").fetchall()

    matches = []
    for row in rows:
        stored_nodes = json.loads(row["activated_nodes"])
        stored_edges = [tuple(edge) for edge in json.loads(row["traversed_edges"])]
        node_score = _jaccard(result.activated_nodes, stored_nodes)
        edge_score = _jaccard(result.traversed_edges, stored_edges)
        score = 0.35 * node_score + 0.65 * edge_score
        matches.append(
            {
                "experience_id": row["id"],
                "text": row["raw_text"],
                "score": round(score, 6),
                "node_overlap": round(node_score, 6),
                "edge_overlap": round(edge_score, 6),
            }
        )

    matches.sort(key=lambda item: (-item["score"], -item["experience_id"]))
    return {
        "input": text.strip(),
        "embedding_model": EMBEDDING_MODEL,
        "stimulus_dimension": STIMULUS_DIM,
        "source_nodes": sources,
        "activated_nodes": result.activated_nodes,
        "traversed_edges": result.traversed_edges,
        "matches": matches[: max(1, int(limit))],
    }


def ask(text: str, limit: int = 5, adapter: OpenAIAdapter | None = None) -> dict:
    adapter = adapter or OpenAIAdapter()
    observation = probe(text, limit=limit, adapter=adapter)
    decoder_view = {
        "current_input": observation["input"],
        "core_observation": {
            "source_node_count": len(observation["source_nodes"]),
            "activated_node_count": len(observation["activated_nodes"]),
            "traversed_edge_count": len(observation["traversed_edges"]),
        },
        "recalled_experiences_selected_by_core": observation["matches"],
        "decoder_rule": "候補はCore活動との重なり順。候補外の知識を追加しない。",
    }
    answer = adapter.decode(decoder_view)
    return {"answer": answer, "observation": observation, "decoder_input": decoder_view}


def stats() -> dict:
    initialize_db()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        distinct = conn.execute("SELECT COUNT(DISTINCT raw_text) FROM experiences").fetchone()[0]
    return {
        "total_experiences": total,
        "distinct_texts": distinct,
        "brain_file": str(BRAIN_FILE),
        "db_file": str(DB_FILE),
        "embedding_model": EMBEDDING_MODEL,
        "decoder_model": DECODER_MODEL,
    }
