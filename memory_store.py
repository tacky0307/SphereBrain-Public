from __future__ import annotations

from pathlib import Path
import sqlite3
import json
from datetime import datetime


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                input_text TEXT,
                source_nodes TEXT NOT NULL,
                activated_nodes TEXT NOT NULL,
                traversed_edges TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """)

    @staticmethod
    def _normalize_edge(edge) -> tuple[int, int]:
        a, b = int(edge[0]), int(edge[1])
        return (a, b) if a <= b else (b, a)

    @classmethod
    def _edge_set(cls, edges) -> set[tuple[int, int]]:
        return {cls._normalize_edge(edge) for edge in edges}

    def add_memory(
        self,
        kind: str,
        input_text: str,
        source_nodes: list[int],
        activated_nodes: list[int],
        traversed_edges: list[tuple[int, int]],
        importance: float = 1.0,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories
                (created_at, kind, input_text, source_nodes, activated_nodes, traversed_edges, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    kind,
                    input_text,
                    json.dumps(source_nodes),
                    json.dumps(activated_nodes),
                    json.dumps(traversed_edges),
                    importance,
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["source_nodes"] = json.loads(item["source_nodes"])
            item["activated_nodes"] = json.loads(item["activated_nodes"])
            item["traversed_edges"] = json.loads(item["traversed_edges"])
            result.append(item)
        return result

    def latest_input_memories(self, limit: int = 40, exclude_id: int | None = None) -> list[dict]:
        params: list[object] = []
        where = "WHERE kind = 'input'"
        if exclude_id is not None:
            where += " AND id <> ?"
            params.append(exclude_id)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where} ORDER BY id DESC LIMIT ?",
                tuple(params),
            ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            item["source_nodes"] = json.loads(item["source_nodes"])
            item["activated_nodes"] = json.loads(item["activated_nodes"])
            item["traversed_edges"] = json.loads(item["traversed_edges"])
            result.append(item)
        return result

    def route_analytics(self, memory_id: int | None = None, comparison_limit: int = 80) -> dict:
        memories = self.latest_input_memories(limit=max(2, comparison_limit + 1))
        if not memories:
            return {
                "available": False,
                "current_text": "",
                "route_count": 0,
                "node_count": 0,
                "previous_similarity": 0.0,
                "previous_common_routes": 0,
                "previous_text": "",
                "new_routes": 0,
                "best_match_text": "",
                "best_match_similarity": 0.0,
                "best_match_common_routes": 0,
                "fingerprint": [],
                "top_routes": [],
            }

        current = memories[0]
        if memory_id is not None:
            match = next((item for item in memories if int(item["id"]) == int(memory_id)), None)
            if match is not None:
                current = match

        candidates = [item for item in memories if int(item["id"]) != int(current["id"])]
        current_edges = self._edge_set(current["traversed_edges"])
        current_nodes = {int(node) for node in current["activated_nodes"]}

        previous = candidates[0] if candidates else None
        previous_edges = self._edge_set(previous["traversed_edges"]) if previous else set()
        previous_common = len(current_edges & previous_edges)
        previous_union = len(current_edges | previous_edges)
        previous_similarity = (previous_common / previous_union * 100.0) if previous_union else 0.0

        seen_before: set[tuple[int, int]] = set()
        for item in candidates:
            seen_before.update(self._edge_set(item["traversed_edges"]))
        new_routes = len(current_edges - seen_before)

        best_match = None
        best_similarity = 0.0
        best_common = 0
        for item in candidates:
            edges = self._edge_set(item["traversed_edges"])
            common = len(current_edges & edges)
            union = len(current_edges | edges)
            similarity = (common / union * 100.0) if union else 0.0
            if similarity > best_similarity:
                best_similarity = similarity
                best_common = common
                best_match = item

        route_frequency: dict[tuple[int, int], int] = {}
        for item in [current] + candidates:
            for edge in self._edge_set(item["traversed_edges"]):
                route_frequency[edge] = route_frequency.get(edge, 0) + 1
        top_routes = [
            {"a": edge[0], "b": edge[1], "count": count}
            for edge, count in sorted(route_frequency.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
        ]

        buckets = [0] * 24
        for node in current_nodes:
            buckets[node % len(buckets)] += 1
        peak = max(buckets) if buckets else 0
        fingerprint = [round(value / peak, 3) if peak else 0.0 for value in buckets]

        return {
            "available": True,
            "current_text": current.get("input_text") or "",
            "route_count": len(current_edges),
            "node_count": len(current_nodes),
            "previous_similarity": round(previous_similarity, 1),
            "previous_common_routes": previous_common,
            "previous_text": (previous.get("input_text") or "") if previous else "",
            "new_routes": new_routes,
            "best_match_text": (best_match.get("input_text") or "") if best_match else "",
            "best_match_similarity": round(best_similarity, 1),
            "best_match_common_routes": best_common,
            "fingerprint": fingerprint,
            "top_routes": top_routes,
        }

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def recent_context_nodes(self, memory_limit: int = 5, node_limit: int = 16) -> list[int]:
        memories = self.recent(memory_limit)
        scores: dict[int, int] = {}
        for memory in memories:
            for node in memory["activated_nodes"]:
                scores[node] = scores.get(node, 0) + 1
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:node_limit]

    def set_value(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_value(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return default if row is None else str(row["value"])
