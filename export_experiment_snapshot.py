from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json
import sqlite3


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"
EXPORT_DIR = DATA / "experiment_exports"


COLUMNS = [
    "snapshot_id",
    "exported_at",
    "record_type",
    "record_id",
    "created_at",
    "kind",
    "input_text",
    "importance",
    "node_a",
    "node_b",
    "weight",
    "usage",
    "node_usage",
    "position_x",
    "position_y",
    "position_z",
    "source_nodes",
    "activated_nodes",
    "traversed_edges",
    "memory_count",
    "node_count",
    "edge_count",
    "total_edge_usage",
    "total_node_usage",
]


def _row(snapshot_id: str, exported_at: str, record_type: str, **values: object) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in COLUMNS}
    row.update(
        {
            "snapshot_id": snapshot_id,
            "exported_at": exported_at,
            "record_type": record_type,
        }
    )
    row.update(values)
    return row


def _load_brain() -> dict:
    if not BRAIN_FILE.exists():
        raise FileNotFoundError(f"脳データが見つかりません: {BRAIN_FILE}")
    return json.loads(BRAIN_FILE.read_text(encoding="utf-8"))


def _load_memories() -> list[dict]:
    if not DB_FILE.exists():
        return []

    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def export_snapshot() -> Path:
    brain = _load_brain()
    memories = _load_memories()

    exported_at = datetime.now().isoformat(timespec="seconds")
    snapshot_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = EXPORT_DIR / f"spherebrain_snapshot_{snapshot_id}.csv"

    node_count = int(brain["node_count"])
    adjacency = brain["adjacency"]
    weights = brain["weights"]
    usage = brain["usage"]
    positions = brain["positions"]
    node_usage = brain.get("node_usage", [0] * node_count)

    edges: list[tuple[int, int]] = []
    for a in range(node_count):
        for b in range(a + 1, node_count):
            if adjacency[a][b]:
                edges.append((a, b))

    rows: list[dict[str, object]] = []
    rows.append(
        _row(
            snapshot_id,
            exported_at,
            "summary",
            memory_count=len(memories),
            node_count=node_count,
            edge_count=len(edges),
            total_edge_usage=sum(int(usage[a][b]) for a, b in edges),
            total_node_usage=sum(int(value) for value in node_usage),
        )
    )

    for node_id in range(node_count):
        position = positions[node_id]
        rows.append(
            _row(
                snapshot_id,
                exported_at,
                "node",
                record_id=node_id,
                node_usage=int(node_usage[node_id]),
                position_x=float(position[0]),
                position_y=float(position[1]),
                position_z=float(position[2]),
            )
        )

    for a, b in edges:
        rows.append(
            _row(
                snapshot_id,
                exported_at,
                "edge",
                record_id=f"{a}-{b}",
                node_a=a,
                node_b=b,
                weight=float(weights[a][b]),
                usage=int(usage[a][b]),
            )
        )

    for memory in memories:
        rows.append(
            _row(
                snapshot_id,
                exported_at,
                "memory",
                record_id=memory.get("id", ""),
                created_at=memory.get("created_at", ""),
                kind=memory.get("kind", ""),
                input_text=memory.get("input_text", ""),
                importance=memory.get("importance", ""),
                source_nodes=memory.get("source_nodes", ""),
                activated_nodes=memory.get("activated_nodes", ""),
                traversed_edges=memory.get("traversed_edges", ""),
            )
        )

    # utf-8-sig にすることで、日本語をExcelで直接開いても文字化けしにくくする。
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output


if __name__ == "__main__":
    try:
        path = export_snapshot()
        print("\n実験スナップショットを出力しました。")
        print(path)
    except Exception as exc:
        print("\nCSV出力に失敗しました。")
        print(exc)
        raise SystemExit(1)
