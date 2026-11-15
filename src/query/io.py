"""Persistence helpers for Role 4 inputs and outputs."""
from __future__ import annotations

import json
from pathlib import Path

from src.graph.entities import Entity


def load_entities(path: str | Path) -> list[Entity]:
    """Load Role 2's canonical entities from JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"entity file not found: {p}; run Role 2 extraction first")
    entities = []
    with p.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entities.append(Entity.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid entity record at {p}:{line_number}") from exc
    return entities


def write_jsonl(records, path: str | Path) -> None:
    """Write JSON-serializable records as UTF-8 JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
