"""Shared selection of deterministic tuning/test question IDs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def filter_records_by_split(
    records: list,
    *,
    split_manifest: str | Path | None = None,
    split_part: str | None = None,
    limit: int | None = None,
) -> list:
    """Filter records by a split manifest, then apply an optional smoke-test limit.

    A split part without a manifest, or a manifest without a part, is rejected to avoid
    accidentally mixing tuning and held-out questions in experiment runs.
    """
    if (split_manifest is None) != (split_part is None):
        raise ValueError("provide both split_manifest and split_part ('tune' or 'test')")
    if split_part not in {None, "tune", "test"}:
        raise ValueError("split_part must be 'tune' or 'test'")
    selected = list(records)
    if split_manifest is not None:
        payload = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        ids = payload.get(f"{split_part}_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"split manifest has no {split_part}_ids")
        requested = set(ids)
        available = {record.question_id for record in selected}
        unknown = requested - available
        if unknown:
            raise ValueError(f"split contains question IDs absent from processed data: {sorted(unknown)[:5]}")
        selected = [record for record in selected if record.question_id in requested]
        if len(selected) != len(requested):
            raise ValueError(f"split manifest contains duplicate {split_part} question IDs")
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative or None")
        selected = selected[:limit]
    return selected
