"""Convert the HuggingFace HotpotQA parquet into the native JSON the loader reads (Role 1).

The official download URLs are dead (see docs/DATA_NOTES.md), but the official
HuggingFace repo ``hotpotqa/hotpot_qa`` still serves the distractor dev set as
parquet. Its columns hold the same data in a column-wise layout::

    supporting_facts = {"title": [...], "sent_id": [...]}
    context          = {"title": [...], "sentences": [[...], ...]}

This script rewrites it into the native layout (``context = [[title, [sents]]]``,
``supporting_facts = [[title, sent_id]]``), so :class:`HotpotQALoader` needs no
changes. Usage (needs ``pip install pyarrow``)::

    curl -L -o data/raw/hotpotqa/validation-00000-of-00001.parquet \\
      https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet
    python -m data.convert_hotpotqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SRC = "data/raw/hotpotqa/validation-00000-of-00001.parquet"
DEFAULT_DEST = "data/raw/hotpotqa/hotpot_dev_distractor_v1.json"


def to_native(row: dict) -> dict:
    """One parquet row -> one record in the original HotpotQA JSON format."""
    facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    return {
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row.get("type"),
        "level": row.get("level"),
        "supporting_facts": [
            [title, int(sent_id)]
            for title, sent_id in zip(facts.get("title") or [], facts.get("sent_id") or [])
        ],
        "context": [
            [title, list(sentences)]
            for title, sentences in zip(context.get("title") or [], context.get("sentences") or [])
        ],
    }


def convert(src: str | Path = DEFAULT_SRC, dest: str | Path = DEFAULT_DEST) -> int:
    import pyarrow.parquet as pq

    rows = pq.read_table(src).to_pylist()
    records = [to_native(r) for r in rows]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)
    print(f"wrote {len(records)} records -> {dest}")
    return len(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--dest", default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    convert(args.src, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
