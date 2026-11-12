"""Run entity extraction + linking over the sampled dev sets (Role 2).

Usage (from the repository root, after ``python -m data.sampling``)::

    python -m src.graph.run_extraction                      # rule baseline, all datasets
    python -m src.graph.run_extraction --dataset musique --limit 20
    python -m src.graph.run_extraction --extractor llm      # needs GEMINI_API_KEY

For each dataset it writes, under ``out_dir/<dataset>/``:

- ``entities.jsonl``  one canonical entity per line (id, name, type, aliases, paragraph_ids)
- ``mentions.jsonl``  one mention per line, with the ``entity_id`` it was linked to
- ``manifest.json``   config, counts and gold-entity recall
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Optional

import yaml

from data.loaders.base import read_question_records, write_jsonl

from .entities import text_sha1
from .eval_extraction import gold_entity_recall
from .extract import QuotaExhausted, build_extractor
from .linking import link_mentions

DEFAULT_CONFIG = "configs/extraction.yaml"


def unique_paragraphs(records: list) -> list:
    """Every distinct paragraph, extracted once even when several questions share it.

    Keyed by (id, text), not id alone, so no text is ever dropped even if an id
    were reused for two texts (Role 1 now guarantees it is not; this stays as a
    safety net, and ``ambiguous_ids`` reports it in the manifest).
    """
    seen = {}
    for record in records:
        for p in record.paragraphs:
            seen.setdefault((p.id, p.text), p)
    return list(seen.values())


def ambiguous_ids(paragraphs: list) -> int:
    """Number of paragraph ids that carry more than one text."""
    counts = Counter(p.id for p in paragraphs)
    return sum(1 for n in counts.values() if n > 1)


def extract_dataset(records: list, extractor, log_every: int = 500, workers: int = 1,
                    batch_size: int = 1, failures: Optional[list] = None) -> tuple:
    """Extract every distinct paragraph and link the mentions.

    For the LLM extractor, ``batch_size`` paragraphs go into one request and
    ``workers`` requests run in parallel threads. Results keep paragraph order, so
    the output is identical to a sequential run. A batch whose extraction raises
    (e.g. retries exhausted) is skipped and its paragraph ids appended to
    ``failures``; thanks to the cache, re-running only redoes those.
    :class:`QuotaExhausted` is never swallowed: it stops the whole run.
    """
    paragraphs = unique_paragraphs(records)
    batch_size = batch_size if hasattr(extractor, "extract_many") else 1
    batches = [paragraphs[i:i + batch_size] for i in range(0, len(paragraphs), batch_size)]

    def one(batch):
        try:
            found = extractor.extract_many(batch) if batch_size > 1 else [extractor.extract(batch[0])]
        except QuotaExhausted:
            raise
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            if failures is None:
                raise
            failures.extend(p.id for p in batch)
            print(f"  failed batch at {batch[0].id}: {type(exc).__name__}: {str(exc)[:200]}")
            return []
        # the hash tells which text a mention's offsets refer to when an id is ambiguous
        return [replace(m, paragraph_sha1=text_sha1(p.text)) for p, ms in zip(batch, found) for m in ms]

    mentions, started, done = [], time.monotonic(), 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for batch, found in zip(batches, pool.map(one, batches)):
            mentions.extend(found)
            before, done = done, done + len(batch)
            if log_every and done // log_every > before // log_every:
                rate = done / max(time.monotonic() - started, 1e-9)
                eta = (len(paragraphs) - done) / rate
                print(f"  {done}/{len(paragraphs)} paragraphs, {len(mentions)} mentions, "
                      f"{rate:.1f}/s, ~{eta / 60:.0f} min left", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    entities, norm_to_id = link_mentions(mentions)
    return paragraphs, mentions, entities, norm_to_id


def run(config_path: str = DEFAULT_CONFIG, datasets=None, limit=None, extractor_name=None,
        out_dir=None) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if extractor_name:
        cfg["extractor"] = extractor_name
    if out_dir:
        cfg["out_dir"] = out_dir
    if (cfg.get("linking") or {}).get("embedding_merge"):
        # wire linking.merge_by_embedding in here once the shared embed() exists
        raise NotImplementedError("linking.embedding_merge needs the shared embed() service (docs/INTERFACE.md)")
    extractor = build_extractor(cfg)
    processed = Path(cfg.get("processed_dir", "data/processed"))
    out_root = Path(cfg.get("out_dir", "data/graph/extraction"))

    report = {}
    for name in datasets or cfg.get("datasets", []):
        path = processed / f"{name}_dev.jsonl"
        if not path.exists():
            print(f"[{name}] skipped: {path} not found (run `python -m data.sampling` first)")
            continue
        records = read_question_records(path)[:limit] if limit else read_question_records(path)
        print(f"[{name}] {len(records)} questions, extractor={cfg['extractor']}", flush=True)
        llm = (cfg.get("llm") or {}) if cfg["extractor"] == "llm" else {}
        workers, batch_size = int(llm.get("workers", 1)), int(llm.get("batch_size", 1))
        failures = []
        try:
            paragraphs, mentions, entities, norm_to_id = extract_dataset(
                records, extractor, log_every=100 if llm else 500, workers=workers,
                batch_size=batch_size, failures=failures)
        except QuotaExhausted as exc:
            print(f"[{name}] STOPPED: daily API quota exhausted. Paragraphs finished so far are "
                  f"cached; re-run the same command once the quota resets (or enable billing).\n"
                  f"  {str(exc)[:300]}")
            return report

        out = out_root / name
        write_jsonl((e.to_dict() for e in entities), out / "entities.jsonl")
        write_jsonl(({**m.to_dict(), "entity_id": norm_to_id.get(m.norm)} for m in mentions),
                    out / "mentions.jsonl")
        manifest = {
            "dataset": name,
            "extractor": cfg["extractor"],
            "llm": cfg.get("llm") if cfg["extractor"] == "llm" else None,
            "questions": len(records),
            "paragraphs": len(paragraphs),
            "paragraph_ids_with_multiple_texts": ambiguous_ids(paragraphs),
            "mentions": len(mentions),
            "entities": len(entities),
            "entity_types": dict(Counter(e.type for e in entities).most_common()),
            "failed_paragraphs": failures,
            "llm_calls": getattr(extractor, "calls", None),
            "llm_invalid_responses": getattr(extractor, "invalid", None),
            "llm_missing_paragraphs": getattr(extractor, "missing", None),
            "llm_dropped_items": getattr(extractor, "dropped", None),
            "gold_entity_recall": gold_entity_recall(records, mentions),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        recall = manifest["gold_entity_recall"].get("all", {})
        print(f"[{name}] {len(paragraphs)} paragraphs -> {len(mentions)} mentions -> "
              f"{len(entities)} entities; gold recall strict={recall.get('strict_recall')} "
              f"lenient={recall.get('lenient_recall')} -> {out}", flush=True)
        incomplete = len(failures) + (getattr(extractor, "missing", 0) or 0) + (getattr(extractor, "invalid", 0) or 0)
        if incomplete:
            print(f"[{name}] WARNING: {incomplete} paragraphs/batches are not extracted yet (failed, "
                  "skipped or invalid answer); re-run to retry them, finished ones come from the cache")
        report[name] = manifest
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", help="dataset name; repeat for several (default: config)")
    parser.add_argument("--limit", type=int, help="only the first N questions (for quick checks)")
    parser.add_argument("--extractor", choices=["rule", "llm"], help="override the config's extractor")
    parser.add_argument("--out-dir", help="override the config's out_dir")
    args = parser.parse_args(argv)
    run(args.config, args.dataset, args.limit, args.extractor, args.out_dir)


if __name__ == "__main__":
    main()
