"""Generate answers for retrieval-output JSONL using the shared Role 6 prompt.

Input rows must contain question, context, and question_id. Outputs preserve all
retrieval fields and add answer, model usage and cited paragraph IDs.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src.context.answerer import GeminiAnswerer
from src.context.prompt import build_qa_prompt, parse_answer_response
from src.common.tokenization import RegexTokenCounter


def answer_file(input_path: str, output_path: str, model: str = "gemini-3.5-flash-lite",
                limit: int | None = None, answerer=None,
                requests_per_minute: float = 1000, concurrency: int = 16):
    rows = []
    with Path(input_path).open(encoding="utf-8") as handle:
        source = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        source = source[:limit]
    answerer = answerer or GeminiAnswerer(model=model, temperature=0.0,
                                         requests_per_minute=requests_per_minute)
    token_counter = RegexTokenCounter()

    def generate_one(index_row):
        index, row = index_row
        prompt = build_qa_prompt(row["question"], row.get("context", ""))
        row["prompt_tokens_shared"] = token_counter(prompt)
        started = time.perf_counter()
        generated = answerer.generate(prompt)
        row["answer_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        row["latency_ms"] = round(float(row.get("retrieval_latency_ms", 0.0)) + row["answer_latency_ms"], 3)
        row["answer_mode"] = "gemini"
        parsed = parse_answer_response(generated.response_text)
        allowed = set(row.get("retrieved_paragraph_ids", []))
        row.update({
            "answer": parsed["answer"],
            "supporting_paragraph_ids": [p for p in parsed["supporting_paragraph_ids"] if p in allowed],
            "evidence_sufficiency": parsed["evidence_sufficiency"],
            "answer_model": generated.model,
            "prompt_tokens_model_reported": generated.prompt_tokens,
            "output_tokens_model_reported": generated.output_tokens,
        })
        return row

    worker_count = max(1, min(concurrency, len(source)))
    if worker_count == 1:
        rows = [generate_one(item) for item in enumerate(source, 1)]
    else:
        print(f"generating {len(source)} answers with {worker_count} concurrent workers",
              flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            rows = list(pool.map(generate_one, enumerate(source, 1)))
    for index in range(25, len(rows) + 1, 25):
        print(f"generated {index}/{len(rows)} answers", flush=True)
    if len(rows) % 25:
        print(f"generated {len(rows)}/{len(rows)} answers", flush=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} answers -> {out}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--requests-per-minute", type=float)
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args(argv)
    context_cfg = yaml.safe_load(Path("configs/context.yaml").read_text(encoding="utf-8")) or {}
    concurrency = args.concurrency
    if concurrency is None:
        concurrency = int(context_cfg.get("concurrency", 16))
    answer_file(
        args.input,
        args.output,
        model=args.model,
        limit=args.limit,
        requests_per_minute=args.requests_per_minute or float(
            context_cfg.get("requests_per_minute", 1000)),
        concurrency=concurrency,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
