"""Optional Gemini answer generation for Role 6."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import hashlib
import json
from pathlib import Path
from typing import Callable

from src.graph.embeddings import load_dotenv


@dataclass
class AnswerResult:
    response_text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""


class AnswerCache:
    """Atomic disk cache for Gemini answers keyed by model and exact prompt."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self._write_lock = threading.Lock()

    def _key(self, model: str, prompt: str) -> str:
        payload = f"{model}\x00{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> AnswerResult | None:
        path = self.dir / f"{self._key(model, prompt)}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return AnswerResult(
                response_text=raw["response_text"],
                prompt_tokens=raw.get("prompt_tokens"),
                output_tokens=raw.get("output_tokens"),
                model=raw.get("model", model),
            )
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return None

    def put(self, model: str, prompt: str, result: AnswerResult) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self._key(model, prompt)}.json"
        payload = {
            "model": model,
            "prompt": prompt,
            "response_text": result.response_text,
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
        }
        with self._write_lock:
            if path.exists():
                return
            tmp = path.with_name(f"{path.name}.{threading.get_ident()}.part")
            try:
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
            except PermissionError:
                tmp.unlink(missing_ok=True)
                if not path.exists():
                    raise


class GeminiAnswerer:
    """Thin injectable wrapper around the Google GenAI text-generation API."""

    def __init__(self, model: str, temperature: float = 0.0, client=None,
                 requests_per_minute: float = 1000,
                 cache_dir: str | Path = "data/cache/answers",
                 use_cache: bool = True):
        load_dotenv()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - needs optional SDK
            raise RuntimeError("install google-genai to use Gemini answer generation") from exc
        self._types = types
        self.client = client or genai.Client()
        self.model = model
        self.temperature = temperature
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.next_at = 0.0
        self._limiter_lock = threading.Lock()
        self.cache = AnswerCache(cache_dir)
        self.use_cache = use_cache
        self.cache_hits = 0
        self.calls = 0

    def _wait_for_slot(self) -> None:
        if not self.interval:
            return
        with self._limiter_lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        time.sleep(max(0.0, start - now))

    def generate(self, prompt: str) -> AnswerResult:
        if self.use_cache:
            cached = self.cache.get(self.model, prompt)
            if cached is not None:
                self.cache_hits += 1
                return cached
        self.calls += 1
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_json_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "supporting_paragraph_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_sufficiency": {
                        "type": "string",
                        "enum": ["sufficient", "insufficient"],
                    },
                },
                "required": ["answer", "supporting_paragraph_ids", "evidence_sufficiency"],
                "additionalProperties": False,
            },
        )
        self._wait_for_slot()
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        if not getattr(response, "text", None):
            raise RuntimeError("Gemini returned an empty answer")
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        result = AnswerResult(
            response_text=response.text,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            model=self.model,
        )
        if self.use_cache:
            self.cache.put(self.model, prompt, result)
        return result


class CallableAnswerer:
    """Adapter for a test function or another generation service."""
    def __init__(self, fn: Callable[[str], str], model: str = "injected"):
        self.fn = fn
        self.model = model

    def generate(self, prompt: str) -> AnswerResult:
        return AnswerResult(self.fn(prompt), model=self.model)
