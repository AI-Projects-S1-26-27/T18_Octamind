"""Optional Gemini answer generation for Role 6."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from src.graph.embeddings import load_dotenv


@dataclass
class AnswerResult:
    response_text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""


class GeminiAnswerer:
    """Thin injectable wrapper around the Google GenAI text-generation API."""

    def __init__(self, model: str, temperature: float = 0.0, client=None,
                 requests_per_minute: float = 1000):
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

    def _wait_for_slot(self) -> None:
        if not self.interval:
            return
        with self._limiter_lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        time.sleep(max(0.0, start - now))

    def generate(self, prompt: str) -> AnswerResult:
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
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
        return AnswerResult(
            response_text=response.text,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            model=self.model,
        )


class CallableAnswerer:
    """Adapter for a test function or another generation service."""
    def __init__(self, fn: Callable[[str], str], model: str = "injected"):
        self.fn = fn
        self.model = model

    def generate(self, prompt: str) -> AnswerResult:
        return AnswerResult(self.fn(prompt), model=self.model)
