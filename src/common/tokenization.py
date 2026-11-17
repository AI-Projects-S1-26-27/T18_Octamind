"""Common token counting for retrieval budgets and evaluation.

The regex tokenizer is deterministic and API-free. It is a project-wide comparison
unit, not Gemini's native tokenizer. When answers are generated with Gemini, the
provider-reported prompt token count is recorded separately where available.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def count_tokens(text: str) -> int:
    """Count word and punctuation tokens using the shared regex tokenizer."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(_TOKEN_RE.findall(text))


class RegexTokenCounter:
    """Callable adapter with a stable name for result manifests."""

    name = "regex_wordpunct_v1"
    is_placeholder = False

    def __call__(self, text: str) -> int:
        return count_tokens(text)


class GeminiTokenCounter:
    """Exact Gemini-model input-token counter backed by the Google GenAI SDK.

    This is intentionally opt-in because it requires credentials and performs a
    remote count request. The same instance should be shared across all systems.
    """

    name = "gemini_count_tokens"
    is_placeholder = False

    def __init__(self, model: str, client=None):
        if not model:
            raise ValueError("model must be non-empty")
        if client is None:
            from src.graph.embeddings import load_dotenv
            load_dotenv()
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - requires optional SDK
                raise RuntimeError("install google-genai to use Gemini token counting") from exc
            client = genai.Client()
        self.model = model
        self.client = client
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if text not in self._cache:
            response = self.client.models.count_tokens(model=self.model, contents=text)
            total = getattr(response, "total_tokens", None)
            if total is None:
                raise RuntimeError("Gemini count_tokens response did not include total_tokens")
            self._cache[text] = int(total)
        return self._cache[text]
