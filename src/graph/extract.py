"""Entity / concept extraction from source paragraphs (Role 2).

Two extractors share one interface, ``extract(paragraph) -> list[Mention]``:

- :class:`RuleExtractor`: dependency-free baseline (paragraph title, capitalized
  spans, dates). Cheap, deterministic, untyped apart from DATE. Used for tests,
  for a quick end-to-end graph, and as the lower bound in the extraction ablation.
- :class:`LLMExtractor`: the main extractor. Prompts the fixed LLM (Gemini Flash,
  temperature 0) for typed mentions per sentence, several paragraphs per request.
  Results are cached on disk per paragraph, so re-runs cost nothing, an interrupted
  run resumes where it stopped, and results are reproducible.

The LLM is passed in as a plain ``generate(prompt) -> str`` callable, so tests use
a fake and nothing here depends on an API key.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from data.loaders.base import Paragraph

from .entities import DEFAULT_TYPE, ENTITY_TYPES, Mention, normalize_name, split_sentences

# --------------------------------------------------------------------------- #
# Rule-based baseline
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[^\W_][\w'’\-.&]*", re.UNICODE)
# lower-case words allowed inside a name: "University of Oxford", "Leonardo da Vinci".
# "the" only after "of" ("University of the Arts"); "and"/"for" are left out because
# they join two separate names far more often than they sit inside one.
_CONNECTORS = {"of", "de", "la", "le", "von", "van", "der", "da", "di", "del", "du"}
_PRONOUNS = {"i", "he", "she", "it", "we", "they", "you", "his", "her", "its", "their"}
# capitalized only because they start a sentence
_SENTENCE_START_WORDS = {
    "the", "a", "an", "he", "she", "it", "they", "his", "her", "its", "their", "in",
    "on", "at", "after", "before", "during", "this", "that", "these", "those", "there",
    "as", "when", "while", "however", "although", "following", "since", "from", "with",
}
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# "3 August 1769", "August 3, 1769", "August 1769", "May 4", "1990s"; a bare month
# name ("May") is not a date
_DATE = re.compile(
    rf"\b(?:\d{{1,2}} (?:{_MONTHS})(?:,? \d{{4}})?"
    rf"|(?:{_MONTHS}) \d{{1,2}}(?:,? \d{{4}})?"
    rf"|(?:{_MONTHS}),? \d{{4}}"
    rf"|(?:1[0-9]|20)\d{{2}}s?)\b"
)


class RuleExtractor:
    name = "rule"

    def extract(self, paragraph: Paragraph) -> list:
        mentions = []
        if paragraph.title:
            # The title names what the paragraph is about, even when the text only
            # says "He" or "The film". sentence_index/start/end = -1: not a text span.
            mentions.append(Mention(paragraph.title, DEFAULT_TYPE, paragraph.id, -1, -1, -1,
                                    source=self.name, confidence=1.0))
        for index, (s_start, s_end) in enumerate(split_sentences(paragraph.text)):
            sentence = paragraph.text[s_start:s_end]
            dates = [(m.start(), m.end()) for m in _DATE.finditer(sentence)]
            for start, end in dates:
                mentions.append(Mention(sentence[start:end], "DATE", paragraph.id, index,
                                        s_start + start, s_start + end, self.name, 0.9))
            for start, end in self._capitalized_spans(sentence, dates):
                mentions.append(Mention(sentence[start:end], DEFAULT_TYPE, paragraph.id, index,
                                        s_start + start, s_start + end, self.name, 0.6))
        return mentions

    @staticmethod
    def _capitalized_spans(sentence: str, dates: list = ()) -> list:
        """Runs of capitalized tokens. A date, or anything but whitespace between two
        tokens ("Paris, France", "Viola (of"), ends the current run."""
        tokens = list(_TOKEN.finditer(sentence))
        spans, run = [], []

        def close():
            # connectors may not end a name
            while run and run[-1].group().lower() in _CONNECTORS | {"the"}:
                run.pop()
            # "In the United States" at a sentence (or quotation) start -> "United States"
            if run and run[0].group().lower() in _SENTENCE_START_WORDS and (
                    run[0].start() == tokens[0].start()
                    or sentence[run[0].start() - 1:run[0].start()] in ("\"", "'", "(", "“")):
                run.pop(0)
                while run and run[0].group().lower() in _CONNECTORS | {"the"}:
                    run.pop(0)
            if len(run) == 1 and (re.fullmatch(_MONTHS, run[0].group().rstrip(".,"))
                                  or run[0].group().lower() in _PRONOUNS):
                run.clear()   # a bare month name or pronoun is not an entity
            if run:
                start, end = run[0].start(), run[-1].end()
                text = sentence[start:end]
                # trailing punctuation, except an initial like "J."
                while text and text[-1] in ".,'’" and not re.search(r"\b[A-Z]\.$", text):
                    text, end = text[:-1], end - 1
                if text:
                    spans.append((start, end))
            run.clear()

        for tok in tokens:
            word = tok.group()
            if run and sentence[run[-1].end():tok.start()].strip():
                close()
            if any(d_start <= tok.start() < d_end for d_start, d_end in dates):
                close()       # dates are extracted separately; never part of a name
                continue
            if word[0].isupper():
                run.append(tok)
            elif run and (word.lower() in _CONNECTORS
                          or (word.lower() == "the" and run[-1].group().lower() == "of")):
                run.append(tok)
            elif (run and word.rstrip(".").isdigit() and len(word.rstrip(".")) <= 2 and not run[-1].group()[0].isdigit()
                  and re.match(r"(?:$|[^\w\s]|\s+[A-Z])", sentence[tok.end():])):
                run.append(tok)   # "PlayStation 2", "Apollo 11." but not "Venice 22 times"
            else:
                close()
        close()
        return spans


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #
PROMPT_VERSION = 2  # bump whenever PROMPT changes, so old cache entries are not reused

PROMPT = """You extract entities and concepts from Wikipedia paragraphs, to build a knowledge graph.

{paragraphs}

For every paragraph, list every named entity, date, quantity and concept that could be the subject
or object of a fact (for example people, places, organizations, works, events, awards, occupations,
languages, genres).

Return only a JSON list. Each item is an object:
  {{"paragraph": <paragraph number>, "sentence": <sentence number>,
    "text": <exact substring of that sentence>, "type": <one of {types}>}}
Rules: copy "text" exactly as written; use the shortest span that names the thing; do not resolve
pronouns; do not invent anything that is not in the sentence."""


def paragraph_block(paragraph: Paragraph, spans: list, number: int = 0) -> str:
    numbered = "\n".join(f"[{i}] {paragraph.text[s:e]}" for i, (s, e) in enumerate(spans))
    return f"Paragraph {number} (title: {paragraph.title})\n{numbered}"


def build_prompt(blocks: list) -> str:
    """One prompt for several paragraph blocks (batching cuts the number of API calls)."""
    return PROMPT.format(paragraphs="\n\n".join(blocks), types=", ".join(ENTITY_TYPES))


class ResponseCache:
    """Per-paragraph cache: one JSON file per paragraph, keyed by
    sha256(model, prompt version, paragraph block), holding that paragraph's items.
    Keying by paragraph, not by request, means results are reused whatever batch
    a paragraph was sent in."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def key(self, model: str, block: str) -> str:
        return hashlib.sha256(f"{model}\x00{PROMPT_VERSION}\x00{block}".encode("utf-8")).hexdigest()

    def get(self, model: str, block: str) -> Optional[list]:
        path = self.dir / f"{self.key(model, block)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["items"]

    def put(self, model: str, block: str, items: list) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.key(model, block)}.json"
        payload = {"model": model, "prompt_version": PROMPT_VERSION, "items": items}
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)   # atomic: an interrupted run never leaves a half-written entry


def _load_json(response: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (response or "").strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_response(response: str) -> list:
    """Parse the LLM's JSON list, tolerating ```json fences and a wrapping object."""
    data = _load_json(response)
    if isinstance(data, dict):
        data = data.get("entities") or data.get("items") or []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def is_valid_response(response: str) -> bool:
    """True if the response is a JSON list (or an object wrapping one)."""
    return isinstance(_load_json(response), (list, dict))


class LLMExtractor:
    """Thread-safe: the runner calls ``extract_many`` from several worker threads."""
    name = "llm"

    def __init__(self, generate: Callable[[str], str], model: str, cache_dir: str | Path,
                 confidence: float = 0.9):
        self.generate = generate
        self.model = model
        self.cache = ResponseCache(cache_dir)
        self.confidence = confidence
        self.calls = 0            # API calls made
        self.invalid = 0          # responses that were not JSON (nothing cached; a re-run retries)
        self.missing = 0          # paragraphs a batched answer skipped (not cached; a re-run retries)
        self.dropped = 0          # items whose text was not found in the sentence (hallucinations)
        self._lock = threading.Lock()

    def _count(self, field: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def extract(self, paragraph: Paragraph) -> list:
        return self.extract_many([paragraph])[0]

    def extract_many(self, paragraphs: list) -> list:
        """Mentions for each paragraph; uncached paragraphs go to the LLM in one request."""
        prepared = [(p, split_sentences(p.text)) for p in paragraphs]
        items = [None] * len(prepared)
        todo = []
        for i, (p, spans) in enumerate(prepared):
            if not spans:
                items[i] = []
                continue
            cached = self.cache.get(self.model, paragraph_block(p, spans))
            if cached is None:
                todo.append(i)
            else:
                items[i] = cached

        if todo:
            blocks = [paragraph_block(prepared[i][0], prepared[i][1], k) for k, i in enumerate(todo)]
            response = self.generate(build_prompt(blocks))
            self._count("calls")
            valid = is_valid_response(response)
            grouped, dropped = {}, 0
            if valid:
                for item in parse_response(response):
                    k = item.pop("paragraph", 0 if len(todo) == 1 else None)
                    try:
                        k = int(k)
                    except (TypeError, ValueError):
                        dropped += 1
                        continue
                    if 0 <= k < len(todo):
                        grouped.setdefault(k, []).append(item)
                    else:
                        dropped += 1
            else:
                self._count("invalid")
            for k, i in enumerate(todo):
                found = grouped.get(k, [])
                items[i] = found
                p, spans = prepared[i]
                if valid and (found or len(todo) == 1):
                    self.cache.put(self.model, paragraph_block(p, spans), found)
                elif valid:
                    self._count("missing")   # empty inside a batch: likely skipped, retry later
            if dropped:
                self._count("dropped", dropped)

        return [self._to_mentions(p, spans, items[i]) for i, (p, spans) in enumerate(prepared)]

    def _to_mentions(self, paragraph: Paragraph, spans: list, items: list) -> list:
        mentions, seen, dropped = [], set(), 0
        title_type = DEFAULT_TYPE
        for item in items:
            try:
                index = int(item.get("sentence"))
            except (TypeError, ValueError):
                dropped += 1
                continue
            text = str(item.get("text", "")).strip()
            if not text or not 0 <= index < len(spans):
                dropped += 1
                continue
            s_start, s_end = spans[index]
            sentence = paragraph.text[s_start:s_end]
            # the model gives no offsets: take the first occurrence not already used, so a
            # name repeated in one sentence yields one mention per occurrence
            haystack, needle = sentence, text
            if haystack.find(needle) < 0:
                haystack, needle = sentence.lower(), text.lower()
            pos = haystack.find(needle)
            if pos < 0:
                dropped += 1
                continue
            while pos >= 0 and (s_start + pos, s_start + pos + len(text)) in seen:
                pos = haystack.find(needle, pos + 1)
            if pos < 0:
                continue      # every occurrence already has a mention (duplicate item)
            start, end = s_start + pos, s_start + pos + len(text)
            seen.add((start, end))
            etype = str(item.get("type", "")).upper()
            etype = etype if etype in ENTITY_TYPES else DEFAULT_TYPE
            if etype != DEFAULT_TYPE and normalize_name(text) == normalize_name(paragraph.title):
                title_type = etype
            mentions.append(Mention(paragraph.text[start:end], etype, paragraph.id, index,
                                    start, end, self.name, self.confidence))
        if dropped:
            self._count("dropped", dropped)
        if paragraph.title:
            # same convention as the rule extractor: the title stands for the paragraph's
            # subject ("He was born ..."); typed when the model typed it in the text
            mentions.insert(0, Mention(paragraph.title, title_type, paragraph.id, -1, -1, -1,
                                       self.name, 1.0))
        return mentions


def load_dotenv(path: str | Path = ".env") -> None:
    """Copy ``KEY=VALUE`` lines from the git-ignored .env into os.environ (existing
    variables win). Enough for API keys; avoids a python-dotenv dependency."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("\"'"))


class QuotaExhausted(RuntimeError):
    """The account's daily request quota is used up; retrying today will not help."""


class RateLimiter:
    """At most ``per_minute`` call starts per minute, shared by all worker threads."""

    def __init__(self, per_minute: Optional[float]):
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        time.sleep(max(0.0, start - now))


def is_daily_quota_error(message: str) -> bool:
    return bool(re.search(r"per ?day|PerDay", message, flags=re.I))


def gemini_generate(model: str, temperature: float = 0.0, *, thinking_level: Optional[str] = None,
                    requests_per_minute: Optional[float] = None, max_attempts: int = 8,
                    timeout_s: float = 180.0) -> Callable[[str], str]:
    """``generate`` callable backed by Gemini. Needs ``pip install google-genai`` and
    GEMINI_API_KEY in the environment or in the git-ignored .env.

    Per-minute rate limits (429) and server errors (5xx) are retried with exponential
    backoff. A *daily* quota error raises :class:`QuotaExhausted` at once, and every
    later call fails fast, so a run stops cleanly instead of retrying for hours.
    """
    load_dotenv()
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as exc:
        raise RuntimeError("the LLM extractor needs `pip install google-genai`") from exc
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("set GEMINI_API_KEY (in .env, never committed) to use the LLM extractor")
    client = genai.Client(http_options=types.HttpOptions(
        timeout=int(timeout_s * 1000),
        retry_options=types.HttpRetryOptions(attempts=1),   # retries are handled below
    ))
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None,
    )
    limiter = RateLimiter(requests_per_minute)
    exhausted = threading.Event()

    def generate(prompt: str) -> str:
        for attempt in range(max_attempts):
            if exhausted.is_set():
                raise QuotaExhausted("daily quota already exhausted in this run")
            limiter.wait()
            try:
                return client.models.generate_content(model=model, contents=prompt, config=config).text or ""
            except errors.APIError as exc:
                if exc.code == 429 and is_daily_quota_error(str(exc)):
                    exhausted.set()
                    raise QuotaExhausted(str(exc)[:400]) from exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt == max_attempts - 1:
                    raise
            except Exception as exc:  # noqa: BLE001 - network errors and timeouts
                transient = re.search(r"timeout|connect|reset|closed", type(exc).__name__ + str(exc), re.I)
                if not transient or attempt == max_attempts - 1:
                    raise
            time.sleep(min(60.0, 2.0 * 2 ** attempt) + random.random())
        raise RuntimeError("unreachable")

    generate.client = client   # keep the client alive as long as the callable
    return generate


def build_extractor(cfg: dict):
    """Build the extractor named by ``cfg["extractor"]`` (configs/extraction.yaml)."""
    kind = cfg.get("extractor", "rule")
    if kind == "rule":
        return RuleExtractor()
    if kind == "llm":
        llm = cfg.get("llm") or {}
        model = llm["model"]
        generate = gemini_generate(
            model, float(llm.get("temperature", 0.0)),
            thinking_level=llm.get("thinking_level"),
            requests_per_minute=llm.get("requests_per_minute"),
            max_attempts=int(llm.get("max_attempts", 8)),
        )
        return LLMExtractor(generate, model=model, cache_dir=llm.get("cache_dir", "data/cache/extraction"))
    raise ValueError(f"unknown extractor {kind!r}; choose 'rule' or 'llm'")
