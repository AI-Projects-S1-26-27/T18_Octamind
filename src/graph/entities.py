"""Entity and mention types, name normalization and sentence splitting (Role 2).

A :class:`Mention` is one occurrence of an entity or concept in one sentence of one
source paragraph. Linking (``src/graph/linking.py``) groups mentions into canonical
:class:`Entity` objects; their ids are the ``head``/``tail`` ids that Role 3 puts on
edges (see docs/INTERFACE.md).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

# Coarse types, enough for linking and for typed relations in Role 3. The LLM
# extractor is asked to use exactly these labels.
ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "WORK",          # films, books, songs, albums, ...
    "EVENT",
    "DATE",
    "NUMBER",
    "CONCEPT",       # anything else worth a node: awards, languages, occupations, ...
)
DEFAULT_TYPE = "CONCEPT"

_LEADING_ARTICLES = ("the ", "a ", "an ")
_EDGE_PUNCT = " \t\n\"'`.,;:!?()[]{}<>"


@dataclass(frozen=True)
class Mention:
    """One entity/concept occurrence. ``start``/``end`` are character offsets
    into the paragraph text (not the sentence)."""
    text: str
    type: str
    paragraph_id: str
    sentence_index: int
    start: int
    end: int
    source: str = "rule"          # which extractor produced it: "rule" or "llm"
    confidence: float = 1.0
    paragraph_sha1: str = ""      # text_sha1 of the paragraph the offsets refer to

    @property
    def norm(self) -> str:
        return normalize_name(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.type,
            "paragraph_id": self.paragraph_id,
            "sentence_index": self.sentence_index,
            "start": self.start,
            "end": self.end,
            "source": self.source,
            "confidence": self.confidence,
            "paragraph_sha1": self.paragraph_sha1,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Mention":
        return cls(**raw)


@dataclass
class Entity:
    """A canonical node of the Semantic Knowledge Network."""
    id: str
    name: str                                    # most frequent surface form
    type: str
    aliases: tuple = ()                          # other surface forms seen
    paragraph_ids: tuple = ()                    # where it is mentioned
    mention_count: int = 0
    embedding: tuple = field(default=(), repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "aliases": list(self.aliases),
            "paragraph_ids": list(self.paragraph_ids),
            "mention_count": self.mention_count,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Entity":
        return cls(
            id=raw["id"],
            name=raw["name"],
            type=raw.get("type", DEFAULT_TYPE),
            aliases=tuple(raw.get("aliases") or ()),
            paragraph_ids=tuple(raw.get("paragraph_ids") or ()),
            mention_count=int(raw.get("mention_count", 0)),
        )


def normalize_name(text: str) -> str:
    """Key used to decide that two surface forms name the same entity.

    Unicode-folds, strips accents, lowercases, drops a leading article and edge
    punctuation, and collapses whitespace: "The  Beatles." and "beatles" match,
    "Pelé" and "Pele" match.
    """
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().strip(_EDGE_PUNCT)
    s = re.sub(r"\s+", " ", s)
    for article in _LEADING_ARTICLES:
        if s.startswith(article) and len(s) > len(article):
            s = s[len(article):]
            break
    return s.strip(_EDGE_PUNCT)


def text_sha1(text: str) -> str:
    """Short content hash that identifies one paragraph text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def entity_id(norm: str) -> str:
    """Stable, readable id for a normalized name, e.g. ``ent:new_york_city``."""
    slug = re.sub(r"[^0-9a-z]+", "_", norm).strip("_")
    return f"ent:{slug or 'unnamed'}"


# A sentence ends at . ! or ? followed by whitespace and an upper-case letter,
# digit or quote. Initials ("J. R. R. Tolkien") and common abbreviations are
# protected below; the rest is good enough for Wikipedia-style paragraphs.
_SENT_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'”)\]]))\s+(?=[\"'“(\[]?[A-Z0-9])")
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "no.",
    "inc.", "ltd.", "co.", "corp.", "bros.", "u.s.", "u.k.", "e.g.", "i.e.", "c.",
    "gen.", "col.", "lt.", "sgt.", "capt.", "gov.", "sen.", "rep.", "rev.", "ft.", "mt.",
}


def split_sentences(text: str) -> list:
    """Split ``text`` into ``(start, end)`` character spans, one per sentence."""
    spans = []
    start = 0
    for m in _SENT_BOUNDARY.finditer(text):
        last_word = text[start:m.start()].rsplit(None, 1)[-1].lower() if text[start:m.start()].strip() else ""
        # do not split after an abbreviation or a single-letter initial ("J.")
        if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word):
            continue
        spans.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans
