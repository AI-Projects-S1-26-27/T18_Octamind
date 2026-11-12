"""Extraction quality against the datasets' gold chains (Role 2).

Gold entity names per question: the head/tail of each gold step (2Wiki
``evidences``; MuSiQue sub-question answers) plus the final answer. A gold name is
*found* if an extracted entity mentioned in one of that question's paragraphs has
the same normalized name as a mention in one of that question's paragraphs (strict), or contains it / is contained in it at word
boundaries (lenient: "1947" vs "June 1947", "Oxford" vs "University of Oxford").

If a gold entity is never extracted, traversal cannot reach it, so strict recall is
an upper bound on path recall for the automatic graph (useful for RQ4).
"""
from __future__ import annotations

import re
from collections import defaultdict

from .entities import normalize_name, text_sha1

_PLACEHOLDER = re.compile(r"^#\d+$")   # MuSiQue "#1" references to an earlier step


def gold_names(record) -> list:
    names = []
    for step in record.gold_chain:
        for value in (step.head, step.tail):
            if value and not _PLACEHOLDER.match(value.strip()):
                names.append(value)
    # yes/no answers (HotpotQA/2Wiki comparison questions) are not entities
    if record.answer and normalize_name(record.answer) not in ("yes", "no"):
        names.append(record.answer)
    return list(dict.fromkeys(n for n in (normalize_name(x) for x in names) if n))


def _contains(a: str, b: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(a)}(?!\w)", b) is not None


def gold_entity_recall(records: list, mentions: list) -> dict:
    """Strict and lenient recall of gold entity names, overall and per hop level.

    Linking merges exactly the mentions with equal normalized names, so the names
    available in a paragraph are its mentions' normalized texts. Paragraphs are
    matched on (id, text hash), because one MuSiQue id can carry several texts.
    """
    by_paragraph = defaultdict(set)    # (paragraph id, text hash) -> normalized names
    for m in mentions:
        if m.norm:
            by_paragraph[(m.paragraph_id, m.paragraph_sha1)].add(m.norm)

    totals = defaultdict(lambda: [0, 0, 0])   # key -> [gold, strict hits, lenient hits]
    for record in records:
        available = set().union(set(), *(by_paragraph[(p.id, text_sha1(p.text))] for p in record.paragraphs))
        for gold in gold_names(record):
            strict = gold in available
            lenient = strict or any(_contains(gold, n) or _contains(n, gold) for n in available)
            for key in ("all", f"hop_{record.hop_level}"):
                totals[key][0] += 1
                totals[key][1] += strict
                totals[key][2] += lenient

    def summary(gold, strict, lenient):
        return {
            "gold_names": gold,
            "strict_recall": round(strict / gold, 4) if gold else None,
            "lenient_recall": round(lenient / gold, 4) if gold else None,
        }

    return {key: summary(*vals) for key, vals in sorted(totals.items())}
