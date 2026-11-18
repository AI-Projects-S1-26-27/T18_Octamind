"""Answer prompt and response parsing for Role 6."""
from __future__ import annotations

import json
import re
from typing import Any


def build_qa_prompt(question: str, evidence_context: str) -> str:
    """Build the shared, evidence-grounded prompt used for every retrieval system."""
    return f"""You are answering a multi-hop question using only the evidence below.

QUESTION:
{question}

EVIDENCE:
{evidence_context if evidence_context.strip() else '[No evidence was retrieved.]'}

INSTRUCTIONS:
- Answer the question directly and concisely.
- Use only facts supported by the supplied evidence.
- If the evidence is insufficient or contradictory, say so in the answer.
- Do not invent facts or citations.
- Return valid JSON with exactly these fields:
  {{"answer": "...", "supporting_paragraph_ids": ["..."], "evidence_sufficiency": "sufficient" or "insufficient"}}
- List only paragraph IDs that appear in the EVIDENCE section.
"""


def parse_answer_response(response: str) -> dict[str, Any]:
    """Parse the expected JSON answer; tolerate fences and extra response text."""
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        payload = None
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                candidate, _ = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
            if isinstance(candidate, list):
                payload = next((item for item in candidate if isinstance(item, dict)), None)
                if payload is not None:
                    break
        if payload is None:
            raise ValueError(f"answer model did not return valid JSON: {exc}") from exc
    if isinstance(payload, str):
        payload = {"answer": payload}
    if isinstance(payload, list):
        payload = next((item for item in payload if isinstance(item, dict)), None)
    if not isinstance(payload, dict):
        raise ValueError("answer model response must be a JSON object")
    answer = payload.get("answer")
    pids = payload.get("supporting_paragraph_ids", [])
    sufficiency = payload.get("evidence_sufficiency", "insufficient")
    if not isinstance(answer, str):
        raise ValueError("answer response field 'answer' must be a string")
    if not isinstance(pids, list) or any(not isinstance(pid, str) for pid in pids):
        raise ValueError("supporting_paragraph_ids must be a list of strings")
    if sufficiency not in {"sufficient", "insufficient"}:
        sufficiency = "insufficient"
    return {
        "answer": answer.strip(),
        "supporting_paragraph_ids": list(dict.fromkeys(pids)),
        "evidence_sufficiency": sufficiency,
    }
