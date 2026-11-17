"""Role 6: evidence serialization, prompt construction and answer generation."""
from .serializer import ContextSerializer, CorpusIndex
from .prompt import build_qa_prompt, parse_answer_response

__all__ = ["ContextSerializer", "CorpusIndex", "build_qa_prompt", "parse_answer_response"]
