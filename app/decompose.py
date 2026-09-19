"""
Rule-based multi-hop question decomposition.

Real multi-hop RAG systems often use an LLM to split a compound question
into independent sub-questions before retrieving evidence for each. This
module demonstrates the same idea with a deterministic, offline rule: split
on a coordinating conjunction ("and") only when it is plausibly joining two
separate questions (the text after "and" starts with a question word or
auxiliary verb). This is intentionally simple and inspectable rather than
a general-purpose parser.
"""
from __future__ import annotations

import re
from typing import List

_SPLIT_RE = re.compile(
    r",?\s+and\s+(?=who|what|when|where|how|which|does|do|is|are|can|should|will)",
    re.IGNORECASE,
)


def decompose(question: str) -> List[str]:
    """Split a compound question into 1+ independent sub-questions."""
    parts = _SPLIT_RE.split(question)
    parts = [p.strip().rstrip("?").strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [question.strip().rstrip("?").strip()]
