"""
Multi-hop RAG pipeline.

Unlike a single-hop RAG demo, this pipeline handles compound questions that
require evidence from more than one document:

  1. `decompose()` splits the question into independent sub-questions.
  2. Each sub-question is retrieved and answered independently against the
     knowledge base (its own "hop").
  3. The per-hop answers are synthesized into one final answer, with a
     citation list that spans every document actually used.

Generation stays extractive (no external LLM call needed): each hop's
answer is the most query-relevant sentence from its retrieved chunk. Swap
`_best_sentence` for an LLM call per hop, or a final synthesis call over all
hop results, to get free-text generation without changing retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from app.decompose import decompose
from app.retriever import ScoredChunk, TfidfRetriever, load_documents, tokenize

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
MIN_SCORE = 0.12


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def _best_sentence(text: str, query_terms: set) -> str:
    sentences = _split_sentences(text)
    return max(
        sentences,
        key=lambda s: len(set(tokenize(s)) & query_terms),
        default=text,
    )


@dataclass
class Citation:
    doc_id: str
    title: str


@dataclass
class Hop:
    sub_question: str
    answer: str
    citation: Citation | None
    score: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    hops: List[Hop] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0


class MultiHopRagPipeline:
    def __init__(self, data_path: Path = DATA_PATH):
        self.documents = load_documents(data_path)
        self.retriever = TfidfRetriever(self.documents)

    def _answer_hop(self, sub_question: str) -> Hop:
        results = self.retriever.search(sub_question, top_k=1)
        if not results or results[0].score < MIN_SCORE:
            return Hop(sub_question=sub_question, answer="", citation=None, score=0.0)
        top: ScoredChunk = results[0]
        query_terms = set(tokenize(sub_question))
        sentence = _best_sentence(top.text, query_terms)
        return Hop(
            sub_question=sub_question,
            answer=sentence,
            citation=Citation(doc_id=top.doc_id, title=top.title),
            score=top.score,
        )

    def ask(self, question: str) -> RagAnswer:
        sub_questions = decompose(question)
        hops = [self._answer_hop(sq) for sq in sub_questions]

        answered_hops = [h for h in hops if h.citation is not None]
        if not answered_hops:
            return RagAnswer(
                question=question,
                answer=(
                    "I couldn't find anything in the knowledge base relevant "
                    "to that question. Try asking about VPN access, laptop "
                    "provisioning, code review, deployment approval, on-call, "
                    "incident response, data classification, or access requests."
                ),
                hops=hops,
                citations=[],
                confidence=0.0,
            )

        citations: List[Citation] = []
        seen = set()
        answer_parts = []
        for hop in hops:
            if hop.citation is None:
                answer_parts.append(
                    f"(no confident answer found for: \"{hop.sub_question}\")"
                )
                continue
            answer_parts.append(f"{hop.answer} [{hop.citation.title}]")
            if hop.citation.doc_id not in seen:
                citations.append(hop.citation)
                seen.add(hop.citation.doc_id)

        confidence = round(sum(h.score for h in answered_hops) / len(hops), 4)
        return RagAnswer(
            question=question,
            answer=" ".join(answer_parts),
            hops=hops,
            citations=citations,
            confidence=confidence,
        )
