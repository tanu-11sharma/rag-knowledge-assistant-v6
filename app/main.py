"""
FastAPI app exposing the multi-hop RAG pipeline as a small HTTP API.

Run:   uvicorn app.main:app --reload
Try:   curl -s -X POST localhost:8000/ask -H "content-type: application/json" \
         -d '{"question": "What review does my code need, and who gives final deployment approval?"}' \
         | python3 -m json.tool
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import MultiHopRagPipeline

app = FastAPI(
    title="Multi-Hop RAG Knowledge Assistant",
    description=(
        "A small retrieval-augmented Q&A agent that decomposes compound "
        "questions into sub-questions, retrieves evidence for each "
        "independently, and synthesizes a cited answer that can span "
        "multiple documents. Demo only — synthetic engineering policy data."
    ),
    version="0.1.0",
)

pipeline = MultiHopRagPipeline()


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    doc_id: str
    title: str


class HopOut(BaseModel):
    sub_question: str
    answer: str
    citation: Optional[CitationOut]
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    hops: List[HopOut]
    citations: List[CitationOut]
    confidence: float


@app.get("/")
def root():
    return {
        "service": "rag-knowledge-assistant-v6 (multi-hop)",
        "status": "ok",
        "docs_loaded": len(pipeline.documents),
    }


@app.get("/documents")
def list_documents():
    return [{"id": d.id, "title": d.title} for d in pipeline.documents]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = pipeline.ask(req.question)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        hops=[
            HopOut(
                sub_question=h.sub_question,
                answer=h.answer,
                citation=CitationOut(doc_id=h.citation.doc_id, title=h.citation.title)
                if h.citation
                else None,
                score=h.score,
            )
            for h in result.hops
        ],
        citations=[CitationOut(doc_id=c.doc_id, title=c.title) for c in result.citations],
        confidence=result.confidence,
    )
