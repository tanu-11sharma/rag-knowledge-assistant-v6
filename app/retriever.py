"""
Lightweight TF-IDF retriever (from scratch, no external embedding API).

Kept deliberately simple so the whole multi-hop pipeline runs offline with
zero API keys: swap this module for a real embedding model + vector DB
without touching decomposition or synthesis.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small stopword list so short questions ("who ...", "what ... need") aren't
# dominated by function words that appear in every document.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "to", "of", "in", "on", "at", "for",
    "and", "or", "but", "if", "so", "my", "me", "i", "it", "its", "this",
    "that", "these", "those", "what", "who", "when", "where", "how",
    "which", "can", "will", "shall", "should", "would", "must",
    "before", "after", "with", "from",
}


def _stem(word: str) -> str:
    """Very light suffix stripping so e.g. 'deployments'/'deployment' and
    'reviewers'/'review' land on the same token. Not a real stemmer —
    just enough to reduce surface-form mismatches in a tiny TF-IDF index."""
    for suffix in ("ations", "ation", "ers", "ing", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> List[str]:
    tokens = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


@dataclass
class Document:
    id: str
    title: str
    text: str


@dataclass
class ScoredChunk:
    doc_id: str
    title: str
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._doc_tokens = [tokenize(d.text) for d in documents]
        self._df = Counter()
        for tokens in self._doc_tokens:
            for term in set(tokens):
                self._df[term] += 1
        self._n_docs = len(documents)
        self._doc_vectors = [self._vectorize(tokens) for tokens in self._doc_tokens]

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((1 + self._n_docs) / (1 + df)) + 1

    def _vectorize(self, tokens: List[str]) -> Counter:
        tf = Counter(tokens)
        vec = Counter()
        for term, count in tf.items():
            vec[term] = count * self._idf(term)
        return vec

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> List[ScoredChunk]:
        q_vec = self._vectorize(tokenize(query))
        scored = []
        for doc, doc_vec in zip(self.documents, self._doc_vectors):
            score = self._cosine(q_vec, doc_vec)
            scored.append(ScoredChunk(doc.id, doc.title, doc.text, score))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


def load_documents(path: Path) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Document(id=d["id"], title=d["title"], text=d["text"]) for d in raw]
