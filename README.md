# RAG Knowledge Assistant v6 — Multi-Hop Retrieval

A small, fully self-contained RAG agent that answers **compound questions requiring evidence from more than one document** — not just single-document lookup — over a synthetic engineering onboarding & security-policy knowledge base.

## What it does

Most simple RAG demos answer one question against one best-matching chunk. This one handles questions like:

> "What review does my code need, and who gives final deployment approval?"

by:

1. **Decomposing** the question into independent sub-questions (`app/decompose.py`) — here, splitting on a coordinating conjunction that joins two separate asks.
2. **Retrieving** evidence for each sub-question independently using a from-scratch TF-IDF + cosine-similarity index (`app/retriever.py`), so each hop can land on a different document.
3. **Synthesizing** a single answer that stitches together each hop's extracted sentence, tagged with its own citation — so a two-part question can correctly cite two different source documents.

The API returns both the final synthesized answer and the full per-hop reasoning trace (`hops`), so you can see exactly which sub-question retrieved which document.

## Why this is relevant

Single-hop "retrieve top-1 chunk, answer from it" is the RAG pattern everyone builds first — but real user questions are often compound, and naive single-hop retrieval either answers only half the question or blends unrelated documents into one noisy retrieval. Multi-hop / question-decomposition RAG (used in production agentic RAG and research-assistant systems) addresses this by treating each sub-question as its own retrieval problem and combining the results explicitly and traceably. This project demonstrates that pattern end to end — decomposition, independent per-hop retrieval, cross-document citation — with zero external API keys, so it runs anywhere in seconds.

## Project structure

```
rag-knowledge-assistant-v6/
├── app/
│   ├── main.py        # FastAPI app (HTTP API)
│   ├── rag.py          # Multi-hop pipeline: decompose -> retrieve per hop -> synthesize
│   ├── decompose.py    # Rule-based compound-question splitter
│   └── retriever.py    # From-scratch TF-IDF retriever (with light stemming)
├── data/
│   └── knowledge_base.json  # 8 synthetic engineering/security policy documents
├── tests/
│   └── test_rag.py     # Decomposition, single-hop, multi-hop, and API tests
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

```bash
curl -s -X POST localhost:8000/ask \
  -H "content-type: application/json" \
  -d '{"question": "What review does my code need, and who gives final deployment approval?"}' \
  | python3 -m json.tool
```

This returns an `answer` that cites both `Code Review Policy` and `Production Deployment Approval`, plus a `hops` array showing each sub-question's own retrieval result and score. A single-hop question (e.g. "How often do VPN credentials rotate?") works the same way but produces just one hop.

Browse the knowledge base: `curl localhost:8000/documents`.

## Test

```bash
pytest -v
```

8 tests cover question decomposition, single-hop retrieval, multi-hop cross-document citation, and the HTTP API — all offline against the bundled synthetic knowledge base, no network or API key needed.

## Disclaimer

This is a demo/portfolio project. All documents in `data/knowledge_base.json` are synthetic policy text written for this demo — they don't describe any real company's actual policies.
