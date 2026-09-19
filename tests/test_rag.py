from fastapi.testclient import TestClient

from app.decompose import decompose
from app.main import app
from app.rag import MultiHopRagPipeline

client = TestClient(app)


def test_decompose_single_hop_question_unchanged():
    q = "What review does my code need before merging?"
    assert decompose(q) == [q.rstrip("?")]


def test_decompose_splits_compound_question():
    q = "What review does my code need, and who gives final deployment approval?"
    parts = decompose(q)
    assert len(parts) == 2
    assert "review" in parts[0].lower()
    assert "approval" in parts[1].lower()


def test_single_hop_answer_has_one_citation():
    pipeline = MultiHopRagPipeline()
    result = pipeline.ask("How often do VPN credentials rotate?")
    assert len(result.hops) == 1
    assert result.citations
    assert result.citations[0].doc_id == "vpn_access"


def test_multi_hop_question_cites_two_documents():
    pipeline = MultiHopRagPipeline()
    result = pipeline.ask(
        "What review does my code need, and who gives final deployment approval?"
    )
    assert len(result.hops) == 2
    doc_ids = {c.doc_id for c in result.citations}
    assert doc_ids == {"code_review_policy", "deployment_approval"}


def test_multi_hop_oncall_and_incident():
    pipeline = MultiHopRagPipeline()
    result = pipeline.ask(
        "Who is on call this week, and how do I escalate a severity 1 incident?"
    )
    doc_ids = {c.doc_id for c in result.citations}
    assert "oncall_rotation" in doc_ids
    assert "incident_response" in doc_ids


def test_irrelevant_question_returns_no_citations():
    pipeline = MultiHopRagPipeline()
    result = pipeline.ask("What is the airspeed velocity of an unladen swallow?")
    assert result.citations == []
    assert "couldn't find" in result.answer.lower()


def test_api_root():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["docs_loaded"] == 8


def test_api_ask_multi_hop():
    resp = client.post(
        "/ask",
        json={
            "question": "What review does my code need, and who gives final deployment approval?"
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hops"]) == 2
    assert len(body["citations"]) == 2
