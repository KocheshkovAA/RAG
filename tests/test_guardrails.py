"""Unit-тесты guardrails — без LLM/Docker, гоняются за секунды."""

from langchain_core.documents import Document

from app.core.guardrails import RetrievalGate, AnswerFaithfulnessGuard, doc_relevance_score


def _doc(score: float, *, rerank: bool = True) -> Document:
    key = "rerank_score" if rerank else "hybrid_score"
    return Document(page_content="chunk", metadata={key: score})


def test_doc_relevance_prefers_rerank():
    d = Document(page_content="x", metadata={"rerank_score": 0.9, "hybrid_score": 0.1})
    assert doc_relevance_score(d) == 0.9


def test_gate_rejects_offtopic_scores():
    """Симулируем кейс Губки Боба: rerank ~0.50 при пороге 0.55."""
    gate = RetrievalGate(min_score=0.55)
    docs = [_doc(0.5011), _doc(0.5006), _doc(0.5004)]
    kept, meta = gate.filter_docs(docs)
    assert kept == []
    assert meta["passed"] is False
    assert meta["reason"] == "below_similarity_threshold"


def test_gate_keeps_confident_docs():
    gate = RetrievalGate(min_score=0.55)
    docs = [_doc(0.82), _doc(0.40), _doc(0.91)]
    kept, meta = gate.filter_docs(docs)
    assert meta["passed"] is True
    assert len(kept) == 2  # 0.40 отфильтрован
    assert meta["max_score"] == 0.91


def test_gate_empty_retrieval():
    gate = RetrievalGate(min_score=0.55)
    kept, meta = gate.filter_docs([])
    assert kept == []
    assert meta["reason"] == "empty_retrieval"


def test_faithfulness_skipped_on_high_confidence():
    guard = AnswerFaithfulnessGuard(enabled=True)
    docs = [_doc(0.85)]
    need, meta = guard.should_verify(docs, degraded=False)
    assert need is False
    assert meta["reason"] == "high_retrieval_confidence"


def test_faithfulness_required_in_gray_zone():
    guard = AnswerFaithfulnessGuard(enabled=True)
    docs = [_doc(0.60)]
    need, meta = guard.should_verify(docs, degraded=False)
    assert need is True
    assert meta["skipped"] is False


def test_faithfulness_required_when_degraded():
    guard = AnswerFaithfulnessGuard(enabled=True)
    docs = [_doc(0.95)]
    need, meta = guard.should_verify(docs, degraded=True)
    assert need is True
    assert meta["reason"] == "degraded_pipeline"
