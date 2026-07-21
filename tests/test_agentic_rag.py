"""Unit-тесты AgenticRAG — без LLM/Docker, гоняются за секунды.

Реальный LLM-клиент (self.llm/self.reformulator в __init__) конструируется
как обычно (это не бьёт по сети — то же самое уже происходит в
AnswerFaithfulnessGuard() в test_guardrails.py), но всё, что реально
вызывает LLM/сеть в ходе теста (reformulator.process, faithfulness_guard.verify,
retriever, reranker), подменяется фейками/моками.
"""

import pytest
from langchain_core.documents import Document

import app.core.agentic_rag as agentic_rag_module
from app.core.agentic_rag import AgenticRAG
from app.core.cache import CacheClient


class _FakeReranker:
    """Пропускает документы как есть — без реального сетевого вызова к vLLM."""

    async def rerank_documents(self, query, docs):
        return docs, False


class _FakeCacheClient:
    """Всегда промах — без реального обращения к Redis."""

    async def get_answer(self, question, route="vector"):
        return None

    async def set_answer(self, question, payload, route="vector"):
        return None


@pytest.fixture(autouse=True)
def _no_network_dependencies(monkeypatch):
    monkeypatch.setattr(agentic_rag_module, "reranker", _FakeReranker())
    monkeypatch.setattr(agentic_rag_module, "cache_client", _FakeCacheClient())


def _doc(rerank_score: float, content: str = "chunk") -> Document:
    return Document(page_content=content, metadata={"rerank_score": rerank_score, "article_name": "Test"})


class _FakeRetriever:
    """Возвращает заранее заданный список документов по каждому запросу."""

    def __init__(self, docs_by_query: dict[str, list[Document]]):
        self.docs_by_query = docs_by_query
        self.calls: list[str] = []

    async def ainvoke(self, query, config=None):
        self.calls.append(query)
        return self.docs_by_query.get(query, [])


class _FakeGate:
    """Тот же контракт, что RetrievalGate.filter_docs, но без реальных порогов конфига."""

    def __init__(self, min_score: float = 0.55):
        self.min_score = min_score

    def filter_docs(self, docs):
        if not docs:
            return [], {"passed": False, "reason": "empty_retrieval", "max_score": 0.0, "min_score": self.min_score}
        scored = [(d.metadata.get("rerank_score", 0.0), d) for d in docs]
        max_score = max(s for s, _ in scored)
        kept = [d for s, d in scored if s >= self.min_score]
        if not kept:
            return [], {
                "passed": False, "reason": "below_similarity_threshold",
                "max_score": max_score, "min_score": self.min_score,
            }
        return kept, {"passed": True, "reason": "ok", "max_score": max_score, "min_score": self.min_score}

    def insufficient_response(self, gate_meta):
        return {
            "answer": "REFUSED", "sources": [],
            "guardrail": {"retrieval_gate": gate_meta, "faithfulness": None, "refused": True},
            "degraded": [], "cached": False,
        }


class _FakeFaithfulnessGuard:
    def should_verify(self, docs, degraded=False, answer=""):
        return False, {"skipped": True, "reason": "test"}

    async def verify(self, question, answer, docs, config=None):
        raise AssertionError("verify() should not be called when should_verify() returns False")

    def refuse_response(self, faith_meta, sources):
        return {
            "answer": "REFUSED_FAITHFULNESS", "sources": sources,
            "guardrail": {"faithfulness": faith_meta, "refused": True},
            "degraded": [], "cached": False,
        }


class _FakeChain:
    async def ainvoke(self, inputs, config=None):
        return "FAKE ANSWER"


class _FakeSourceExtractor:
    def extract(self, docs):
        return [{"article_name": d.metadata.get("article_name")} for d in docs]


class _FakeReformulator:
    def __init__(self, responses: list[str | None]):
        self.responses = list(responses)
        self.calls: list[tuple] = []

    async def process(self, question, previous_query, gate_meta, config=None):
        self.calls.append((question, previous_query))
        return self.responses.pop(0) if self.responses else None


class _FakeVectorRAG:
    def __init__(self, retriever, retrieval_gate, faithfulness_guard, chain, source_extractor):
        self.retriever = retriever
        self.retrieval_gate = retrieval_gate
        self.faithfulness_guard = faithfulness_guard
        self.chain = chain
        self.source_extractor = source_extractor


def _make_agentic(retriever) -> AgenticRAG:
    fake_rag = _FakeVectorRAG(
        retriever=retriever,
        retrieval_gate=_FakeGate(min_score=0.55),
        faithfulness_guard=_FakeFaithfulnessGuard(),
        chain=_FakeChain(),
        source_extractor=_FakeSourceExtractor(),
    )
    agentic = AgenticRAG(fake_rag)
    # Реальный QueryReformulator бьёт по сети — подменяем на фейк в каждом тесте отдельно.
    return agentic


async def test_agentic_stops_immediately_when_gate_passes():
    question = "Кто такие некроны?"
    retriever = _FakeRetriever({question: [_doc(0.82)]})
    agentic = _make_agentic(retriever)
    agentic.reformulator = _FakeReformulator([])  # не должен вызываться

    result = await agentic.answer(question)

    assert result["guardrail"]["refused"] is False
    assert result["agentic"]["stopped_reason"] == "gate_passed"
    assert result["agentic"]["iterations"] == 1
    assert result["answer"] == "FAKE ANSWER"
    assert len(retriever.calls) == 1


async def test_agentic_reformulates_then_stops_at_max_iterations():
    question = "Кто такой Ярик?"
    # Ни один вариант запроса никогда не находит документов с достаточным score.
    retriever = _FakeRetriever({question: [_doc(0.1)]})
    agentic = _make_agentic(retriever)
    agentic.max_iterations = 3
    agentic.reformulator = _FakeReformulator(["запрос 2", "запрос 3"])

    result = await agentic.answer(question)

    assert result["guardrail"]["refused"] is True
    assert result["agentic"]["stopped_reason"] == "max_iterations"
    assert result["agentic"]["iterations"] == 3
    assert result["agentic"]["queries_tried"] == [question, "запрос 2", "запрос 3"]


async def test_agentic_stops_when_reformulator_gives_nothing_new():
    question = "Офф-топик вопрос"
    retriever = _FakeRetriever({question: [_doc(0.1)]})
    agentic = _make_agentic(retriever)
    agentic.max_iterations = 3
    agentic.reformulator = _FakeReformulator([None])

    result = await agentic.answer(question)

    assert result["guardrail"]["refused"] is True
    assert result["agentic"]["stopped_reason"] == "no_new_query"
    assert result["agentic"]["iterations"] == 1


def test_agentic_reuses_gate_and_guard_by_reference():
    fake_rag = _FakeVectorRAG(
        retriever=_FakeRetriever({}),
        retrieval_gate=_FakeGate(),
        faithfulness_guard=_FakeFaithfulnessGuard(),
        chain=_FakeChain(),
        source_extractor=_FakeSourceExtractor(),
    )
    agentic = AgenticRAG(fake_rag)

    assert agentic.retrieval_gate is fake_rag.retrieval_gate
    assert agentic.faithfulness_guard is fake_rag.faithfulness_guard


def test_cache_keys_differ_by_route():
    vector_key = CacheClient._key("answer:vector", "Кто такие некроны?")
    agentic_key = CacheClient._key("answer:agentic", "Кто такие некроны?")
    assert vector_key != agentic_key
