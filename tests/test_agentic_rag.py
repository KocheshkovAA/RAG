"""Unit-тесты AgenticRAG — без LLM/Docker, гоняются за секунды.

Реальный LLM-клиент (self.llm/self.llm_with_tools в __init__) конструируется
как обычно (это не бьёт по сети — то же самое уже происходит в
AnswerFaithfulnessGuard() в test_guardrails.py), но всё, что реально
вызывает LLM/сеть в ходе теста (llm_with_tools.ainvoke, faithfulness_guard.verify,
retriever, reranker), подменяется фейками/моками.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage

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


class _FakeToolLLM:
    """Скриптует последовательность ходов модели; ainvoke() отдаёт по одному AIMessage за раз."""

    def __init__(self, turns: list[AIMessage]):
        self.turns = list(turns)
        self.calls: list[list] = []  # история сообщений на каждый вызов — для проверки ToolMessage

    async def ainvoke(self, messages, config=None):
        self.calls.append(list(messages))
        return self.turns.pop(0)


def _ai_tool_call(query: str, call_id: str = "call_1", thought: str = "") -> AIMessage:
    return AIMessage(
        content=thought,
        tool_calls=[{"name": "SearchKnowledgeBase", "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


def _ai_tool_calls(*queries_and_ids: tuple[str, str], thought: str = "") -> AIMessage:
    return AIMessage(
        content=thought,
        tool_calls=[
            {"name": "SearchKnowledgeBase", "args": {"query": q}, "id": cid, "type": "tool_call"}
            for q, cid in queries_and_ids
        ],
    )


def _ai_stop(text: str = "Готово") -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


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
    # Реальный self.llm_with_tools бьёт по сети — подменяем на фейк в каждом тесте отдельно.
    return agentic


async def test_agentic_stops_after_model_confirms_enough_context():
    question = "Кто такие некроны?"
    retriever = _FakeRetriever({question: [_doc(0.82)]})
    agentic = _make_agentic(retriever)
    # Гейт проходит уже на первом раунде, но модель всё равно получает следующий
    # ход и там сама решает остановиться (без вызова инструмента) — раннего
    # выхода по "гейт прошёл" больше нет, решение всегда за моделью.
    agentic.llm_with_tools = _FakeToolLLM([_ai_tool_call(question), _ai_stop()])

    result = await agentic.answer(question)

    assert result["guardrail"]["refused"] is False
    assert result["agentic"]["stopped_reason"] == "model_stopped"
    assert result["agentic"]["iterations"] == 2
    assert result["answer"] == "FAKE ANSWER"
    assert len(retriever.calls) == 1


async def test_agentic_keeps_calling_tool_until_max_iterations():
    question = "Кто такой Ярик?"
    # Ни один вариант запроса никогда не находит документов с достаточным score.
    retriever = _FakeRetriever({question: [_doc(0.1)]})
    agentic = _make_agentic(retriever)
    agentic.max_iterations = 3
    agentic.llm_with_tools = _FakeToolLLM(
        [_ai_tool_call(question), _ai_tool_call("запрос 2"), _ai_tool_call("запрос 3")]
    )

    result = await agentic.answer(question)

    assert result["guardrail"]["refused"] is True
    assert result["agentic"]["stopped_reason"] == "max_iterations"
    assert result["agentic"]["iterations"] == 3
    assert [tc["query"] for tc in result["agentic"]["tool_calls_made"]] == [question, "запрос 2", "запрос 3"]


async def test_agentic_refuses_when_model_never_searches():
    question = "Офф-топик вопрос"
    retriever = _FakeRetriever({})
    agentic = _make_agentic(retriever)
    agentic.max_iterations = 3
    agentic.llm_with_tools = _FakeToolLLM([_ai_stop()])

    result = await agentic.answer(question)

    assert result["guardrail"]["refused"] is True
    assert result["agentic"]["stopped_reason"] == "model_stopped"
    assert result["agentic"]["iterations"] == 1
    assert result["guardrail"]["retrieval_gate"]["reason"] == "no_search_attempted"
    assert len(retriever.calls) == 0


async def test_agentic_runs_parallel_tool_calls_in_one_round():
    question = "Кто такие Магнус и Ангрон?"
    retriever = _FakeRetriever({
        "Магнус Красный": [_doc(0.9, content="magnus")],
        "Ангрон": [_doc(0.9, content="angron")],
    })
    agentic = _make_agentic(retriever)
    fake_llm = _FakeToolLLM([
        _ai_tool_calls(("Магнус Красный", "call_1"), ("Ангрон", "call_2")),
        _ai_stop(),
    ])
    agentic.llm_with_tools = fake_llm

    result = await agentic.answer(question)

    assert sorted(retriever.calls) == ["Ангрон", "Магнус Красный"]
    assert result["agentic"]["iterations"] == 2
    assert len(result["agentic"]["tool_calls_made"]) == 2

    # Второй вызов ainvoke должен содержать по ToolMessage на каждый tool_call_id.
    second_call_messages = fake_llm.calls[1]
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_messages} == {"call_1", "call_2"}


async def test_agentic_captures_thought_per_round():
    question = "Кто такие некроны?"
    retriever = _FakeRetriever({question: [_doc(0.82)]})
    agentic = _make_agentic(retriever)
    agentic.llm_with_tools = _FakeToolLLM([
        _ai_tool_call(question, thought="Ищу базовую информацию о некронах."),
        _ai_stop("Данных достаточно, гейт пройден, останавливаюсь."),
    ])

    result = await agentic.answer(question)

    assert result["agentic"]["thoughts"] == [
        {"round": 1, "thought": "Ищу базовую информацию о некронах."},
        {"round": 2, "thought": "Данных достаточно, гейт пройден, останавливаюсь."},
    ]


async def test_agentic_skips_empty_thoughts():
    question = "Кто такие некроны?"
    retriever = _FakeRetriever({question: [_doc(0.82)]})
    agentic = _make_agentic(retriever)
    # Модель не всегда обязана прислать мысль (например, реальный провайдер
    # иногда шлёт пустой content вместе с tool_calls) — пустые ходы не должны
    # засорять thoughts фиктивными записями.
    agentic.llm_with_tools = _FakeToolLLM([_ai_tool_call(question), _ai_stop("")])

    result = await agentic.answer(question)

    assert result["agentic"]["thoughts"] == []


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
