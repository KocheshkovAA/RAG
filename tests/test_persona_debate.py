"""Unit-тесты PersonaDebate (LangGraph) — без реального LLM/Docker, в стиле
tests/test_agentic_rag.py. Реальный LLM подменяется FakeListChatModel
(доступен в установленной версии langchain_core, не требует сети)."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.persona_debate import PersonaDebate, _format_transcript
from app.core.personas import Persona

_TEST_PERSONAS = [
    Persona(id="p1", display_name="Первый", voice_prompt="Говори кратко."),
    Persona(id="p2", display_name="Второй", voice_prompt="Говори длинно."),
    Persona(id="p3", display_name="Третий", voice_prompt="Говори загадками."),
]


def _doc(content: str = "chunk") -> Document:
    return Document(page_content=content, metadata={"article_name": "Test", "rerank_score": 0.9})


class _FakeGate:
    def filter_docs(self, docs):
        if not docs:
            return [], {"passed": False, "reason": "empty_retrieval", "max_score": 0.0, "min_score": 0.55}
        return docs, {"passed": True, "reason": "ok", "max_score": 0.9, "min_score": 0.55}

    def insufficient_response(self, gate_meta):
        return {
            "answer": "ОТКАЗ_РЕТРИВ", "sources": [],
            "guardrail": {"retrieval_gate": gate_meta, "faithfulness": None, "refused": True},
            "degraded": [], "cached": False,
        }


class _FakeFaithfulnessGuard:
    def __init__(self, need_check: bool = False, verify_results: list[bool] | None = None):
        self.need_check = need_check
        self.verify_results = list(verify_results or [])
        self.should_verify_calls = 0
        self.verify_calls: list[str] = []
        self.insufficient_message = "ОТКАЗ_ДОСТОВЕРНОСТЬ"

    def should_verify(self, docs, degraded=False):
        self.should_verify_calls += 1
        return self.need_check, {"skipped": not self.need_check}

    async def verify(self, question, answer, docs, config=None):
        self.verify_calls.append(answer)
        passed = self.verify_results.pop(0) if self.verify_results else True
        return passed, None, {"passed": passed}


class _FakeContextBuilder:
    def build(self, docs):
        return "FAKE_CONTEXT"


class _FakeSourceExtractor:
    def extract(self, docs):
        return [{"article_name": "Test"}]


class _FakeVectorRAG:
    def __init__(self, gate, guard, docs=None):
        self.retrieval_gate = gate
        self.faithfulness_guard = guard
        self.context_builder = _FakeContextBuilder()
        self.source_extractor = _FakeSourceExtractor()
        self._docs = docs if docs is not None else [_doc()]

    async def get_relevant_documents(self, question, callbacks=None):
        return self._docs, []


def _make_debate(guard=None, gate=None, docs=None, responses=None, personas=None):
    guard = guard or _FakeFaithfulnessGuard()
    gate = gate or _FakeGate()
    fake_rag = _FakeVectorRAG(gate=gate, guard=guard, docs=docs)
    llm = FakeListChatModel(responses=responses or ["ответ"] * 10)
    return PersonaDebate(fake_rag, personas=personas or _TEST_PERSONAS, llm=llm)


def test_format_transcript_empty():
    assert _format_transcript([]) == ""


def test_format_transcript_with_turns():
    turns = [
        {"persona_display_name": "Орк", "answer": "ВААГХ"},
        {"persona_display_name": "Эльдар", "answer": "Печально."},
    ]
    text = _format_transcript(turns)
    assert "Орк: ВААГХ" in text
    assert "Эльдар: Печально." in text
    assert text.index("Орк") < text.index("Эльдар")


async def test_debate_refuses_on_empty_retrieval():
    debate = _make_debate(docs=[])
    events = [e async for e in debate.stream("вопрос")]
    assert events[-1]["type"] == "refused"
    assert not any(e["type"] == "turn" for e in events)


async def test_debate_produces_turns_in_order():
    debate = _make_debate(responses=["ответ1", "ответ2", "ответ3"])
    events = [e async for e in debate.stream("вопрос")]
    turns = [e["turn"] for e in events if e["type"] == "turn"]
    assert [t["persona"] for t in turns] == ["p1", "p2", "p3"]
    assert [t["answer"] for t in turns] == ["ответ1", "ответ2", "ответ3"]
    assert all(t["refused"] is False for t in turns)


async def test_should_verify_called_once_per_debate():
    guard = _FakeFaithfulnessGuard(need_check=True)
    debate = _make_debate(guard=guard, responses=["a", "b", "c"])
    _ = [e async for e in debate.stream("вопрос")]
    assert guard.should_verify_calls == 1


async def test_faithfulness_failure_replaces_turn_and_continues():
    guard = _FakeFaithfulnessGuard(need_check=True, verify_results=[True, False, True])
    debate = _make_debate(guard=guard, responses=["a", "b", "c"])
    events = [e async for e in debate.stream("вопрос")]
    turns = [e["turn"] for e in events if e["type"] == "turn"]

    assert len(turns) == 3
    assert turns[1]["refused"] is True
    assert turns[1]["answer"] == guard.insufficient_message
    assert turns[0]["refused"] is False
    assert turns[2]["refused"] is False


async def test_stream_event_order():
    debate = _make_debate(responses=["a", "b", "c"])
    event_types = [e["type"] async for e in debate.stream("вопрос")]
    assert event_types == ["turn", "turn", "turn", "done"]


def test_reuses_gate_and_guard_by_reference():
    guard = _FakeFaithfulnessGuard()
    gate = _FakeGate()
    fake_rag = _FakeVectorRAG(gate=gate, guard=guard)
    debate = PersonaDebate(fake_rag, personas=_TEST_PERSONAS, llm=FakeListChatModel(responses=["x"]))
    assert debate.retrieval_gate is gate
    assert debate.faithfulness_guard is guard
