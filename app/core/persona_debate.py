"""MVP "дебатов персонажей" на LangGraph.

Пайплайн разделён на факт и форму:
1. retrieve — обычный ретрив + retrieval gate (как у vector-маршрута).
2. draft — ОДИН нейтральный ответ (без персонажа), который проходит
   faithfulness-проверку ОДИН раз. Если не прошёл — отказ на весь раунд, до
   персонажей дело не доходит.
3. персонажи по очереди пересказывают уже верифицированный draft в своей
   манере, каждый видя финальные реплики предыдущих в этом же раунде и держа
   в уме, что факты менять нельзя — заново их не проверяем.

Это осознанный отход от прежней версии, где faithfulness гонялась по три
раза на уже стилизованный (со сленгом/пафосом) текст каждого персонажа —
подача сбивала LLM-judge с толку не хуже, чем реальная галлюцинация, и стоила
трёх LLM-вызовов вместо одного на один и тот же контекст/вопрос.

RetrievalGate/AnswerFaithfulnessGuard переиспользуются по ссылке, не
пересоздаются — тот же принцип, что и у AgenticRAG.

Единственное место в проекте, где реальная нелинейная топология (несколько
узлов-агентов + общее состояние) оправдывает графовый фреймворк — обычный
retry-цикл (см. AgenticRAG) в этом не нуждался.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional, TYPE_CHECKING, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph

from app.chains.prompts import QA_SYSTEM_PROMPT
from app.core.config import settings
from app.core.llm import llm_factory
from app.core.personas import PERSONAS, Persona
from app.core.usage import new_usage_handler, summarize_usage

if TYPE_CHECKING:
    # Только для type hint — как и в agentic_rag.py/orchestrator.py, не тянем
    # vectorrag.py на уровне модуля (создаёт rag_chain = RAG() с сетевым
    # обращением к Qdrant при конструировании Retriever).
    from app.core.vectorrag import RAG

logger = logging.getLogger(__name__)


class DebateState(TypedDict, total=False):
    question: str
    callbacks: list
    context: str
    gated_docs: list
    gate_meta: dict
    degraded: list[str]
    sources: list
    need_check: bool
    draft: str
    faithfulness_meta: dict
    turns: list[dict]
    refused: bool
    refusal_answer: str


_DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
])


def _persona_prompt(persona: Persona) -> ChatPromptTemplate:
    system = QA_SYSTEM_PROMPT + "\n\nГОЛОС ПЕРСОНАЖА:\n" + persona.voice_prompt
    return ChatPromptTemplate.from_messages([
        ("system", system),
        (
            "human",
            "Вопрос: {question}\n\n"
            "Проверенный ответ (единственный источник фактов — пересказывай "
            "его, не добавляя и не меняя факты):\n{draft}\n\n"
            "{transcript}"
            "Перескажи этот ответ в характере персонажа. Если выше есть реплики "
            "других участников дебатов — отреагируй на них в своей манере "
            "(согласись, съязви, возрази по тону), но не трогай факты.",
        ),
    ])


def _format_transcript(turns: list[dict]) -> str:
    """Ответы предыдущих персонажей этого раунда — то, что видит следующий
    персонаж, чтобы реагировать/контрастировать."""
    if not turns:
        return ""
    lines = ["Ответы предыдущих участников дебатов в этом раунде:"]
    for t in turns:
        lines.append(f"— {t['persona_display_name']}: {t['answer']}")
    return "\n".join(lines) + "\n\n"


class PersonaDebate:
    def __init__(
        self,
        vector_rag: "RAG",
        personas: Optional[list[Persona]] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.vector_rag = vector_rag
        # Переиспользуем по ссылке — те же пороги/инстансы, что и vector-маршрут.
        self.retrieval_gate = vector_rag.retrieval_gate
        self.faithfulness_guard = vector_rag.faithfulness_guard
        self.context_builder = vector_rag.context_builder
        self.source_extractor = vector_rag.source_extractor

        self.personas = personas if personas is not None else PERSONAS
        self.llm = llm if llm is not None else llm_factory.get_llm(temperature=0.4, role="persona")
        self._draft_chain = _DRAFT_PROMPT | self.llm | StrOutputParser()
        self._chains = {
            p.id: _persona_prompt(p) | self.llm | StrOutputParser()
            for p in self.personas
        }
        self.graph = self._build_graph()

    def _make_persona_node(self, persona: Persona):
        async def node(state: DebateState) -> DebateState:
            transcript = _format_transcript(state.get("turns", []))
            answer = await self._chains[persona.id].ainvoke(
                {
                    "draft": state["draft"],
                    "question": state["question"],
                    "transcript": transcript,
                },
                config={"callbacks": state.get("callbacks", [])},
            )
            turn = {
                "persona": persona.id,
                "persona_display_name": persona.display_name,
                "answer": answer,
                "refused": False,
            }
            return {**state, "turns": state.get("turns", []) + [turn]}

        return node

    async def _retrieve_node(self, state: DebateState) -> DebateState:
        docs, degraded = await self.vector_rag.get_relevant_documents(
            state["question"], callbacks=state.get("callbacks", [])
        )
        gated_docs, gate_meta = self.retrieval_gate.filter_docs(docs)

        if not gated_docs:
            resp = self.retrieval_gate.insufficient_response(gate_meta)
            return {
                **state,
                "refused": True,
                "refusal_answer": resp["answer"],
                "gate_meta": gate_meta,
                "degraded": degraded,
                "turns": [],
            }

        context = self.context_builder.build(gated_docs)
        sources = self.source_extractor.extract(gated_docs)
        need_check, _skip_meta = self.faithfulness_guard.should_verify(
            gated_docs, degraded=bool(degraded)
        )
        return {
            **state,
            "gated_docs": gated_docs,
            "gate_meta": gate_meta,
            "degraded": degraded,
            "context": context,
            "sources": sources,
            "need_check": need_check,
            "turns": [],
            "refused": False,
        }

    async def _draft_node(self, state: DebateState) -> DebateState:
        draft = await self._draft_chain.ainvoke(
            {"context": state["context"], "question": state["question"]},
            config={"callbacks": state.get("callbacks", [])},
        )

        if state.get("need_check"):
            passed, _verdict, faith_meta = await self.faithfulness_guard.verify(
                state["question"], draft, state["gated_docs"],
                config={"callbacks": state.get("callbacks", [])},
            )
        else:
            passed, faith_meta = True, {
                "enabled": settings.FAITHFULNESS_CHECK_ENABLED,
                "passed": True,
                "skipped": True,
            }

        if not passed:
            return {
                **state,
                "refused": True,
                "refusal_answer": self.faithfulness_guard.insufficient_message,
                "faithfulness_meta": faith_meta,
            }

        return {**state, "draft": draft, "faithfulness_meta": faith_meta, "refused": False}

    def _build_graph(self):
        graph = StateGraph(DebateState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("draft", self._draft_node)
        graph.set_entry_point("retrieve")

        node_ids = [p.id for p in self.personas]
        for persona in self.personas:
            graph.add_node(persona.id, self._make_persona_node(persona))

        first_id = node_ids[0]

        graph.add_conditional_edges(
            "retrieve",
            lambda state: END if state.get("refused") else "draft",
            {"draft": "draft", END: END},
        )
        graph.add_conditional_edges(
            "draft",
            lambda state: END if state.get("refused") else first_id,
            {first_id: first_id, END: END},
        )
        for a, b in zip(node_ids, node_ids[1:]):
            graph.add_edge(a, b)
        graph.add_edge(node_ids[-1], END)

        return graph.compile()

    async def stream(self, question: str, usage_handler=None) -> AsyncIterator[dict[str, Any]]:
        usage_handler = usage_handler or new_usage_handler()
        handler = CallbackHandler()
        initial_state: DebateState = {"question": question, "callbacks": [handler, usage_handler]}

        seen_turns = 0
        final_state: DebateState = initial_state

        async for update in self.graph.astream(initial_state, stream_mode="values"):
            final_state = update
            turns = update.get("turns", [])
            while seen_turns < len(turns):
                yield {"type": "turn", "turn": turns[seen_turns]}
                seen_turns += 1

            if update.get("refused"):
                yield {"type": "refused", "answer": update.get("refusal_answer")}
                return

        yield {
            "type": "done",
            "sources": final_state.get("sources", []),
            "degraded": final_state.get("degraded", []),
            "guardrail": {
                "retrieval_gate": final_state.get("gate_meta", {}),
                "faithfulness": final_state.get("faithfulness_meta", {}),
            },
            "token_usage": summarize_usage(usage_handler),
        }
