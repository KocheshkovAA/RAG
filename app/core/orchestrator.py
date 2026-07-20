from enum import Enum
import logging
import time

from pydantic import BaseModel, Field
from langfuse import observe, propagate_attributes

from app.core.llm import llm_factory
from app.core.vectorrag import RAG
from app.core.lightrag_client import LightRAGClient
from app.core.guardrails import RetrievalGate
from app.core.tracing import score_ask_result

logger = logging.getLogger(__name__)


class RAGRoute(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


class RouteDecision(BaseModel):
    """Схема для классификации входящего вопроса пользователя по вселенной Warhammer 40k."""

    reasoning: str = Field(
        ...,
        description="Краткое обоснование, почему выбран этот путь (векторный поиск или графовый)",
    )
    route: RAGRoute = Field(
        ...,
        description="Выбранный маршрут: 'vector' для простых фактов, 'graph' для сложных связей и лора",
    )


class WarhammerOrchestrator:
    def __init__(self, vector_rag: RAG, light_rag: LightRAGClient):
        self.vector_rag = vector_rag
        self.light_rag = light_rag
        self.llm = llm_factory.get_llm(temperature=0)
        self.retrieval_gate = RetrievalGate()

        self.system_prompt = (
            "Ты — логический модуль системы Warhammer 40k Lore Knowledge Base. "
            "Твоя задача — выбрать лучший инструмент для ответа на вопрос.\n\n"
            "Выбирай 'vector' (Vector RAG), если:\n"
            "- Нужен конкретный факт (Кто убил? В каком году? На какой планете?)\n"
            "- Нужно описание конкретного юнита или персонажа.\n"
            "- Вопрос НЕ про Warhammer 40k (кулинария, мемы, мусорный текст, другие вселенные) — "
            "всегда vector, чтобы сработали пороги отказа.\n\n"
            "Выбирай 'graph' (Graph RAG), если:\n"
            "- Нужно понять связи между сущностями Warhammer (Как связаны Магнус и Ариман?)\n"
            "- Нужен анализ событий Warhammer (Как Ересь Хоруса повлияла на Империум?)\n"
            "- Вопрос сложный, охватывает несколько эпох или организаций Warhammer.\n\n"
            "Важно: graph только для вопросов по лору Warhammer 40k."
        )

    @observe(name="Router Decision")
    async def classify_route(self, question: str) -> RAGRoute:
        try:
            structured_llm = self.llm.with_structured_output(RouteDecision)
            messages = [
                ("system", self.system_prompt),
                ("human", f"Вопрос: {question}"),
            ]
            decision = await structured_llm.ainvoke(messages)
            return decision.route
        except Exception as e:
            logger.warning("Router failed (%s), defaulting to vector", e)
            return RAGRoute.VECTOR

    @observe(name="Domain Relevance Probe")
    async def _domain_probe(self, question: str) -> tuple[bool, dict]:
        """
        Перед LightRAG проверяем, что в векторной базе есть релевантный контекст.
        Иначе graph-путь обходит пороги и отвечает на оффтоп.
        """
        docs, degraded = await self.vector_rag.get_relevant_documents(question)
        kept, gate_meta = self.retrieval_gate.filter_docs(docs)
        return bool(kept), {
            "retrieval_gate": gate_meta,
            "degraded": degraded,
            "kept_docs": len(kept),
        }

    def _refuse(self, gate_meta: dict, degraded: list[str], started: float, mode: str) -> dict:
        resp = self.retrieval_gate.insufficient_response(gate_meta)
        resp["degraded"] = degraded
        resp["mode"] = mode
        resp["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return resp

    @observe(name="Global Orchestrator")
    async def answer(self, question: str):
        started = time.perf_counter()
        with propagate_attributes(tags=["orchestrator", "warhammer"]):
            route = await self.classify_route(question)

            if route == RAGRoute.GRAPH:
                ok, probe = await self._domain_probe(question)
                if not ok:
                    result = self._refuse(
                        probe["retrieval_gate"],
                        probe.get("degraded") or [],
                        started,
                        mode="graph-refused",
                    )
                    score_ask_result(question, result)
                    return result

                result = await self.light_rag.query(question, mode="hybrid")
                if result.get("_fallback_to_vector"):
                    vector_result = await self.vector_rag.answer(question)
                    degraded = list(vector_result.get("degraded", []))
                    if "lightrag" not in degraded:
                        degraded.append("lightrag")
                    vector_result["degraded"] = degraded
                    vector_result["mode"] = "vector-fallback"
                    score_ask_result(question, vector_result)
                    return vector_result

                result.setdefault(
                    "guardrail",
                    {"refused": False, "retrieval_gate": probe["retrieval_gate"]},
                )
                result.setdefault("degraded", [])
                result.setdefault("cached", False)
                result["mode"] = result.get("mode", "lightrag-hybrid")
                result["latency_ms"] = int((time.perf_counter() - started) * 1000)
                result["sources"] = _normalize_sources(result.get("sources"))
                score_ask_result(question, result)
                return result

            result = await self.vector_rag.answer(question)
            result["mode"] = "vector"
            score_ask_result(question, result)
            return result


def _normalize_sources(sources) -> list:
    if not sources:
        return []
    if isinstance(sources, str):
        return [{"article_name": "N/A", "url": "", "title": "", "score": None, "snippet": sources[:500]}]
    out = []
    for s in sources:
        if isinstance(s, dict):
            out.append(s)
        else:
            out.append({"article_name": "N/A", "url": "", "title": str(s)[:200], "score": None})
    return out
