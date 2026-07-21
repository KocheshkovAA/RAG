"""Агентный RAG-маршрут.

LLM решает, КАК искать (переформулирует запрос и повторяет ретрив), но решение
"можно ли отвечать" остаётся за теми же guardrails, что и у vector-маршрута —
RetrievalGate и AnswerFaithfulnessGuard переиспользуются по ссылке (не
пересоздаются), поэтому эта агентность не может обойти пороги так, как это
когда-то было возможно у graph-маршрута до фикса domain-probe.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.core.cache import cache_client
from app.core.config import settings
from app.core.llm import llm_factory
from app.core.query_processor import QueryReformulator
from app.core.reranker import reranker
from app.core.usage import new_usage_handler, summarize_usage

if TYPE_CHECKING:
    # Только для type hint — реальный импорт vectorrag.py на уровне модуля
    # эагерно создаёт rag_chain = RAG() с сетевым обращением к Qdrant при
    # конструировании Retriever. Незачем тянуть это ради аннотации типа:
    # AgenticRAG получает vector_rag как обычный duck-typed параметр.
    from app.core.vectorrag import RAG

logger = logging.getLogger(__name__)


class AgenticRAG:
    def __init__(self, vector_rag: RAG):
        self.vector_rag = vector_rag
        # Переиспользуем те же инстансы, что и vector-маршрут — не копии с
        # возможно разошедшимися порогами.
        self.retriever = vector_rag.retriever
        self.retrieval_gate = vector_rag.retrieval_gate
        self.faithfulness_guard = vector_rag.faithfulness_guard
        self.chain = vector_rag.chain
        self.source_extractor = vector_rag.source_extractor

        self.llm = llm_factory.get_llm(temperature=0, role="agentic")
        self.reformulator = QueryReformulator(self.llm)
        self.max_iterations = max(1, settings.AGENTIC_MAX_ITERATIONS)

    @observe(name="Agentic RAG Pipeline")
    async def answer(self, question: str, usage_handler=None, include_debug_docs: bool = False):
        started = time.perf_counter()
        with propagate_attributes(tags=["rag", "warhammer", "agentic"]):
            cached = await cache_client.get_answer(question, route="agentic")
            if cached and cached.get("answer"):
                cached = dict(cached)
                cached["cached"] = True
                cached["latency_ms"] = int((time.perf_counter() - started) * 1000)
                return cached

            handler = CallbackHandler()
            usage_handler = usage_handler or new_usage_handler()
            callbacks = [handler, usage_handler]

            queries_tried = [question]
            current_query = question
            all_docs: list = []
            seen_contents: set[str] = set()
            gated_docs: list = []
            gate_meta: dict = {}
            degraded: list[str] = []
            stop_reason = "max_iterations"

            for i in range(self.max_iterations):
                new_docs = await self.retriever.ainvoke(current_query, config={"callbacks": callbacks})
                for doc in new_docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)

                # Реранк всегда против исходного вопроса — гейт и финальный ответ
                # должны оцениваться по релевантности тому, что реально спросили,
                # а не переформулированному под-запросу.
                reranked, rerank_degraded = await reranker.rerank_documents(question, all_docs)
                if rerank_degraded and "reranker" not in degraded:
                    degraded.append("reranker")

                gated_docs, gate_meta = self.retrieval_gate.filter_docs(reranked)
                if gated_docs:
                    stop_reason = "gate_passed"
                    break
                if i == self.max_iterations - 1:
                    stop_reason = "max_iterations"
                    break

                reformulated = await self.reformulator.process(
                    question=question,
                    previous_query=current_query,
                    gate_meta=gate_meta,
                    config={"callbacks": callbacks},
                )
                if not reformulated or reformulated in queries_tried:
                    stop_reason = "no_new_query"
                    break
                current_query = reformulated
                queries_tried.append(current_query)

            agentic_meta = {
                "iterations": len(queries_tried),
                "queries_tried": queries_tried,
                "stopped_reason": stop_reason,
            }

            if not gated_docs:
                resp = self.retrieval_gate.insufficient_response(gate_meta)
                resp["degraded"] = degraded
                resp["agentic"] = agentic_meta
                resp["token_usage"] = summarize_usage(usage_handler)
                resp["latency_ms"] = int((time.perf_counter() - started) * 1000)
                return resp

            answer = await self.chain.ainvoke(
                {"docs": gated_docs, "question": question},
                config={"callbacks": callbacks},
            )
            sources = self.source_extractor.extract(gated_docs)

            need_check, skip_meta = self.faithfulness_guard.should_verify(
                gated_docs,
                degraded=bool(degraded),
                answer=answer,
            )
            if need_check:
                passed, _verdict, faith_meta = await self.faithfulness_guard.verify(
                    question, answer, gated_docs, config={"callbacks": callbacks}
                )
                if not passed:
                    refused = self.faithfulness_guard.refuse_response(faith_meta, sources)
                    refused["guardrail"]["retrieval_gate"] = gate_meta
                    refused["degraded"] = degraded
                    refused["agentic"] = agentic_meta
                    refused["token_usage"] = summarize_usage(usage_handler)
                    refused["latency_ms"] = int((time.perf_counter() - started) * 1000)
                    return refused
            else:
                faith_meta = {
                    "enabled": settings.FAITHFULNESS_CHECK_ENABLED,
                    "passed": True,
                    **skip_meta,
                }

            result = {
                "answer": answer,
                "sources": sources,
                "guardrail": {
                    "retrieval_gate": gate_meta,
                    "faithfulness": faith_meta,
                    "refused": False,
                },
                "degraded": degraded,
                "cached": False,
                "agentic": agentic_meta,
                "token_usage": summarize_usage(usage_handler),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

            if not degraded and not result["guardrail"]["refused"]:
                try:
                    await cache_client.set_answer(question, result, route="agentic")
                except Exception as e:
                    logger.warning("Failed to cache answer: %s", e)

            if include_debug_docs:
                result["_debug_docs"] = gated_docs

            return result
