from operator import itemgetter
from typing import List, Any
import time
import asyncio
import logging

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import propagate_attributes, observe
from langfuse.langchain import CallbackHandler

from app.core.query_processor import QueryOptimizer
from app.core.retriever import Retriever
from app.core.llm import llm_factory
from app.chains.prompts import QA_SYSTEM_PROMPT
from app.core.postprocessors.context_builder import ContextBuilder
from app.core.postprocessors.source_extractor import SourceExtractor
from app.core.reranker import reranker
from app.core.config import settings
from app.core.guardrails import RetrievalGate, AnswerFaithfulnessGuard
from app.core.cache import cache_client

logger = logging.getLogger(__name__)


class RAG:
    def __init__(self):
        self.llm = llm_factory.get_llm(temperature=0.15)
        self.retriever = Retriever.from_collection(k=30)
        self.context_builder = ContextBuilder()
        self.source_extractor = SourceExtractor()
        self.retrieval_gate = RetrievalGate()
        self.faithfulness_guard = AnswerFaithfulnessGuard()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            ("human", "Контекст:\n{context}\n\nВопрос: {question}")
        ])

        self.chain = (
            {
                "context": itemgetter("docs") | RunnableLambda(self.context_builder.build),
                "question": itemgetter("question"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @observe(name="Retrieve Only")
    async def get_relevant_documents(
        self, question: str, handler=None
    ) -> tuple[List[Any], list[str]]:
        """
        Expansion -> Multi-Retrieval -> Rerank.
        Returns: (docs, degraded_components)
        """
        degraded: list[str] = []

        if settings.QUERY_OPTIMIZER_ENABLED:
            optimizer = QueryOptimizer(self.llm)
            expanded_queries = await optimizer.process(question)
        else:
            expanded_queries = [question]

        tasks = [
            self.retriever.ainvoke(q, config={"callbacks": [handler] if handler else []})
            for q in expanded_queries
        ]
        docs_results = await asyncio.gather(*tasks)

        all_docs = []
        seen_contents = set()
        for sublist in docs_results:
            for doc in sublist:
                if doc.page_content not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)

        if not all_docs:
            return [], degraded

        final_docs, rerank_degraded = await reranker.rerank_documents(question, all_docs)
        if rerank_degraded:
            degraded.append("reranker")
        return final_docs, degraded

    @observe(name="RAG Pipeline")
    async def answer(self, question: str):
        """Cache → retrieval gate → generate → conditional faithfulness."""
        started = time.perf_counter()
        with propagate_attributes(tags=["rag", "warhammer", "production"]):
            cached = await cache_client.get_answer(question)
            if cached and cached.get("answer"):
                cached = dict(cached)
                cached["cached"] = True
                cached["latency_ms"] = int((time.perf_counter() - started) * 1000)
                return cached

            handler = CallbackHandler()
            final_docs, degraded = await self.get_relevant_documents(
                question, handler=handler
            )

            gated_docs, gate_meta = self.retrieval_gate.filter_docs(final_docs)
            if not gated_docs:
                resp = self.retrieval_gate.insufficient_response(gate_meta)
                resp["degraded"] = degraded
                resp["latency_ms"] = int((time.perf_counter() - started) * 1000)
                return resp

            answer = await self.chain.ainvoke(
                {"docs": gated_docs, "question": question},
                config={"callbacks": [handler]},
            )
            sources = self.source_extractor.extract(gated_docs)

            need_check, skip_meta = self.faithfulness_guard.should_verify(
                gated_docs,
                degraded=bool(degraded),
                answer=answer,
            )
            if need_check:
                passed, _verdict, faith_meta = await self.faithfulness_guard.verify(
                    question, answer, gated_docs
                )
                if not passed:
                    refused = self.faithfulness_guard.refuse_response(faith_meta, sources)
                    refused["guardrail"]["retrieval_gate"] = gate_meta
                    refused["degraded"] = degraded
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
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }

            # Не кэшируем отказы и деградированные ответы (чтобы не «заморозить» сбой)
            if not degraded and not result["guardrail"]["refused"]:
                try:
                    await cache_client.set_answer(question, result)
                except Exception as e:
                    logger.warning("Failed to cache answer: %s", e)

            return result


rag_chain = RAG()
