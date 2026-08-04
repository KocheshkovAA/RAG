"""Агентный RAG-маршрут.

Настоящий tool-calling цикл: LLM получает инструмент `search_knowledge_base`
через `bind_tools` и сама решает, сколько раз и с какими запросами его вызвать
(в том числе несколько параллельных вызовов за один ход — для составных
вопросов с несколькими сущностями) и когда данных достаточно, чтобы
остановиться. Решение "можно ли отвечать" при этом остаётся за теми же
guardrails, что и у vector-маршрута — RetrievalGate и AnswerFaithfulnessGuard
переиспользуются по ссылке (не пересоздаются), финальный ответ всегда идёт
через тот же `self.chain` (строгий context-only промпт), поэтому агентность
не может обойти пороги так, как это когда-то было возможно у graph-маршрута
до фикса domain-probe. Свободный текст модели в ходах, где она не вызывает
инструмент, никогда не используется как ответ пользователю — только сигнал
"хватит искать".
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field

from app.core.cache import cache_client
from app.core.config import settings
from app.core.llm import llm_factory
from app.core.reranker import reranker
from app.core.usage import new_usage_handler, summarize_usage

if TYPE_CHECKING:
    # Только для type hint — реальный импорт vectorrag.py на уровне модуля
    # эагерно создаёт rag_chain = RAG() с сетевым обращением к Qdrant при
    # конструировании Retriever. Незачем тянуть это ради аннотации типа:
    # AgenticRAG получает vector_rag как обычный duck-typed параметр.
    from app.core.vectorrag import RAG

logger = logging.getLogger(__name__)


class SearchKnowledgeBase(BaseModel):
    """Найти релевантные фрагменты в базе знаний Warhammer 40k (гибридный
    dense+sparse поиск по вики). Вызывай отдельно для каждой сущности/
    подвопроса составного вопроса — можно вызвать несколько раз за один ход."""

    query: str = Field(
        description="Конкретный поисковый запрос: имя персонажа/фракции/термина "
        "или отдельный подвопрос, на русском языке"
    )


AGENTIC_SYSTEM_PROMPT = (
    "Ты — исследовательский агент базы знаний Warhammer 40k. Твоя задача — "
    "собрать достаточно контекста для ответа на вопрос пользователя, используя "
    "инструмент search_knowledge_base.\n\n"
    "Правила:\n"
    "- Если вопрос составной (несколько сущностей/дат/названий), вызови "
    "инструмент отдельно на каждую часть — можно несколько вызовов за один ход.\n"
    "- После каждого вызова ты увидишь, сколько найдено и прошёл ли порог "
    "релевантности (гейт). Если гейт прошёл, но покрыта не вся часть вопроса — "
    "доищи оставшееся.\n"
    "- Когда данных достаточно (или дальнейший поиск явно не помогает), "
    "останови вызовы инструмента — просто ответь коротко без вызова инструмента.\n"
    "- Твой собственный текстовый ответ в этом чате НЕ показывается пользователю "
    "и не используется как ответ — финальный ответ строится отдельно из "
    "найденного контекста. Твоя единственная задача здесь — искать."
)


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
        self.llm_with_tools = self.llm.bind_tools([SearchKnowledgeBase])
        self.max_iterations = max(1, settings.AGENTIC_MAX_ITERATIONS)
        self.max_tool_calls_per_round = max(1, settings.AGENTIC_MAX_TOOL_CALLS_PER_ROUND)

    @observe(name="Agentic RAG Pipeline")
    async def answer(self, question: str, usage_handler=None, include_debug_docs: bool = False):
        started = time.perf_counter()
        with propagate_attributes(tags=["rag", "warhammer", "agentic"]):
            # Кэш-хит выходит раньше, чем ниже добавляется _debug_docs — если
            # это eval-прогон с include_debug_docs, кэш обходим (см. тот же
            # паттерн в vectorrag.py RAG.answer).
            cached = None if include_debug_docs else await cache_client.get_answer(question, route="agentic")
            if cached and cached.get("answer"):
                cached = dict(cached)
                cached["cached"] = True
                cached["latency_ms"] = int((time.perf_counter() - started) * 1000)
                return cached

            handler = CallbackHandler()
            usage_handler = usage_handler or new_usage_handler()
            callbacks = [handler, usage_handler]

            messages = [SystemMessage(content=AGENTIC_SYSTEM_PROMPT), HumanMessage(content=question)]
            all_docs: list = []
            seen_contents: set[str] = set()
            gated_docs: list = []
            gate_meta: dict = {
                "passed": False,
                "reason": "no_search_attempted",
                "max_score": 0.0,
                "min_score": getattr(self.retrieval_gate, "min_score", None),
            }
            degraded: list[str] = []
            tool_calls_made: list[dict] = []
            stop_reason = "max_iterations"
            rounds_used = 0

            for round_idx in range(self.max_iterations):
                rounds_used = round_idx + 1
                ai_msg = await self.llm_with_tools.ainvoke(messages, config={"callbacks": callbacks})
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    stop_reason = "model_stopped"
                    break

                calls = ai_msg.tool_calls[: self.max_tool_calls_per_round]
                overflow = ai_msg.tool_calls[self.max_tool_calls_per_round :]

                async def _run_search(tool_call):
                    query = (tool_call.get("args") or {}).get("query") or question
                    try:
                        docs = await self.retriever.ainvoke(query, config={"callbacks": callbacks})
                        return tool_call, query, docs, None
                    except Exception as e:
                        logger.warning("search_knowledge_base failed for %r: %s", query, e)
                        return tool_call, query, [], str(e)

                results = await asyncio.gather(*(_run_search(tc) for tc in calls))

                round_info: dict[str, tuple[str, int, int, str | None]] = {}
                for tool_call, query, docs, err in results:
                    new_count = 0
                    for doc in docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
                            new_count += 1
                    round_info[tool_call["id"]] = (query, len(docs), new_count, err)
                    tool_calls_made.append(
                        {
                            "round": round_idx + 1,
                            "query": query,
                            "found": len(docs),
                            "new": new_count,
                            "error": err,
                        }
                    )

                # Реранк всегда против исходного вопроса — гейт и финальный ответ
                # должны оцениваться по релевантности тому, что реально спросили,
                # а не под-запросу конкретного инструмента.
                reranked, rerank_degraded = await reranker.rerank_documents(question, all_docs)
                if rerank_degraded and "reranker" not in degraded:
                    degraded.append("reranker")
                gated_docs, gate_meta = self.retrieval_gate.filter_docs(reranked)

                gate_status = "пройден" if gate_meta.get("passed") else "не пройден"
                for tool_call in calls:
                    query, found, new_count, err = round_info[tool_call["id"]]
                    if err:
                        content = f"Ошибка поиска: {err}"
                    else:
                        content = (
                            f"По запросу «{query}» найдено {found} фрагментов ({new_count} новых, "
                            f"всего накоплено {len(all_docs)}). Гейт релевантности: {gate_status} "
                            f"(score={gate_meta.get('max_score')}, порог={gate_meta.get('min_score')})."
                        )
                    messages.append(
                        ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_call["name"])
                    )
                for tool_call in overflow:
                    messages.append(
                        ToolMessage(
                            content="Пропущено: превышен лимит параллельных вызовов за один ход.",
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        )
                    )

                if round_idx == self.max_iterations - 1:
                    stop_reason = "max_iterations"
                    break

            agentic_meta = {
                "iterations": rounds_used,
                "tool_calls_made": tool_calls_made,
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
