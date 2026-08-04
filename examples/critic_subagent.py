"""Sub-agent spike: Critic — второй, независимый "взгляд" на черновик ответа.

Демонстрирует границу agent / sub-agent / guard из `docs/agent-design-decisions.md` (п.6
плана): `AnswerFaithfulnessGuard` (`app/core/guardrails.py`) тоже делает LLM-вызовы, но
остаётся guard'ом, а не sub-agent'ом — узкая единственная задача (заземлён ли ответ в
контексте буквально), решение в конце концов бинарное (passed/not passed по порогу),
встроен в критический путь каждого запроса как обязательная точка. `CriticSubAgent` —
шире по объёму суждения (полнота ответа на составной вопрос, ясность формулировок,
не выдаёт ли черновик общее знание о вселенной за специфичный факт), трёхвариантный
вердикт (`approve`/`revise`/`refuse`) с текстовым фидбеком, который предназначен
управлять СЛЕДУЮЩИМ шагом генерации, а не быть терминальным условием самим по себе.

НЕ подключён к основному пайплайну — задокументированный спайк, доказывающий, что решение
"здесь нужен guard, а не sub-agent" в проде осознанное, а не потому что sub-agent'ы не
умеем. Многошаговый цикл "draft -> critic -> revise -> critic" тоже НЕ реализован — это
отдельная, более крупная задача, которую этот спайк сознательно не берёт на себя (см.
"Практический минимум" в `docs/agent-design-decisions.md`).

Запуск с реальным пайплайном (нужен настроенный .env / контейнер api):
    python -m examples.critic_subagent "Кто такие некроны?"
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Literal, Protocol

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class CriticVerdict(BaseModel):
    """Вердикт критика по черновику ответа: шире, чем бинарная заземлённость — оценивает
    полноту, ясность и фактическую уверенность, а не дословное цитирование (это отдельно
    проверяет AnswerFaithfulnessGuard, критик эту работу не дублирует)."""

    verdict: Literal["approve", "revise", "refuse"] = Field(
        description="approve — черновик готов как есть; revise — есть конкретные исправимые "
        "проблемы (см. feedback); refuse — вопрос не стоит пытаться закрыть этим контекстом, "
        "честный отказ лучше натянутого ответа."
    )
    concerns: List[str] = Field(
        default_factory=list,
        description="Конкретные проблемы черновика: пропущенные части составного вопроса, "
        "неуверенные/двусмысленные формулировки, общее знание о вселенной, выданное за "
        "специфичный факт из контекста.",
    )
    feedback: str = Field(
        default="",
        description="Короткая инструкция для следующей попытки генерации, если verdict=revise. "
        "Пусто при approve/refuse.",
    )


CRITIC_SYSTEM_PROMPT = (
    "Ты — независимый критик черновика ответа в RAG-системе по лору Warhammer 40k. Тебе "
    "известны исходный вопрос, найденный контекст и уже сгенерированный черновик ответа "
    "(дословная заземлённость черновика в контексте уже проверена отдельно — не дублируй "
    "эту работу, сосредоточься на качестве и полноте).\n\n"
    "Оцени черновик по трём осям:\n"
    "- Полнота: отвечает ли черновик на ВСЕ части вопроса, если вопрос составной?\n"
    "- Ясность: нет ли двусмысленных или противоречащих друг другу формулировок?\n"
    "- Уверенность: не выдаёт ли черновик общее знание о вселенной Warhammer 40k за "
    "специфичный факт из контекста, даже если формально что-то похожее процитировано?\n\n"
    "Верни approve, если проблем нет; revise — если есть конкретные исправимые проблемы "
    "(укажи их в concerns и краткую инструкцию в feedback); refuse — если контекста в "
    "принципе недостаточно, а обтекаемые формулировки в черновике маскируют нехватку данных."
)


class _LLMLike(Protocol):
    def with_structured_output(self, schema): ...


class CriticSubAgent:
    """Тонкая обёртка вокруг одного структурированного LLM-вызова — намеренно минимальная,
    чтобы не превращать пример в multi-agent платформу (см. докстринг модуля)."""

    def __init__(self, llm: _LLMLike):
        structured = llm.with_structured_output(CriticVerdict)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CRITIC_SYSTEM_PROMPT),
                ("human", "ВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{context}\n\nЧЕРНОВИК ОТВЕТА:\n{draft}"),
            ]
        )
        self.chain = prompt | structured

    async def review(self, question: str, context: str, draft: str, config=None) -> CriticVerdict:
        return await self.chain.ainvoke(
            {"question": question, "context": context, "draft": draft}, config=config
        )


async def _main(question: str) -> None:
    from app.core.llm import llm_factory
    from app.core.vectorrag import rag_chain

    docs, _degraded = await rag_chain.get_relevant_documents(question)
    gated, _gate_meta = rag_chain.retrieval_gate.filter_docs(docs)
    if not gated:
        print("Ретрив пуст — критику показывать нечего.")
        return

    context = rag_chain.faithfulness_guard.context_builder.build(gated)
    draft = await rag_chain.chain.ainvoke({"docs": gated, "question": question})
    print(f"ЧЕРНОВИК:\n{draft}\n")

    critic = CriticSubAgent(llm_factory.get_llm(temperature=0, role="faithfulness"))
    verdict = await critic.review(question, context, draft)
    print(f"ВЕРДИКТ: {verdict.verdict}")
    if verdict.concerns:
        print("ЗАМЕЧАНИЯ:")
        for c in verdict.concerns:
            print(f"  - {c}")
    if verdict.feedback:
        print(f"ФИДБЕК: {verdict.feedback}")


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else "Кто такие некроны?"))
