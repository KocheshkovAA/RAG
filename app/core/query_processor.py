import json
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import observe
from pydantic import BaseModel, Field

from typing import List
from pydantic import BaseModel, Field

class ExpandedQuery(BaseModel):
    """Список поисковых запросов для улучшения ретрива в Warhammer 40k Wiki."""
    queries: List[str] = Field(description="Список из 1-3 переформулированных или уточняющих поисковых запросов")

class QueryOptimizer:
    def __init__(self, llm):
        # Привязываем схему прямо к модели
        # Это заставит LLM выдавать структурированный ответ
        self.structured_llm = llm.with_structured_output(ExpandedQuery)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Твоя задача — оптимизировать запрос для поиска в базе знаний.\n"
                "Если вопрос сложный, разбей его на 2-3 простых под-вопросов.\n"
                "Если вопрос простой, переформулируй его для более точного поиска.\n"
            )),
            ("human", "{question}")
        ])
        
        # Теперь цепочка возвращает сразу объект Pydantic
        self.chain = self.prompt | self.structured_llm

    @observe(name="Query Expansion")
    async def process(self, question: str, config=None) -> List[str]:
        try:
            # На выходе уже будет инстанс ExpandedQuery, парсить руками не надо
            result: ExpandedQuery = await self.chain.ainvoke({"question": question}, config=config)
            return result.queries
        except Exception as e:
            # Если даже структурированный вывод подвел (редко, но бывает)
            print(f"Query optimization failed: {e}")
            return [question]


class ReformulatedQuery(BaseModel):
    """Единичный переформулированный запрос для retry-цикла ретрива."""
    query: str = Field(description="Переформулированный поисковый запрос")
    reasoning: str = Field(description="Кратко: почему предыдущий запрос не дал релевантных результатов")


class QueryReformulator:
    """Переформулирует запрос по фидбеку о неудаче ретрива (не путать с QueryOptimizer —
    тот делает expansion на 1-3 параллельных под-запроса без обратной связи)."""

    def __init__(self, llm):
        self.structured_llm = llm.with_structured_output(ReformulatedQuery)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Твоя задача — переформулировать поисковый запрос, который не нашёл "
                "достаточно релевантных документов в базе знаний по Warhammer 40k.\n"
                "Тебе известен исходный вопрос пользователя, последний поисковый запрос "
                "и оценка релевантности найденных документов (чем ниже — тем хуже).\n"
                "Предложи ОДИН новый вариант запроса: другие ключевые слова, синоним "
                "имени/термина, более широкая или более узкая формулировка — в зависимости "
                "от того, что вероятнее сработает."
            )),
            ("human", (
                "Исходный вопрос: {question}\n"
                "Предыдущий запрос: {previous_query}\n"
                "Максимальный скор найденных документов: {max_score} (нужно >= {min_score})\n"
            )),
        ])

        self.chain = self.prompt | self.structured_llm

    @observe(name="Query Reformulation")
    async def process(self, question: str, previous_query: str, gate_meta: dict, config=None) -> str | None:
        try:
            result: ReformulatedQuery = await self.chain.ainvoke(
                {
                    "question": question,
                    "previous_query": previous_query,
                    "max_score": gate_meta.get("max_score"),
                    "min_score": gate_meta.get("min_score"),
                },
                config=config,
            )
            reformulated = (result.query or "").strip()
            return reformulated or None
        except Exception as e:
            print(f"Query reformulation failed: {e}")
            return None