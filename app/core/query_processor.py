from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langfuse import observe
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