from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from app.config import MAX_RESPONSE_LENGTH

TELEGRAM_PROMPT_TEMPLATE = """Ты AI-ассистент, отвечающий на вопросы. Формат ответа:

1. **Основные требования**:
   - Язык: русский
   - Текст: простой текст без форматирования
   - Длина: не более {max_length} символов
   - Четкая структура: используй списки и короткие абзацы

2. **Как оформлять ответ**:
   - Главную мысль выделяй первым тезисом
   - Дополнительные пункты оформляй списком
   - Каждый пункт списка - одна законченная мысль
   - Между абзацами оставляй пустую строку
   - Избегай сложных конструкций

Контекст: {context}

Вопрос: {question}

Сформулируй четкий ответ:"""

def build_rag_chain(llm, retriever):
    prompt = PromptTemplate(
        template=TELEGRAM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"max_length": str(MAX_RESPONSE_LENGTH)}
    )

    combine_documents_chain = load_qa_chain(
        llm=llm.bind(max_length=MAX_RESPONSE_LENGTH),
        chain_type="stuff", 
        prompt=prompt
    )

    return RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True,
        input_key="question"
    )
