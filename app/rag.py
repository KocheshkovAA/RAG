from app.config import MAX_RESPONSE_LENGTH
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

TELEGRAM_PROMPT_TEMPLATE = """Ты AI-ассистент, отвечающий на вопросы про warhammer 40000. Формат ответа:

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

Вопрос: {input}

Сформулируй четкий ответ:"""

def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        TELEGRAM_PROMPT_TEMPLATE,
        partial_variables={"max_length": str(MAX_RESPONSE_LENGTH)}
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain
