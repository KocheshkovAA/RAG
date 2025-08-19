from app.config import MAX_RESPONSE_LENGTH
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

TELEGRAM_PROMPT_TEMPLATE = """Ты эксперт по вселенной Warhammer 40,000.

Используй приведённый контекст, чтобы ответить на вопрос. Отвечай на русском языке.

- Если контекст пустой, то отвечай, что не знаешь ответа.
- Максимальная длина ответа: {max_length} символов
- Не упоминай в ответе вопрос и контекст

Контекст:
{context}

Вопрос:
{input}

Ответ:
"""

def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(
        TELEGRAM_PROMPT_TEMPLATE,
        partial_variables={"max_length": str(MAX_RESPONSE_LENGTH)}
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain
