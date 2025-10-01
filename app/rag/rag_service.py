import logging
import asyncio
from app.config import CHROMA_PERSIST_DIR
from app.chunks_loader import DatabaseTextLoader
from app.rag.retriever import build_or_load_vectorstore
from app.rag.llm import get_llm
from app.rag.rag_chain import build_rag_chain
from app.formatter import TelegramMarkdownFormatter

logger = logging.getLogger(__name__)

# Инициализация
if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
    logger.info("Loading existing vectorstore from %s", CHROMA_PERSIST_DIR)
    retriever = build_or_load_vectorstore([])
else:
    logger.info("Creating new vectorstore")
    loader = DatabaseTextLoader()
    chunks, _ = loader.load_and_split_documents()
    retriever = build_or_load_vectorstore(chunks)
    logger.info("Vectorstore created at %s", CHROMA_PERSIST_DIR)

llm = get_llm()
rag_chain = build_rag_chain(llm, retriever)


def format_sources(source_documents):
    unique_sources = []
    seen = set()

    for doc in source_documents:
        title = doc.metadata.get("document_title", doc.metadata.get("title", "Без названия"))
        source = doc.metadata.get("source")
        if not source:
            continue

        key = (title, source)
        if key not in seen:
            seen.add(key)
            unique_sources.append(key)

    if not unique_sources:
        return ""

    sources_text = "\n\nИспользованные источники:\n"
    sources_text += "\n".join(
        f"{i}. [{title}]({source})" for i, (title, source) in enumerate(unique_sources, 1)
    )
    return sources_text


async def get_rag_answer(user_input: str):
    result = await asyncio.to_thread(rag_chain.invoke, {"input": user_input})
    raw_response = result.get("answer", "Не удалось получить ответ")
    sources = format_sources(result.get("context", []))
    return TelegramMarkdownFormatter.format_into_chunks(raw_response + sources)
