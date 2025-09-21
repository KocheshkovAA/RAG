import os
import re
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties

from app.formatter import TelegramMarkdownFormatter
from app.loader import DatabaseTextLoader
from app.embedder import build_or_load_vectorstore
from app.llm import get_llm
from app.rag import build_rag_chain
from app.config import CHROMA_PERSIST_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
for name in ["pymorphy2", "sentence_transformers", "app.NER"]:
    logging.getLogger(name).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
    logger.info("Loading existing vectorstore from %s", CHROMA_PERSIST_DIR)
    retriever = build_or_load_vectorstore([])
else:
    logger.info("Creating new vectorstore")
    loader = DatabaseTextLoader()
    chunks, _ = loader.load_and_split_documents()
    retriever = build_or_load_vectorstore(chunks)
    logger.info("Vectorstore created and persisted at %s", CHROMA_PERSIST_DIR)

llm = get_llm()
rag_chain = build_rag_chain(llm, retriever)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2)
)
dp = Dispatcher(storage=MemoryStorage())

@dp.message()
async def handle_message(message: Message):
    try:
        logger.info("Received message from user %d: %s", message.from_user.id, message.text)

        result = rag_chain.invoke({"input": message.text})
        raw_response = result.get("answer", "Failed to get answer")
        source_documents = result.get("context", [])

        if source_documents:
            logger.info("Source chunks text:")
            for i, doc in enumerate(source_documents, 1):
                print(f"\n--- Chunk {i} ---\n{doc.page_content}\n--- End Chunk {i} ---\n")


        # собираем уникальные источники в порядке появления
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

        sources_text = ""
        if unique_sources:
            sources_text = "\n\nИспользованные источники:\n"
            sources_text += "\n".join(
                f"{i}. [{title}]({source})"
                for i, (title, source) in enumerate(unique_sources, 1)
            )

        response_chunks = TelegramMarkdownFormatter.format_into_chunks(
            raw_response + sources_text
        )
        for chunk in response_chunks:
            await message.answer(chunk)

        logger.info("Response sent to user %d", message.from_user.id)

    except Exception as e:
        logger.error("Error processing message: %s", str(e), exc_info=True)
        error_msg = TelegramMarkdownFormatter.format(f"🚫 Error: {str(e)}")
        await message.answer(error_msg)


async def main():
    logger.info("Starting bot...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical("Fatal error: %s", str(e), exc_info=True)
