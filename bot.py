import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from app.formatter import TelegramMarkdownFormatter
from app.loader import load_and_split_documents
from app.embedder import build_or_load_vectorstore
from app.llm import get_llm
from app.rag import build_rag_chain
from app.config import CHROMA_PERSIST_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Initialize vectorstore
logger.info("Initializing vectorstore...")
if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
    logger.info("Loading existing vectorstore from %s", CHROMA_PERSIST_DIR)
    vectorstore = build_or_load_vectorstore([])
else:
    logger.info("Creating new vectorstore")
    chunks = load_and_split_documents()
    vectorstore = build_or_load_vectorstore(chunks)
    logger.info("Vectorstore created and persisted at %s", CHROMA_PERSIST_DIR)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = get_llm()
rag_chain = build_rag_chain(llm, retriever)
logger.info("RAG chain initialized successfully")

# Initialize bot
logger.info("Initializing Telegram bot...")
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
        logger.debug("RAG response: %s", raw_response)
        
        formatted_response = TelegramMarkdownFormatter.format(raw_response)
        await message.answer(formatted_response)
        logger.info("Response sent to user %d", message.from_user.id)
        
    except Exception as e:
        logger.error("Error processing message: %s", str(e), exc_info=True)
        error_msg = TelegramMarkdownFormatter.format(f"🚫 Error: {str(e)}")
        await message.answer(error_msg)

async def main():
    logger.info("Starting bot...")
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Webhook deleted, starting polling")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        logger.info("Launching bot application")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical("Fatal error: %s", str(e), exc_info=True)