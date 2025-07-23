import os
import asyncio
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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Инициализация RAG системы
if CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
    vectorstore = build_or_load_vectorstore([])
else:
    chunks = load_and_split_documents()
    vectorstore = build_or_load_vectorstore(chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = get_llm()
rag_chain = build_rag_chain(llm, retriever)

# Инициализация бота с MarkdownV2
bot = Bot(
    token=TELEGRAM_TOKEN, 
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2)
)
dp = Dispatcher(storage=MemoryStorage())

@dp.message()
async def handle_message(message: Message):
    try:
        # Получаем ответ от RAG системы
        result = rag_chain.invoke({"question": message.text})
        raw_response = result.get("result", "Не удалось получить ответ")
        
        formatted_response = TelegramMarkdownFormatter.format(raw_response)
        print(formatted_response)
        # Отправляем ответ
        await message.answer(formatted_response)
        
    except Exception as e:
        error_msg = TelegramMarkdownFormatter.format(f"🚫 Ошибка: {str(e)}")
        await message.answer(
            error_msg
        )

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())