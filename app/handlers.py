import asyncio
import logging
from aiogram import Dispatcher
from aiogram.types import Message, ContentType

from app.utils import send_typing_action, safe_send_error
from app.rag.rag_service import get_rag_answer

logger = logging.getLogger(__name__)


def register_handlers(dp: Dispatcher):
    @dp.message()
    async def handle_message(message: Message):
        try:
            if message.content_type != ContentType.TEXT:
                await message.answer("Я могу отвечать только на текстовые сообщения 📝")
                return

            if message.text.startswith("/"):
                await message.answer("Привет! Я бот по Warhammer 40k. Задай мне любой вопрос о вселенной.")
                return

            logger.info("Received message from %d: %s", message.from_user.id, message.text)

            # запускаем "печатает..."
            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(send_typing_action(message.bot, message.chat.id, stop_typing))

            response_chunks = await get_rag_answer(message.text)

            stop_typing.set()
            await typing_task

            for chunk in response_chunks:
                await message.answer(chunk)

            logger.info("Response sent to user %d", message.from_user.id)

        except Exception as e:
            logger.error("Error processing message: %s", str(e), exc_info=True)
            await safe_send_error(message, str(e))
