import asyncio
import logging
from aiogram import Bot
from app.formatter import TelegramMarkdownFormatter

logger = logging.getLogger(__name__)
bot: Bot = None  # будет установлен при запуске

async def send_typing_action(bot: Bot, chat_id: int, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id, action="typing")
        except Exception as e:
            logger.warning("Failed to send typing action: %s", e)
        await asyncio.sleep(5)



async def safe_send_error(message, error: str):
    error_msg = TelegramMarkdownFormatter.format(f"🚫 Error: {error}")
    await message.answer(error_msg)
