import re
from typing import List, Tuple
from app.config import MAX_MESSAGE_LENGTH

class TelegramMarkdownFormatter:
    """Форматирование текста для Telegram MarkdownV2"""

    _ESCAPE_CHARS = '_[]()~`>#+-=|{}.!'
    _CODE_BLOCK_PATTERN = r'```(.*?)```'
    _code_blocks: List[str] = []

    @classmethod
    def format(cls, text: str) -> str:
        """Форматирует текст и возвращает одну строку"""
        if not text:
            return text

        text = cls._preserve_code_blocks(text)
        formatted = cls._process_text(text)
        formatted = cls._restore_code_blocks(formatted)
        return formatted

    @classmethod
    def format_into_chunks(cls, text: str, max_length: int = MAX_MESSAGE_LENGTH) -> List[str]:
        """Форматирует текст и разбивает его на чанки для Telegram так,
        чтобы не резать ссылки и другие сущности"""
        formatted = cls.format(text)
        chunks = []

        while formatted:
            if len(formatted) <= max_length:
                chunks.append(formatted)
                break

            # Берём кусок до max_length
            cut = formatted[:max_length]

            # Проверяем, не разорвали ли ссылку [text](url)
            last_open = cut.rfind("[")
            last_close = cut.rfind(")")

            if last_open != -1 and (last_close == -1 or last_close < last_open):
                # Нашли начало ссылки, но не конец — обрезаем до last_open
                cut = cut[:last_open]

            # Если вдруг отрезали слишком мало (например, из-за длинной ссылки),
            # тогда лучше взять полный URL и перенести в следующий чанк
            if not cut.strip():
                # Находим конец ссылки за пределами max_length
                end_link = formatted.find(")", max_length)
                if end_link != -1:
                    cut = formatted[:end_link+1]
                else:
                    cut = formatted[:max_length]  # fallback

            chunks.append(cut)
            formatted = formatted[len(cut):]

        return chunks

    
    @classmethod
    def _preserve_code_blocks(cls, text: str) -> str:
        """Сохраняет код-блоки перед обработкой"""
        cls._code_blocks = []
        def replace_code(match):
            cls._code_blocks.append(match.group(0))
            return f'__CODE_BLOCK_{len(cls._code_blocks)-1}__'
            
        return re.sub(cls._CODE_BLOCK_PATTERN, replace_code, text, flags=re.DOTALL)
    
    @classmethod
    def _restore_code_blocks(cls, text: str) -> str:
        """Восстанавливает код-блоки после обработки"""
        for i, code in enumerate(cls._code_blocks):
            text = text.replace(f'__CODE_BLOCK_{i}__', code)
        return text
    
    @classmethod
    def _process_text(cls, text: str) -> str:
        """Обработка текста (без код-блоков)"""
        formatted_text = []
        i = 0
        n = len(text)
        
        while i < n:
            # Пропускаем временные метки код-блоков
            if text.startswith('__CODE_BLOCK_', i):
                end = text.find('__', i + 13)
                if end != -1:
                    formatted_text.append(text[i:end+2])
                    i = end + 2
                    continue
            
            # Обработка ссылок [текст](url)
            if text[i] == '[':
                i, link_part = cls._process_link(text, i, n)
                if link_part:
                    formatted_text.append(link_part)
                    continue
            
            # Обработка заголовков (начинаются с #)
            if text[i] == '#':
                i, header_part = cls._process_header(text, i, n)
                if header_part:
                    formatted_text.append(header_part)
                    continue
            
            # Обработка жирного текста **text**
            if i + 1 < n and text[i] == '*' and text[i+1] == '*':
                i, bold_part = cls._process_bold(text, i, n)
                if bold_part:
                    formatted_text.append(bold_part)
                    continue
            
            # Экранирование обычных символов
            char = text[i]
            if char in cls._ESCAPE_CHARS:
                formatted_text.append(f'\\{char}')
            else:
                formatted_text.append(char)
            i += 1
        
        return ''.join(formatted_text)
    
    @classmethod
    def _process_link(cls, text: str, i: int, n: int) -> Tuple[int, str]:
        """Обработка ссылки [текст](url)"""
        j = i + 1
        while j < n and text[j] != ']':
            j += 1
        
        if j < n and text[j] == ']' and j + 1 < n and text[j+1] == '(':
            k = j + 2
            while k < n and text[k] != ')':
                k += 1
            
            if k < n and text[k] == ')':
                link_text = text[i+1:j]
                url = text[j+2:k]
                
                escaped_link_text = []
                for char in link_text:
                    if char in cls._ESCAPE_CHARS:
                        escaped_link_text.append(f'\\{char}')
                    else:
                        escaped_link_text.append(char)
                
                return k + 1, f'[{"".join(escaped_link_text)}]({url})'
        
        return i, ''
    
    @classmethod
    def _process_header(cls, text: str, i: int, n: int) -> Tuple[int, str]:
        """Обработка заголовка (# Header)"""
        header_level = 0
        start = i
        while i < n and text[i] == '#':
            header_level += 1
            i += 1
        
        while i < n and text[i] == ' ':
            i += 1
        
        j = i
        while j < n and text[j] != '\n':
            j += 1
        
        header_text = []
        k = i
        while k < j:
            char = text[k]
            if char in cls._ESCAPE_CHARS:
                header_text.append(f'\\{char}')
            else:
                header_text.append(char)
            k += 1
        
        if header_text:
            return j, f'*{"".join(header_text)}*'
        
        return start, ''
    
    @classmethod
    def _process_bold(cls, text: str, i: int, n: int) -> Tuple[int, str]:
        """Обработка жирного текста (**bold**)"""
        j = i + 2
        while j < n and not (text[j] == '*' and j + 1 < n and text[j+1] == '*'):
            j += 1
        
        if j + 1 < n and text[j] == '*' and text[j+1] == '*':
            bold_text = []
            k = i + 2
            while k < j:
                char = text[k]
                if char in cls._ESCAPE_CHARS:
                    bold_text.append(f'\\{char}')
                else:
                    bold_text.append(char)
                k += 1
            
            return j + 2, f'*{"".join(bold_text)}*'
        
        return i, ''
    
    @classmethod
    def _truncate(cls, text: str) -> str:
        """Обрезка длинных сообщений"""
        if len(text) > MAX_MESSAGE_LENGTH:
            return text[:MAX_MESSAGE_LENGTH-50] + "...\n\n[ответ сокращен]"
        return text