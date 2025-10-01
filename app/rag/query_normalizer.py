import logging
import json
from langchain.prompts import PromptTemplate
from langchain_gigachat.chat_models import GigaChat
from langchain_core.utils.json import parse_json_markdown
from app.config import GIGA_KEY

logger = logging.getLogger(__name__)

# Инициализация модели
giga = GigaChat(
    credentials=GIGA_KEY,
    verify_ssl_certs=False
)

split_prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
Ты эксперт по извлечению сущностей для LLM.

Задача:
1. Если вопрос сложный — разбей его на несколько (2–3) логичных под-вопросов.
2. Если вопрос простой и самодостаточный — оставь его одним под-вопросом.
3. Выдели все имена, термины, события и важные сущности **только для исходного вопроса**. Используй именительный падеж.
4. Под-вопросы формулируй компактно.

Вход: Где сражались Абаддон с Жиллиманом?
Выход — строго JSON:
{{
  "entities": ["Абаддон", "Жиллиман"],
  "questions": [
    {{
      "text": "Где сражался Абаддон?"
    }},
    {{
      "text": "Где сражался Жиллиман?"
    }}
  ]
}}

Вопрос пользователя:
{question}
"""
)

def split_and_extract_entities(user_question: str) -> dict:
    """
    Принимает вопрос, возвращает под-вопросы и список сущностей для исходного вопроса.
    Всегда возвращает словарь.
    """
    prompt = split_prompt_template.format(question=user_question)

    try:
        raw_response = giga.invoke(prompt).content
    except Exception as e:
        logger.error(f"Ошибка при запросе к LLM: {e}")
        return {"entities": [], "questions": []}

    try:
        parsed = parse_json_markdown(raw_response)
    except Exception as e:
        logger.warning(f"Ошибка парсинга JSON: {e}")
        parsed = {"entities": [], "questions": []}

    # Если parsed оказался None или не словарём — приводим к словарю
    if not isinstance(parsed, dict):
        parsed = {"entities": [], "questions": []}

    # Очистка текста под-вопросов и сущностей
    parsed["entities"] = [e.strip() for e in parsed.get("entities", []) if e.strip()]
    if "questions" in parsed:
        for q in parsed["questions"]:
            q["text"] = q.get("text", "").strip()

    return parsed


if __name__ == "__main__":
    question = "Кто такой Абаддон и какие сражения он возглавлял в Готической войне?"
    result = split_and_extract_entities(question)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
