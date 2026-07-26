from langchain_core.language_models.chat_models import BaseChatModel
from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI
from app.core.config import settings

# Роль -> (поле провайдера, поле модели) в Settings. Пустое значение поля
# означает "наследовать" глобальный LLM_PROVIDER/модель по умолчанию.
_ROLE_SETTINGS = {
    "router": ("ROUTER_LLM_PROVIDER", "ROUTER_LLM_MODEL"),
    "generation": ("GENERATION_LLM_PROVIDER", "GENERATION_LLM_MODEL"),
    "faithfulness": ("FAITHFULNESS_LLM_PROVIDER", "FAITHFULNESS_LLM_MODEL"),
    "agentic": ("AGENTIC_LLM_PROVIDER", "AGENTIC_LLM_MODEL"),
    "persona": ("PERSONA_LLM_PROVIDER", "PERSONA_LLM_MODEL"),
    # judge — единственная роль, где пустое значение НЕ означает "наследовать
    # LLM_PROVIDER/generation": судья по умолчанию обязан быть другой моделью,
    # чем generation, иначе eval рискует оценивать сам себя (self-bias). Оба
    # поля ниже имеют собственные непустые дефолты в Settings.
    "judge": ("JUDGE_LLM_PROVIDER", "JUDGE_LLM_MODEL"),
}


def _resolve_provider_model(role: str | None, model_name: str | None) -> tuple[str, str]:
    """Провайдер + итоговая модель для роли — без создания клиента.
    Единственный источник правды для этого резолвинга: и get_llm(), и
    resolve_role_config() (инспектор для логирования конфига eval-прогонов)
    используют именно эту функцию, чтобы не разъезжались две копии логики."""
    provider = settings.LLM_PROVIDER

    if role in _ROLE_SETTINGS:
        provider_field, model_field = _ROLE_SETTINGS[role]
        provider = getattr(settings, provider_field) or provider
        model_name = model_name or getattr(settings, model_field) or None

    if provider == "gigachat":
        return provider, model_name or settings.GIGACHAT_MODEL_NAME
    elif provider == "openrouter":
        return provider, model_name or settings.LLM_MODEL_NAME
    elif provider == "routerai":
        return provider, model_name or settings.ROUTERAI_MODEL_NAME
    elif provider == "vllm-local":
        return provider, model_name or settings.VLLM_LOCAL_MODEL_NAME

    raise ValueError(f"Unknown LLM provider: {provider}")


def resolve_role_config(role: str) -> dict:
    """Провайдер/модель, которые реально будут использованы для этой роли,
    без создания клиента — для записи в конфиг eval-прогона (см.
    app/eval/evaluate_routes.py), чтобы потом можно было сравнить "было/стало"
    при смене провайдера/промптов."""
    provider, model = _resolve_provider_model(role, None)
    return {"provider": provider, "model": model}


class LLMFactory:
    @staticmethod
    def get_llm(
        temperature: float = 0.12,
        model_name: str | None = None,
        role: str | None = None,
    ) -> BaseChatModel:
        provider, target_model = _resolve_provider_model(role, model_name)

        if provider == "gigachat":
            return GigaChat(
                credentials=settings.GIGACHAT_CREDENTIALS,
                verify_ssl_certs=False,
                model=target_model,
                temperature=temperature,
            )

        elif provider == "openrouter":
            return ChatOpenAI(
                model=target_model,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
            )

        elif provider == "routerai":
            return ChatOpenAI(
                model=target_model,
                openai_api_key=settings.ROUTERAI_API_KEY,
                openai_api_base="https://routerai.ru/api/v1",
                temperature=temperature,
                timeout=settings.ROUTERAI_TIMEOUT_SEC,
            )

        elif provider == "vllm-local":
            # Self-hosted vLLM на этом же хосте (см. docker-compose.yml, сервис vllm-llm) —
            # OpenAI-совместимый эндпоинт, ключ не нужен, но ChatOpenAI требует непустую строку.
            # enable_thinking=False обязателен: Qwen3 по умолчанию генерирует <think>...</think>
            # перед ответом — на живых запросах (26.07) это иногда съедало весь max_tokens ещё
            # внутри блока рассуждений, оставляя ответ пустым/обрезанным непредсказуемо (не каждый
            # раз одинаково на один и тот же вопрос). Отключаем reasoning для этой роли явно, а не
            # полагаемся на дефолт модели.
            return ChatOpenAI(
                model=target_model,
                openai_api_key="not-needed",
                openai_api_base=settings.VLLM_LOCAL_BASE_URL,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                # Без явного max_tokens модель генерирует, пока не упрётся в ОБЩИЙ бюджет
                # --max-model-len (вход+выход вместе) — при длинном retrieved-контексте на
                # вывод оставалось мало места, и связные ответы обрывались непредсказуемо
                # (см. живые прогоны 26.07). Фиксированный потолок делает бюджет вывода
                # постоянным независимо от длины промпта.
                max_tokens=settings.VLLM_LOCAL_MAX_TOKENS,
            )

        raise ValueError(f"Unknown LLM provider: {provider}")

llm_factory = LLMFactory()