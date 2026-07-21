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
}


class LLMFactory:
    @staticmethod
    def get_llm(
        temperature: float = 0.12,
        model_name: str | None = None,
        role: str | None = None,
    ) -> BaseChatModel:
        provider = settings.LLM_PROVIDER

        if role in _ROLE_SETTINGS:
            provider_field, model_field = _ROLE_SETTINGS[role]
            provider = getattr(settings, provider_field) or provider
            model_name = model_name or getattr(settings, model_field) or None

        if provider == "gigachat":
            target_model = model_name or settings.GIGACHAT_MODEL_NAME 
            
            return GigaChat(
                credentials=settings.GIGACHAT_CREDENTIALS,
                verify_ssl_certs=False,
                model=target_model,
                temperature=temperature,
            )
        
        elif provider == "openrouter":
            target_model = model_name or settings.LLM_MODEL_NAME
            
            return ChatOpenAI(
                model=target_model,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=temperature,
            )

        raise ValueError(f"Unknown LLM provider: {provider}")

llm_factory = LLMFactory()