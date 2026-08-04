from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG API"
    
    # --- Внутренние URL для Docker-сети ---
    QDRANT_URL: str = Field(default="http://qdrant:6333")
    TEI_URL: str = Field(default="http://tei:80")
    # "tei" — нативный TEI-эндпоинт /embed ({"inputs": [...]} -> [[float]]).
    # "openai" — OpenAI-совместимый /v1/embeddings (для vLLM --runner pooling,
    # см. docker-compose.yml vllm-biencoder — GPU-обход бага TEI на GeForce).
    TEI_API_FORMAT: str = Field(default="tei")
    REDIS_URL: str = Field(default="redis://redis:6379/0")
    
    # --- RERANKER ---
    RERANKER_ENABLED: bool = Field(default=True)
    RERANKER_TOP_K: int = 5
    RERANKER_URL: str = Field(default="http://vllm-reranker:8000")
    RERANKER_MODEL: str = Field(default="bge-reranker-v2-m3")
    RERANKER_BATCH_SIZE: int = 2
    RERANK_MIN_SCORE: float = Field(default=0.55)
    RERANKER_TIMEOUT_SEC: float = Field(default=15.0)

    # --- GUARDRAILS ---
    # 0.55 отсекает «случайный» top-k (~0.50 на оффтопе вроде Губки Боба)
    # Калибровано под rerank_score (сигмоида reranker'а)
    RETRIEVAL_MIN_SCORE: float = Field(default=0.55)
    # Без reranker'а (circuit open / RERANKER_ENABLED=False) в metadata остаётся
    # только hybrid_score — это RRF-скор по рангу (0.5/0.333/0.25...), а не
    # семантическая близость: top-1 хит скорим ~0.5 независимо от того,
    # релевантен он вопросу или нет. Поэтому тот же порог 0.55 в деградации
    # отсекает вообще всё, включая нормальные вопросы по теме. Порог ниже —
    # это просто пол «хоть что-то нашлось», а не confidence gate; реальная
    # защита от оффтопа в этом режиме — обязательная faithfulness-проверка
    # (AnswerFaithfulnessGuard.should_verify всегда True при degraded=True).
    RETRIEVAL_MIN_SCORE_NO_RERANK: float = Field(default=0.2)
    FAITHFULNESS_CHECK_ENABLED: bool = Field(default=True)
    FAITHFULNESS_MIN_SCORE: float = Field(default=0.7)
    # Серая зона 0.55–0.80: генерация + LLM-верификация; выше 0.80 — можно пропустить verify
    FAITHFULNESS_SKIP_ABOVE: float = Field(default=0.80)
    INSUFFICIENT_INFO_MESSAGE: str = Field(
        default=(
            "В данных инфо-хранилища недостаточно надёжных сведений по данному запросу. "
            "Не могу подтвердить факты из доступного контекста."
        )
    )

    # --- RATE LIMITING ---
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    # Формат slowapi: "<число>/<second|minute|hour|day>"
    RATE_LIMIT_ASK: str = Field(default="20/minute")
    # /debate дороже /ask (несколько персон = несколько LLM-вызовов на один HTTP-запрос),
    # поэтому лимит ниже, а не переиспользует RATE_LIMIT_ASK.
    RATE_LIMIT_DEBATE: str = Field(default="5/minute")

    # --- CACHE / RESILIENCE ---
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_TTL_ANSWER_SEC: int = Field(default=600)
    CACHE_TTL_DOCS_SEC: int = Field(default=300)
    REQUEST_TIMEOUT_SEC: float = Field(default=45.0)
    CIRCUIT_FAILURE_THRESHOLD: int = Field(default=5)
    CIRCUIT_RECOVERY_SEC: float = Field(default=30.0)
    TEI_TIMEOUT_SEC: float = Field(default=15.0)
    QDRANT_TIMEOUT_SEC: float = Field(default=10.0)

    QUERY_OPTIMIZER_ENABLED: bool = False
    
    DATA_PATH: str = "data/processed/processed_chunks.jsonl"
    VECTOR_SIZE: int = 1024
    
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="http://langfuse:3000")
    # false = не слать трейсы (удобно при make up без debug-профиля)
    LANGFUSE_ENABLE_TRACE: bool = Field(default=True)
    LLM_PROVIDER: str = "gigachat"

    DATASET_PATH: str = "data/eval/warhammer40k_eval_60q.jsonl"
    
    GIGACHAT_CREDENTIALS: str = Field(default="")
    GIGACHAT_MODEL_NAME: str = Field(default="Gigachat")

    OPENROUTER_API_KEY: str = Field(default="")
    LLM_MODEL_NAME: str = "qwen/qwen-2.5-72b-instruct"

    # RouterAI (routerai.ru) — OpenAI-совместимый агрегатор, доступный из России
    # в отличие от OpenRouter. Платный (нет бесплатного тира), но недорогой на
    # масштабе eval-прогонов (десятки вопросов).
    ROUTERAI_API_KEY: str = Field(default="")
    ROUTERAI_MODEL_NAME: str = Field(default="deepseek/deepseek-v4-pro")
    # deepseek-v4-pro — reasoning-модель (генерирует "мышление" перед ответом),
    # без явного таймаута ChatOpenAI может ждать неопределённо долго.
    ROUTERAI_TIMEOUT_SEC: float = Field(default=60.0)

    # Self-hosted vLLM на этом же хосте (см. docker-compose.yml, сервис vllm-llm) — учебный
    # эксперимент по self-hosted инференсу (training/vllm_inference_deepdive.ipynb). Не в
    # core-стеке, поднимается отдельно. Порт 8003 — то, что реально слушает vllm-llm.
    VLLM_LOCAL_BASE_URL: str = Field(default="http://vllm-llm:8000/v1")
    VLLM_LOCAL_MODEL_NAME: str = Field(default="Qwen/Qwen3-4B-AWQ")
    # Фиксированный потолок вывода — независимый от длины промпта (см. VLLM_LOCAL_MAX_TOKENS
    # в app/core/llm.py). 500 — с запасом под наблюдаемую длину полных связных ответов (~400-530
    # токенов на живых прогонах 26.07), с местом под ~2500 токенов входа в max-model-len=3072.
    VLLM_LOCAL_MAX_TOKENS: int = Field(default=500)

    # Модель/провайдер по задаче (router/generation/faithfulness) — пустая строка
    # значит "наследовать LLM_PROVIDER/{GIGACHAT,LLM}_MODEL_NAME по умолчанию".
    # Позволяет назначить более сильную или локальную модель под конкретную задачу
    # без изменения кода. LightRAG (граф) настраивается отдельно через свой .env
    # и сюда не относится — сейчас там уже GigaChat-2-Pro.
    ROUTER_LLM_PROVIDER: str = Field(default="")
    ROUTER_LLM_MODEL: str = Field(default="")

    GENERATION_LLM_PROVIDER: str = Field(default="")
    GENERATION_LLM_MODEL: str = Field(default="")

    FAITHFULNESS_LLM_PROVIDER: str = Field(default="")
    FAITHFULNESS_LLM_MODEL: str = Field(default="")

    AGENTIC_LLM_PROVIDER: str = Field(default="")
    AGENTIC_LLM_MODEL: str = Field(default="")

    # --- AGENTIC ROUTE ---
    # По умолчанию выключено: роутер сам не выбирает agentic, пока не включим явно.
    AGENTIC_ROUTE_ENABLED: bool = Field(default=False)
    # Сколько раундов tool-calling разрешено модели (search_knowledge_base),
    # прежде чем принудительно остановить цикл независимо от того, что хочет
    # модель. Условие остановки в рамках раунда — тот же RetrievalGate, что у
    # vector-маршрута (не отдельный порог, чтобы не разъезжались калибровки).
    AGENTIC_MAX_ITERATIONS: int = Field(default=3)
    # Потолок параллельных вызовов search_knowledge_base за один ход модели —
    # защита от патологического "вызвать инструмент 20 раз за раз".
    AGENTIC_MAX_TOOL_CALLS_PER_ROUND: int = Field(default=4)

    PERSONA_LLM_PROVIDER: str = Field(default="")
    PERSONA_LLM_MODEL: str = Field(default="")

    # --- JUDGE (LLM-as-judge для eval, app/eval/evaluate_generation.py::WarJudge) ---
    # В отличие от остальных ролей выше, тут дефолты НЕ пустые: судья по
    # умолчанию обязан быть другой моделью/провайдером, чем generation
    # (там по умолчанию gigachat) и чем graph-маршрут (LightRAG использует
    # GigaChat-2-Pro напрямую, минуя эту настройку) — иначе eval рискует
    # оценивать сам себя (self-bias), особенно вредно именно на graph-eval.
    # OpenRouter недоступен из РФ — судья на routerai.ru (OpenAI-совместимый
    # агрегатор, платный, но недорогой на масштабе eval-прогонов).
    # deepseek-chat (V3, не reasoning) вместо deepseek-v4-pro: v4-pro генерирует
    # "мышление" перед каждым ответом (поле reasoning) — на масштабе eval-прогона
    # (десятки вопросов) это заметно замедляет прогон без явной пользы для
    # задачи judge; deepseek-chat отвечает за ~1-2с вместо заметно дольше.
    JUDGE_LLM_PROVIDER: str = Field(default="routerai")
    JUDGE_LLM_MODEL: str = Field(default="deepseek/deepseek-chat")

    # --- PERSONA DEBATE (MVP) ---
    # Отдельный explicit-эндпоинт (/v1/debate), роутер его не выбирает и не
    # знает о нём вообще — это operational kill-switch для дорогой фичи
    # (до 2×N последовательных LLM-вызовов на раунд), а не gate для авто-роутинга.
    PERSONA_DEBATE_ENABLED: bool = Field(default=True)

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    QDRANT_COLLECTION: str = "warhammer_wiki"
    LIGHTRAG_BASE_URL: str = "http://lightrag:9621"
    LIGHTRAG_TIMEOUT_SEC: float = Field(default=60.0)

settings = Settings()
