from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG API"
    
    # --- Внутренние URL для Docker-сети ---
    QDRANT_URL: str = Field(default="http://qdrant:6333")
    TEI_URL: str = Field(default="http://tei:80")
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
    
    COLLECTION_NAME: str = "warhammer_wiki"
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
    # Сколько раз можно переформулировать запрос и повторить ретрив, прежде чем
    # сдаться. Условие остановки — тот же RetrievalGate, что у vector-маршрута
    # (не отдельный порог, чтобы не разъезжались калибровки).
    AGENTIC_MAX_ITERATIONS: int = Field(default=3)

    PERSONA_LLM_PROVIDER: str = Field(default="")
    PERSONA_LLM_MODEL: str = Field(default="")

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
