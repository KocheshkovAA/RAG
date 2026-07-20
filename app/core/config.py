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
    RETRIEVAL_MIN_SCORE: float = Field(default=0.55)
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

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    QDRANT_COLLECTION: str = "warhammer_wiki"
    LIGHTRAG_BASE_URL: str = "http://lightrag:9621"
    LIGHTRAG_TIMEOUT_SEC: float = Field(default=60.0)

settings = Settings()
