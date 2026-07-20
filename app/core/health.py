"""Dependency health probes for readiness checks."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from qdrant_client import QdrantClient

from app.core.cache import cache_client
from app.core.config import settings
from app.core.resilience import CircuitBreaker

logger = logging.getLogger(__name__)

# Shared breakers (импортируются из сервисов)
reranker_breaker = CircuitBreaker(
    name="reranker",
    failure_threshold=settings.CIRCUIT_FAILURE_THRESHOLD,
    recovery_timeout_sec=settings.CIRCUIT_RECOVERY_SEC,
)
lightrag_breaker = CircuitBreaker(
    name="lightrag",
    failure_threshold=settings.CIRCUIT_FAILURE_THRESHOLD,
    recovery_timeout_sec=settings.CIRCUIT_RECOVERY_SEC,
)
tei_breaker = CircuitBreaker(
    name="tei",
    failure_threshold=settings.CIRCUIT_FAILURE_THRESHOLD,
    recovery_timeout_sec=settings.CIRCUIT_RECOVERY_SEC,
)


async def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code < 500
    except Exception:
        return False


async def check_dependencies() -> dict[str, Any]:
    qdrant_ok = False
    try:
        client = QdrantClient(url=settings.QDRANT_URL, timeout=2.0)
        client.get_collections()
        qdrant_ok = True
    except Exception as e:
        logger.debug("Qdrant health failed: %s", e)

    tei_ok = await _http_ok(f"{settings.TEI_URL.rstrip('/')}/health")
    rerank_ok = await _http_ok(f"{settings.RERANKER_URL.rstrip('/')}/health")
    lightrag_ok = await _http_ok(f"{settings.LIGHTRAG_BASE_URL.rstrip('/')}/health")
    redis_ok = await cache_client.ping() if settings.CACHE_ENABLED else True

    langfuse_ok = None
    if settings.LANGFUSE_ENABLE_TRACE and settings.LANGFUSE_PUBLIC_KEY:
        langfuse_ok = await _http_ok(f"{settings.LANGFUSE_HOST.rstrip('/')}/api/public/health")
        if langfuse_ok is False:
            # некоторые сборки отдают /health на корне
            langfuse_ok = await _http_ok(f"{settings.LANGFUSE_HOST.rstrip('/')}/")

    checks = {
        "qdrant": qdrant_ok,
        "tei": tei_ok,
        "reranker": rerank_ok,
        "lightrag": lightrag_ok,
        "redis": redis_ok,
        "langfuse": langfuse_ok,
    }
    # Критичные для vector RAG: qdrant + tei. Reranker/LightRAG/Langfuse — не блокируют ready.
    critical_ok = qdrant_ok and tei_ok
    return {
        "ready": critical_ok,
        "checks": checks,
        "tracing_enabled": settings.LANGFUSE_ENABLE_TRACE,
        "circuits": {
            "reranker": reranker_breaker.snapshot(),
            "lightrag": lightrag_breaker.snapshot(),
            "tei": tei_breaker.snapshot(),
        },
    }
