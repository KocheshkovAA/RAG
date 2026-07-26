import httpx
import logging
from langfuse import observe

from app.core.config import settings
from app.core.health import lightrag_breaker
from app.core.resilience import call_with_circuit

logger = logging.getLogger(__name__)


class LightRAGClient:
    def __init__(self):
        self.base_url = settings.LIGHTRAG_BASE_URL
        self.timeout = httpx.Timeout(settings.LIGHTRAG_TIMEOUT_SEC, connect=5.0)

    @observe(name="LightRAG Query")
    async def query(self, question: str, mode: str = "mix") -> dict:
        payload = {
            "query": question,
            "mode": mode,
            "enable_rerank": True,
            "top_k": 40,
            # Без этого references приходят с content=null (только file_path/id) —
            # faithfulness-гвард и eval-харнесс остаются без реального текста чанков.
            "include_chunk_content": True,
        }

        async def _call() -> dict:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/query",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "answer": data.get("answer", data.get("response", "Нет ответа")),
                    "sources": data.get("references", []),
                    "mode": f"lightrag-{mode}",
                    "degraded": [],
                    "cached": False,
                }

        def _unavailable() -> dict:
            return {
                "answer": None,
                "sources": [],
                "mode": "lightrag-unavailable",
                "degraded": ["lightrag"],
                "cached": False,
                "_fallback_to_vector": True,
            }

        try:
            result, degraded = await call_with_circuit(
                lightrag_breaker,
                _call,
                fallback=_unavailable,
            )
            if degraded and result.get("_fallback_to_vector"):
                return result
            if degraded:
                result = dict(result)
                result.setdefault("degraded", []).append("lightrag")
            return result
        except Exception as e:
            logger.error("LightRAG error: %s", e)
            return _unavailable()
