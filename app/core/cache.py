"""Redis cache with graceful degradation when Redis is unavailable."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheClient:
    def __init__(self) -> None:
        self.enabled = settings.CACHE_ENABLED
        self.ttl_answer = settings.CACHE_TTL_ANSWER_SEC
        self.ttl_docs = settings.CACHE_TTL_DOCS_SEC
        self._redis = None
        self._available = False

        if not self.enabled:
            return

        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            self._available = True
        except Exception as e:
            logger.warning("Redis init failed, cache disabled: %s", e)
            self._available = False

    @staticmethod
    def _key(prefix: str, text: str) -> str:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
        return f"rag:{prefix}:{digest}"

    async def ping(self) -> bool:
        if not self._redis:
            return False
        try:
            return bool(await self._redis.ping())
        except Exception:
            self._available = False
            return False

    async def get_json(self, prefix: str, text: str) -> Optional[dict[str, Any]]:
        if not self.enabled or not self._available or not self._redis:
            return None
        try:
            raw = await self._redis.get(self._key(prefix, text))
            if not raw:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("Cache get failed: %s", e)
            self._available = False
            return None

    async def set_json(
        self, prefix: str, text: str, value: dict[str, Any], ttl: int
    ) -> None:
        if not self.enabled or not self._available or not self._redis:
            return
        try:
            await self._redis.set(
                self._key(prefix, text),
                json.dumps(value, ensure_ascii=False),
                ex=ttl,
            )
        except Exception as e:
            logger.warning("Cache set failed: %s", e)
            self._available = False

    async def get_answer(self, question: str, route: str = "vector") -> Optional[dict[str, Any]]:
        return await self.get_json(f"answer:{route}", question)

    async def set_answer(self, question: str, payload: dict[str, Any], route: str = "vector") -> None:
        await self.set_json(f"answer:{route}", question, payload, self.ttl_answer)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()


cache_client = CacheClient()
