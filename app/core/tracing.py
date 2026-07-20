"""Langfuse tracing helpers: init, business scores, flush."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_enabled: Optional[bool] = None


def _keys_configured() -> bool:
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


def init_tracing() -> bool:
    """
    Вызывать на старте API.
    Если Langfuse недоступен / выключен — отключаем экспорт, чтобы не спамить лог.
    """
    global _enabled

    # SDK читает LANGFUSE_BASE_URL; у нас в .env исторически LANGFUSE_HOST
    if settings.LANGFUSE_HOST and not os.getenv("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = settings.LANGFUSE_HOST

    if not settings.LANGFUSE_ENABLE_TRACE or not _keys_configured():
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        _enabled = False
        logger.info("Langfuse tracing disabled (flag/keys)")
        return False

    try:
        from langfuse import get_client

        client = get_client()
        ok = bool(client.auth_check())
        _enabled = ok
        if ok:
            logger.info("Langfuse connected: %s", settings.LANGFUSE_HOST)
        else:
            os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
            logger.warning("Langfuse auth_check failed — tracing off")
        return ok
    except Exception as e:
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        _enabled = False
        logger.warning("Langfuse unreachable (%s) — tracing off", e)
        return False


def is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        return bool(settings.LANGFUSE_ENABLE_TRACE and _keys_configured())
    return _enabled


def flush() -> None:
    if not is_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        logger.debug("Langfuse flush failed: %s", e)


def score_ask_result(question: str, result: dict[str, Any]) -> None:
    """Пишет бизнес-метрики на текущий trace (для дашбордов Langfuse)."""
    if not is_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        guard = result.get("guardrail") or {}
        gate = guard.get("retrieval_gate") or {}
        faith = guard.get("faithfulness") or {}
        refused = bool(guard.get("refused"))
        latency_ms = int(result.get("latency_ms") or 0)
        cached = bool(result.get("cached"))
        degraded = result.get("degraded") or []
        mode = result.get("mode") or "unknown"

        client.update_current_span(
            input={"question": question},
            output={
                "answer": (result.get("answer") or "")[:2000],
                "mode": mode,
                "refused": refused,
                "cached": cached,
                "degraded": degraded,
                "sources_count": len(result.get("sources") or []),
            },
            metadata={
                "mode": mode,
                "latency_ms": latency_ms,
                "retrieval_max_score": gate.get("max_score"),
                "faithfulness_score": faith.get("faithfulness_score"),
                "faithfulness_skipped": faith.get("skipped"),
            },
        )

        client.score_current_trace(
            name="latency_ms",
            value=float(latency_ms),
            comment="End-to-end ask latency",
        )
        client.score_current_trace(
            name="refused",
            value=1.0 if refused else 0.0,
            comment="Guardrail refused answer",
        )
        client.score_current_trace(
            name="cache_hit",
            value=1.0 if cached else 0.0,
        )
        client.score_current_trace(
            name="degraded",
            value=1.0 if degraded else 0.0,
            comment=",".join(degraded) if degraded else "ok",
        )
        if gate.get("max_score") is not None:
            client.score_current_trace(
                name="retrieval_max_score",
                value=float(gate["max_score"]),
            )
        if faith.get("faithfulness_score") is not None:
            client.score_current_trace(
                name="faithfulness_score",
                value=float(faith["faithfulness_score"]),
            )
        elif faith.get("skipped"):
            client.score_current_trace(
                name="faithfulness_skipped",
                value=1.0,
                comment=str(faith.get("reason", "")),
            )
    except Exception as e:
        logger.debug("Langfuse score_ask_result failed: %s", e)
