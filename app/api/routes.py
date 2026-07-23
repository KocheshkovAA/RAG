import json
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.vectorrag import rag_chain
from app.core.agentic_rag import AgenticRAG
from app.core.persona_debate import PersonaDebate
from app.core.lightrag_client import LightRAGClient
from app.core.orchestrator import WarhammerOrchestrator
from app.core.config import settings
from app.core.resilience import with_timeout
from app.core.health import check_dependencies
from app.core.tracing import flush, is_enabled

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address, enabled=settings.RATE_LIMIT_ENABLED)

light_rag = LightRAGClient()
agentic_rag = AgenticRAG(rag_chain)
orchestrator = WarhammerOrchestrator(vector_rag=rag_chain, light_rag=light_rag, agentic_rag=agentic_rag)
persona_debate = PersonaDebate(rag_chain)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Опционально для Langfuse sessions (если фронт передаёт)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)


@router.post("/ask")
@limiter.limit(settings.RATE_LIMIT_ASK)
async def ask(request: Request, payload: QuestionRequest):
    try:
        # session/user — атрибуты трейса (если tracing включён)
        if is_enabled() and (payload.session_id or payload.user_id):
            from langfuse import propagate_attributes

            attrs = {}
            if payload.session_id:
                attrs["session_id"] = payload.session_id
            if payload.user_id:
                attrs["user_id"] = payload.user_id
            with propagate_attributes(**attrs):
                result = await with_timeout(
                    orchestrator.answer(payload.question),
                    settings.REQUEST_TIMEOUT_SEC,
                    "ask",
                )
        else:
            result = await with_timeout(
                orchestrator.answer(payload.question),
                settings.REQUEST_TIMEOUT_SEC,
                "ask",
            )

        flush()
        return result
    except TimeoutError as e:
        logger.error("Request timeout: %s", e)
        flush()
        raise HTTPException(
            status_code=504,
            detail={
                "error": "request_timeout",
                "message": str(e),
                "timeout_sec": settings.REQUEST_TIMEOUT_SEC,
            },
        )
    except Exception as e:
        logger.exception("Ask failed")
        flush()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "upstream_unavailable",
                "message": "Сервис временно недоступен. Попробуйте позже.",
                "cause": str(e)[:200],
            },
        )


class DebateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)


@router.post("/debate")
async def debate(request: DebateRequest):
    """Стримящий NDJSON-эндпоинт (не обычный JSON, как /ask) — каждая строка
    ответа это отдельное событие {"type": "turn"|"refused"|"done", ...},
    персонажи отправляются по мере готовности, а не все разом в конце."""
    if not settings.PERSONA_DEBATE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail={"error": "feature_disabled", "message": "Debate feature is disabled."},
        )

    async def event_stream():
        try:
            async for event in persona_debate.stream(request.question):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.exception("Debate failed")
            yield json.dumps({"type": "error", "message": str(e)[:200]}, ensure_ascii=False) + "\n"
        finally:
            flush()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.get("/ready")
async def ready():
    status = await check_dependencies()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status)
    return status
