import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.vectorrag import rag_chain
from app.core.lightrag_client import LightRAGClient
from app.core.orchestrator import WarhammerOrchestrator
from app.core.config import settings
from app.core.resilience import with_timeout
from app.core.health import check_dependencies
from app.core.tracing import flush, is_enabled

logger = logging.getLogger(__name__)

router = APIRouter()

light_rag = LightRAGClient()
orchestrator = WarhammerOrchestrator(vector_rag=rag_chain, light_rag=light_rag)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Опционально для Langfuse sessions (если фронт передаёт)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)


@router.post("/ask")
async def ask(request: QuestionRequest):
    try:
        # session/user — атрибуты трейса (если tracing включён)
        if is_enabled() and (request.session_id or request.user_id):
            from langfuse import propagate_attributes

            attrs = {}
            if request.session_id:
                attrs["session_id"] = request.session_id
            if request.user_id:
                attrs["user_id"] = request.user_id
            with propagate_attributes(**attrs):
                result = await with_timeout(
                    orchestrator.answer(request.question),
                    settings.REQUEST_TIMEOUT_SEC,
                    "ask",
                )
        else:
            result = await with_timeout(
                orchestrator.answer(request.question),
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


@router.get("/ready")
async def ready():
    status = await check_dependencies()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status)
    return status
