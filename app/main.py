from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import router, limiter
from app.core.cache import cache_client
from app.core.tracing import init_tracing, flush

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API starting")
    init_tracing()
    yield
    flush()
    await cache_client.close()
    logger.info("RAG API stopped")


app = FastAPI(title="RAG", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(router, prefix="/v1")


@app.get("/health")
def health():
    """Liveness: процесс жив (для k8s/docker)."""
    return {"status": "ok"}
