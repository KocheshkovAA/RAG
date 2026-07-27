# app/embeddings.py
from typing import List
from langchain_core.embeddings import Embeddings
from tenacity.retry import retry_if_exception_type
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langfuse import observe, get_client
from app.core.config import settings


class TEIEmbeddings(Embeddings):
    """Dense embeddings через TEI inference сервер (async-first)"""

    def __init__(self):
        self.api_format = settings.TEI_API_FORMAT
        base = settings.TEI_URL.rstrip("/")
        self.url = f"{base}/v1/embeddings" if self.api_format == "openai" else f"{base}/embed"
        self.query_prefix = (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
        )
        self.client = httpx.AsyncClient(
            timeout=settings.TEI_TIMEOUT_SEC,
            limits=httpx.Limits(max_connections=100),
        )

    def _request_body(self, texts: List[str]) -> dict:
        if self.api_format == "openai":
            return {"model": "biencoder-finetuned", "input": texts}
        return {"inputs": texts}

    def _parse_response(self, data) -> List[List[float]]:
        if self.api_format == "openai":
            return [row["embedding"] for row in data["data"]]
        return data

    @observe(name="TEI embed_documents", capture_output=False)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._sync_embed(texts, is_query=False)

    @observe(name="TEI embed_query", capture_output=False)
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([f"{self.query_prefix}{text}"])[0]

    def _sync_embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        processed = texts if not is_query else [f"{self.query_prefix}{t}" for t in texts]
        resp = httpx.post(
            f"{self.url}",
            json=self._request_body(processed),
            timeout=settings.TEI_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    @observe(name="TEI aembed_documents", capture_output=False)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError)
        ),
        reraise=True,
    )
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        from app.core.health import tei_breaker

        if not tei_breaker.allow():
            raise RuntimeError("TEI circuit is open")

        lf = get_client()
        lf.update_current_trace(input={"count": len(texts), "type": "documents"})

        try:
            resp = await self.client.post(f"{self.url}", json=self._request_body(texts))
            resp.raise_for_status()
            tei_breaker.record_success()
            return self._parse_response(resp.json())
        except Exception:
            tei_breaker.record_failure()
            raise

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([f"{self.query_prefix}{text}"]))[0]


class BM25SparseEmbeddings:
    """Sparse BM25-подобные embeddings (fastembed)"""

    def __init__(self):
        self.model = SparseTextEmbedding(model_name="Qdrant/bm25")

    @observe(name="BM25 embed_documents", capture_output=False)
    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        embeddings = list(self.model.embed(texts))
        return [
            SparseVector(
                indices=e.indices.tolist(),
                values=e.values.tolist()
            )
            for e in embeddings
        ]

    def embed_query(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]
    

