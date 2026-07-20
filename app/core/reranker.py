import httpx
import math
from typing import List, Any, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings
from app.core.health import reranker_breaker
from app.core.resilience import call_with_circuit
from langfuse import observe


class Reranker:
    def __init__(self):
        self.enabled = settings.RERANKER_ENABLED
        self.vllm_url = (
            f"{settings.RERANKER_URL.rstrip('/')}/v1/rerank"
            if settings.RERANKER_URL
            else ""
        )
        self.top_k = settings.RERANKER_TOP_K or 5
        self.batch_size = settings.RERANKER_BATCH_SIZE or 32
        self.model_name = settings.RERANKER_MODEL
        self.client = httpx.AsyncClient(timeout=settings.RERANKER_TIMEOUT_SEC)

    def _fallback_docs(self, docs: List[Any]) -> List[Any]:
        """Деградация: исходный порядок hybrid-скоров, без rerank_score."""
        return docs[: self.top_k]

    @observe(name="Reranker: Process Documents")
    async def rerank_documents(
        self, query: str, docs: List[Any]
    ) -> Tuple[List[Any], bool]:
        """
        Returns: (docs, degraded).
        При падении/open circuit — top-k без реранка (graceful degradation).
        """
        if not docs:
            return [], False

        if not self.enabled or not self.vllm_url:
            return self._fallback_docs(docs), False

        async def _run() -> List[Any]:
            texts = [doc.page_content for doc in docs]
            scores = await self._get_scores(query, texts)
            scored_docs = sorted(
                zip(scores, docs),
                key=lambda x: x[0],
                reverse=True,
            )
            final_docs = []
            for score, doc in scored_docs[: self.top_k]:
                doc.metadata["rerank_score"] = round(float(score), 4)
                final_docs.append(doc)
            return final_docs

        result, degraded = await call_with_circuit(
            reranker_breaker,
            _run,
            fallback=lambda: self._fallback_docs(docs),
        )
        return result, degraded

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError)
        ),
        reraise=True,
    )
    async def _get_scores(self, query: str, texts: List[str]) -> List[float]:
        all_scores = [0.0] * len(texts)

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            response = await self.client.post(
                self.vllm_url,
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": batch_texts,
                },
            )
            response.raise_for_status()
            data = response.json()
            results = sorted(data.get("results", []), key=lambda x: x["index"])
            for idx, res in enumerate(results):
                all_scores[i + idx] = 1 / (1 + math.exp(-res["relevance_score"]))

        return all_scores


reranker = Reranker()
