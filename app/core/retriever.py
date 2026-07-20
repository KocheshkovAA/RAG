from typing import List, Optional, Dict, Any
import asyncio
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.runnables import RunnableConfig
from qdrant_client import QdrantClient
from langfuse import observe, get_client, propagate_attributes

from app.core.embedder import TEIEmbeddings, BM25SparseEmbeddings
from app.core.config import settings

class RetrievalMetrics:
    """Вспомогательный класс для расчета статистики поиска"""
    def __init__(self):
        self.num_results: int = 0
        self.avg_score: float = 0.0
        self.max_score: float = 0.0
        self.scores: List[float] = []

    def update(self, docs_with_scores: List[tuple]) -> None:
        if not docs_with_scores:
            return
            
        self.scores = [score for _, score in docs_with_scores]
        self.num_results = len(self.scores)
        
        if self.scores:
            self.avg_score = sum(self.scores) / len(self.scores)
            self.max_score = max(self.scores)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_results": self.num_results,
            "avg_score": round(self.avg_score, 4),
            "max_score": round(self.max_score, 4),
        }

class Retriever(BaseRetriever):
    vector_store: Optional[QdrantVectorStore] = None
    embeddings_dense: Optional[TEIEmbeddings] = None
    embeddings_sparse: Optional[BM25SparseEmbeddings] = None

    langfuse: Optional[Any] = None

    k: int = 8
    fetch_k_dense: int = 30
    fetch_k_sparse: int = 30
    alpha: float = 0.7  

    def __init__(
        self,
        collection_name: str = settings.QDRANT_COLLECTION,
        qdrant_client: Optional[QdrantClient] = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        
        self.embeddings_dense = TEIEmbeddings()
        self.embeddings_sparse = BM25SparseEmbeddings()
        self.langfuse = get_client()

        client = qdrant_client or QdrantClient(
            url=settings.QDRANT_URL,
            timeout=settings.QDRANT_TIMEOUT_SEC,
        )

        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings_dense, 
            sparse_embedding=self.embeddings_sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="text-dense", 
            sparse_vector_name="text-sparse",
            content_payload_key="content",
        )

    @observe(name="Retriever → Hybrid Search", capture_input=False, capture_output=False)
    def _get_relevant_documents(
        self,
        query: str,
        *,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> List[Document]:
        with propagate_attributes(tags=["retrieval", "hybrid", "qdrant"]):
            # Логируем входные данные в текущий спан
            self.langfuse.update_current_span(
                input={
                    "query": query,
                    "k": self.k,
                    "fetch_k": self.fetch_k_dense + self.fetch_k_sparse
                }
            )

            # Основной поиск
            docs = self._retrieve_hybrid(query, **kwargs)

            return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> List[Document]:
        # Sync Qdrant/TEI клиент — в threadpool, чтобы не блокировать event loop
        return await asyncio.to_thread(
            self._get_relevant_documents, query, config=config, **kwargs
        )

    def _retrieve_hybrid(self, query: str, **kwargs) -> List[Document]:
        qdrant_filter = kwargs.get("filter")

        # Выполняем поиск через обертку LangChain
        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.fetch_k_dense + self.fetch_k_sparse,
            filter=qdrant_filter,
            query_filter=qdrant_filter,
        )

        # Расчет метрик
        metrics = RetrievalMetrics()
        metrics.update(docs_with_scores)
        
        # Логируем метрики в Langfuse
        self._log_metrics_to_langfuse(metrics)

        # Формируем финальный список документов
        top_docs = docs_with_scores[: self.k]
        result_docs = []
        for doc, score in top_docs:
            doc.metadata["hybrid_score"] = round(float(score), 4)
            doc.metadata["search_type"] = "hybrid"
            result_docs.append(doc)

        return result_docs

    def _log_metrics_to_langfuse(self, metrics: RetrievalMetrics) -> None:
        metrics_dict = metrics.to_dict()

        # Записываем метрики в метаданные спана
        self.langfuse.update_current_span(metadata=metrics_dict)

        # Создаем скоры для удобного мониторинга на дашбордах
        if metrics.max_score > 0:
            self.langfuse.score_current_span(
                name="retrieval_max_score",
                value=round(metrics.max_score, 4),
                comment="Максимальная релевантность из Qdrant"
            )

        if metrics.num_results > 0:
            self.langfuse.score_current_span(
                name="retrieval_avg_score",
                value=round(metrics.avg_score, 4),
                comment="Средняя релевантность выборки"
            )

    @classmethod
    def from_collection(
        cls,
        collection_name: str = settings.QDRANT_COLLECTION,
        qdrant_url: Optional[str] = None,
        **kwargs
    ):
        client = QdrantClient(url=qdrant_url or settings.QDRANT_URL)
        return cls(collection_name=collection_name, qdrant_client=client, **kwargs)