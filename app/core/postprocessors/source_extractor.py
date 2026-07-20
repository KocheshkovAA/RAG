from typing import List, Dict, Any
from langchain_core.documents import Document


class SourceExtractor:
    """Извлекает источники / метаданные для ответа API"""

    @staticmethod
    def extract(docs: List[Document]) -> List[Dict[str, Any]]:
        result = []
        for doc in docs:
            meta = doc.metadata or {}
            score = meta.get("rerank_score", meta.get("hybrid_score", 0.0))
            result.append(
                {
                    "article_name": meta.get("article_name", "N/A"),
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "score": round(float(score), 4),
                    "rerank_score": meta.get("rerank_score"),
                    "hybrid_score": meta.get("hybrid_score"),
                }
            )
        return result