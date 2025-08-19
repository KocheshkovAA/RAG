import re
import pickle
import logging
from pathlib import Path
from typing import List

from pymorphy2 import MorphAnalyzer
from pydantic import Field
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.schema import BaseRetriever
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

from app.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

VECTORSTORE_FILE = CHROMA_PERSIST_DIR / "bm25_vectorstore.pkl"
morph = MorphAnalyzer()

# ---------- базовые утилиты ----------
def _tokenize_ru(text: str) -> list[str]:
    # токенизация + лемматизация; отбрасываем очень короткие токены
    return [morph.parse(w)[0].normal_form
            for w in re.findall(r"\w+", text.lower())
            if len(w) > 2]

def lemmatize_text(text: str) -> str:
    return " ".join(_tokenize_ru(text))

# ---------- BM25 индекс ----------
def build_bm25_retriever(documents: list[Document]) -> BM25Retriever:
    if VECTORSTORE_FILE.exists():
        logger.info("Loading an existing BM25 retriever")
        with open(VECTORSTORE_FILE, "rb") as f:
            return pickle.load(f)

    logger.info("Building a new BM25 retriever")

    lemmatized_docs = [
        Document(
            page_content=lemmatize_text(doc.page_content),
            metadata={"original": doc.page_content, **doc.metadata}
        )
        for doc in documents
    ]

    

    retriever = BM25Retriever.from_documents(lemmatized_docs)
    retriever.k = 200

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORSTORE_FILE, "wb") as f:
        pickle.dump(retriever, f)

    return retriever

# ---------- PRF на TF-IDF ----------
def _build_prf_expansion_terms(
    query: str,
    docs: List[Document],
    top_terms: int = 8,
) -> list[str]:
    """
    Возвращает список терминов (леммы) для расширения запроса.
    Используем TF-IDF по top-N документах.
    """
    if not docs:
        return []

    # корпус = тексты top-N (оригиналы, а не лемматизированные)
    texts = [d.metadata.get("original", d.page_content) for d in docs]

    # Векторизатор: используем наш токенизатор, без стоп-слов (они уже выпиливаются лемматизацией и df)

    russian_stopwords = set(get_stop_words("ru")) | {"заголовок", "статья"} 
    vectorizer = TfidfVectorizer(tokenizer=_tokenize_ru, lowercase=True, stop_words=russian_stopwords)
    tfidf = vectorizer.fit_transform(texts)  # shape: (N_docs, V)

    # Средний вес термина по всем документах (centroid в терм-пространстве TF-IDF)
    mean_scores = tfidf.mean(axis=0).A1  # -> np.array shape (V,)
    vocab = vectorizer.get_feature_names_out()

    # термины исходного запроса (чтобы не дублировать)
    q_terms = set(_tokenize_ru(query))

    # сортируем по весу, исключая термины из запроса
    ranked = [(term, score) for term, score in zip(vocab, mean_scores) if term not in q_terms]
    ranked.sort(key=lambda x: x[1], reverse=True)

    return [t for t, _ in ranked[:top_terms]]

def _expand_query_with_terms(
    query: str,
    terms: list[str],
    weights: list[float] | None = None,
    max_repeat: int = 3,
) -> str:
    """
    Для BM25 нет явных весов, поэтому моделируем веса повторением терминов.
    Если weights не заданы — используем единичные.
    """
    if not terms:
        return query

    if weights is None:
        weights = [1.0] * len(terms)

    # нормируем в [0, 1]
    w_min, w_max = min(weights), max(weights)
    if w_max == w_min:
        reps = [1] * len(terms)
    else:
        reps = [1 + int((w - w_min) / (w_max - w_min) * (max_repeat - 1)) for w in weights]

    expanded_tail = []
    for term, r in zip(terms, reps):
        expanded_tail.extend([term] * r)

    return (query + " " + " ".join(expanded_tail)).strip()

# ---------- Каскад: BM25 → PRF(BM25) → CrossEncoder ----------
class BM25PrfRerankRetriever(BaseRetriever):
    bm25_retriever: BM25Retriever = Field(...)
    reranker: CrossEncoder = Field(...)

    # Stage 1: начальный BM25
    top_k_stage1: int = Field(default=200)

    # PRF параметры
    prf_enable: bool = Field(default=True)
    prf_top_docs: int = Field(default=10)   # на скольких документах считаем TF-IDF
    prf_top_terms: int = Field(default=8)   # сколько слов добавить
    prf_max_repeat: int = Field(default=3)  # макс. кратность добавления термина

    # Stage 2: финальный срез + порог
    top_k_final: int = Field(default=20)
    score_threshold: float = Field(default=0.3)

    def _stage1(self, query: str) -> List[Document]:
        return self.bm25_retriever.get_relevant_documents(lemmatize_text(query))[:self.top_k_stage1]

    def _apply_prf(self, query: str, candidates: List[Document]) -> str:
        if not self.prf_enable:
            return query
        top_for_prf = candidates[: self.prf_top_docs]
        terms = _build_prf_expansion_terms(query, top_for_prf, top_terms=self.prf_top_terms)

        # можно прокинуть веса (если захочешь) — сейчас используем равные
        q_expanded = _expand_query_with_terms(
            query=query,
            terms=terms,
            weights=None,
            max_repeat=self.prf_max_repeat
        )
        return q_expanded

    def _stage2(self, query: str) -> List[Document]:
        # повторный BM25 уже по расширенному запросу
        return self.bm25_retriever.get_relevant_documents(lemmatize_text(query))[:self.top_k_stage1]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) первичный BM25
        initial = self._stage1(query)

        # 2) PRF: строим q'
        q_prime = self._apply_prf(query, initial)
        print(q_prime)

        # 3) вторичный BM25 на q'
        candidates = self._stage2(q_prime)

        # 4) реранкинг CrossEncoder (по ОРИГИНАЛАМ текстов)
        pairs = [(query, doc.metadata.get("original", doc.page_content)) for doc in candidates]
        scores = self.reranker.predict(pairs)

        reranked = [
            (Document(page_content=doc.metadata.get("original", doc.page_content), metadata=doc.metadata), score)
            for doc, score in zip(candidates, scores)
            if score >= self.score_threshold
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked[: self.top_k_final]]

# ---------- фабрика ----------
def build_or_load_vectorstore(documents: list[Document]) -> BM25PrfRerankRetriever:
    logger.info("Create or download Cascade Retriever (BM25 → PRF → Reranker)")

    bm25_retriever = build_bm25_retriever(documents)

    logger.info("Load CrossEncoder (reranker)")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    return BM25PrfRerankRetriever(
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        top_k_stage1=50,
        top_k_final=6,
        prf_enable=True,
        prf_top_docs=30,
        prf_top_terms=7,
        prf_max_repeat=3,
        score_threshold=0.3,
    )
