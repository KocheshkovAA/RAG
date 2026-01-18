import logging
from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever
from pydantic import Field

from app.rag.embedding_model import MLMEmbeddings
from app.rag.NER import normalize_text_entities
from app.config import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR
from app.rag.query_normalizer import split_and_extract_entities
from app.graph.node import get_node_info, calculate_graph_metrics
from langsmith import traceable

logger = logging.getLogger(__name__)

# --- Эмбеддинги ---
embedding_model = MLMEmbeddings(EMBEDDING_MODEL_NAME)


class HybridRetriever(BaseRetriever):
    """Retriever с гибридным поиском по под-вопросам, сущностям и графовой структуре."""
    
    vectorstore: Chroma = Field(...)
    top_k_vector: int = Field(default=50)
    top_k_final: int = Field(default=10)

    @traceable
    def _search_by_questions(self, questions: List[Dict[str, str]]) -> List[Tuple[Document, float]]:
        """Поиск релевантных документов по под-вопросам."""
        docs_collected = []
        for q in questions:
            text = q.get("text", "").strip()
            if text:
                results = self.vectorstore.similarity_search_with_score(text, k=self.top_k_vector)
                docs_collected.extend(results)
        return docs_collected

    @traceable
    def _search_by_entities(self, entities: List[str]) -> List[Tuple[Document, float]]:
        """Поиск релевантных документов по сущностям."""
        docs_collected = []
        for ent in entities:
            ent = ent.strip()
            if not ent:
                continue
            results = self.vectorstore.similarity_search_with_score(normalize_text_entities(ent), k=self.top_k_vector)
            docs_collected.extend(results)
        return docs_collected

    def _merge_chunks(self, docs_collected: List[Tuple[Document, float]]) -> Dict[str, List[Tuple[Document, float]]]:
        """Объединение чанков по документам и фильтрация дубликатов."""
        doc_to_chunks = defaultdict(list)
        seen_chunks = set()

        for chunk, score in docs_collected:
            title = chunk.metadata.get("title", "Без названия")
            key = (title, chunk.page_content.strip())
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            doc_to_chunks[title].append((chunk, score))
        return doc_to_chunks

    @traceable
    def _add_graph_info(self, filtered_titles: List[str], doc_to_chunks: Dict[str, List[Tuple[Document, float]]]):
        """Добавляет информацию о графе и промежуточные узлы."""
        merged_docs = []
        candidate_nodes = list(doc_to_chunks.keys())
        node_scores, paths_dict, intermediate_nodes = calculate_graph_metrics(candidate_nodes, max_length=5)

        for title in filtered_titles:
            chunks_scores = doc_to_chunks[title]
            first_chunk, _ = chunks_scores[0]
            base_metadata = first_chunk.metadata.copy()
            detailed = len(chunks_scores) > 1
            graph_info = get_node_info(title, detailed=detailed)
            merged_text_parts = [f"[Graph Info: {graph_info}]" if graph_info else ""]
            merged_text_parts.extend(chunk.page_content for chunk, _ in chunks_scores)
            merged_docs.append(Document(page_content="\n".join(merged_text_parts), metadata=base_metadata))

        # Добавляем промежуточные узлы
        for interm_node in intermediate_nodes:
            node_info_text = get_node_info(interm_node) or ""
            merged_docs.append(Document(
                page_content=f"[Intermediate Node: {interm_node}]\n{node_info_text}",
                metadata={"title": interm_node}
            ))

        # Добавляем пути между узлами
        if paths_dict:
            path_lines = [
                " ".join(path)
                for (node1, node2), path in paths_dict.items()
                if node1 < node2
            ]
            merged_docs.append(Document(
                page_content=f"[Graph Paths]\n" + "\n".join(path_lines),
                metadata={"title": "Graph Paths", "type": "graph_paths"}
            ))

        # Сортировка документов по графовому весу
        merged_docs.sort(key=lambda doc: node_scores.get(doc.metadata.get("title", ""), 0.0), reverse=True)
        return merged_docs, node_scores

    @traceable
    def _filter_top_k(self, doc_to_chunks: Dict[str, List[Tuple[Document, float]]], node_scores: Dict[str, float]) -> List[str]:
        """Фильтруем топ-K документов по графовому весу и количеству чанков."""
        filtered_titles = [
            title for title, chunks_scores in doc_to_chunks.items()
            if node_scores.get(title, 0.0) > 0 or len(chunks_scores) > 1
        ]
        if len(filtered_titles) > self.top_k_final:
            filtered_titles = sorted(
                filtered_titles,
                key=lambda t: node_scores.get(t, 0.0) + len(doc_to_chunks[t]),
                reverse=True
            )[:self.top_k_final]
        if not filtered_titles:
            filtered_titles = list(doc_to_chunks.keys())
        return filtered_titles

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Главная функция поиска релевантных документов для запроса."""
        parsed = split_and_extract_entities(query)
        logger.debug(f"Parsed query: {parsed}")
        questions = parsed.get("questions", [])
        questions.append({"text": query})
        entities = parsed.get("entities", [])

        if not questions:
            logger.warning(f"Не удалось извлечь сущности из запроса: {query}")
            return self.vectorstore.similarity_search(query, k=self.top_k_vector)

        docs_by_questions = self._search_by_questions(questions)
        docs_by_entities = self._search_by_entities(entities)
        all_docs = docs_by_questions + docs_by_entities

        doc_to_chunks = self._merge_chunks(all_docs)
        filtered_titles = self._filter_top_k(doc_to_chunks, {title: 0 for title in doc_to_chunks})  # node_scores добавим ниже

        merged_docs, node_scores = self._add_graph_info(filtered_titles, doc_to_chunks)
        logger.info(f"Total connected documents: {len(merged_docs)} / {len(doc_to_chunks)}")
        return merged_docs[:self.top_k_final]


def build_or_load_vectorstore(documents: List[Document]) -> HybridRetriever:
    """Создаёт или загружает векторное хранилище Chroma."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if any(CHROMA_PERSIST_DIR.iterdir()):
        logger.info("Loading existing Chroma vectorstore")
        vectorstore = Chroma(persist_directory=str(CHROMA_PERSIST_DIR), embedding_function=embedding_model)
    else:
        logger.info("Building new Chroma vectorstore")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=str(CHROMA_PERSIST_DIR)
        )

    return HybridRetriever(vectorstore=vectorstore, top_k_vector=6, top_k_final=10)
