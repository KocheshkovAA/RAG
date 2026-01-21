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
from app.rag.llm import get_llm
from app.rag.agent import GraphContextOptimizer
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
    def _prepare_agent_payload(self, doc_to_chunks: Dict) -> Dict:
        candidate_nodes = list(doc_to_chunks.keys())
        node_scores, paths_dict, intermediate_nodes = calculate_graph_metrics(candidate_nodes)
        
        all_nodes = []
        unique_titles = list(doc_to_chunks.keys()) + [n for n in intermediate_nodes if n not in doc_to_chunks]
        
        for i, title in enumerate(unique_titles):
            is_detailed = title in doc_to_chunks
            node_data = get_node_info(title, detailed=is_detailed)
            
            if node_data:
                all_nodes.append({
                    "id": f"node_{i+1}", 
                    "score": round(node_scores.get(title, 0.0), 3) if is_detailed else 0.0,
                    "graph_info": node_data
                })

        return {"nodes": all_nodes, "paths": paths_dict}
    
    @traceable
    def _assemble_final_context(self, clean_payload: Dict, doc_to_chunks: Dict) -> List[Document]:
        """Собирает текст только если есть валидная ссылка в источнике."""
        final_docs = []
        nodes = clean_payload.get("nodes", [])

        for n in nodes:
            info = n.get("graph_info", {})
            title = info.get("title")
            if not title: continue

            graph_description = info.get('text', '').strip()
            chunks_with_scores = doc_to_chunks.get(title, [])

            source_url = None
            base_metadata = {}

            if chunks_with_scores:
                first_meta = chunks_with_scores[0][0].metadata
                source_url = first_meta.get("source") or first_meta.get("url")
                base_metadata = first_meta.copy()
            else:
                source_url = info.get('source') or info.get('url')
                base_metadata = {"title": title, "source": source_url}

            if not source_url or not str(source_url).startswith("http"):
                logger.warning(f"Skipping node {title}: no valid HTTP source found.")
                continue

            if not graph_description and not chunks_with_scores:
                continue

            content_parts = [f"=== СУЩНОСТЬ: {title} ==="]
            
            if graph_description:
                content_parts.append(f"[ОПИСАНИЕ]: {graph_description}")

            if chunks_with_scores:
                content_parts.append("[ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ ИЗ АРХИВОВ]:")
                for i, (chunk, _) in enumerate(chunks_with_scores, 1):
                    content_parts.append(f"Фрагмент {i}:\n{chunk.page_content}")

            final_docs.append(Document(
                page_content="\n\n".join(content_parts),
                metadata=base_metadata
            ))

        return final_docs

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
        parsed = split_and_extract_entities(query)
        questions = parsed.get("questions", []) + [{"text": query}]
        entities = parsed.get("entities", [])

        docs_by_questions = self._search_by_questions(questions)
        docs_by_entities = self._search_by_entities(entities)
        
        doc_to_chunks = self._merge_chunks(docs_by_questions + docs_by_entities)

        agent_payload = self._prepare_agent_payload(doc_to_chunks)

        optimizer = GraphContextOptimizer(model=get_llm())
        clean_payload = optimizer.optimize(query, agent_payload)

        final_docs = self._assemble_final_context(clean_payload, doc_to_chunks)

        return final_docs



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
