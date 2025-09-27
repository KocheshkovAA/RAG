import logging
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever
from pydantic import Field
from app.embedding_model import MLMEmbeddings
from app.NER import normalize_text_entities
from app.config import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR
from app.llm_tools.query_normalizer import split_and_extract_entities  # твой модуль для вопросов
from app.graph.node import get_node_info, calculate_graph_metrics

from collections import defaultdict, Counter
logger = logging.getLogger(__name__)

# --- эмбеддинги ---
embedding_model = MLMEmbeddings(EMBEDDING_MODEL_NAME)

class HybridRetriever(BaseRetriever):
    vectorstore: Chroma = Field(...)

    top_k_vector: int = Field(default=50)
    top_k_final: int = Field(default=10)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Извлекает под-вопросы и сущности из исходного вопроса, ищет релевантные документы 
        по под-вопросам и по сущностям, сортирует по графовой связанности.
        """
        # 1) Разбиваем вопрос и извлекаем сущности только для исходного вопроса
        parsed = split_and_extract_entities(query)
        #parsed = {'entities': ['Аба́ддон', 'Жиллима́н'], 'questions': [{'text': 'Кто такой Аба́ддон?'}, {'text': 'Какие способности и качества есть у Аба́ддона?'}, {'text': 'Кто такой Жиллима́н?'}, {'text': 'Какие способности и качества есть у Жиллима́на?'}, {'text': 'Кого традиционно считают победителем в битве между Аба́ддоном и Жиллима́ном?'}]}
        print(parsed)
        questions = parsed.get("questions", [])
        questions.append({"text": query})
        all_entities = set(parsed.get("entities", []))

        if not questions:
            logger.warning(f"Не удалось извлечь сущности из запроса: {query}")
            return self.vectorstore.similarity_search(query, k=self.top_k_vector)

        docs_collected = []

        # 2) Поиск по под-вопросам
        for q in questions:
            sub_query = q.get("text", "").strip()
            if sub_query:
                results = self.vectorstore.similarity_search_with_score(sub_query, k=self.top_k_vector)
                docs_collected.extend([(doc, score) for doc, score in results])

        # 3) Поиск по сущностям (только из исходного вопроса)
        for ent in all_entities:
            if not ent.strip():
                continue
            results = self.vectorstore.similarity_search_with_score(normalize_text_entities(ent.strip()), k=self.top_k_vector)
            docs_collected.extend([(doc, score) for doc, score in results])

        doc_to_chunks = defaultdict(list)
        seen_chunks = set()

        for chunk, score in docs_collected:
            doc_title = chunk.metadata.get("title", "Без названия")
            key = (doc_title, chunk.page_content.strip())  # уникальность по названию + тексту чанка
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            doc_to_chunks[doc_title].append((chunk, score))


        candidate_nodes = list(doc_to_chunks.keys())
        node_scores, paths_dict, intermediate_nodes = calculate_graph_metrics(candidate_nodes, max_length=5)

        filtered_titles = [
            title
            for title, chunks_scores in doc_to_chunks.items()
            if node_scores.get(title, 0.0) > 0 or len(chunks_scores) > 1
        ]

        # если их слишком много, можно ограничить топ-k
        if len(filtered_titles) > self.top_k_final:
            # например, сортируем по сумме node_score + len(chunks)
            filtered_titles = sorted(
                filtered_titles,
                key=lambda t: node_scores.get(t, 0.0) + len(doc_to_chunks[t]),
                reverse=True
            )[:self.top_k_final]


        # если все нули — берем все узлы
        if not filtered_titles:
            filtered_titles = candidate_nodes

        merged_docs = []
        for title in filtered_titles:
            chunks_scores = doc_to_chunks[title]

            # Берём метадату из первого чанка
            first_chunk, _ = chunks_scores[0]
            base_metadata = first_chunk.metadata.copy()

            # добавляем графовую информацию
            graph_info = get_node_info(title)
            merged_text_parts = []
            if graph_info:
                merged_text_parts.append(f"[Graph Info: {graph_info}]")

            merged_text_parts.extend(chunk.page_content for chunk, _ in chunks_scores)
            merged_text = "\n".join(merged_text_parts)

            merged_docs.append(
                Document(
                    page_content=merged_text,
                    metadata=base_metadata
                )
            )

        if intermediate_nodes:
            for interm_node in intermediate_nodes:
                node_info_text = get_node_info(interm_node) or ""
                merged_docs.append(
                    Document(
                        page_content=f"[Intermediate Node: {interm_node}]\n{node_info_text}",
                        metadata={
                            "title": interm_node
                        }
                    )
                )

        # --- Формируем отдельный чанк с путями между найденными узлами ---
        if paths_dict:
            path_lines = []
            for (node1, node2), path in paths_dict.items():
                # чтобы не дублировать симметричные пути, можно проверять node1 < node2
                if node1 < node2:
                    path_lines.append(" ".join(path))
            paths_text = "\n".join(path_lines)
            merged_docs.append(
                Document(
                    page_content=f"[Graph Paths]\n{paths_text}",
                    metadata={"title": "Graph Paths", "type": "graph_paths"}
                )
            )

        # сортировка документов по графовому весу
        merged_docs.sort(
            key=lambda doc: node_scores.get(doc.metadata.get("title", ""), 0.0),
            reverse=True
        )

        logger.info(f"Total connected documents: {len(merged_docs)} / {len(candidate_nodes)}")
        return merged_docs[:self.top_k_final]


def build_or_load_vectorstore(documents: List[Document]) -> HybridRetriever:
    logger.info("Building Chroma vectorstore")
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if any(CHROMA_PERSIST_DIR.iterdir()):
        logger.info("Loading existing Chroma vectorstore")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embedding_model,
        )
    else:
        logger.info("Building new Chroma vectorstore")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )

    return HybridRetriever(
        vectorstore=vectorstore,
        top_k_vector=6,
        top_k_final=10,
    )
