import sqlite3
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
from app.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_vectorstore_and_sync_entities(
    embedding_model,
    CHROMA_PERSIST_DIR,
    db_path="warhammer_articles.db"
):
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    if not any(CHROMA_PERSIST_DIR.iterdir()):
        raise RuntimeError("Chroma persist dir is empty — сначала нужно построить векторстор!")

    logger.info("Loading existing Chroma vectorstore")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embedding_model
    )

    # --- выгружаем документы из Chroma ---
    all_docs = vectorstore.get(include=["metadatas", "documents"])
    metadatas = all_docs["metadatas"]

    logger.info(f"Loaded {len(metadatas)} documents from Chroma, syncing entities to DB")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # проверяем, есть ли колонка entities
    cursor.execute("PRAGMA table_info(articles)")
    cols = [c[1] for c in cursor.fetchall()]
    if "entities" not in cols:
        logger.info("Adding 'entities' column to articles table")
        cursor.execute("ALTER TABLE articles ADD COLUMN entities TEXT")

    # обновляем сущности для каждой статьи
    for m in metadatas:
        article_id = m.get("article_id")
        entities = m.get("entities", "")
        if article_id is not None:
            cursor.execute(
                "UPDATE articles SET entities = ? WHERE id = ?",
                (entities, article_id)
            )

    conn.commit()
    conn.close()
    logger.info("Entities successfully synced to database")

    return vectorstore

if __name__ == "__main__":

    try:
        load_vectorstore_and_sync_entities(
            embedding_model=embedding_model,
            CHROMA_PERSIST_DIR=CHROMA_PERSIST_DIR,
        )
        logger.info("Sync finished successfully ✅")
    except Exception as e:
        logger.error(f"Failed to sync entities: {e}")