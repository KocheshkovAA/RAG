import sqlite3
import logging
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настройка логгера для модуля
logger = logging.getLogger(__name__)

class DatabaseTextLoader:
    def __init__(self, db_path='warhammer_articles.db'):
        self.db_path = db_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n"],
        )
        logger.info(f"Initialized DatabaseTextLoader with database at: {db_path}")

    def load_and_split_documents(self, limit=50000):
        """Loads the first `limit` articles from database with sources in metadata.
        Returns a tuple of (chunks, titles) where both are in the same Document format."""
        chunks = []
        titles = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            logger.info("Connected to database, starting data loading")

            # Добавляем article_url в выборку
            cursor.execute(f'''
                SELECT a.id, a.original_title, a.content, a.article_url,
                    GROUP_CONCAT(s.source_text, '|||') as sources
                FROM articles a
                LEFT JOIN sources s ON a.id = s.article_id
                GROUP BY a.id
                LIMIT ?
            ''', (limit,))

            articles = cursor.fetchall()
            logger.info(f"Found {len(articles)} articles in database (limit={limit})")

            for article_id, title, content, article_url, sources in articles:
                metadata = {
                    'article_id': article_id,
                    'title': title,
                    'source': article_url if article_url else 'https://warhammer40k.fandom.com/ru/wiki/Warhammer_40000_Wiki',
                    'sources': sources.replace(';;;', ', ') if sources else None
                }

                # Создаем документ для заголовка
                title_doc = Document(
                    page_content=title,
                    metadata=metadata.copy()
                )
                titles.append(title_doc)

                # Создаем документ для контента и разбиваем на чанки
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )

                article_chunks = self.splitter.split_documents([doc])

                for chunk in article_chunks:
                    chunk.page_content = f"[ЗАГОЛОВОК СТАТЬИ]: {title} {title} {title} {title} {title}\n{chunk.page_content}"
                    chunks.append(chunk)

                logger.debug(f"Processed article: {title} ({len(article_chunks)} chunks)")

            logger.info(f"Successfully loaded {len(titles)} titles and {len(chunks)} chunks")
            return chunks, titles

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return [], []
        finally:
            if 'conn' in locals():
                conn.close()
                logger.info("Database connection closed")
