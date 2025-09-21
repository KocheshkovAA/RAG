import sqlite3
import logging
import json
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Настройка логгера для модуля
logger = logging.getLogger(__name__)

class DatabaseTextLoader:
    def __init__(self, db_path='warhammer_articles.db'):
        self.db_path = db_path
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n"],
        )
        logger.info(f"Initialized DatabaseTextLoader with database at: {db_path}")
        
        # Инициализируем структуру БД
        self._init_database()

    def _init_database(self):
        """Создает таблицу для чанков если она не существует"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS article_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER,
                    char_count INTEGER,
                    title TEXT,  
                    article_url TEXT,
                    sources TEXT,
                    entities TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_id) REFERENCES articles (id)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_article_chunks_article_id 
                ON article_chunks (article_id)
            ''')
            
            conn.commit()
            logger.info("Database structure initialized (article_chunks table)")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def _check_chunks_exist(self):
        """Проверяет, есть ли уже чанки в базе данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM article_chunks')
            count = cursor.fetchone()[0]
            
            logger.info(f"Found {count} existing chunks in database")
            return count > 0
            
        except sqlite3.Error as e:
            logger.error(f"Error checking chunks existence: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()

    def _process_entities(self, entities_json):
        """Преобразует entities в строку, разделенную запятыми"""
        if not entities_json or not entities_json.strip():
            return ""
        
        try:
            # Пробуем разные форматы entities
            if entities_json.startswith('['):
                # Если это JSON массив, извлекаем только имена
                entities_list = json.loads(entities_json)
                if isinstance(entities_list, list):
                    # Извлекаем имена из словарей или берем строки как есть
                    entity_names = []
                    for item in entities_list:
                        if isinstance(item, dict) and 'name' in item:
                            entity_names.append(item['name'])
                        elif isinstance(item, str):
                            entity_names.append(item)
                    return ", ".join(entity_names)
                else:
                    return str(entities_list)
            else:
                # Если это строка с разделителями, возвращаем как есть
                return entities_json.strip()
                
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Invalid entities format: {e}")
            # Возвращаем исходную строку как есть
            return entities_json.strip() if entities_json else ""

    def load_and_split_documents(self, limit=50000, overwrite=False):
        """
        Загружает статьи и создает чанки. 
        Если чанки уже существуют и overwrite=False, загружает из БД.
        Если overwrite=True или чанков нет, создает заново.
        """
        
        # Проверяем, есть ли уже чанки
        chunks_exist = self._check_chunks_exist()
        
        # Если чанки есть и не требуется перезапись - загружаем из БД
        if chunks_exist and not overwrite:
            logger.info("Chunks already exist, loading from database...")
            chunks = self.load_chunks_from_db(limit=limit)
            titles = self._load_titles_from_db(limit=limit)
            return chunks, titles
        
        # Если чанков нет или требуется перезапись - создаем заново
        logger.info("Creating chunks from articles...")
        chunks = []
        titles = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            logger.info("Connected to database, starting data loading")

            if overwrite and chunks_exist:
                cursor.execute('DELETE FROM article_chunks')
                conn.commit()
                logger.info("Cleared existing chunks from database")

            cursor.execute(f'''
                SELECT a.id, a.final_title, a.content, a.article_url, a.entities,
                    GROUP_CONCAT(s.source_text, '|||') as sources
                FROM articles a
                LEFT JOIN sources s ON a.id = s.article_id
                GROUP BY a.id
                LIMIT ?
            ''', (limit,))

            articles = cursor.fetchall()
            logger.info(f"Found {len(articles)} articles in database (limit={limit})")

            total_chunks_saved = 0

            for article_id, final_title, content, article_url, entities_json, sources in articles:
                # Преобразуем entities в строку
                entities_string = self._process_entities(entities_json)
                
                metadata = {
                    'article_id': article_id,
                    'title': final_title, 
                    'source': article_url if article_url else 'https://warhammer40k.fandom.com/ru/wiki/Warhammer_40000_Wiki',
                    'sources': sources.replace('|||', ', ') if sources else None,
                    'entities': entities_string
                }

                title_doc = Document(page_content=final_title, metadata=metadata.copy())
                titles.append(title_doc)

                # Пропускаем пустой контент
                if not content or not content.strip():
                    logger.warning(f"Empty content for article {article_id}")
                    continue

                doc = Document(page_content=content, metadata=metadata)
                
                try:
                    article_chunks = self.splitter.split_documents([doc])
                except Exception as e:
                    logger.error(f"Error splitting article {article_id}: {e}")
                    continue

                article_chunks_to_save = []

                for chunk_index, chunk in enumerate(article_chunks):
                    if len(chunk.page_content.strip()) < 100:
                        continue 

                    chunk_text = chunk.page_content.strip()
                    
                    # Подготавливаем данные для сохранения в БД
                    article_chunks_to_save.append((
                        article_id,
                        chunk_text,
                        chunk_index,
                        len(chunk_text.split()),
                        len(chunk_text),
                        final_title,
                        article_url,
                        sources,
                        entities_string
                    ))
                    
                    # Обновляем метаданные
                    chunk.metadata.update({
                        'title': final_title,
                        'entities': entities_string
                    })
                    chunk.page_content = chunk_text
                    chunks.append(chunk)

                # Сохраняем чанки
                if article_chunks_to_save:
                    try:
                        cursor.executemany('''
                            INSERT INTO article_chunks 
                            (article_id, chunk_text, chunk_index, token_count, char_count, 
                             title, article_url, sources, entities)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', article_chunks_to_save)
                        conn.commit()
                        
                        total_chunks_saved += len(article_chunks_to_save)
                        logger.debug(f"Saved {len(article_chunks_to_save)} chunks for article: {final_title}")
                    
                    except sqlite3.Error as e:
                        logger.error(f"Error saving chunks for article {article_id}: {e}")
                        conn.rollback()

            logger.info(f"Successfully saved {total_chunks_saved} chunks to database")
            
            # Фильтруем пустые чанки
            valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            logger.info(f"Returning {len(titles)} titles and {len(valid_chunks)} valid chunks")
            
            return valid_chunks, titles

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if 'conn' in locals():
                conn.rollback()
            return [], []
        finally:
            if 'conn' in locals():
                conn.close()
                logger.info("Database connection closed")

    def _load_titles_from_db(self, limit=1000):
        """Загружает заголовки статей из базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT article_id, title, article_url, sources, entities
                FROM article_chunks
                LIMIT ?
            ''', (limit,))
            
            titles_data = cursor.fetchall()
            
            titles = []
            for article_id, title, article_url, sources, entities_string in titles_data:
                metadata = {
                    'article_id': article_id,
                    'title': title,
                    'source': article_url,
                    'sources': sources,
                    'entities': entities_string
                }
                
                titles.append(Document(page_content=title, metadata=metadata))
            
            logger.info(f"Loaded {len(titles)} titles from database")
            return titles
            
        except sqlite3.Error as e:
            logger.error(f"Error loading titles from database: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

    def load_chunks_from_db(self, article_id=None, limit=10000):
        """Загружает чанки из базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, chunk_text, article_id, chunk_index, title, 
                       article_url, sources, entities
                FROM article_chunks
                WHERE LENGTH(chunk_text) > 10
            '''
            params = []
            
            if article_id:
                query += ' AND article_id = ?'
                params.append(article_id)
            
            query += ' ORDER BY article_id, chunk_index LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            chunks_data = cursor.fetchall()
            
            chunks = []
            for chunk_id, chunk_text, article_id, chunk_index, title, article_url, sources, entities_string in chunks_data:
                metadata = {
                    'chunk_id': chunk_id,
                    'article_id': article_id,
                    'chunk_index': chunk_index,
                    'title': title,  
                    'source': article_url,
                    'sources': sources,
                    'entities': entities_string
                }
                
                if chunk_text.strip():
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
            
            logger.info(f"Loaded {len(chunks)} chunks from database")
            return chunks
            
        except sqlite3.Error as e:
            logger.error(f"Error loading chunks from database: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()

    def get_chunks_with_entity(self, entity_name, entity_type=None, limit=1000):
        """Возвращает чанки, содержащие указанную entity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT id, chunk_text, article_id, chunk_index, title, 
                       article_url, sources, entities
                FROM article_chunks
                WHERE entities LIKE ? AND LENGTH(chunk_text) > 10
                LIMIT ?
            '''
            
            search_pattern = f'%{entity_name}%'
            
            cursor.execute(query, (search_pattern, limit))
            chunks_data = cursor.fetchall()
            
            chunks = []
            for chunk_id, chunk_text, article_id, chunk_index, title, article_url, sources, entities_string in chunks_data:
                metadata = {
                    'chunk_id': chunk_id,
                    'article_id': article_id,
                    'chunk_index': chunk_index,
                    'title': title,
                    'source': article_url,
                    'sources': sources,
                    'entities': entities_string
                }
                
                if chunk_text.strip():
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
            
            logger.info(f"Found {len(chunks)} chunks with entity: {entity_name}")
            return chunks
            
        except sqlite3.Error as e:
            logger.error(f"Error searching chunks by entity: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()