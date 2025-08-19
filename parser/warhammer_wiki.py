import re
import sqlite3
import requests
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import quote
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warhammer_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WarhammerDatabase:
    def __init__(self, db_name='warhammer_articles.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        
        # Main articles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_title TEXT NOT NULL,
            final_title TEXT NOT NULL,
            content TEXT NOT NULL,
            content_length INTEGER,
            article_url TEXT NOT NULL,
            redirects_count INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(final_title)
        )
        ''')
        
        # Sources table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            source_text TEXT NOT NULL,
            FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
        )
        ''')
        
        # Full-text search table
        cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts 
        USING fts5(title, content, tokenize="porter unicode61")
        ''')
        
        # Update history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS update_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            articles_count INTEGER,
            run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created/verified")

    def save_article(self, original_title, final_title, content, redirects=0):
        cursor = self.conn.cursor()
        
        try:
            # Формируем безопасный URL статьи
            safe_title = quote(final_title.replace(' ', '_'))
            article_url = f"https://warhammer40k.fandom.com/ru/wiki/{safe_title}"
            
            # Insert or update article с новым полем article_url
            cursor.execute('''
            INSERT OR REPLACE INTO articles 
            (original_title, final_title, article_url, content, content_length, redirects_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (original_title, final_title, article_url, content, len(content), redirects))
            
            # Получаем ID статьи
            article_id = cursor.lastrowid
            if article_id == 0:
                cursor.execute('SELECT id FROM articles WHERE final_title = ?', (final_title,))
                article_id = cursor.fetchone()[0]
            
            # Извлекаем и сохраняем источники
            self._extract_and_save_sources(cursor, article_id, content)
            
            # Обновляем индекс полнотекстового поиска
            cursor.execute('''
            INSERT OR REPLACE INTO articles_fts (rowid, title, content)
            VALUES (?, ?, ?)
            ''', (article_id, final_title, content))
            
            self.conn.commit()
            logger.debug(f"Successfully saved article: {final_title} ({article_url})")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error saving article {final_title}: {e}")
            return False

    def _extract_and_save_sources(self, cursor, article_id, content):
        """Extracts and saves sources from article content"""
        cursor.execute('DELETE FROM sources WHERE article_id = ?', (article_id,))
        
        lines = content.split('\n')
        sources = []
        in_source_block = False
        current_source = []
        
        for line in lines:
            # Start of source block
            if line.strip().upper().startswith('ИСТОЧНИК'):
                in_source_block = True
                current_source = []
                continue
            
            # End of source block (empty line or new heading)
            if in_source_block and (line.strip().isupper()):
                in_source_block = False
                if current_source:
                    for source_line in current_source:
                        if source_line.strip() and source_line.strip() != '':
                            sources.append(source_line.strip())
                continue
            
            # Collect source lines
            if in_source_block and line.strip():
                current_source.append(line.strip())
        
        # Add last source if block wasn't closed
        if in_source_block and current_source:
            for source_line in current_source:
                if source_line.strip() != '':
                    sources.append(source_line.strip())
        
        # Save each source separately
        for source in sources:
            cursor.execute(
                'INSERT INTO sources (article_id, source_text) VALUES (?, ?)',
                (article_id, source)
            )
        logger.debug(f"Extracted {len(sources)} sources for article ID {article_id}")

    def log_update(self, count):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO update_history (articles_count) VALUES (?)
        ''', (count,))
        self.conn.commit()
        logger.info(f"Logged update with {count} articles processed")

class FandomParser:
    def __init__(self, db):
        self.base_url = "https://warhammer40k.fandom.com/ru/api.php"
        self.session = requests.Session()
        self.db = db
        self.session.headers.update({
            "User-Agent": "MyRAGBot/1.0 (contact@example.com)",
            "Accept": "application/json"
        })
        logger.info("FandomParser initialized")

    def get_article_text(self, title, max_redirects=3, redirect_chain=None):
        """Recursively gets article text with redirect handling"""
        if redirect_chain is None:
            redirect_chain = []
            
        if max_redirects <= 0:
            logger.warning(f"Redirect limit reached for: {title}")
            return None, redirect_chain

        params = {
            "action": "parse",
            "page": title,
            "format": "json",
            "prop": "text|redirects",
            "disabletoc": 1,
            "redirects": True
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                logger.error(f"API error for '{title}': {data['error']['info']}")
                return None, redirect_chain

            parse_data = data.get("parse", {})
            
            # Handle redirects
            if "redirects" in parse_data and parse_data["redirects"]:
                new_title = parse_data["redirects"][0]["to"]
                logger.info(f"Redirect: {title} → {new_title}")
                redirect_chain.append((title, new_title))
                return self.get_article_text(new_title, max_redirects-1, redirect_chain)

            # Get content
            html_content = parse_data.get("text", {}).get("*")
            if not html_content:
                logger.warning(f"Empty content for: {title}")
                return None, redirect_chain

            return self.clean_html(html_content), redirect_chain

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error for '{title}': {str(e)}")
            return None, redirect_chain
        except Exception as e:
            logger.error(f"Unexpected error for '{title}': {str(e)}", exc_info=True)
            return None, redirect_chain

    def clean_html(self, html):
        """Cleans HTML and extracts text"""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for element in soup([
            "script", "style", "table", "div.portable-infobox", 
            "div.references", "span.mw-editsection", "div.notice",
            "div.hatnote", "div.redirectMsg", "nav", "footer", 
            "aside", "figure", "img", "svg", "noscript"
        ]):
            element.decompose()
        
        # Extract structured text
        text_parts = []
        
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
            text = element.get_text(' ', strip=True)
            if text and len(text) > 10:  # Ignore short fragments
                if element.name.startswith('h'):
                    text = f"\n{text.upper()}\n"
                text_parts.append(text)
        
        return '\n'.join(text_parts)

    def fetch_all_articles(self, limit=None):
        """Gets list of all articles with pagination"""
        articles = []
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": 500,
            "format": "json"
        }

        try:
            while True:
                response = self.session.get(self.base_url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()

                if "query" not in data or "allpages" not in data["query"]:
                    break

                articles.extend(page["title"] for page in data["query"]["allpages"])
                
                if limit and len(articles) >= limit:
                    articles = articles[:limit]
                    break

                if "continue" not in data:
                    break
                    
                params.update(data["continue"])
                time.sleep(1.5)

        except Exception as e:
            logger.error(f"Error fetching article list: {str(e)}", exc_info=True)

        logger.info(f"Fetched {len(articles)} articles")
        return articles

    def process_and_save_articles(self, limit=None):
        """Main method for processing and saving articles"""
        articles = self.fetch_all_articles(limit)
        success_count = 0
        
        for i, title in enumerate(articles, 1):
            try:
                start_time = time.time()
                
                # Get text and redirect history
                content, redirects = self.get_article_text(title)
                
                if content:
                    # Determine final title (after all redirects)
                    final_title = redirects[-1][1] if redirects else title
                    
                    # Save to database
                    if self.db.save_article(
                        original_title=title,
                        final_title=final_title,
                        content=content,
                        redirects=len(redirects)
                    ):
                        success_count += 1
                
                elapsed = time.time() - start_time
                logger.info(f"[{i}/{len(articles)}] {'✓' if content else '✗'} {title} ({elapsed:.1f}s)")
                
            except Exception as e:
                logger.error(f"Critical error processing '{title}': {str(e)}", exc_info=True)
            finally:
                time.sleep(1.5)  # Respect Crawl-delay
        
        # Log update results
        self.db.log_update(success_count)
        logger.info(f"Completed! Successfully saved {success_count}/{len(articles)} articles")
        
        return success_count

def resume_from_article(start_title, limit=None):
    """Resumes parsing from specific article"""
    try:
        db = WarhammerDatabase()
        parser = FandomParser(db)
        
        # Get all articles
        all_articles = parser.fetch_all_articles()
        
        # Find starting position
        try:
            start_idx = all_articles.index(start_title) + 1  # Start from next article
            logger.info(f"Resuming from article #{start_idx}: {all_articles[start_idx]}")
        except ValueError:
            logger.error(f"Article '{start_title}' not found in list")
            return

        # Process remaining articles
        articles_to_process = all_articles[start_idx:start_idx+limit] if limit else all_articles[start_idx:]
        total_count = len(all_articles)
        
        for i, title in enumerate(articles_to_process, start_idx+1):
            try:
                start_time = time.time()
                content, redirects = parser.get_article_text(title)
                
                if content:
                    final_title = redirects[-1][1] if redirects else title
                    db.save_article(title, final_title, content, len(redirects))
                
                logger.info(f"[{i}/{total_count}] Processed: {title} ({time.time()-start_time:.1f}s)")
                time.sleep(1.5)
                
            except Exception as e:
                logger.error(f"Error processing {title}: {str(e)}")
                time.sleep(5)  # Longer delay on error

    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        if 'db' in locals():
            db.conn.close()

if __name__ == "__main__":
    db = WarhammerDatabase()
    parser = FandomParser(db)
    parser.process_and_save_articles()  # загрузит все статьи по умолчанию (limit=None)
    db.close()