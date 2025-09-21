import sqlite3
import json
from pathlib import Path

DB_PATH = "warhammer_articles.db"
OUTPUT_FILE = Path("entities.txt")

def dump_entities(db_path=DB_PATH, output_file=OUTPUT_FILE):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT entities FROM articles")
    rows = cursor.fetchall()

    all_entities = set()
    for row in rows:
        entities = row[0]
        if not entities:
            continue

        try:
            # если у тебя entities в JSON-формате
            parsed = json.loads(entities)
            if isinstance(parsed, list):
                all_entities.update(parsed)
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        all_entities.update(v)
                    elif isinstance(v, str):
                        all_entities.add(v)
        except Exception:
            # если это просто строка через запятую
            parts = [e.strip() for e in entities.split(",") if e.strip()]
            all_entities.update(parts)

    conn.close()

    sorted_entities = sorted(all_entities)

    with open(output_file, "w", encoding="utf-8") as f:
        for e in sorted_entities:
            f.write(e + "\n")

    print(f"Сущности сохранены в {output_file.resolve()} ({len(sorted_entities)} шт.)")

if __name__ == "__main__":
    dump_entities()
