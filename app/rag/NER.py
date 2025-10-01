# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Tuple, Set
import re
import sqlite3
import logging

from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc
from razdel import tokenize as razdel_tokenize
from rapidfuzz import fuzz
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import pickle

# ---------- Логгер ----------
logger = logging.getLogger(__name__)

GAZETTEER_FILE = "gazetteer.pkl"
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()

# ---------- Загрузка заголовков из БД ----------
def extract_named_entities(text: str) -> Set[str]:
    """Возвращает множество нормализованных именованных сущностей из текста."""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    entities = set()
    for span in doc.spans:
        span.normalize(morph_vocab)
        entities.add(span.normal)
    return entities

STOP_WORDS = set(get_stop_words("ru"))  # русские стоп-слова

def load_titles_with_entities(db_path: str = 'warhammer_articles.db', limit: int = 50000) -> List[str]:
    enriched_entities = set()
    morph = MorphAnalyzer()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT original_title, final_title, entities 
            FROM articles 
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        logger.info(f"Loaded {len(rows)} rows from database")

        for orig, final, ent_str in rows:
            for title in [orig, final]:
                if title:
                    title_clean = title.strip()
                    words = re.findall(r'\w+', title_clean)
                    for w in words:
                        if len(w) > 1 and w.lower() not in STOP_WORDS:
                            p = morph.parse(w)[0]
                            if 'NOUN' in p.tag:
                                # сохраняем исходное написание с заглавной буквы
                                enriched_entities.add(w.capitalize())
            if ent_str:
                links = [e.strip() for e in ent_str.split(",") if e.strip()]
                for link in links:
                    words = re.findall(r'\w+', link)
                    for w in words:
                        lw = w.lower()
                        if len(lw) > 1 and lw not in STOP_WORDS:
                            p = morph.parse(lw)[0]
                            if 'NOUN' in p.tag:
                                enriched_entities.add(p.normal_form)

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    return list(enriched_entities)


def build_or_load_gazetteer(db_path='warhammer_articles.db', limit=50000):
    try:
        # если файл есть — читаем
        with open(GAZETTEER_FILE, "rb") as f:
            gazetteer = pickle.load(f)
            logger.info(f"Loaded gazetteer from {GAZETTEER_FILE}")
            return gazetteer
    except FileNotFoundError:
        # иначе строим из БД
        gazetteer = load_titles_with_entities(db_path=db_path, limit=limit)
        with open(GAZETTEER_FILE, "wb") as f:
            pickle.dump(gazetteer, f)
        logger.info(f"Saved gazetteer to {GAZETTEER_FILE}")
        return gazetteer

# Пример загрузки
GAZETTEER = build_or_load_gazetteer()

# ---------- Natasha / Morph ----------
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()
morph = MorphAnalyzer()

@dataclass
class Entity:
    text: str
    span: Tuple[int, int]
    canonical: str | None = None
    score: float | None = None
    source: str = "gazetteer"

# ---------- Утилиты ----------
def _lemmatize(s: str) -> str:
    tokens = [t.text for t in razdel_tokenize(s.lower()) if re.search(r"\w", t.text)]
    lemmas = [morph.parse(tok)[0].normal_form for tok in tokens]
    return " ".join(lemmas)

# ---------- NER через Natasha ----------
def natasha_ner(text: str) -> List[Entity]:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    ents: List[Entity] = []
    for span in doc.spans:
        span.normalize(morph_vocab)
        ents.append(Entity(
            text=text[span.start:span.stop],
            span=(span.start, span.stop),
            canonical=span.normal,
            source="natasha"
        ))
    return ents

# ---------- NER через словарь ----------
def gazetteer_ner(text: str, cutoff: int = 82) -> List[Entity]:
    tokens = list(razdel_tokenize(text))
    lowered_tokens = [tok.text.lower() for tok in tokens]
    ents: List[Entity] = []

    for i in range(len(tokens)):
        for j in range(i+1, min(i+6, len(tokens))+1):  # ngram длиной до 5 слов
            fragment = " ".join(lowered_tokens[i:j])
            orig_fragment = text[tokens[i].start:tokens[j-1].stop]  # корректный фрагмент в тексте
            for name in GAZETTEER:
                score = fuzz.ratio(fragment, name.lower())
                if score >= cutoff:
                    start = tokens[i].start
                    end = tokens[j-1].stop
                    ents.append(Entity(
                        text=orig_fragment,
                        span=(start, end),
                        canonical=name,
                        score=float(score),
                        source="gazetteer"
                    ))
    return ents


# ---------- Объединение ----------
def merge_entities(entities: List[Entity]) -> List[Entity]:
    by_key = {}
    for e in entities:
        key = (e.text.lower(), e.span, e.source)
        by_key[key] = e

    ents = list(by_key.values())
    ents.sort(key=lambda e: (e.span[0], -e.span[1]))

    result: List[Entity] = []
    for e in ents:
        overlapped = False
        for r in result:
            s1, e1 = e.span
            s2, e2 = r.span
            if (s1 < e2 and s2 < e1):
                def quality(x: Entity):
                    return (x.score or 0.0, len(x.text))
                if quality(e) > quality(r):
                    result.remove(r)
                    result.append(e)
                overlapped = True
                break
        if not overlapped:
            result.append(e)
    return result

# ---------- Основная функция ----------
def extract_entities(text: str) -> List[Entity]:
    #ents1 = natasha_ner(text)
    ents2 = gazetteer_ner(text)
    return merge_entities( ents2)

def inflect_to_case(word: str, canonical: str) -> str:
    """
    Приводит canonical к падежу исходного слова.
    Сохраняет заглавную букву, если исходное слово было с большой.
    """
    parsed = morph.parse(word)[0]
    target = morph.parse(canonical)[0]

    # граммемы исходного слова
    features = set()
    if parsed.tag.number:
        features.add(parsed.tag.number)
    if parsed.tag.case:
        features.add(parsed.tag.case)
    if parsed.tag.gender:
        features.add(parsed.tag.gender)

    inflected = target.inflect(features)
    if inflected:
        result = inflected.word
    else:
        result = canonical  # fallback

    # сохраняем капитализацию исходного слова
    if canonical[0].isupper() | word[0].isupper():
        result = result.capitalize()
    return result

    
def normalize_text_entities(text: str, cutoff: int = 82) -> str:
    """
    Находит сущности в тексте и заменяет их на каноническое написание.
    Использует gazetteer. Сохраняет капитализацию из словаря.
    """
    # 1) извлекаем сущности
    entities = extract_entities(text)
    if not entities:
        return text

    # 2) сортируем по позиции, чтобы заменять справа налево
    entities.sort(key=lambda e: e.span[0], reverse=True)

    corrected_text = text
    for e in entities:
        if e.canonical:
            start, end = e.span
            orig_word = corrected_text[start:end]
            corrected_word = inflect_to_case(orig_word, e.canonical)
            corrected_text = corrected_text[:start] + corrected_word + corrected_text[end:]


    return corrected_text

# -------- Пример --------
if __name__ == "__main__":
    sample = """
    Аба́ддон Робаут Жиллиман жиллиман жиллимана жилиман гиллиман гилиман?? КТо такой нургл?
    """
    ents = extract_entities(sample)
    for e in ents:
        print(f"[{e.source}] -> '{e.text}' canon='{e.canonical}' span={e.span} score={e.score}")

    normalized = normalize_text_entities(sample)
    print(normalized)
