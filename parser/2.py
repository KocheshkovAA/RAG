# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re

from natasha import (
    Segmenter, MorphVocab, NewsEmbedding,
    NewsNERTagger, Doc
)
from razdel import tokenize as razdel_tokenize
from rapidfuzz import process, fuzz
from pymorphy2 import MorphAnalyzer

# ---------- Доменные словари (пример) ----------
GAZETTEER: Dict[str, List[str]] = {
    "FACTION": [
        "Империум Человечества", "Адептус Астартес", "Астра Милитарум",
        "Черный Легион", "Тау", "Эльдары", "Темные Эльдары", "Орки",
        "Некроны", "Тираниды", "Хаос", "Сыны Хоруса", "Альфа-Легион"
    ],
    "PLANET": [
        "Терра", "Кадия", "Катачан", "Кальт", "Просперо", "Кедриа",
        "Марс", "Армагеддон", "Кадиме", "Катачон"
    ],
    "CHARACTER": [
        "Абаддон Разоритель", "Император Человечества", "Хорус Луперкаль",
        "Робут Жиллимэн", "Сангвиний", "Леман Русс", "Магнус Красный",
        "Конрад Керз", "Мортарион", "Ангрон", "Пертурабо", "Фулгрим"
    ],
    "ORDER": [
        "Ордо Маллеус", "Ордо Ксенос", "Ордо Еретикус",
        "Инквизиция", "Серафимы", "Систеры Битвы", "Адептус Механикус"
    ],
    "TITLE": [
        "Первый капитан", "Первый Верховный Полководец", "Примарх",
        "Святой", "Святой Мститель", "Warmaster", "Верховный Лорд Терры"
    ]
}

CANON_MAP: Dict[str, str] = {
    "абаддон": "Абаддон Разоритель",
    "абаддон разоритель": "Абаддон Разоритель",
    "император": "Император Человечества",
    "черный легион": "Черный Легион",
    "сыны хоруса": "Сыны Хоруса",
}

# ---------- Natasha / Morph ----------
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph_vocab = MorphVocab()
morph = MorphAnalyzer()

@dataclass
class Entity:
    text: str
    label: str
    span: Tuple[int, int]
    canonical: str | None = None
    score: float | None = None
    source: str = "natasha"

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def _lemmatize(s: str) -> str:
    tokens = [t.text for t in razdel_tokenize(s.lower()) if re.search(r"\w", t.text)]
    lemmas = [morph.parse(tok)[0].normal_form for tok in tokens]
    return " ".join(lemmas)

def natasha_ner(text: str) -> List[Entity]:
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    ents: List[Entity] = []
    for span in doc.spans:
        span.normalize(morph_vocab)
        ents.append(Entity(
            text=text[span.start:span.stop],
            label=span.type,
            span=(span.start, span.stop),
            canonical=span.normal,
            source="natasha"
        ))
    return ents

def gazetteer_ner(text: str, cutoff: int = 85) -> List[Entity]:
    """
    Фаззи-поиск только для проверки сходства,
    а в span кладём именно найденное имя из газеттира.
    """
    lowered = text.lower()
    ents: List[Entity] = []

    for label, names in GAZETTEER.items():
        for name in names:
            # Сравниваем с текстом (весь текст, а не n-граммы)
            score = fuzz.partial_ratio(name.lower(), lowered)
            if score < cutoff:
                continue

            # ищем само name в тексте
            idx = lowered.find(name.lower())
            if idx == -1:
                # fallback по леммам
                idx = lowered.find(_lemmatize(name))
            if idx == -1:
                span = (0, 0)
            else:
                span = (idx, idx + len(name))

            canon = CANON_MAP.get(_lemmatize(name), name)

            ents.append(Entity(
                text=name,
                label=label,
                span=span,
                canonical=canon,
                score=float(score),
                source="gazetteer"
            ))
    return ents

def merge_entities(entities: List[Entity]) -> List[Entity]:
    by_key = {}
    for e in entities:
        key = (e.text.lower(), e.label, e.source, e.span)
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

def extract_entities(text: str) -> List[Entity]:
    ents1 = natasha_ner(text)
    ents2 = gazetteer_ner(text, cutoff=85)
    return merge_entities(ents1 + ents2)

# -------- Пример --------
if __name__ == "__main__":
    sample = """
    Кто такой Аббадон? Это Абаддон Разоритель из Черного Легиона, первый Верховный Полководец Хаоса.
    Его путь начался в Сынах Хоруса. Говорят, что он вёл крестовые походы с Терры до Кадии.
    Инквизиция Ордо Маллеус и Адептус Астартес сражались против него на Армагеддоне.
    """
    ents = extract_entities(sample)
    for e in ents:
        print(f"[{e.source}] {e.label:10} -> '{e.text}' canon='{e.canonical}' span={e.span} score={e.score}")
