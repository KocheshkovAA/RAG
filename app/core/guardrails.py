"""Production guardrails: retrieval gate + conditional faithfulness check."""

import difflib
import logging
import re
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import llm_factory
from app.core.postprocessors.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z]{3,}")


def is_spelling_variant_in_context(claim: str, context: str, threshold: float = 0.82) -> bool:
    """True, если claim похож на опечатку/вариант написания сущности, уже есть в контексте.

    Верификатор иногда помечает как unsupported_claim имя, написанное в вопросе
    с опечаткой (например "Ярик" вместо "Яррик") — это ложное срабатывание,
    а не признак галлюцинации.
    """
    claim_norm = claim.strip().lower()
    if not claim_norm:
        return False
    return any(
        difflib.SequenceMatcher(None, claim_norm, word).ratio() >= threshold
        for word in _WORD_RE.findall(context.lower())
    )


class AtomicClaim(BaseModel):
    """Одно атомарное фактическое утверждение из ответа + его опора в контексте."""

    claim: str = Field(
        description="Одно атомарное фактическое утверждение из ответа: одна дата/номер/имя/связь за раз"
    )
    supporting_quote: str = Field(
        default="",
        description="Дословная цитата из КОНТЕКСТА, слово в слово подтверждающая claim. "
        "Пустая строка, если дословного подтверждения в контексте нет.",
    )


class FaithfulnessVerdict(BaseModel):
    """Вердикт пост-проверки: разбиение ответа на атомарные утверждения с опорой на контекст.

    Грунтованность каждого claim считается программно по supporting_quote (сверка
    с контекстом), а не по самооценке модели — модель, дающая holistic is_grounded=true,
    систематически пропускает искажённые атрибуты существующих сущностей (например,
    подмену номера легиона), потому что оценивает "упомянута ли сущность", а не
    "подтверждён ли именно этот факт дословно".
    """

    claims: List[AtomicClaim] = Field(
        default_factory=list,
        description="Разбиение ОТВЕТА на атомарные фактические утверждения, "
        "каждое — с дословной подтверждающей цитатой из контекста",
    )
    reasoning: str = Field(description="Краткое обоснование итогового вердикта")


_MARKUP_RE = re.compile(r"[*_`\"'«»\-–—]")


def _normalize_for_match(text: str) -> str:
    return " ".join(_MARKUP_RE.sub("", text.lower()).split())


def is_quote_grounded(quote: str, context: str, min_coverage: float = 0.85) -> bool:
    """True, если quote — почти дословный фрагмент context (терпимо к markdown/пунктуации,
    но не к пересказу или выдуманной цитате).

    Модель иногда самооценивает is_grounded=true для ответа, где сущность упомянута
    в контексте, но конкретный атрибут (номер легиона, дата, число) подменён — сверка
    цитаты against контекста, а не доверие вердикту модели, ловит именно такие случаи.

    autojunk=False обязателен: реальный context — это тысячи символов конкатенированных
    чанков, и при len(context) >= 200 SequenceMatcher по умолчанию включает эвристику
    "популярные символы — мусор", которая на практике режет find_longest_match до ~70%
    даже для цитаты, дословно и полностью содержащейся в контексте (воспроизведено на
    реальном примере: 317-символьная verbatim-цитата давала match.size=222 с autojunk=True
    и 317 — с autojunk=False). Без этой опции гейт ложно отклонял корректно заземлённые
    ответы почти на каждом сколько-нибудь длинном контексте.
    """
    quote_norm = _normalize_for_match(quote)
    if len(quote_norm) < 4:
        return False
    context_norm = _normalize_for_match(context)
    match = difflib.SequenceMatcher(None, quote_norm, context_norm, autojunk=False).find_longest_match(
        0, len(quote_norm), 0, len(context_norm)
    )
    return match.size >= min_coverage * len(quote_norm)


class ClaimEntailment(BaseModel):
    """Вердикт по одной паре (утверждение, цитата): подтверждает ли цитата именно это
    утверждение дословно и по фактам, а не только тематически."""

    entailed: bool = Field(
        description="True только если цитата ТОЧНО подтверждает именно это утверждение — "
        "включая числа, имена, даты, названия. Тематическое совпадение (та же сущность, "
        "но другое число/имя/дата) — False."
    )


class EntailmentCheck(BaseModel):
    """Список вердиктов entailment — по одному на каждую входную пару (утверждение, цитата),
    в том же порядке, что и на входе."""

    verdicts: List[ClaimEntailment] = Field(
        description="Ровно один вердикт на каждую пару (утверждение, цитата) из входного "
        "списка, СТРОГО в том же порядке — не пропускай и не переставляй пары."
    )


def doc_relevance_score(doc: Document) -> float:
    meta = doc.metadata or {}
    if "rerank_score" in meta:
        return float(meta["rerank_score"])
    if "hybrid_score" in meta:
        return float(meta["hybrid_score"])
    return 0.0


class RetrievalGate:
    def __init__(
        self,
        min_score: Optional[float] = None,
        min_score_no_rerank: Optional[float] = None,
        insufficient_message: Optional[str] = None,
    ):
        configured = (
            min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE
        )
        self.min_score = max(configured, settings.RERANK_MIN_SCORE)
        self.min_score_no_rerank = (
            min_score_no_rerank
            if min_score_no_rerank is not None
            else settings.RETRIEVAL_MIN_SCORE_NO_RERANK
        )
        self.insufficient_message = (
            insufficient_message or settings.INSUFFICIENT_INFO_MESSAGE
        )

    @observe(name="Guardrail: Retrieval Gate")
    def filter_docs(self, docs: List[Document]) -> tuple[List[Document], dict[str, Any]]:
        if not docs:
            return [], {
                "passed": False,
                "reason": "empty_retrieval",
                "max_score": 0.0,
                "min_score": self.min_score,
            }

        # Без rerank_score в metadata остаётся только hybrid (RRF) скор — он
        # ранговый, не семантический, и несравним по шкале с rerank_score.
        # Порог 0.55 калиброван под reranker; на hybrid-only берём отдельный,
        # более мягкий порог (см. комментарий у RETRIEVAL_MIN_SCORE_NO_RERANK).
        has_rerank = any("rerank_score" in (d.metadata or {}) for d in docs)
        active_min_score = self.min_score if has_rerank else self.min_score_no_rerank
        scoring = "rerank" if has_rerank else "hybrid_only"

        scored = [(doc_relevance_score(d), d) for d in docs]
        max_score = max(s for s, _ in scored)
        kept = [d for s, d in scored if s >= active_min_score]

        if not kept:
            return [], {
                "passed": False,
                "reason": "below_similarity_threshold",
                "max_score": round(max_score, 4),
                "kept": 0,
                "dropped": len(docs),
                "min_score": active_min_score,
                "scoring": scoring,
            }

        return kept, {
            "passed": True,
            "reason": "ok",
            "max_score": round(max_score, 4),
            "kept": len(kept),
            "dropped": len(docs) - len(kept),
            "min_score": active_min_score,
            "scoring": scoring,
        }

    def insufficient_response(self, gate_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": self.insufficient_message,
            "sources": [],
            "guardrail": {
                "retrieval_gate": gate_meta,
                "faithfulness": None,
                "refused": True,
            },
            "degraded": [],
            "cached": False,
        }


class AnswerFaithfulnessGuard:
    """Пост-проверка с условным пропуском при высокой уверенности retrieval."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        min_score: Optional[float] = None,
        insufficient_message: Optional[str] = None,
    ):
        self.enabled = (
            enabled if enabled is not None else settings.FAITHFULNESS_CHECK_ENABLED
        )
        self.min_score = (
            min_score if min_score is not None else settings.FAITHFULNESS_MIN_SCORE
        )
        self.skip_above = settings.FAITHFULNESS_SKIP_ABOVE
        self.insufficient_message = (
            insufficient_message or settings.INSUFFICIENT_INFO_MESSAGE
        )
        self.context_builder = ContextBuilder()
        self.llm = llm_factory.get_llm(temperature=0, role="faithfulness")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты — строгий верификатор фактов для RAG.\n"
                    "Разбей ОТВЕТ на атомарные фактические утверждения (claims): каждое имя, "
                    "дата, номер (легиона, тома, года), число или связь между сущностями — "
                    "отдельным пунктом. Вводные фразы и связки в claims не выделяй.\n"
                    "Для КАЖДОГО claim найди дословную (verbatim) цитату из КОНТЕКСТА, которая "
                    "прямо его подтверждает, и укажи её в supporting_quote ТОЧНО как в контексте — "
                    "без пересказа, без изменения чисел/имён.\n"
                    "Если в контексте нет дословного подтверждения именно этому утверждению — даже "
                    "если упоминается похожая или связанная сущность — оставь supporting_quote "
                    "пустой строкой. Не подставляй правдоподобную, но ненайденную цитату.",
                ),
                (
                    "human",
                    "ВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{context}\n\nОТВЕТ:\n{answer}",
                ),
            ]
        )
        # Второй, отдельный LLM-вызов — намеренно отдельный от prompt/chain выше, а не
        # дополнительное поле в той же структуре. is_quote_grounded() проверяет только
        # "эта цитата реально есть в контексте дословно" (ловит выдуманные цитаты), но
        # не проверяет "эта цитата подтверждает именно ЭТОТ claim" — модель может
        # приложить настоящую, но не по теме цитату к испорченному утверждению (реальный
        # текст про другой номер/дату), и такое чисто строковая проверка не поймает.
        # Отдельный узкий вызов на короткие пары (claim, quote) — тот же паттерн, что
        # использует RAGAS для statement-level entailment, вместо holistic self-report
        # в одном вызове с генерацией самих claims (что и было исходной проблемой).
        self.entailment_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Для каждой пронумерованной пары (УТВЕРЖДЕНИЕ, ЦИТАТА) реши: подтверждает ли "
                    "цитата ИМЕННО это утверждение — дословно и по фактам, а не тематически. "
                    "Если цитата про ту же сущность, но называет другое число, имя, дату или "
                    "название — entailed=false. Верни ровно один вердикт на каждую пару, строго "
                    "в том же порядке, ничего не пропускай.",
                ),
                ("human", "{pairs}"),
            ]
        )

    async def _check_entailment(self, claims: List["AtomicClaim"], config=None) -> list[bool]:
        """Возвращает по одному bool на claim (в том же порядке): entailed ли цитата этому
        claim'у. Fail-safe при любой проблеме (ошибка вызова, несовпадение числа вердиктов
        с числом claims) — считаем НЕ entailed, а не доверяем на слово: для guardrail'а
        безопаснее лишний раз отказать, чем пропустить неподтверждённый факт."""
        if not claims:
            return []
        pairs_text = "\n\n".join(
            f"{i + 1}. УТВЕРЖДЕНИЕ: {c.claim}\nЦИТАТА: {c.supporting_quote}"
            for i, c in enumerate(claims)
        )
        try:
            structured = self.llm.with_structured_output(EntailmentCheck)
            chain = self.entailment_prompt | structured
            result: EntailmentCheck = await chain.ainvoke({"pairs": pairs_text}, config=config)
        except Exception as e:
            logger.warning("entailment check failed: %s", e)
            return [False] * len(claims)

        if len(result.verdicts) != len(claims):
            logger.warning(
                "entailment check returned %d verdicts for %d claims", len(result.verdicts), len(claims)
            )
            return [False] * len(claims)
        return [v.entailed for v in result.verdicts]

    def should_verify(
        self,
        docs: List[Document],
        *,
        degraded: bool = False,
        answer: str = "",
    ) -> tuple[bool, dict[str, Any]]:
        """
        Пропускаем дорогую LLM-проверку только если:
        - есть rerank scores
        - max score >= FAITHFULNESS_SKIP_ABOVE
        - пайплайн не деградировал (без реранка риск выше)
        """
        if not self.enabled:
            return False, {"skipped": True, "reason": "disabled"}

        if degraded:
            return True, {"skipped": False, "reason": "degraded_pipeline"}

        scores = [doc_relevance_score(d) for d in docs]
        max_score = max(scores) if scores else 0.0
        has_rerank = any("rerank_score" in (d.metadata or {}) for d in docs)

        if has_rerank and max_score >= self.skip_above:
            return False, {
                "skipped": True,
                "reason": "high_retrieval_confidence",
                "max_score": round(max_score, 4),
                "skip_above": self.skip_above,
            }

        return True, {
            "skipped": False,
            "reason": "low_or_missing_confidence",
            "max_score": round(max_score, 4),
            "has_rerank": has_rerank,
        }

    @observe(name="Guardrail: Faithfulness Check")
    async def verify(
        self, question: str, answer: str, docs: List[Document], config=None
    ) -> tuple[bool, FaithfulnessVerdict | None, dict[str, Any]]:
        context = self.context_builder.build(docs)

        try:
            # GigaChat требует description у схемы — создаём tool внутри try
            structured = self.llm.with_structured_output(FaithfulnessVerdict)
            chain = self.prompt | structured
            verdict: FaithfulnessVerdict = await chain.ainvoke(
                {"question": question, "context": context, "answer": answer},
                config=config,
            )
        except Exception as e:
            return False, None, {
                "enabled": True,
                "passed": False,
                "skipped": False,
                "error": str(e),
                "reason": "verifier_error",
            }

        # Два независимых слоя, оба должны пройти, иначе claim unsupported:
        # 1) is_quote_grounded (дёшево, без LLM) — цитата реально есть в контексте
        #    дословно, ловит выдуманные цитаты.
        # 2) _check_entailment (второй LLM-вызов, только для claims, прошедших слой 1 —
        #    не тратим его на уже сфабрикованные цитаты) — цитата подтверждает ИМЕННО
        #    этот claim, а не просто существует где-то в контексте. Ловит случай, когда
        #    модель прикладывает настоящую, но не по теме цитату к испорченному факту
        #    (например, реальная цитата про другой номер легиона) — чисто строковая
        #    проверка слоя 1 такое пропускает, потому что не смотрит на текст claim'а.
        # Опечатки/варианты написания сущности прощаем отдельно, на тексте самого claim.
        quote_ok = [is_quote_grounded(c.supporting_quote, context) for c in verdict.claims]
        to_check = [c for c, ok in zip(verdict.claims, quote_ok) if ok]
        entailment_results = iter(await self._check_entailment(to_check, config=config))
        entailed = [next(entailment_results) if ok else False for ok in quote_ok]

        unsupported = [
            c.claim
            for c, ok, ent in zip(verdict.claims, quote_ok, entailed)
            if not (ok and ent)
            and not is_spelling_variant_in_context(c.claim, context)
        ]

        total_claims = len(verdict.claims)
        faithfulness_score = 1.0 if total_claims == 0 else (total_claims - len(unsupported)) / total_claims
        is_grounded = len(unsupported) == 0

        passed = is_grounded and faithfulness_score >= self.min_score
        meta = {
            "enabled": True,
            "passed": passed,
            "skipped": False,
            "faithfulness_score": round(faithfulness_score, 4),
            "is_grounded": is_grounded,
            "unsupported_claims": unsupported,
            "claims_checked": total_claims,
            "reasoning": verdict.reasoning,
            "min_score": self.min_score,
        }
        return passed, verdict, meta

    def refuse_response(
        self, faithfulness_meta: dict[str, Any], sources: List[dict]
    ) -> dict[str, Any]:
        return {
            "answer": self.insufficient_message,
            "sources": sources,
            "guardrail": {
                "faithfulness": faithfulness_meta,
                "refused": True,
            },
            "degraded": [],
            "cached": False,
        }
