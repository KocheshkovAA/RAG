"""Production guardrails: retrieval gate + conditional faithfulness check."""

from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.llm import llm_factory
from app.core.postprocessors.context_builder import ContextBuilder


class FaithfulnessVerdict(BaseModel):
    """Вердикт пост-проверки: все ли факты ответа подтверждены retrieved-контекстом."""

    is_grounded: bool = Field(
        description="True только если каждое фактическое утверждение ответа прямо следует из контекста"
    )
    faithfulness_score: float = Field(
        description="Доля утверждений, подтверждённых контекстом (0-1)",
        ge=0.0,
        le=1.0,
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Утверждения из ответа, которых нет в контексте",
    )
    reasoning: str = Field(description="Краткое обоснование вердикта")


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
        self.llm = llm_factory.get_llm(temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты — строгий верификатор фактов для RAG. "
                    "Проверь, что ОТВЕТ полностью опирается на КОНТЕКСТ.\n"
                    "Штрафуй: выдуманные даты, номера, имена, цифры и связи вне контекста.\n"
                    "Если контекст недостаточен — is_grounded=false.",
                ),
                (
                    "human",
                    "ВОПРОС:\n{question}\n\nКОНТЕКСТ:\n{context}\n\nОТВЕТ:\n{answer}",
                ),
            ]
        )

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
        self, question: str, answer: str, docs: List[Document]
    ) -> tuple[bool, FaithfulnessVerdict | None, dict[str, Any]]:
        context = self.context_builder.build(docs)

        try:
            # GigaChat требует description у схемы — создаём tool внутри try
            structured = self.llm.with_structured_output(FaithfulnessVerdict)
            chain = self.prompt | structured
            verdict: FaithfulnessVerdict = await chain.ainvoke(
                {"question": question, "context": context, "answer": answer}
            )
        except Exception as e:
            return False, None, {
                "enabled": True,
                "passed": False,
                "skipped": False,
                "error": str(e),
                "reason": "verifier_error",
            }

        passed = (
            verdict.is_grounded
            and verdict.faithfulness_score >= self.min_score
            and len(verdict.unsupported_claims) == 0
        )
        meta = {
            "enabled": True,
            "passed": passed,
            "skipped": False,
            "faithfulness_score": verdict.faithfulness_score,
            "is_grounded": verdict.is_grounded,
            "unsupported_claims": verdict.unsupported_claims,
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
