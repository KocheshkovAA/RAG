"""Юнит-тесты чистой логики app/eval/evaluate_routes.py (сборка строки результата +
агрегаты по агентным решениям) — без LLM/сети/Docker, только на синтетических result-дictах
в форме, которую реально отдают RAG.answer()/AgenticRAG.answer()/orchestrator._answer_graph()."""

import pytest

from app.core.guardrails import refusal_reason
from app.core.orchestrator import RAGRoute
from app.eval import evaluate_routes
from app.eval.evaluate_routes import (
    build_result_row,
    compute_agent_decision_aggregates,
)


def _question(id_=1, question="вопрос?"):
    return {"id": id_, "question": question}


# --------------------------------------------------------------------------- refusal_reason


def test_refusal_reason_none_when_not_refused():
    result = {"guardrail": {"refused": False}}
    assert refusal_reason(result) is None


def test_refusal_reason_retrieval_gate():
    result = {
        "guardrail": {
            "refused": True,
            "retrieval_gate": {"passed": False, "reason": "below_similarity_threshold"},
            "faithfulness": None,
        }
    }
    assert refusal_reason(result) == "retrieval_gate:below_similarity_threshold"


def test_refusal_reason_faithfulness():
    result = {
        "guardrail": {
            "refused": True,
            "faithfulness": {"passed": False},
        }
    }
    # Продовый verify() не кладёт "reason" в обычный (не verifier_error) провал —
    # должен быть честный дефолт, а не KeyError/None.
    assert refusal_reason(result) == "faithfulness:unsupported_claims"


def test_refusal_reason_faithfulness_verifier_error():
    result = {
        "guardrail": {
            "refused": True,
            "faithfulness": {"passed": False, "reason": "verifier_error"},
        }
    }
    assert refusal_reason(result) == "faithfulness:verifier_error"


def test_refusal_reason_graph_empty_falls_back_to_mode():
    # graph-empty: domain probe ПРОШЁЛ (retrieval_gate.passed=True), но LightRAG сам
    # текстом ответил "нет данных" — ни retrieval_gate, ни faithfulness это не объясняют.
    result = {
        "mode": "graph-empty",
        "guardrail": {
            "refused": True,
            "retrieval_gate": {"passed": True, "reason": "ok"},
            "faithfulness": None,
        },
    }
    assert refusal_reason(result) == "mode:graph-empty"


# --------------------------------------------------------------------------- build_result_row


def test_build_result_row_vector_answer():
    result = {
        "answer": "Некроны — раса роботов.",
        "latency_ms": 120,
        "guardrail": {"refused": False, "retrieval_gate": {"passed": True}, "faithfulness": {"skipped": True}},
        "degraded": [],
        "token_usage": {"total": {"total_tokens": 500}},
    }
    row = build_result_row(_question(), RAGRoute.VECTOR, result)

    assert row["route"] == "vector"
    assert row["refused"] is False
    assert row["refusal_reason"] is None
    assert row["iterations"] is None
    assert row["tool_calls_total"] == 0
    assert row["faithfulness_checked"] is False


def test_build_result_row_agentic_with_wasted_retry():
    result = {
        "answer": None,
        "latency_ms": 900,
        "guardrail": {"refused": True, "retrieval_gate": {"passed": False, "reason": "below_similarity_threshold"}},
        "degraded": [],
        "agentic": {
            "iterations": 3,
            "stopped_reason": "max_iterations",
            "tool_calls_made": [
                {"round": 1, "query": "a", "found": 2, "new": 2, "error": None},
                {"round": 2, "query": "b", "found": 0, "new": 0, "error": None},
                {"round": 3, "query": "c", "found": 0, "new": 0, "error": None},
            ],
            "thoughts": [{"round": 1, "thought": "ищу"}, {"round": 2, "thought": "ещё"}],
        },
    }
    row = build_result_row(_question(), RAGRoute.AGENTIC, result)

    assert row["stopped_reason"] == "max_iterations"
    assert row["tool_calls_total"] == 3
    assert row["tool_calls_wasted"] == 2  # раунды 2 и 3 не добавили новой evidence
    assert row["retry_rounds"] == 2  # раунды 2 и 3 (round > первого раунда=1)
    assert row["retry_added_evidence"] is False
    assert row["thought_count"] == 2


def test_build_result_row_agentic_with_justified_retry():
    result = {
        "answer": "ответ",
        "latency_ms": 500,
        "guardrail": {"refused": False, "retrieval_gate": {"passed": True}, "faithfulness": {"skipped": False, "passed": True}},
        "degraded": [],
        "agentic": {
            "iterations": 2,
            "stopped_reason": "model_stopped",
            "tool_calls_made": [
                {"round": 1, "query": "a", "found": 1, "new": 1, "error": None},
                {"round": 1, "query": "b", "found": 1, "new": 1, "error": None},
            ],
            "thoughts": [],
        },
    }
    row = build_result_row(_question(), RAGRoute.AGENTIC, result)

    # Оба tool call в раунде 1 (параллельные) -> нет раундов "после первого" -> retry не было.
    assert row["retry_rounds"] == 0
    assert row["retry_added_evidence"] is None
    assert row["faithfulness_checked"] is True


# --------------------------------------------------------------------------- compute_agent_decision_aggregates


def test_compute_agent_decision_aggregates_empty():
    agg = compute_agent_decision_aggregates([])
    assert agg["refusal_reasons"] == {}
    assert agg["faithfulness_check_rate"] == 0.0
    assert agg["faithfulness_catch_rate"] is None
    assert agg["retry_justified_rate"] is None


def test_compute_agent_decision_aggregates_counts_reasons_and_retry():
    rows = [
        {"refusal_reason": "retrieval_gate:empty_retrieval", "faithfulness_checked": False, "tool_calls_total": 0, "retry_rounds": 0},
        {"refusal_reason": "faithfulness:unsupported_claims", "faithfulness_checked": True, "tool_calls_total": 1, "tool_calls_wasted": 0, "retry_rounds": 0},
        {"refusal_reason": None, "faithfulness_checked": True, "tool_calls_total": 2, "tool_calls_wasted": 1, "retry_rounds": 1, "retry_added_evidence": False},
        {"refusal_reason": None, "faithfulness_checked": True, "tool_calls_total": 2, "tool_calls_wasted": 0, "retry_rounds": 1, "retry_added_evidence": True},
    ]
    agg = compute_agent_decision_aggregates(rows)

    assert agg["refusal_reasons"] == {"retrieval_gate:empty_retrieval": 1, "faithfulness:unsupported_claims": 1}
    assert agg["faithfulness_check_rate"] == 3 / 4
    # faith_checked = строки 2,3,4 (3 штуки); из них "faithfulness:*" причина отказа только у строки 2.
    assert agg["faithfulness_catch_rate"] == 1 / 3
    assert agg["retry_occurred_n"] == 2
    assert agg["retry_justified_rate"] == 1 / 2
    assert agg["avg_wasted_tool_calls"] == (0 + 1 + 0) / 3


# --------------------------------------------------------------------------- _evaluate_with_retry


async def test_evaluate_with_retry_succeeds_without_retry(monkeypatch):
    async def _fake(orchestrator, q, route):
        return {"answer": "ok"}

    monkeypatch.setattr(evaluate_routes, "evaluate_question_for_route", _fake)

    result = await evaluate_routes._evaluate_with_retry(None, {"id": 1}, RAGRoute.VECTOR)

    assert result == {"answer": "ok"}


async def test_evaluate_with_retry_retries_on_generic_exception(monkeypatch):
    """Регресс: полный 60-вопросный прогон потерял строку на 'NoneType' object is not
    iterable — не воспроизвелось на изолированном повторе, транзиентный сбой. Раньше
    ретраилось только по тексту 'rate limit', любая другая ошибка падала с первой попытки."""
    calls = []

    async def _fake(orchestrator, q, route):
        calls.append(1)
        if len(calls) < 2:
            raise TypeError("'NoneType' object is not iterable")
        return {"answer": "ok"}

    sleeps = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(evaluate_routes, "evaluate_question_for_route", _fake)
    monkeypatch.setattr(evaluate_routes.asyncio, "sleep", _fake_sleep)

    result = await evaluate_routes._evaluate_with_retry(None, {"id": 31}, RAGRoute.VECTOR)

    assert result == {"answer": "ok"}
    assert len(calls) == 2
    assert sleeps == [3]  # не rate-limit -> короткий backoff (3с), не 15*2^0=15с


async def test_evaluate_with_retry_uses_longer_backoff_for_rate_limit(monkeypatch):
    calls = []

    async def _fake(orchestrator, q, route):
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("429 Too Many Requests")
        return {"answer": "ok"}

    sleeps = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(evaluate_routes, "evaluate_question_for_route", _fake)
    monkeypatch.setattr(evaluate_routes.asyncio, "sleep", _fake_sleep)

    await evaluate_routes._evaluate_with_retry(None, {"id": 1}, RAGRoute.VECTOR)

    assert sleeps == [15]  # rate-limit -> экспоненциальный backoff, 15 * 2**0


async def test_evaluate_with_retry_gives_up_after_max_attempts(monkeypatch):
    async def _fake(orchestrator, q, route):
        raise TypeError("boom")

    async def _fake_sleep(seconds):
        pass

    monkeypatch.setattr(evaluate_routes, "evaluate_question_for_route", _fake)
    monkeypatch.setattr(evaluate_routes.asyncio, "sleep", _fake_sleep)

    with pytest.raises(TypeError, match="boom"):
        await evaluate_routes._evaluate_with_retry(None, {"id": 1}, RAGRoute.VECTOR, max_attempts=2)
