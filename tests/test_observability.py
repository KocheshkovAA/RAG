"""Юнит-тесты app/core/observability.py — без сети/LLM/Docker, синтетические result-дicты."""

import json
import logging

from app.core.observability import log_ask_result


def _base_result(**overrides):
    result = {
        "answer": "ответ",
        "mode": "vector",
        "cached": False,
        "degraded": [],
        "latency_ms": 123,
        "guardrail": {
            "refused": False,
            "retrieval_gate": {"passed": True, "max_score": 0.81},
            "faithfulness": {"skipped": True},
        },
        "token_usage": {"total": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
    }
    result.update(overrides)
    return result


def test_log_ask_result_emits_one_json_line(caplog):
    with caplog.at_level(logging.INFO, logger="app.observability.ask"):
        log_ask_result("Кто такие некроны?", _base_result())

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["route"] == "vector"
    assert payload["refused"] is False
    assert payload["refusal_reason"] is None
    assert payload["latency_ms"] == 123
    assert payload["token_usage"] == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    assert payload["agent"]["reasoning_mode"] == "single_shot"
    assert payload["agent"]["tool_calls"] == []
    assert payload["reflection"]["retrieval_gate_passed"] is True


def test_log_ask_result_captures_refusal_reason(caplog):
    result = _base_result(
        guardrail={
            "refused": True,
            "retrieval_gate": {"passed": False, "reason": "below_similarity_threshold", "max_score": 0.1},
            "faithfulness": None,
        }
    )
    with caplog.at_level(logging.INFO, logger="app.observability.ask"):
        log_ask_result("офф-топик", result)

    payload = json.loads(caplog.records[0].message)
    assert payload["refused"] is True
    assert payload["refusal_reason"] == "retrieval_gate:below_similarity_threshold"


def test_log_ask_result_captures_agentic_fields(caplog):
    result = _base_result(
        mode="agentic",
        agentic={
            "iterations": 2,
            "stopped_reason": "model_stopped",
            "tool_calls_made": [
                {"round": 1, "query": "Магнус", "found": 3, "new": 3, "error": None},
                {"round": 1, "query": "Ангрон", "found": 2, "new": 2, "error": None},
            ],
            "thoughts": [{"round": 1, "thought": "ищу обоих персонажей"}],
        },
    )
    with caplog.at_level(logging.INFO, logger="app.observability.ask"):
        log_ask_result("Кто такие Магнус и Ангрон?", result)

    payload = json.loads(caplog.records[0].message)
    assert payload["agent"]["reasoning_mode"] == "react"
    assert payload["agent"]["iterations"] == 2
    assert payload["agent"]["stopped_reason"] == "model_stopped"
    assert len(payload["agent"]["tool_calls"]) == 2
    assert payload["agent"]["retry_rounds"] == 0  # оба вызова в раунде 1 — параллельные, не retry
    assert payload["agent"]["thought_count"] == 1


def test_log_ask_result_never_raises_on_malformed_result(caplog):
    with caplog.at_level(logging.WARNING, logger="app.observability.ask"):
        log_ask_result("вопрос", {"guardrail": "не dict, а строка — намеренно битый ввод"})

    # Не упало — залогировало предупреждение вместо json-строки результата.
    assert any("log_ask_result failed" in r.message for r in caplog.records)
