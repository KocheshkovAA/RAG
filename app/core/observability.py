"""Structured observability logging for the agent loop — independent of Langfuse.

`score_ask_result()` (`app/core/tracing.py`) only writes business metrics to Langfuse, and
Langfuse self-disables cleanly when not configured/reachable (`init_tracing()`) — meaning without
a live, authenticated Langfuse instance there is currently NO durable trail of what the agent
actually did on a given request. `log_ask_result()` closes that gap with one structured JSON log
line per `/v1/ask`, covering the same "agent decision" fields eval already computes for offline
runs (`app/eval/evaluate_routes.py`) — route, tool calls, retries, reflection result, refusal
reason, token/cost/latency — so this data exists even when nobody has `make debug-up` running.
Greppable via `docker compose logs api | grep '"logger": "app.observability.ask"'` or any log
shipper pointed at stdout; deliberately not a new external dependency.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.guardrails import refusal_reason

logger = logging.getLogger("app.observability.ask")
# Ничего в проекте не вызывает logging.basicConfig() — эффективный уровень root-логгера
# остаётся дефолтным WARNING, и обычный logger.info() здесь молча проглатывался бы (не
# доходил ни до одного handler'а). Раз весь смысл этого модуля — гарантированно видимый
# канал наблюдаемости независимо от остальной конфигурации логирования проекта, логгер
# настраивает сам себе level+handler, а не полагается на глобальный setup.
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


def _agent_decision_summary(result: dict[str, Any]) -> dict[str, Any]:
    agentic_meta = result.get("agentic") or {}
    tool_calls_made = agentic_meta.get("tool_calls_made") or []
    rounds = sorted({tc["round"] for tc in tool_calls_made})
    first_round = rounds[0] if rounds else None
    retry_calls = [tc for tc in tool_calls_made if first_round is not None and tc["round"] > first_round]

    return {
        # "single_shot" здесь означает "без ReAct-цикла" (vector/graph) — не то же самое, что
        # CoT/ToT из docs/agent-architecture.md; в проде реализован только ReAct.
        "reasoning_mode": "react" if agentic_meta else "single_shot",
        "iterations": agentic_meta.get("iterations"),
        "stopped_reason": agentic_meta.get("stopped_reason"),
        "tool_calls": [
            {
                "round": tc.get("round"),
                "query": tc.get("query"),
                "found": tc.get("found"),
                "new": tc.get("new"),
                "error": tc.get("error"),
            }
            for tc in tool_calls_made
        ],
        "retry_rounds": len({tc["round"] for tc in retry_calls}),
        "thought_count": len(agentic_meta.get("thoughts") or []),
    }


def _reflection_summary(result: dict[str, Any]) -> dict[str, Any]:
    guardrail = result.get("guardrail") or {}
    gate = guardrail.get("retrieval_gate") or {}
    faith = guardrail.get("faithfulness") or {}
    return {
        "retrieval_gate_passed": gate.get("passed"),
        "retrieval_max_score": gate.get("max_score"),
        "faithfulness_skipped": faith.get("skipped"),
        "faithfulness_score": faith.get("faithfulness_score"),
        "unsupported_claims_count": len(faith.get("unsupported_claims") or []),
        "claims_checked": faith.get("claims_checked"),
    }


def log_ask_result(question: str, result: dict[str, Any]) -> None:
    """Один структурированный JSON-лог на каждый ответ — не должен уметь ронять запрос
    (см. try/except): наблюдаемость — побочный эффект, а не часть critical path."""
    try:
        guardrail = result.get("guardrail") or {}
        token_usage = (result.get("token_usage") or {}).get("total") or {}
        payload = {
            "question": question[:300],
            "route": result.get("mode"),
            "refused": bool(guardrail.get("refused")),
            "refusal_reason": refusal_reason(result),
            "degraded": result.get("degraded") or [],
            "cached": bool(result.get("cached")),
            "latency_ms": result.get("latency_ms"),
            "token_usage": token_usage,
            "agent": _agent_decision_summary(result),
            "reflection": _reflection_summary(result),
        }
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logger.warning("log_ask_result failed: %s", e)
