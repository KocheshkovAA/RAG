"""Сравнение трёх маршрутов (vector / graph / agentic) на одном датасете.

По умолчанию гоняет только vector и agentic (дёшево, без зависимости от
живого LightRAG). graph добавляется явно через --routes vector,graph,agentic —
он делает свои внешние LLM-вызовы и требует поднятого сервиса lightrag.

Классификатор (WarhammerOrchestrator.classify_route) намеренно обходится —
каждый маршрут вызывается напрямую (_answer_vector/_answer_graph/_answer_agentic),
чтобы в сравнении не было стоимости/вариативности роутера.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.core.guardrails import refusal_reason
from app.core.llm import resolve_role_config
from app.core.vectorrag import rag_chain
from app.core.agentic_rag import AgenticRAG
from app.core.lightrag_client import LightRAGClient
from app.core.orchestrator import WarhammerOrchestrator, RAGRoute
from app.core.usage import new_usage_handler
from app.eval.metrics import compute_retrieval_metrics
from app.eval.evaluate_generation import WarJudge
from app.eval.run_manifest import new_run_id, write_manifest

K_VALUES = [3, 5, 10, 20]
RESULTS_DIR = Path(project_root) / "app/eval/results"

ALL_ROUTES = [RAGRoute.VECTOR, RAGRoute.GRAPH, RAGRoute.AGENTIC]

# Роли, чей провайдер/модель реально влияют на цифры в этом харнессе.
# "router" намеренно не включён — классификатор здесь обходится (см. докстринг
# модуля), "persona" сюда не относится (это /v1/debate, не /v1/ask).
RELEVANT_ROLES = ["generation", "faithfulness", "agentic", "judge"]


def _expected_titles_quotes(question_data: dict) -> tuple[list[str], list[str]]:
    expected_titles = question_data.get("article_title", [])
    if isinstance(expected_titles, str):
        expected_titles = [expected_titles]
    expected_quotes = question_data.get("quote", [])
    if isinstance(expected_quotes, str):
        expected_quotes = [expected_quotes]
    expected_quotes = [q for q in expected_quotes if q and isinstance(q, str)]
    return expected_titles, expected_quotes


def build_result_row(question_data: dict, route: RAGRoute, result: dict) -> dict:
    """Общая сборка строки для evaluate_routes.py и evaluate_capstone.py — не только
    "хватило ли контекста" (retrieval-метрики), но и агентные решения: почему отказал,
    сколько раундов поиска потрачено впустую (retry без новой evidence), включалась ли
    faithfulness-проверка. Без этого eval видит только конечный результат, а не то, как
    агент до него дошёл — то, ради чего в первую очередь стоит расширять eval для agentic-
    маршрута, а не только для retrieval (см. docs/agent-architecture.md)."""
    token_total = (result.get("token_usage") or {}).get("total") if result.get("token_usage") else None
    guardrail = result.get("guardrail") or {}
    agentic_meta = result.get("agentic") or {}
    faith = guardrail.get("faithfulness") or {}

    tool_calls_made = agentic_meta.get("tool_calls_made") or []
    rounds = sorted({tc["round"] for tc in tool_calls_made})
    first_round = rounds[0] if rounds else None
    retry_calls = [tc for tc in tool_calls_made if first_round is not None and tc["round"] > first_round]

    row = {
        "id": question_data.get("id"),
        "question": question_data["question"],
        "route": route.value,
        "answer": result.get("answer"),
        "latency_ms": result.get("latency_ms"),
        "refused": bool(guardrail.get("refused")),
        "refusal_reason": refusal_reason(result),
        "degraded": result.get("degraded") or [],
        "token_usage": token_total,
        "iterations": agentic_meta.get("iterations"),
        "stopped_reason": agentic_meta.get("stopped_reason"),
        "tool_calls_total": len(tool_calls_made),
        "tool_calls_wasted": sum(1 for tc in tool_calls_made if not tc.get("error") and tc.get("new", 0) == 0),
        "retry_rounds": len({tc["round"] for tc in retry_calls}),
        "retry_added_evidence": (any(tc.get("new", 0) > 0 for tc in retry_calls) if retry_calls else None),
        "thought_count": len(agentic_meta.get("thoughts") or []),
        "faithfulness_checked": bool(faith) and not faith.get("skipped", True),
        "contexts": [],
    }

    debug_docs = result.get("_debug_docs")
    if debug_docs:
        expected_titles, expected_quotes = _expected_titles_quotes(question_data)
        retrieved_titles = [d.metadata.get("article_name", "UNKNOWN") for d in debug_docs]
        retrieved_contents = [d.page_content for d in debug_docs]
        row["contexts"] = retrieved_contents
        row.update(
            compute_retrieval_metrics(retrieved_titles, retrieved_contents, expected_titles, expected_quotes, K_VALUES)
        )

    return row


def compute_agent_decision_aggregates(rows: list[dict]) -> dict:
    """Агрегаты поверх build_result_row: разбивка причин отказа, как часто faithfulness-
    проверка реально ловит проблему (а не просто включается), окупается ли retry поиском
    новых данных. ToT/branching здесь нет метрики намеренно — паттерн не реализован
    (см. docs/agent-architecture.md, раздел Reasoning modes), нечего измерять."""
    refusal_reasons = Counter(r["refusal_reason"] for r in rows if r.get("refusal_reason"))
    faith_checked = [r for r in rows if r.get("faithfulness_checked")]
    faith_caught = sum(1 for r in faith_checked if (r.get("refusal_reason") or "").startswith("faithfulness"))
    tool_rows = [r for r in rows if r.get("tool_calls_total")]
    retry_rows = [r for r in rows if r.get("retry_rounds")]
    retry_justified = sum(1 for r in retry_rows if r.get("retry_added_evidence"))

    return {
        "refusal_reasons": dict(refusal_reasons),
        "faithfulness_check_rate": len(faith_checked) / len(rows) if rows else 0.0,
        "faithfulness_catch_rate": (faith_caught / len(faith_checked)) if faith_checked else None,
        "avg_wasted_tool_calls": _mean([r["tool_calls_wasted"] for r in tool_rows]) if tool_rows else None,
        "retry_occurred_n": len(retry_rows),
        "retry_justified_rate": (retry_justified / len(retry_rows)) if retry_rows else None,
    }


def print_agent_decision_details(summaries: dict[str, dict]) -> None:
    for route_name, s in summaries.items():
        reasons = s.get("refusal_reasons") or {}
        print(f"\n[{route_name}] причины отказа: {reasons if reasons else '(отказов нет)'}")
        fc_rate = s.get("faithfulness_check_rate")
        fcatch_rate = s.get("faithfulness_catch_rate")
        print(
            f"[{route_name}] faithfulness: проверялась в {fc_rate:.1%} ответов, "
            f"поймала проблему в {fcatch_rate:.1%} проверенных" if fcatch_rate is not None
            else f"[{route_name}] faithfulness: проверялась в {fc_rate:.1%} ответов, проблем не поймано"
        )
        if s.get("retry_occurred_n"):
            rj = s.get("retry_justified_rate")
            print(
                f"[{route_name}] retry (>1 раунда поиска): {s['retry_occurred_n']} вопросов, "
                f"из них оправдан (нашёл новую evidence) в {rj:.1%}; "
                f"среднее число впустую потраченных tool-раундов: {s.get('avg_wasted_tool_calls'):.2f}"
            )


async def evaluate_question_for_route(orchestrator: WarhammerOrchestrator, question_data: dict, route: RAGRoute) -> dict:
    question = question_data["question"]
    usage_handler = new_usage_handler()
    started = time.perf_counter()

    if route == RAGRoute.VECTOR:
        result = await orchestrator._answer_vector(question, usage_handler=usage_handler, include_debug_docs=True)
    elif route == RAGRoute.GRAPH:
        result = await orchestrator._answer_graph(
            question, started, usage_handler=usage_handler, include_debug_docs=True
        )
    elif route == RAGRoute.AGENTIC:
        result = await orchestrator._answer_agentic(question, usage_handler=usage_handler, include_debug_docs=True)
    else:
        raise ValueError(f"Unknown route: {route}")

    return build_result_row(question_data, route, result)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return s[idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def _evaluate_with_retry(orchestrator: WarhammerOrchestrator, q: dict, route: RAGRoute,
                                max_attempts: int = 4) -> dict:
    """GigaChat отдаёт 429 под нагрузкой полного 56-вопросного прогона (не
    наблюдается на маленьких sample-прогонах) — экспоненциальный backoff для rate-limit.

    Retry также и на прочих исключениях (короче backoff) — регресс: полный 60-вопросный
    прогон один раз потерял строку на 'NoneType' object is not iterable, не
    воспроизвелось на изолированном повторе того же вопроса сразу после — похоже на
    разовый транзиентный сбой конкретного LLM-вызова под нагрузкой, не системную ошибку.
    Раньше такие ошибки не ретраились вообще (условие ловило только rate-limit по тексту),
    из-за чего одна такая накладка молча теряла строку из отчёта и слегка смещала агрегаты."""
    for attempt in range(max_attempts):
        try:
            return await evaluate_question_for_route(orchestrator, q, route)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
            wait_s = 15 * (2 ** attempt) if is_rate_limit else 3 * (attempt + 1)
            reason = "rate-limited" if is_rate_limit else f"error ({type(e).__name__}: {e})"
            print(f"  [{route.value}] q{q.get('id')} {reason}, retry {attempt + 1}/{max_attempts} in {wait_s}s")
            await asyncio.sleep(wait_s)


async def run_route(orchestrator: WarhammerOrchestrator, questions: list[dict], route: RAGRoute,
                     judge: WarJudge, results_path: Path) -> dict:
    rows = []
    for q in questions:
        try:
            row = await _evaluate_with_retry(orchestrator, q, route)
        except Exception as e:
            print(f"  [{route.value}] q{q.get('id')} failed: {e}")
            continue
        if row.get("answer") and not row["refused"] and row.get("contexts"):
            score = await judge.evaluate_single_row(row)
            if score:
                row["judge_faithfulness"] = score.faithfulness
                row["judge_answer_relevance"] = score.answer_relevance
                row["judge_context_relevance"] = score.context_relevance
                row["judge_language_quality"] = score.language_quality
                row["judge_critique"] = score.critique
        rows.append(row)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: v for k, v in row.items() if k != "contexts"}, ensure_ascii=False) + "\n")

    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    tokens = [r["token_usage"]["total_tokens"] for r in rows if r.get("token_usage")]
    iterations = [r["iterations"] for r in rows if r.get("iterations") is not None]
    faithfulness = [r["judge_faithfulness"] for r in rows if "judge_faithfulness" in r]
    answer_rel = [r["judge_answer_relevance"] for r in rows if "judge_answer_relevance" in r]
    context_rel = [r["judge_context_relevance"] for r in rows if "judge_context_relevance" in r]
    language_quality = [r["judge_language_quality"] for r in rows if "judge_language_quality" in r]
    refusal_rate = sum(1 for r in rows if r["refused"]) / len(rows) if rows else 0.0

    has_retrieval_metrics = any(f"title_hit@5" in r for r in rows)

    return {
        "n": len(rows),
        "title_hit@5": _mean([r.get("title_hit@5", 0) for r in rows]) if has_retrieval_metrics else None,
        "title_mrr@5": _mean([r.get("title_mrr@5", 0) for r in rows]) if has_retrieval_metrics else None,
        "citation_recall@5": _mean([r.get("citation_recall@5", 0) for r in rows]) if has_retrieval_metrics else None,
        "faithfulness": _mean(faithfulness) if faithfulness else None,
        "answer_relevance": _mean(answer_rel) if answer_rel else None,
        "context_relevance": _mean(context_rel) if context_rel else None,
        "language_quality": _mean(language_quality) if language_quality else None,
        "avg_latency_ms": _mean(latencies),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_total_tokens": _mean(tokens) if tokens else None,
        "avg_iterations": _mean(iterations) if iterations else None,
        "refusal_rate": refusal_rate,
        **compute_agent_decision_aggregates(rows),
    }


def print_comparison_table(summaries: dict[str, dict]):
    cols = [
        ("title_hit@5", "{:.3f}"), ("title_mrr@5", "{:.3f}"), ("citation_recall@5", "{:.3f}"),
        ("faithfulness", "{:.3f}"), ("answer_relevance", "{:.3f}"), ("context_relevance", "{:.3f}"),
        ("language_quality", "{:.3f}"),
        ("avg_latency_ms", "{:.0f}"), ("p95_latency_ms", "{:.0f}"),
        ("avg_total_tokens", "{:.0f}"), ("avg_iterations", "{:.2f}"), ("refusal_rate", "{:.2f}"),
    ]
    header = f"{'route':<10} " + " ".join(f"{name:<18}" for name, _ in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for route_name, s in summaries.items():
        cells = []
        for name, fmt in cols:
            v = s.get(name)
            cells.append(f"{'n/a':<18}" if v is None else f"{fmt.format(v):<18}")
        print(f"{route_name:<10} " + " ".join(cells))
    print("=" * len(header))


async def main():
    parser = argparse.ArgumentParser(description="Сравнение vector/graph/agentic маршрутов")
    parser.add_argument(
        "--routes", default="vector,agentic",
        help="Через запятую: vector,graph,agentic (по умолчанию без graph — дешевле и без LightRAG)",
    )
    args = parser.parse_args()
    selected = [RAGRoute(r.strip()) for r in args.routes.split(",") if r.strip()]

    run_id = new_run_id()
    results_path = RESULTS_DIR / f"route_comparison_{run_id}.jsonl"

    with open(settings.DATASET_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    light_rag = LightRAGClient()
    agentic_rag = AgenticRAG(rag_chain)
    orchestrator = WarhammerOrchestrator(vector_rag=rag_chain, light_rag=light_rag, agentic_rag=agentic_rag)
    judge = WarJudge()

    summaries = {}
    for route in selected:
        print(f"\n>>> Прогон маршрута: {route.value} ({len(questions)} вопросов)")
        summaries[route.value] = await run_route(orchestrator, questions, route, judge, results_path)

    print_comparison_table(summaries)
    print_agent_decision_details(summaries)
    manifest_path = write_manifest(
        RESULTS_DIR, run_id,
        dataset_path=settings.DATASET_PATH,
        routes=[r.value for r in selected],
        roles={role: resolve_role_config(role) for role in RELEVANT_ROLES},
        results_path=str(results_path.relative_to(project_root)),
        summaries=summaries,
    )
    print(f"\nПодробности по каждому вопросу: {results_path}")
    print(f"Конфиг прогона (модели/провайдеры по ролям): {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
