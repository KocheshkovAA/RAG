"""Capstone-прогон: реальная классификация (classify_route), а не bypass по
одному route, как в evaluate_routes.py. Единственный смысл этого скрипта —
показать "вот что выдаёт весь задеплоенный пайплайн" на полном датасете:
каждый вопрос идёт через тот же путь, что и живой /v1/ask (LLM-роутер решает
vector/graph/agentic сам), а не через принудительно выбранный маршрут.

Переиспользует evaluate_question_for_route/WarJudge из evaluate_routes.py —
единственная разница здесь в том, ЧТО передаётся как route: результат
classify_route(), а не значение из --routes.

Запуск: python -m app.eval.evaluate_capstone
"""
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.core.llm import resolve_role_config
from app.core.vectorrag import rag_chain
from app.core.agentic_rag import AgenticRAG
from app.core.lightrag_client import LightRAGClient
from app.core.orchestrator import WarhammerOrchestrator, RAGRoute
from app.core.usage import new_usage_handler
from app.eval.evaluate_generation import WarJudge
from app.eval.evaluate_routes import (
    RELEVANT_ROLES, _mean, _percentile, build_result_row, compute_agent_decision_aggregates,
    print_agent_decision_details,
)
from app.eval.run_manifest import new_run_id, write_manifest

RESULTS_DIR = Path(project_root) / "app/eval/results"


async def evaluate_question_classified(orchestrator: WarhammerOrchestrator, question_data: dict) -> dict:
    question = question_data["question"]
    usage_handler = new_usage_handler()
    started = time.perf_counter()

    route = await orchestrator.classify_route(question)

    if route == RAGRoute.GRAPH:
        result = await orchestrator._answer_graph(question, started, usage_handler=usage_handler, include_debug_docs=True)
    elif route == RAGRoute.AGENTIC:
        result = await orchestrator._answer_agentic(question, usage_handler=usage_handler, include_debug_docs=True)
    else:
        result = await orchestrator._answer_vector(question, usage_handler=usage_handler, include_debug_docs=True)

    return build_result_row(question_data, route, result)


async def _evaluate_with_retry(orchestrator: WarhammerOrchestrator, q: dict, max_attempts: int = 4) -> dict:
    for attempt in range(max_attempts):
        try:
            return await evaluate_question_classified(orchestrator, q)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "Too Many Requests" in str(e)
            if not is_rate_limit or attempt == max_attempts - 1:
                raise
            wait_s = 15 * (2 ** attempt)
            print(f"  q{q.get('id')} rate-limited, retry {attempt + 1}/{max_attempts} in {wait_s}s")
            await asyncio.sleep(wait_s)


def print_summary(summary: dict, route_counts: Counter):
    cols = [
        ("title_hit@5", "{:.3f}"), ("title_mrr@5", "{:.3f}"), ("citation_recall@5", "{:.3f}"),
        ("faithfulness", "{:.3f}"), ("answer_relevance", "{:.3f}"), ("context_relevance", "{:.3f}"),
        ("language_quality", "{:.3f}"),
        ("avg_latency_ms", "{:.0f}"), ("p95_latency_ms", "{:.0f}"),
        ("avg_total_tokens", "{:.0f}"), ("refusal_rate", "{:.2f}"),
    ]
    print("\n" + "=" * 100)
    print("CAPSTONE — вся система, реальная классификация маршрута")
    print(f"n={summary['n']}, распределение маршрутов: {dict(route_counts)}")
    print("-" * 100)
    for name, fmt in cols:
        v = summary.get(name)
        print(f"  {name:<20} {'n/a' if v is None else fmt.format(v)}")
    print("=" * 100)


async def main():
    run_id = new_run_id()
    results_path = RESULTS_DIR / f"capstone_{run_id}.jsonl"

    with open(settings.DATASET_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    light_rag = LightRAGClient()
    agentic_rag = AgenticRAG(rag_chain)
    orchestrator = WarhammerOrchestrator(vector_rag=rag_chain, light_rag=light_rag, agentic_rag=agentic_rag)
    judge = WarJudge()

    print(f">>> Capstone-прогон: {len(questions)} вопросов, реальная classify_route()")
    rows = []
    for q in questions:
        try:
            row = await _evaluate_with_retry(orchestrator, q)
        except Exception as e:
            print(f"  q{q.get('id')} failed: {e}")
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

    route_counts = Counter(r["route"] for r in rows)
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    tokens = [r["token_usage"]["total_tokens"] for r in rows if r.get("token_usage")]
    faithfulness = [r["judge_faithfulness"] for r in rows if "judge_faithfulness" in r]
    answer_rel = [r["judge_answer_relevance"] for r in rows if "judge_answer_relevance" in r]
    context_rel = [r["judge_context_relevance"] for r in rows if "judge_context_relevance" in r]
    language_quality = [r["judge_language_quality"] for r in rows if "judge_language_quality" in r]
    refusal_rate = sum(1 for r in rows if r["refused"]) / len(rows) if rows else 0.0
    has_retrieval_metrics = any("title_hit@5" in r for r in rows)

    summary = {
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
        "refusal_rate": refusal_rate,
        **compute_agent_decision_aggregates(rows),
    }

    print_summary(summary, route_counts)
    print_agent_decision_details({"classified": summary})
    manifest_path = write_manifest(
        RESULTS_DIR, run_id,
        dataset_path=settings.DATASET_PATH,
        routes=["classified"],
        roles={role: resolve_role_config(role) for role in RELEVANT_ROLES + ["router"]},
        results_path=str(results_path.relative_to(project_root)),
        summaries={"classified": summary, "route_distribution": dict(route_counts)},
    )
    print(f"\nПодробности по каждому вопросу: {results_path}")
    print(f"Конфиг прогона: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
