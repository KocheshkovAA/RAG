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


async def evaluate_question_for_route(orchestrator: WarhammerOrchestrator, question_data: dict, route: RAGRoute) -> dict:
    question = question_data["question"]
    usage_handler = new_usage_handler()
    started = time.perf_counter()

    if route == RAGRoute.VECTOR:
        result = await orchestrator._answer_vector(question, usage_handler=usage_handler, include_debug_docs=True)
    elif route == RAGRoute.GRAPH:
        result = await orchestrator._answer_graph(question, started, usage_handler=usage_handler)
    elif route == RAGRoute.AGENTIC:
        result = await orchestrator._answer_agentic(question, usage_handler=usage_handler, include_debug_docs=True)
    else:
        raise ValueError(f"Unknown route: {route}")

    token_total = (result.get("token_usage") or {}).get("total") if result.get("token_usage") else None

    row = {
        "id": question_data.get("id"),
        "question": question,
        "route": route.value,
        "answer": result.get("answer"),
        "latency_ms": result.get("latency_ms"),
        "refused": bool((result.get("guardrail") or {}).get("refused")),
        "degraded": result.get("degraded") or [],
        "token_usage": token_total,
        "iterations": (result.get("agentic") or {}).get("iterations"),
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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return s[idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def run_route(orchestrator: WarhammerOrchestrator, questions: list[dict], route: RAGRoute,
                     judge: WarJudge, results_path: Path) -> dict:
    rows = []
    for q in questions:
        try:
            row = await evaluate_question_for_route(orchestrator, q, route)
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
