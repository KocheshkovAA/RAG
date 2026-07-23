import sys
import os
import asyncio
import json
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client, propagate_attributes

# Настройка путей
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.core.llm import resolve_role_config
from app.core.vectorrag import rag_chain
from app.core.reranker import reranker
from app.eval.metrics import compute_retrieval_metrics
from app.eval.run_manifest import new_run_id, write_manifest

K_VALUES = [3, 5, 10, 20]
from datetime import datetime

RESULTS_DIR = Path(project_root) / "app/eval/results"
# Один run_id на весь прогон (обе фазы — без/с реранкером — пишут в один и тот
# же файл, как и раньше), чтобы разные вызовы main() не затирали друг друга.
RUN_ID = new_run_id()
RESULTS_PATH = RESULTS_DIR / f"eval_full_data_{RUN_ID}.jsonl"
langfuse_client = get_client()

@observe(name="Retrieval + Generation Collector")
async def evaluate_one(question_data, use_rerank: bool, collect_answers: bool = False):
    q_id = question_data.get("id", "??")
    question = question_data["question"]

    # ── Подготовка эталонов ──
    expected_titles = question_data.get("article_title", [])
    if isinstance(expected_titles, str): expected_titles = [expected_titles]

    expected_quotes = question_data.get("quote", [])
    if isinstance(expected_quotes, str): expected_quotes = [expected_quotes]
    expected_quotes = [q for q in expected_quotes if q and isinstance(q, str)]

    # Настройки
    reranker.enabled = use_rerank
    settings.QUERY_OPTIMIZER_ENABLED = False

    try:
        handler = CallbackHandler()
        # 1. Получаем документы через твой RAG класс
        final_docs, _degraded = await rag_chain.get_relevant_documents(
            question, handler=handler
        )

        if not final_docs:
            return {"error": "No documents retrieved"}

        retrieved_titles = [doc.metadata.get("article_name", "UNKNOWN") for doc in final_docs]
        retrieved_contents = [doc.page_content for doc in final_docs]

        metrics = compute_retrieval_metrics(
            retrieved_titles, retrieved_contents, expected_titles, expected_quotes, K_VALUES
        )

        # 4. ── Генерация ответа и сохранение дампа для RAGAS ──
        if collect_answers:
            # Важно: вызываем саму генерацию ответа
            answer_text = await rag_chain.chain.ainvoke(
                {"docs": final_docs, "question": question},
                config={"callbacks": [handler]} # Используем тот же хендлер для Langfuse
            )
            
            dump_entry = {
                "id": q_id,
                "question": question,
                "answer": answer_text,
                "contexts": retrieved_contents,  # Сохраняем все чанки (переменная K для RAGAS)
                "retrieval_metrics": metrics,     # Все твои расчеты Hit/Recall/MRR
                "expected_titles": expected_titles,
                "retrieved_titles": retrieved_titles,
                "rerank_enabled": use_rerank,
                "timestamp": datetime.now().isoformat()
            }

            # Пишем в JSONL (папка results должна существовать)
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(dump_entry, ensure_ascii=False) + "\n")
        
        return metrics

    except Exception as e:
        print(f"q{q_id} | Ошибка: {e}")
        return {"error": str(e)}

async def run_evaluation(use_rerank: bool, collect_answers: bool = False):
    dataset_path = Path(settings.DATASET_PATH)
    with open(dataset_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"\n🚀 Оценка | Rerank = {use_rerank} | Вопросов: {len(questions)}")

    results = []
    
    for q in tqdm_asyncio(questions, desc=f"Rerank={use_rerank}"):
        res = await evaluate_one(q, use_rerank, collect_answers)
        results.append(res)
        
        if collect_answers:
            await asyncio.sleep(0.1) 

    valid = [r for r in results if r and "error" not in r]
    if not valid: 
        print("❌ Ошибка: Не получено ни одного валидного результата.")
        return {}

    n = len(valid)
    # Собираем среднее по всем метрикам
    aggregated = {key: sum(r.get(key, 0.0) for r in valid) / n for key in valid[0].keys()}
    return aggregated

def print_table(agg, use_rerank: bool):
    mode = "WITH RERANKER" if use_rerank else "WITHOUT RERANKER"
    print("\n" + "═" * 100)
    print(f"          SUMMARY — {mode}")
    print("═" * 100)
    print(f"{'Метрика':<28} {'@3':<10} {'@5':<10} {'@10':<10} {'@20':<10}")
    print("-" * 100)
    
    for base_metric in ["title_hit", "title_recall", "title_precision", "title_mrr", 
                        "citation_hit", "citation_recall", "citation_precision", "citation_mrr"]:
        row = f"{base_metric:<28} "
        for k in K_VALUES:
            val = agg.get(f"{base_metric}@{k}", 0)
            row += f"{val:<10.3f} "
        print(row)

async def main():
    # 1. Без реранкера
    agg_no = await run_evaluation(use_rerank=False)
    print_table(agg_no, False)

    # 2. С реранкером
    agg_yes = await run_evaluation(use_rerank=True, collect_answers=True)
    print_table(agg_yes, True)

    # 3. Сравнение
    print("\n" + "═" * 100)
    print(f"{'СРАВНЕНИЕ @5':<28} | {'Base':<10} | {'Rerank':<10} | {'Delta'}")
    print("-" * 100)
    comparison_at_5 = {}
    for m in ["title_hit@5", "title_mrr@5", "citation_recall@5", "citation_precision@5"]:
        v1, v2 = agg_no.get(m, 0), agg_yes.get(m, 0)
        comparison_at_5[m] = {"no_rerank": v1, "rerank": v2, "delta": v2 - v1}
        print(f"{m:<28} | {v1:<10.3f} | {v2:<10.3f} | {v2-v1:+10.3f}")

    manifest_path = write_manifest(
        RESULTS_DIR, RUN_ID,
        dataset_path=str(settings.DATASET_PATH),
        roles={"generation": resolve_role_config("generation")},
        results_path=str(RESULTS_PATH.relative_to(project_root)),
        summary_no_rerank=agg_no,
        summary_rerank=agg_yes,
        comparison_at_5=comparison_at_5,
    )
    print(f"\nДамп для RAGAS/judge: {RESULTS_PATH}")
    print(f"Конфиг прогона: {manifest_path}")

if __name__ == "__main__":
    asyncio.run(main())