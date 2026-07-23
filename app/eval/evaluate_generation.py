import json
import asyncio
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Any, Optional, List

# Добавляем корень проекта в sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.llm import llm_factory
from langfuse import observe
from pydantic import BaseModel, Field

class JudgeScore(BaseModel):
    """Схема оценки ответа экспертом"""
    context_relevance: float = Field(description="Полезность контекста (0-1)")
    faithfulness: float = Field(description="Отсутствие галлюцинаций (0-1)")
    answer_relevance: float = Field(description="Полнота ответа на вопрос (0-1)")
    language_quality: float = Field(
        description=(
            "Качество русского языка ответа (0-1): согласование слов, порядок слов, "
            "отсутствие калек с английского — не про факты и полноту, а чисто про грамотность"
        )
    )
    critique: str = Field(description="Обоснование на русском языке")

class WarJudge:
    def __init__(self):
        # role="judge" — намеренно не model_name="GigaChat-Pro": та же модель,
        # что и generation, инфлирует себе оценку (self-bias). См. комментарий
        # у JUDGE_LLM_PROVIDER/JUDGE_LLM_MODEL в app/core/config.py.
        self.llm = llm_factory.get_llm(temperature=0.0, role="judge")
        self.structured_llm = self.llm.with_structured_output(JudgeScore)

    @observe(name="Judge: Evaluate Response")
    async def evaluate_single_row(self, row: dict) -> Optional[JudgeScore]:
        system_prompt = (
            "Ты — эксперт-валидатор данных по вселенной Warhammer 40,000. "
            "Проведи строгий аудит ответа на основе предоставленного контекста.\n"
            "Отдельно оцени language_quality — качество русского языка ответа "
            "(согласование слов, порядок слов, отсутствие калек с английского). "
            "Это независимая ось: грамматически кривой, но фактически верный ответ "
            "должен получить низкий language_quality при высоком faithfulness, и наоборот."
        )
        
        context_text = "\n---\n".join(row.get("contexts", [])[:3])
        
        user_content = f"ВОПРОС: {row['question']}\nКОНТЕКСТ: {context_text}\nОТВЕТ: {row['answer']}"

        try:
            score: JudgeScore = await self.structured_llm.ainvoke([
                ("system", system_prompt), 
                ("user", user_content)
            ])

            return score
        except Exception as e:
            print(f"❌ Error evaluating {row.get('id')}: {e}")
            return None

def _latest_eval_dump(results_dir: Path) -> Optional[Path]:
    """evaluate_retrieval.py теперь пишет дамп с run_id в имени
    (eval_full_data_<run_id>.jsonl), поэтому без явного --input берём самый
    свежий по mtime. Старое фиксированное имя eval_full_data.jsonl тоже
    подхватится, если вдруг осталось с прошлых запусков."""
    candidates = sorted(results_dir.glob("eval_full_data_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        legacy = results_dir / "eval_full_data.jsonl"
        if legacy.exists():
            candidates = [legacy]
    return candidates[0] if candidates else None


async def run_mega_eval(input_path: Optional[str] = None):
    results_dir = Path("app/eval/results")
    resolved_input = Path(input_path) if input_path else _latest_eval_dump(results_dir)

    if resolved_input is None:
        print(f"❌ Не найдено ни одного eval_full_data*.jsonl в {results_dir} — "
              f"сначала прогони evaluate_retrieval.py")
        return
    input_path = resolved_input
    # Имя выхода наследует run_id входного файла, чтобы не затирать предыдущие оценки
    output_path = results_dir / (input_path.stem.replace("eval_full_data", "judge_results") + ".csv")

    if not input_path.exists():
        print(f"❌ Файл {input_path} не найден!")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    judge = WarJudge()
    evaluated_data = []

    print(f"🚀 Начало оценки | Объектов: {len(data)}")
    print("═" * 80)
    print(f"{'ID':<10} | {'Faith':<6} | {'Relv':<6} | {'Ctx':<6} | {'Lang':<6} | {'Critique'}")
    print("─" * 80)

    for row in data:
        score = await judge.evaluate_single_row(row)
        if score:
            # Печать в консоль всех метрик
            print(f"{str(row.get('id')):<10} | {score.faithfulness:<6.2f} | "
                  f"{score.answer_relevance:<6.2f} | {score.context_relevance:<6.2f} | "
                  f"{score.language_quality:<6.2f} | {score.critique[:50]}...")

            # Собираем данные для сохранения
            result_row = {
                **row,
                "judge_faithfulness": score.faithfulness,
                "judge_answer_relevance": score.answer_relevance,
                "judge_context_relevance": score.context_relevance,
                "judge_language_quality": score.language_quality,
                "judge_critique": score.critique
            }
            evaluated_data.append(result_row)

    # Сохранение результатов
    if evaluated_data:
        df = pd.DataFrame(evaluated_data)
        df.to_csv(output_path, index=False)

        # Финальная статистика
        print("═" * 80)
        print(f"📊 СРЕДНИЕ ПОКАЗАТЕЛИ:")
        print(f"Faithfulness: {df['judge_faithfulness'].mean():.2f}")
        print(f"Relevance:    {df['judge_answer_relevance'].mean():.2f}")
        print(f"Context:      {df['judge_context_relevance'].mean():.2f}")
        print(f"Language:     {df['judge_language_quality'].mean():.2f}")
        print(f"\n✅ Результаты сохранены в: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-judge оценка дампа из evaluate_retrieval.py")
    parser.add_argument(
        "--input", default=None,
        help="Путь к eval_full_data_<run_id>.jsonl (по умолчанию — самый свежий в app/eval/results)",
    )
    args = parser.parse_args()
    asyncio.run(run_mega_eval(args.input))