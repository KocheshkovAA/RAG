"""Сравнение промпт-классификатора (WarhammerOrchestrator.classify_route,
живой LLM-вызов) и обученного router LoRA (training/router_lora.ipynb) на
одном и том же честном held-out сплите (training/router_dataset_held_out.jsonl
— размечен дистилляцией продакшн-промпта, ни один пример не участвовал в
обучении LoRA, см. build_router_dataset.py).

Промпт-классификатор и LoRA-модель живут в разных, несовместимых окружениях
(langchain/gigachat в api-контейнере vs unsloth/transformers в training/.venv)
— поэтому сравнение разбито на два шага:

  1. Здесь, внутри api-контейнера или где доступен app.core.orchestrator:
     считает accuracy промпт-классификатора и (если файл уже есть) LoRA —
     оба выводятся рядом.
  2. training/predict_router_lora.py (запускается через training/.venv)
     генерирует предсказания LoRA и сохраняет их в JSON, который эта
     команда потом просто читает.

Запуск (внутри api-контейнера, есть настоящий LLM_PROVIDER):
    docker compose exec api python -m app.eval.compare_router

Если сначала прогнать training/predict_router_lora.py — в выводе появится
и колонка LoRA рядом с промпт-классификатором.
"""

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.core.orchestrator import WarhammerOrchestrator, RAGRoute

HELD_OUT_PATH = Path(project_root) / "training/router_dataset_held_out.jsonl"
LORA_PREDICTIONS_PATH = Path(project_root) / "training/router_lora_predictions.jsonl"


def load_held_out() -> list[dict]:
    rows = [json.loads(l) for l in HELD_OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def load_lora_predictions() -> dict[int, str] | None:
    if not LORA_PREDICTIONS_PATH.exists():
        return None
    rows = [json.loads(l) for l in LORA_PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {row["id"]: row["predicted_route"] for row in rows}


def _accuracy_and_confusion(rows: list[dict], predictions: dict[int, str]) -> tuple[float, Counter]:
    correct = 0
    confusion = Counter()  # (true, pred) -> count
    for row in rows:
        pred = predictions.get(row["id"])
        true = row["output"]
        confusion[(true, pred)] += 1
        if pred == true:
            correct += 1
    return correct / len(rows) if rows else 0.0, confusion


def print_confusion(name: str, accuracy: float, confusion: Counter):
    print(f"\n{name}: accuracy = {accuracy:.2%}")
    header_label = "true \\ pred"
    print(f"{header_label:<12}{'vector':<10}{'graph':<10}")
    for true_label in ("vector", "graph"):
        row = [confusion.get((true_label, pred_label), 0) for pred_label in ("vector", "graph")]
        print(f"{true_label:<12}{row[0]:<10}{row[1]:<10}")


async def run_prompt_classifier(rows: list[dict]) -> dict[int, str]:
    # vector_rag/light_rag/agentic_rag не нужны для classify_route — оно
    # использует только self.llm/self.system_prompt, безопасно передать None.
    orchestrator = WarhammerOrchestrator(vector_rag=None, light_rag=None, agentic_rag=None)
    predictions = {}
    for row in rows:
        route = await orchestrator.classify_route(row["input"])
        predictions[row["id"]] = route.value
    return predictions


async def main():
    rows = load_held_out()
    print(f"Held-out примеров: {len(rows)}")

    prompt_predictions = await run_prompt_classifier(rows)
    accuracy, confusion = _accuracy_and_confusion(rows, prompt_predictions)
    print_confusion("Промпт-классификатор (текущий продакшн)", accuracy, confusion)

    lora_predictions = load_lora_predictions()
    if lora_predictions is None:
        print(
            f"\nLoRA-предсказаний нет ({LORA_PREDICTIONS_PATH} не найден) — "
            "сначала прогоните training/predict_router_lora.py в training/.venv"
        )
        return

    lora_accuracy, lora_confusion = _accuracy_and_confusion(rows, lora_predictions)
    print_confusion("Router LoRA", lora_accuracy, lora_confusion)

    print(f"\nИтог: промпт-классификатор {accuracy:.2%} vs LoRA {lora_accuracy:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
