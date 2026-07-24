"""Матрица сравнений поверх evaluate_routes.py: {LLM-провайдер/модель} x
{маршрут vector/graph/agentic} x {эмбеддинги: база vs дообученные}.

Каждый сценарий — это отдельный запуск evaluate_routes.py в сабпроцессе со
своими env-переопределениями (settings — pydantic, читает env один раз при
импорте, поэтому сравнение конфигов в одном процессе не сработает: нужно
поднимать отдельный интерпретатор на сценарий). Это не лишний слой сложности,
а прямое следствие того, как Settings уже устроен в проекте (см.
app/core/llm.py:_ROLE_SETTINGS) — переиспользуем его, а не изобретаем
параллельный механизм конфигурации.

Ось "с дообучением / без" для ЭМБЕДДИНГОВ требует отдельной Qdrant-коллекции,
переиндексированной дообученной моделью — сравнение query-энкодера без
переиндексации бессмысленно (документы останутся в старом векторном
пространстве). Перед сценарием с QDRANT_COLLECTION-оверрайдом:
  1. Поднять второй TEI (или временно подменить model-id) на дообученном
     чекпоинте (training/biencoder_output/final после biencoder_finetune.ipynb).
  2. TEI_URL=http://localhost:<port> QDRANT_COLLECTION=warhammer_wiki_finetuned \\
     python scripts/ingest.py
Без этого шага сценарий с другой QDRANT_COLLECTION просто упадёт на
"коллекция не существует" — это ожидаемо, не баг.

Ось "с дообучением / без" для РОУТЕРА (LoRA-классификатор vs промпт-
классификатор) — другая по природе задача (accuracy классификации, не RAG-
метрики), сравнивается отдельным скриптом app/eval/compare_router.py.

Запуск: python -m app.eval.compare_configs --scenarios gigachat-baseline,openrouter-frontier
Без --scenarios — прогоняет все сценарии из SCENARIOS ниже.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

RESULTS_DIR = Path(project_root) / "app/eval/results"

# Каждый сценарий — независимый прогон evaluate_routes.py. env — оверрайды
# ПОВЕРХ текущего окружения/.env (пустые поля означают "как сейчас настроено").
# Добавляйте свои сценарии сюда же — это единственное место конфигурации.
SCENARIOS = [
    {
        "name": "gigachat-baseline",
        "routes": "vector,agentic",
        "env": {},
    },
    {
        "name": "openrouter-frontier",
        "routes": "vector,agentic",
        "env": {
            "GENERATION_LLM_PROVIDER": "openrouter",
            "GENERATION_LLM_MODEL": "qwen/qwen-2.5-72b-instruct",
        },
    },
    {
        # Требует предварительной переиндексации — см. докстринг модуля.
        "name": "finetuned-embeddings",
        "routes": "vector,agentic",
        "env": {
            "QDRANT_COLLECTION": "warhammer_wiki_finetuned",
            "TEI_URL": "http://localhost:8081",
        },
    },
]

MANIFEST_PATH_RE = re.compile(r"Конфиг прогона.*?:\s*(\S+\.json)")


def run_scenario(scenario: dict) -> dict | None:
    name = scenario["name"]
    env = {**os.environ, **scenario["env"]}
    cmd = [sys.executable, "-m", "app.eval.evaluate_routes", "--routes", scenario["routes"]]

    print(f"\n{'=' * 70}\n>>> Сценарий: {name}  (routes={scenario['routes']}, env={scenario['env']})\n{'=' * 70}")
    proc = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
    print(proc.stdout[-3000:])
    if proc.returncode != 0:
        print(f"[{name}] ПРОВАЛ (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        return None

    match = MANIFEST_PATH_RE.search(proc.stdout)
    if not match:
        print(f"[{name}] не удалось найти путь к манифесту в выводе — пропускаю агрегацию")
        return None

    manifest_path = Path(match.group(1))
    if not manifest_path.exists():
        print(f"[{name}] манифест не найден по пути {manifest_path}")
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scenario"] = name
    manifest["scenario_env"] = scenario["env"]
    return manifest


def print_master_table(manifests: list[dict]):
    cols = [
        ("title_hit@5", "{:.3f}"), ("citation_recall@5", "{:.3f}"),
        ("faithfulness", "{:.3f}"), ("answer_relevance", "{:.3f}"),
        ("language_quality", "{:.3f}"), ("avg_latency_ms", "{:.0f}"),
        ("avg_total_tokens", "{:.0f}"), ("refusal_rate", "{:.2f}"),
    ]
    header = f"{'scenario':<22}{'route':<10} " + " ".join(f"{name:<18}" for name, _ in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for manifest in manifests:
        for route_name, s in manifest.get("summaries", {}).items():
            cells = []
            for name, fmt in cols:
                v = s.get(name)
                cells.append(f"{'n/a':<18}" if v is None else f"{fmt.format(v):<18}")
            print(f"{manifest['scenario']:<22}{route_name:<10} " + " ".join(cells))
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Матрица сравнений LLM/эмбеддингов/маршрутов")
    parser.add_argument(
        "--scenarios", default=None,
        help="Через запятую — имена сценариев из SCENARIOS (по умолчанию все)",
    )
    args = parser.parse_args()

    known = {s["name"]: s for s in SCENARIOS}
    if args.scenarios:
        names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"Неизвестные сценарии: {unknown}. Доступны: {list(known)}")
        selected = [known[n] for n in names]
    else:
        selected = SCENARIOS

    manifests = []
    for scenario in selected:
        manifest = run_scenario(scenario)
        if manifest:
            manifests.append(manifest)

    if not manifests:
        raise SystemExit("Ни один сценарий не завершился успешно — сравнивать нечего")

    print_master_table(manifests)

    from app.eval.run_manifest import new_run_id
    combined_path = RESULTS_DIR / f"compare_{new_run_id()}.json"
    combined_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСводный отчёт: {combined_path}")


if __name__ == "__main__":
    main()
