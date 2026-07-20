export DOCKER_API_VERSION := 1.44

.PHONY: help debug-up up down eval test smoke ui

help:                   ## Показать все доступные команды
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

debug-up:               ## Стек + Langfuse/Clickhouse/Minio (нужен для трейсинга)
	docker compose --profile debug up -d

up:                     ## Запустить основной стек без debug
	docker compose up -d

down:                   ## Остановить и удалить все контейнеры
	docker compose down --remove-orphans --volumes

eval:                   ## Полная оценка ретривера (медленно, ~60 вопросов)
	docker compose exec api python /app/app/eval/evaluate_retrieval.py

test:                   ## Быстрые unit-тесты guardrails (без LLM)
	docker compose exec api python -m pytest /app/tests -q

smoke:                  ## Smoke-кейсы по живому /v1/ask (оффтоп + факт)
	docker compose exec api python -m app.eval.smoke_api --api-url http://api:8000

ui:                     ## Поднять/пересоздать Chainlit UI (http://localhost:8501)
	docker compose up -d --build chainlit

logs:                   ## Показать логи
	docker compose logs -f
