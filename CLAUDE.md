# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hybrid RAG system answering Warhammer 40k lore questions in Russian, backed by a wiki scrape from `warhammer40k.fandom.com/ru`. It routes each question to either classic vector retrieval or graph retrieval (LightRAG) depending on whether the question needs multi-hop entity/event reasoning, and refuses to answer when retrieval/faithfulness confidence is too low rather than hallucinating.

## Commands

Everything runs through Docker Compose; the `api` container is where app code executes.

```bash
make up          # start core stack (api, qdrant, tei, vllm-reranker, redis, lightrag, gigachat-adapter)
make debug-up    # core stack + Langfuse/Clickhouse/Minio (tracing profile)
make down        # stop and remove all containers + volumes
make ui          # build/start the Chainlit chat UI (http://localhost:8501)
make eval        # full retriever+generation evaluation over data/eval dataset (~60 questions, slow)
make test        # fast guardrail unit tests, no LLM/Docker deps: docker compose exec api python -m pytest /app/tests -q
make smoke       # smoke tests against a live /v1/ask (offtopic + factual cases), needs the stack up
make logs        # follow logs for all services
```

Unit tests (`tests/`) only touch `app/core/guardrails.py` and have no network/LLM dependency, so they can also run directly on the host if the Python deps are installed: `pytest tests/test_guardrails.py -q`, or `pytest tests/test_guardrails.py::test_gate_rejects_offtopic_scores` for a single case. Config is in `pytest.ini` (`testpaths = tests`, `asyncio_mode = auto`).

`tei` and `vllm-reranker` require an NVIDIA GPU (see `deploy.resources.reservations.devices` in `docker-compose.yml`); there's a commented-out CPU variant of the `tei` service for machines without one.

## Architecture

### Request flow

`app/main.py` (FastAPI) → `app/api/routes.py` (`POST /v1/ask`) → `WarhammerOrchestrator.answer()` in `app/core/orchestrator.py`:

1. An LLM call classifies the question as `vector` (single facts, unit/character descriptions, off-topic — off-topic is deliberately forced to `vector` so the retrieval-gate refusal path catches it) or `graph` (relationships, causal chains across factions/eras).
2. `vector` route → `RAG.answer()` in `app/core/vectorrag.py`.
3. `graph` route → first runs a vector-side "domain probe" (must pass the retrieval gate) to stop off-topic questions from bypassing guardrails via the graph path, then queries `LightRAGClient` (`app/core/lightrag_client.py`, HTTP to the separate `lightrag` container). If LightRAG is circuit-broken/unavailable it falls back to the vector pipeline and tags the response `degraded: ["lightrag"]`.

Every response is a dict with `answer`, `sources`, `guardrail` (retrieval_gate + faithfulness verdicts), `degraded`, `cached`, `mode`, `latency_ms` — the Chainlit UI and smoke tests both key off this shape.

### Vector RAG pipeline (`app/core/vectorrag.py`)

Redis answer cache → `Retriever` (hybrid dense+sparse Qdrant search, `app/core/retriever.py`: dense via TEI embeddings + sparse via fastembed BM25, RRF-style hybrid through `langchain_qdrant`) → `Reranker` (`app/core/reranker.py`, calls a vLLM `bge-reranker-v2-m3` endpoint, sigmoid-normalized scores) → `RetrievalGate` guardrail → LLM generation (`app/chains/prompts.py` system prompt, strict "context-only" instructions) → conditional `AnswerFaithfulnessGuard` verification → `SourceExtractor`.

Successful, non-degraded, non-refused answers are cached in Redis (`app/core/cache.py`); refusals/degraded results are never cached.

### Guardrails (`app/core/guardrails.py`)

Two independent gates, both configured in `app/core/config.py`:
- `RetrievalGate`: drops documents below `RETRIEVAL_MIN_SCORE`/`RERANK_MIN_SCORE` (uses `rerank_score` if present, else `hybrid_score`); empty result after filtering → refusal with `INSUFFICIENT_INFO_MESSAGE`.
- `AnswerFaithfulnessGuard`: an LLM verifies the generated answer's claims are grounded in the retrieved context. Skipped only when *not* degraded, reranked, and max score ≥ `FAITHFULNESS_SKIP_ABOVE` (0.55–0.80 is treated as a "gray zone" always requiring verification).

These are the two things `tests/test_guardrails.py` exercises directly (no Docker/LLM needed).

### Resilience (`app/core/resilience.py`, `app/core/health.py`)

Shared `CircuitBreaker` instances per external dependency (`reranker`, `lightrag`, `tei`, defined in `health.py`) wrap calls via `call_with_circuit`, which returns `(result, degraded: bool)` and falls back gracefully (e.g. reranker degrades to un-reranked top-k, LightRAG degrades to the vector pipeline) instead of failing the request. `GET /v1/ready` reports per-dependency health plus circuit state; `GET /health` is a bare liveness check. Request-level timeouts go through `with_timeout` (`REQUEST_TIMEOUT_SEC`).

### LLM provider (`app/core/llm.py`)

`LLMFactory` switches on `settings.LLM_PROVIDER`: `gigachat` (via `langchain_gigachat`, routed through the `gigachat-adapter` container) or `openrouter` (via `langchain_openai.ChatOpenAI` pointed at OpenRouter). Used for routing classification, answer generation, and faithfulness verification — each call site can pass its own `temperature`.

### Tracing (`app/core/tracing.py`)

Langfuse is optional and self-disables cleanly if keys/`LANGFUSE_ENABLE_TRACE` aren't set or `auth_check()` fails (`init_tracing()` at API startup). When enabled, `score_ask_result()` writes business metrics (latency, refused, cache_hit, degraded, retrieval/faithfulness scores) onto the current trace, and `session_id`/`user_id` from the request are propagated as trace attributes. Langfuse itself (+Clickhouse/Postgres/Minio) only runs under `make debug-up` (`profiles: ["debug"]` in `docker-compose.yml`).

### Data pipeline (`scripts/`)

`loader.py` scrapes the fandom wiki via its MediaWiki API into `raw_warhammer_data.jsonl` → `parser.py` (`WarhammerWikiParser`) converts HTML to Markdown, splits on headers then recursively by length, extracts infobox data as its own chunk type, and writes `data/processed/processed_chunks.jsonl` → `ingest.py` embeds (dense+sparse) and upserts into the Qdrant `warhammer_wiki` collection (dropping/recreating it each run; indexing threshold is disabled during bulk load and restored after).

### Evaluation (`app/eval/`)

- `evaluate_retrieval.py`: runs the real `rag_chain` retrieval against `DATASET_PATH` (60 hand-labeled questions with expected article titles/quotes), computes hit/recall/precision/MRR at k∈{3,5,10,20} for both title-match and citation-match, with/without reranking, and dumps per-question generations for RAGAS-style judging.
- `smoke_api.py` (`make smoke`): black-box cases from `data/eval/smoke_cases.json` against a *live* `/v1/ask`, asserting expected refusal/answer behavior, latency ceilings, and required source counts.

### UI (`app/ui/chainlit_app.py`)

A thin Chainlit chat frontend — it does not duplicate the pipeline, it only calls `/v1/ask` on the `api` service and renders `answer`/`sources`/guardrail metadata. Runs as its own `chainlit` Compose service (`make ui`).

### Config

All runtime configuration is a single `pydantic-settings` `Settings` object in `app/core/config.py`, loaded from `.env` (see `.env.example` for the full var list, including per-service Docker ports and Langfuse init vars). Internal service URLs (Qdrant, TEI, reranker, LightRAG, Redis) default to Docker Compose service names, so code run outside the `api` container needs those overridden.
