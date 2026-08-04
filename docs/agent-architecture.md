# Архитектура агента

Этот документ отвечает на вопрос «как устроен агент внутри», а не «что он умеет». Он описывает state,
control flow, reasoning-режимы (CoT/ReAct/Reflection/ToT как архитектурные паттерны, а не buzzword'ы),
tools, guards и stop conditions **как они реально реализованы в коде на сегодня** — без забегания вперёд
недоделанных частей. Там, где что-то не реализовано, это прямо помечено как нереализованное.

Для обзора компонентов и request flow на уровне сервисов см. [README](../README.md) и `CLAUDE.md`; этот
документ — про то, что происходит *внутри* `WarhammerOrchestrator` и `AgenticRAG`.

## 1. Agent state

Состояние агента не хранится в одном объекте — оно живёт в переменных внутри `answer()` на каждом уровне
пайплайна и явно прокидывается в ответ API как поле `agentic`/`guardrail`. Он не персистентный между
запросами (см. [раздел 7](#7-memory--context-management) — это осознанное ограничение, не пробел).

Ключевые поля state и где они живут:

| Поле | Где | Смысл |
|---|---|---|
| `question` | вход `answer()` | исходный вопрос пользователя, неизменный на всё время цикла |
| `route` | `RAGRoute` (`orchestrator.py`) | `vector` / `graph` / `agentic` — решение роутера, принимается один раз до входа в конкретный маршрут |
| `messages` | `AgenticRAG.answer()` | история ReAct-хода: `SystemMessage` → `HumanMessage(question)` → чередование `AIMessage`(tool_calls) / `ToolMessage`(observation) |
| `all_docs` / `seen_contents` | `AgenticRAG.answer()` | аккумулятор найденных документов по всем раундам, с дедупликацией по content — это единственная «память» между раундами поиска |
| `gated_docs` / `gate_meta` | результат `RetrievalGate.filter_docs()` | что реально прошло порог релевантности и почему |
| `tool_calls_made` | `AgenticRAG.answer()` | лог вызовов инструмента: раунд, запрос, сколько найдено/нового, ошибка |
| `rounds_used` / `stop_reason` | `AgenticRAG.answer()` | сколько раундов ушло и почему цикл остановился (см. [Stop conditions](#stop-conditions)) |
| `faith_meta` | результат `AnswerFaithfulnessGuard.verify()` | вердикт пост-проверки заземлённости ответа |
| `degraded` | список строк (`"reranker"`, `"lightrag"`, ...) | какие внешние зависимости отказали и были обойдены fallback'ом |
| финальный статус | `guardrail.refused: bool` | отказ (`INSUFFICIENT_INFO_MESSAGE`) vs. финальный ответ |

Явно нет в state: сырого chain-of-thought модели наружу (см. [CoT](#cot--внутренняя-декомпозиция)),
персистентной памяти между вопросами, и отдельного "confidence"-скаляра — вместо него используются
конкретные пороговые метрики (`gate_meta.max_score`, `faith_meta.faithfulness_score`).

## 2. Control flow

```
question
  │
  ▼
classify_route()  ── LLM-классификатор → RAGRoute {vector | graph | agentic}
  │
  ├─ vector  → RAG.answer()            (app/core/vectorrag.py)
  ├─ graph   → domain probe → LightRAGClient, fallback на vector при недоступности
  └─ agentic → AgenticRAG.answer()     (app/core/agentic_rag.py)  ← ReAct-цикл, см. ниже
```

Внутри `agentic`-маршрута (основной ReAct-цикл проекта) control flow на каждый раунд:

```
round:
  1. model turn:      llm_with_tools.ainvoke(messages) → AIMessage
  2. branch:
       нет tool_calls  → stop_reason = "model_stopped", выход из цикла
       есть tool_calls → до AGENTIC_MAX_TOOL_CALLS_PER_ROUND штук исполняются параллельно (asyncio.gather)
  3. observe:          retriever.ainvoke(query) на каждый вызов → docs, дедуп в all_docs
  4. rerank + gate:     rerank(question, all_docs) → RetrievalGate.filter_docs() → gate_meta
  5. reflect:           ToolMessage с found/new/gate-статусом дописывается в messages —
                        модель "видит" это на следующем раунде и решает: доискать / остановиться
  6. stop check:        round_idx == max_iterations-1 → stop_reason = "max_iterations"
  │
  ▼ (после выхода из цикла)
gated_docs пуст?  → RetrievalGate.insufficient_response()  (отказ, refused=true)
gated_docs есть   → self.chain.ainvoke(...) → answer
                     → AnswerFaithfulnessGuard.should_verify() → при необходимости verify()
                     → verify провалился → refuse_response() (отказ)
                     → verify прошёл / пропущен → финальный ответ, sources, кэш
```

Важное свойство: генерация финального ответа **всегда** идёт через тот же `self.chain`
(context-only промпт), что и у vector-маршрута — ReAct-цикл решает только *что искать*, не
*как отвечать*. Guardrails (`RetrievalGate`, `AnswerFaithfulnessGuard`) переиспользуются по
ссылке от `vector_rag`, а не пересоздаются с независимыми порогами — это то, что не даёт
агентности "уговорить" систему обойти пороги отказа.

## 3. Reasoning modes

### CoT — Thought-леg внутри ReAct-хода

Реализовано как недостающая часть уже существовавшего ReAct-цикла, а не отдельная надстройка:
классический ReAct (Yao et al., 2022) — это тройка **Thought → Action → Observation**, и до этого
изменения у `AgenticRAG` были только Action (`tool_calls`) и Observation (`ToolMessage`) — Thought
не запрашивался и не сохранялся вообще, независимо от того, что провайдер мог класть в `content`.

Теперь `AGENTIC_SYSTEM_PROMPT` явно требует от модели одно предложение мысли в `content` на КАЖДОМ
ходу (и при вызове инструмента, и при остановке): что собирается искать дальше и почему, либо почему
данных уже достаточно. `AgenticRAG.answer()` читает `ai_msg.content`, кладёт непустые мысли в
`agentic_meta["thoughts"]` (список `{round, thought}`) — это отдельная от `tool_calls_made` структура,
по одной записи на ход модели, а не на отдельный вызов инструмента (в раунде может быть несколько
параллельных tool calls с одной общей мыслью). Пустой `content` не создаёт фиктивную запись
(`test_agentic_skips_empty_thoughts`) — честность важнее заполненности лога.

Ограничения, которые стоит проговаривать честно:
- Это мысль **в момент действия** (ReAct-flavored CoT), а не предварительный план на весь вопрос —
  архитектурно ближе к "explain-then-act each step", чем к явному plan-and-execute с ревизией.
- Мысль по-прежнему не показывается пользователю и не влияет на дальнейший control flow (не читается
  кодом, кроме логирования) — это debug/trace-артефакт, полезный для отладки и для объяснения решений
  агента на собеседовании, а не входной сигнал для guardrails.
- `RouteDecision.reasoning` и `FaithfulnessVerdict.reasoning` (`orchestrator.py`, `guardrails.py`)
  остаются отдельным случаем — поле-обоснование, сгенерированное в одном вызове вместе с готовым
  решением (structured output), а не Thought, предшествующий действию. Их не стоит путать с этим CoT.
- `QueryOptimizer` (`app/core/query_processor.py`) — по-прежнему чистая decomposition без объяснения
  (`ExpandedQuery.queries`), Thought там не добавлен (не ReAct-цикл, добавлять некуда).

### ReAct — основной runtime-паттерн

`AgenticRAG` (`app/core/agentic_rag.py`) — это буквальный ReAct-цикл: `bind_tools([SearchKnowledgeBase])`
на LLM, model turn → tool call(s) → observation (`ToolMessage`) → следующий model turn, до
`model_stopped` или `max_iterations`. В отличие от более простых "single-shot RAG" реализаций:

- модель сама решает, **сколько раз** и **с какими запросами** вызывать поиск — не скриптованный retry;
- поддерживаются **параллельные** вызовы инструмента за один ход (для составных вопросов с
  несколькими сущностями — `_ai_tool_calls` в тестах, `asyncio.gather` в реализации);
- observation — это не сырой список документов, а агрегированный сигнал (сколько найдено/нового,
  прошёл ли гейт релевантности) — модель ориентируется на *решение гейта*, а не на сырой текст чанков;
- собственный текстовый ответ модели в ReAct-ходах (теперь используемый как Thought, см. ниже) никогда
  не используется как ответ пользователю — это архитектурная гарантия, что "решение искать/не искать"
  и "финальный ответ" — разные шаги с разными guardrails на выходе второго.

### Reflection — самопроверка после observation/final draft

Reflection реализован не как отдельный LLM-вызов "подумай ещё раз", а как два независимых guard'а,
встроенных в control flow как обязательные точки проверки:

- **После retrieval, перед генерацией**: `RetrievalGate.filter_docs()` — реф­лексия вида "хватает ли
  найденного, чтобы вообще пытаться отвечать". В ReAct-цикле результат этой рефлексии дополнительно
  скармливается обратно модели как `ToolMessage`, так что она может отреагировать retry — это то, что
  отличает reflection в agentic-маршруте от reflection в vector-маршруте (там это разовая проверка).
- **После генерации, перед выдачей ответа**: `AnswerFaithfulnessGuard.verify()` — разбивает ответ на
  атомарные claims (первый LLM-вызов), затем грунтует каждый claim ДВУМЯ независимыми слоями, оба
  должны пройти: 1) `is_quote_grounded` — дёшево, без LLM, дословная цитата реально есть в контексте
  (ловит выдуманные цитаты, `autojunk=False`-фикс из `039e4d8`); 2) `_check_entailment` — второй,
  отдельный LLM-вызов (только для claims, прошедших слой 1), который проверяет, что найденная цитата
  подтверждает ИМЕННО этот claim, а не просто существует в контексте (`03296ac`) — ловит случай, когда
  модель прикладывает настоящую, но не по теме цитату к неверному факту (см. коммит `eae0fff` — holistic
  `is_grounded=true` систематически пропускал такие случаи, потому что оценивал "упомянута ли сущность",
  а не "подтверждён ли именно этот факт"; слой 1 в одиночку эту же дыру не закрывал — воспроизведено
  живьём: реальная цитата про арену была приложена к claim'у про рейды в реальное пространство).
  `should_verify()` — это policy, когда рефлексия обязательна (деградация пайплайна, низкий/отсутствующий
  rerank-score) и когда её можно пропустить ради латентности (высокая уверенность retrieval + не
  деградировано).

Обе точки reflection ведут к одному из трёх исходов: **retry** (в ReAct-цикле — ещё один раунд tool
calling), **refusal** (`insufficient_response()` / `refuse_response()`), либо **пропуск дальше**.

### ToT — optional branching для multi-hop

**Не реализовано.** В плане: 2-3 кандидатных декомпозиции сложного multi-hop вопроса, оценка по
evidence/latency/confidence, выбор лучшей ветки — целенаправленно не включается везде из-за стоимости
(лишние LLM-вызовы на каждый запрос). Ближайший реализованный сосед — `QueryOptimizer` (CoT-decomposition
на 1-3 под-запроса), но это не branching с оценкой и выбором, а безусловное объединение результатов всех
под-запросов. Кандидат на место для ToT, если/когда будет реализован: между `classify_route()` и входом
в `agentic`-маршрут, для вопросов, где `graph` недоступен (LightRAG деградирован) и нужно решить, какую
из нескольких стратегий декомпозиции вопроса стоит попробовать в ReAct-цикле первой.

## 4. Tools

| Инструмент | Где | Action space |
|---|---|---|
| `search_knowledge_base` (`SearchKnowledgeBase`) | `AgenticRAG`, через `bind_tools` | один параметр `query: str`, вызывает `Retriever.ainvoke` (hybrid dense+sparse Qdrant) — read-only, без побочных эффектов |
| vector retrieval (не как tool, а как прямой пайплайн) | `RAG.get_relevant_documents()` | тот же `Retriever`, но вызывается детерминированным кодом (`QueryOptimizer` expansion), не решением LLM за ход |
| graph retrieval | `LightRAGClient.query()` (`app/core/lightrag_client.py`) | HTTP к отдельному контейнеру `lightrag`, read-only; за circuit breaker'ом (`app/core/health.py`), деградирует на vector |
| `ask_warhammer_lore` (MCP tool) | `mcp_server/server.py` | внешний вход в весь пайплайн (роутинг+ретрив+guardrails) через MCP streamable-HTTP; сам по себе никакой логики не содержит, тонкий HTTP-клиент к `/v1/ask` |
| `debate_warhammer_lore` (MCP tool) | `mcp_server/server.py` | то же, но к `/v1/debate` (`persona_debate.py`) — другая форма ответа поверх тех же guardrails |
| `warhammer_service_status` (MCP tool) | `mcp_server/server.py` | health-check `/v1/ready`, не проходит через агентный цикл |

Все read-only. Ни один инструмент не имеет write-доступа к внешнему состоянию (нет tool'ов, которые
что-то изменяют — только читают Qdrant/LightRAG/HTTP-статус). Это прямо ограничивает threat model —
полный разбор (prompt injection, tool misuse, зацикливание, hallucinated action, context poisoning,
избыточные MCP-права) см. [docs/agent-threat-model.md](agent-threat-model.md).

## 5. Guards

- **`RetrievalGate`** (`app/core/guardrails.py`) — порог по score (`rerank_score`, иначе `hybrid_score`
  с отдельным более мягким порогом `RETRIEVAL_MIN_SCORE_NO_RERANK`). Пустой результат после фильтрации →
  `insufficient_response()`, `guardrail.refused = true`. Единственный источник истины про "достаточно ли
  найдено" — используется и как терминальное условие (vector/graph), и как промежуточный сигнал внутри
  ReAct-раундов (agentic).
- **`AnswerFaithfulnessGuard`** (`app/core/guardrails.py`) — пост-генерационная верификация в два
  LLM-вызова + два слоя проверки на claim: (1) декомпозиция ответа на атомарные claims + дословная
  supporting_quote на каждый; (2) грунтованность каждого claim требует ОБА: `is_quote_grounded`
  (программная сверка цитаты с контекстом, `difflib.SequenceMatcher(..., autojunk=False)`) И
  `_check_entailment` (второй, отдельный LLM-вызов на короткие пары claim+quote — цитата должна
  подтверждать именно этот claim, не просто существовать в контексте). Ни разу не булевый самоотчёт
  модели в один присест. `should_verify()` пропускает всю проверку только при
  `rerank_score >= FAITHFULNESS_SKIP_ABOVE` (0.80) и отсутствии деградации — иначе (в том числе весь
  диапазон 0.55-0.80, "серая зона") проверка обязательна.
- **Refusal policy**: единое сообщение `INSUFFICIENT_INFO_MESSAGE` на оба guard'а — с точки зрения
  пользователя "не нашли релевантный контекст" и "нашли, но не смогли заземлить ответ" неотличимы,
  осознанно, чтобы не давать сигналов, полезных для промпт-инъекций/probing.

Оба guard'а переиспользуются **по ссылке** между vector/graph/agentic маршрутами (не пересоздаются
с независимыми инстансами) — это структурное свойство, не просто DRY: агентность не может обойти
пороги, потому что пороги не принадлежат агентному коду.

## 6. Stop conditions

| Условие | Где | Что происходит |
|---|---|---|
| `model_stopped` | `AgenticRAG` | модель сделала ход без tool_calls — явное решение "хватит искать" |
| `max_iterations` (`AGENTIC_MAX_ITERATIONS`, default 3) | `AgenticRAG` | принудительная остановка независимо от желания модели — верхняя граница стоимости/латентности одного запроса |
| лимит параллельных вызовов за ход (`AGENTIC_MAX_TOOL_CALLS_PER_ROUND`, default 4) | `AgenticRAG` | не stop-condition цикла, а защита от патологического "вызвать инструмент 20 раз за раз" — избыточные вызовы получают `ToolMessage` "пропущено" вместо исполнения |
| retrieval gate fail | `RetrievalGate` | пустой `gated_docs` после фильтрации → отказ, дальше в цикл не идём |
| faithfulness fail | `AnswerFaithfulnessGuard` | ответ сгенерирован, но не прошёл верификацию → отказ вместо выдачи непроверенного ответа |
| tool/зависимость упала | `try/except` в `_run_search` (agentic), `CircuitBreaker`/`call_with_circuit` (reranker/lightrag/tei, `app/core/health.py`) | не роняет запрос — деградация (`degraded: [...]`) и fallback (un-reranked top-k, vector вместо graph) |
| latency/cost budget | `REQUEST_TIMEOUT_SEC` (default 45s) через `with_timeout()` на уровне HTTP-роута (`app/api/routes.py`) | обрезает запрос целиком на уровне API, а не внутри агентного цикла — то есть это внешний, а не архитектурный agent-level stop condition сейчас |
| rate limit | `slowapi`, `RATE_LIMIT_ASK` (default 20/мин на IP) | защита на уровне API, не agent-level, но релевантна как cost/abuse guard вокруг всего цикла |

Явный пробел: нет agent-level стоп-условия "no new evidence" (например, раунд, где ни один tool call не
добавил новых документов в `all_docs`) — сейчас это неявно ловится тем, что гейт не меняется и модель
рано или поздно сама решает остановиться (`model_stopped`) либо упирается в `max_iterations`, но нет
явного счётчика "N раундов подряд без прогресса" как принудительного условия.

## 7. Framework-less mini-agent (спайк)

`examples/minimal_agent_loop.py` — тот же набор компонентов (state, tool registry, planner/router,
executor, ReAct-цикл, reflection/retry, stop conditions), но написанный вручную, без LangGraph/
`bind_tools`/AgentExecutor: LLM отдаёт сырой текст, JSON-решение парсится и валидируется руками
(`_parse_decision`), reflection — эвристика на пересечении слов ответа и наблюдений, а не LLM-вызов.
Игрушечные инструменты (калькулятор, статичный справочник), без Qdrant/сети — юнит-тесты
(`tests/test_minimal_agent_loop.py`) гоняются за миллисекунды. Цель — не заменить `AgenticRAG`, а
доказать, что LangChain/`bind_tools` в проде — осознанный выбор (готовая сериализация tool-calling,
меньше ручного парсинга), а не то, что скрывает непонимание, как это устроено внутри.

## 8. Memory / context management

### Что входит в per-turn context

Два раздельных канала, не один — это структурное свойство, не случайность:

- **`messages`** (история ReAct-хода в `AgenticRAG`) — `SystemMessage` → `HumanMessage(question)` →
  чередование `AIMessage` (thought + tool_calls) / `ToolMessage`. В `ToolMessage` идёт не сырой
  список найденных документов, а агрегированная сводка: сколько найдено/нового, прошёл ли гейт,
  score/порог (см. п.2 Control flow). Модель на следующем раунде принимает решение по этой сводке,
  а не по тексту чанков.
- **`gated_docs`** (аккумулятор `Document`) — реальный текст контекста для финальной генерации,
  передаётся в `self.chain.ainvoke({"docs": gated_docs, "question": question})` **отдельным,
  самостоятельным вызовом**, без истории ReAct-раундов вообще. Финальный ответ строится с нуля из
  вопроса + отфильтрованных документов — черновики мыслей/поисковых итераций в этот вызов не
  просачиваются.

### Что НЕ входит

- Сырой dump всех найденных документов в историю ReAct-хода — только сводка (см. выше).
- Chain-of-thought/reasoning модели как часть финального промпта — `Thought` (п.3, CoT) существует
  только внутри `messages` ReAct-цикла, наружу (в ответ пользователю) никогда не попадает; это
  прямо оговорено в `AGENTIC_SYSTEM_PROMPT`: "твой текстовый ответ в этом чате НЕ показывается
  пользователю".
- **История разговора между запросами — отсутствует полностью.** `session_id`/`user_id` в
  `QuestionRequest` (`app/api/routes.py`) используются ТОЛЬКО как атрибуты Langfuse-трейса
  (`propagate_attributes`) — нигде не читаются обратно, чтобы подмешать предыдущие вопросы/ответы
  в контекст. Каждый `/v1/ask` полностью независим от истории, даже в рамках одного `session_id`.
  Явно нет multi-turn conversation memory — не потому что забыли, а потому что её никто не строил;
  на собеседовании стоит проговаривать это прямо, а не позволять "session_id" вводить в заблуждение.

### Чем memory отличается от RAG-контекста

В проекте нет **memory** в смысле agent-литературы (persistent, cross-session, специфичные для
пользователя факты/предпочтения) — только **RAG-контекст**: evidence, собранный заново на каждый
вопрос из статичного корпуса, живущий ровно на время одного запроса и выбрасываемый после ответа.
Redis-кэш (`app/core/cache.py`) — не память в этом смысле: это чистый computation cache,
ключ — `sha256(question)` (одинаковый вопрос → тот же кэшированный ответ), TTL `CACHE_TTL_ANSWER_SEC`
(600с по умолчанию), не привязан к пользователю/сессии. Одинаковый вопрос от двух разных
пользователей получит один и тот же закэшированный ответ — это подтверждает, что это кэш
вычисления, а не персонализированная память.

### Как ограничивается раздувание истории

- `ToolMessage` — агрегированная сводка (found/new/gate), не текст чанков (см. выше) — рост
  `messages` за раунд ограничен размером сводки, не размером retrieval-результата.
- `max_iterations` (default 3) — жёсткий потолок на число раундов, а значит и на длину `messages`.
- `thoughts` — одно предложение на ход по промпту (см. п.3, CoT), не развёрнутое рассуждение.
- Ответ пользователю — `sources` (`SourceExtractor.extract()`) отдаёт `article_name`/`url`/`score`,
  не полный текст документов — размер payload ответа не растёт с размером retrieved-контекста.
- `ContextBuilder.max_chars` (15000, `app/core/postprocessors/context_builder.py`) — жёсткий потолок
  символов на финальный контекст генерации, обрезает по границе документа, а не посередине.

### Как это связано с безопасностью

Внешний контекст (retrieved-документы) никогда не получает права управлять tools/actions —
структурное свойство, не эвристика: единственный источник `tool_calls` — структурированное решение
LLM через `bind_tools`-схему (`SearchKnowledgeBase`), проверяемую провайдером до того, как код агента
вообще увидит вызов. Код нигде не парсит текст retrieved-документов в поисках команд/имён
инструментов — retrieved-текст всегда идёт в фиксированный слот промпта (`{context}` в
`QA_SYSTEM_PROMPT`, содержимое `ToolMessage`), никогда не интерпретируется как новая инструкция самим
кодом. LLM в принципе может *попытаться* последовать инструкции, спрятанной в retrieved-тексте (это
остаточный риск, не закрытый структурно — см. `docs/agent-threat-model.md`, п.1 Prompt injection), но
даже в этом случае худшее, что она может сделать — сформировать вызов `search_knowledge_base` со
странным `query` или заявить неподтверждённый факт, который затем ловит `AnswerFaithfulnessGuard`
(defense-in-depth, не по замыслу изначально, но фактически работает и против этого).

---

*Связанные документы: [docs/agent-threat-model.md](agent-threat-model.md) — security/failure modes
(п.4 плана, готово); [docs/agent-design-decisions.md](agent-design-decisions.md) — agent vs tool
vs sub-agent vs deterministic code (п.6 плана, готово); [docs/prompt-regression.md](prompt-regression.md)
и [docs/rnd-decision-log.md](rnd-decision-log.md) — п.8 плана, готово. Ещё не написан:
`docs/agent-framework-comparison.md` (п.9 плана). Этот документ описывает
state/control-flow/reasoning modes/tools/guards/stop-conditions/framework-less-спайк/memory —
пп. 1, 2 и 7 плана.*
