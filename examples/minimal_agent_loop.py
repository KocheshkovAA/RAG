"""Framework-less agent loop: state, tool registry, executor, ReAct, reflection, stop conditions.

Написано без LangGraph/AgentExecutor/CrewAI — вручную, чтобы показать, что именно эти фреймворки
автоматизируют под капотом: сборку каталога инструментов в промпт, разбор структурированного решения
модели за ход, диспетчеризацию к инструменту, обратную связь в контекст. LLM-клиент переиспользует
`app.core.llm.llm_factory` (тот же провайдер/конфиг, что и весь проект) — "framework-less" здесь
означает "без агентного оркестратора", а не "без LLM SDK".

В отличие от `app.core.agentic_rag.AgenticRAG` (продовый ReAct-цикл на `bind_tools`, реальные Qdrant-
инструменты, реальные guardrails), этот модуль — изолированный спайк на игрушечных инструментах
(калькулятор + статичный справочник), без сети и Docker-зависимостей, специально чтобы юнит-тесты
(`tests/test_minimal_agent_loop.py`) гонялись мгновенно и чтобы код целиком читался за одну посадку.

Компоненты (по пунктам плана):
- state            -> AgentState
- planner/router    -> MinimalAgent._decide (один LLM-ход решает: tool_call/final/refuse и какой tool)
- tool registry     -> Tool, ToolRegistry
- executor          -> MinimalAgent._execute_tool
- ReAct loop        -> MinimalAgent.run (thought -> action -> observation, по кругу)
- reflection/retry  -> _is_grounded + один допустимый повтор ответа, не подтверждённого наблюдениями
- stop conditions   -> STOP_REASONS ниже

Запуск с реальным LLM (нужен настроенный .env / контейнер api):
    python -m examples.minimal_agent_loop "Сколько будет 12 умножить на 7?"
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

# Пояснение к каждой причине остановки — см. docs/agent-architecture.md, раздел Stop conditions,
# для того же набора идей на продовом ReAct-цикле.
STOP_REASONS = {
    "final": "модель дала финальный ответ, прошедший reflection-проверку",
    "refused_by_model": "модель сама решила, что не может/не должна отвечать",
    "failed_reflection": "финальный ответ не прошёл reflection дважды подряд — отказ вместо непроверенного ответа",
    "malformed_output": "модель не выдала валидное решение после исчерпания попыток парсинга",
    "unknown_action": "решение распарсилось, но action вне {tool_call, final, refuse}",
    "repeated_tool_failure": "два вызова инструмента подряд завершились ошибкой",
    "no_new_evidence": "наблюдение раунда дословно совпало с предыдущим — дальнейший поиск бесполезен",
    "max_rounds": "исчерпан бюджет раундов, не дожидаясь решения модели",
    "latency_budget": "исчерпан бюджет времени на весь цикл",
}


# --------------------------------------------------------------------------- tool registry


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # имя параметра -> краткое описание, для рендера в промпт
    fn: Callable[..., str]  # синхронный вызов; кидает исключение при ошибке, возвращает текст-наблюдение


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def catalog(self) -> str:
        lines = []
        for tool in self._tools.values():
            params = ", ".join(f"{k} ({v})" for k, v in tool.params.items()) or "без параметров"
            lines.append(f"- {tool.name}: {tool.description}. Параметры: {params}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- state


@dataclass
class AgentState:
    query: str
    round: int = 0
    thoughts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    final_answer: Optional[str] = None
    refused: bool = False


# --------------------------------------------------------------------------- LLM decision parsing


class MalformedDecisionError(Exception):
    pass


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> dict:
    """Достаёт JSON-объект из сырого текстового ответа модели: сперва пробует
    ```json fenced block```, потом весь текст как есть. Никакого structured-output
    API модели (with_structured_output/bind_tools) — парсинг ручной, специально."""
    raw = raw.strip()
    match = _JSON_FENCE_RE.search(raw)
    candidate = match.group(1) if match else raw
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise MalformedDecisionError(f"не удалось распарсить JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise MalformedDecisionError("решение должно быть JSON-объектом")
    return parsed


_VALID_ACTIONS = {"tool_call", "final", "refuse"}


def _parse_decision(raw: str) -> dict:
    parsed = _extract_json_object(raw)
    action = parsed.get("action")
    if action not in _VALID_ACTIONS:
        raise MalformedDecisionError(f"action={action!r} не входит в {_VALID_ACTIONS}")
    if action == "tool_call" and not parsed.get("tool"):
        raise MalformedDecisionError("action=tool_call без поля 'tool'")
    parsed.setdefault("thought", "")
    parsed.setdefault("args", {})
    return parsed


# --------------------------------------------------------------------------- reflection


_WORD_RE = re.compile(r"[а-яёa-z]{4,}", re.IGNORECASE)


def _is_grounded(answer: str, observations: list[dict[str, Any]]) -> bool:
    """Грубая эвристика вместо LLM-верификатора (ср. AnswerFaithfulnessGuard в основном
    проекте, где эту роль играет отдельный LLM-вызов с дословной цитатой): без единого
    вызова инструмента ответ по определению не заземлён; при наличии наблюдений требуем
    хотя бы одно значимое слово ответа дословно встретить в тексте наблюдений."""
    if not observations:
        return False
    answer_words = {w.lower() for w in _WORD_RE.findall(answer)}
    if not answer_words:
        return False
    combined = " ".join(o["observation"] for o in observations).lower()
    return any(word in combined for word in answer_words)


# --------------------------------------------------------------------------- LLM protocol (для fake в тестах)


class _AIMessageLike(Protocol):
    content: str


class _LLMLike(Protocol):
    def ainvoke(self, messages: list[dict[str, str]]) -> Awaitable[_AIMessageLike]: ...


def _system_prompt(tool_catalog: str) -> str:
    return (
        "Ты — агент, решающий задачу пошагово. У тебя есть инструменты:\n"
        f"{tool_catalog}\n\n"
        "На КАЖДОМ ходу отвечай РОВНО одним JSON-объектом (можно в ```json блоке), без текста "
        "вокруг, одной из трёх форм:\n"
        '  {"thought": "...", "action": "tool_call", "tool": "имя", "args": {...}}\n'
        '  {"thought": "...", "action": "final", "answer": "..."}\n'
        '  {"thought": "...", "action": "refuse"}\n'
        "Используй final только когда ответ подтверждён результатом инструмента — если инструмент "
        "ещё не вызывался или не дал релевантных данных, вызови инструмент или ответь refuse, "
        "но не выдумывай ответ."
    )


# --------------------------------------------------------------------------- agent


class MinimalAgent:
    def __init__(
        self,
        tools: ToolRegistry,
        llm: _LLMLike,
        *,
        max_rounds: int = 4,
        max_parse_retries: int = 2,
        max_reflection_retries: int = 1,
        max_seconds: float = 60.0,
    ) -> None:
        self.tools = tools
        self.llm = llm
        self.max_rounds = max_rounds
        self.max_parse_retries = max_parse_retries
        self.max_reflection_retries = max_reflection_retries
        self.max_seconds = max_seconds

    async def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        history: list[dict[str, str]] = [
            {"role": "system", "content": _system_prompt(self.tools.catalog())},
            {"role": "user", "content": query},
        ]
        started = time.monotonic()
        consecutive_tool_failures = 0
        reflection_retries_used = 0

        for round_idx in range(self.max_rounds):
            if time.monotonic() - started > self.max_seconds:
                state.stop_reason = "latency_budget"
                state.refused = True
                break

            state.round = round_idx + 1
            decision = await self._decide(history)
            if decision is None:
                state.stop_reason = "malformed_output"
                state.refused = True
                break

            state.thoughts.append({"round": state.round, "thought": decision["thought"]})
            history.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})

            action = decision["action"]

            if action == "refuse":
                state.stop_reason = "refused_by_model"
                state.refused = True
                break

            if action == "final":
                answer = (decision.get("answer") or "").strip()
                if _is_grounded(answer, state.observations):
                    state.final_answer = answer
                    state.stop_reason = "final"
                    break
                if reflection_retries_used >= self.max_reflection_retries:
                    state.stop_reason = "failed_reflection"
                    state.refused = True
                    break
                reflection_retries_used += 1
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "Reflection: ответ не подтверждён текстом наблюдений (нет общих значимых "
                            "слов). Обоснуй ответ данными из наблюдений или ответь refuse."
                        ),
                    }
                )
                continue

            if action != "tool_call":
                state.stop_reason = "unknown_action"
                state.refused = True
                break

            observation = self._execute_tool(decision["tool"], decision["args"])
            failed = observation.startswith("Ошибка")
            consecutive_tool_failures = consecutive_tool_failures + 1 if failed else 0

            duplicate = bool(state.observations) and state.observations[-1]["observation"] == observation
            state.observations.append({"round": state.round, "tool": decision["tool"], "observation": observation})
            history.append({"role": "user", "content": f"Наблюдение: {observation}"})

            if consecutive_tool_failures >= 2:
                state.stop_reason = "repeated_tool_failure"
                state.refused = True
                break
            if duplicate:
                state.stop_reason = "no_new_evidence"
                state.refused = True
                break

        # Дошли сюда без break — либо раунды исчерпаны после tool_call, либо (edge case)
        # reflection затребовала повтор на последнем разрешённом раунде и `continue`
        # увёл прямиком на выход из range() без явного стоп-условия. В обоих случаях
        # это один и тот же стоп-condition: бюджет раундов кончился, не дожидаясь решения.
        if state.stop_reason is None:
            state.stop_reason = "max_rounds"
            state.refused = True

        return state

    async def _decide(self, history: list[dict[str, str]]) -> Optional[dict]:
        local_history = list(history)
        for _attempt in range(self.max_parse_retries + 1):
            msg = await self.llm.ainvoke(local_history)
            try:
                return _parse_decision(msg.content)
            except MalformedDecisionError as e:
                logger.warning("malformed agent decision: %s", e)
                local_history.append({"role": "assistant", "content": msg.content})
                local_history.append(
                    {"role": "user", "content": f"Решение не распознано ({e}). Ответь строго одним JSON-объектом."}
                )
        return None

    def _execute_tool(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Ошибка: инструмента '{name}' не существует."
        try:
            return tool.fn(**args)
        except Exception as e:  # инструмент — чужой код, не доверяем ему не падать
            return f"Ошибка инструмента '{name}': {e}"


# --------------------------------------------------------------------------- демо-инструменты + CLI


def _calculator(expression: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        raise ValueError("выражение содержит недопустимые символы")
    return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 — вход уже сужен regex'ом выше


_WIKI_FACTS = {
    "некроны": "Некроны — древняя раса роботизированных скелетов, некогда органическая цивилизация "
    "некронтир, обменявшая плоть на бессмертие в металле по договору со Звёздными Богами.",
    "магнус": "Магнус Красный — примарх Тысячи Сынов, кибернетический циклоп, мастер тайных искусств, "
    "чьё вмешательство в Варп на Никее считается одним из триггеров Ереси Хоруса.",
}


def _wiki_lookup(term: str) -> str:
    fact = _WIKI_FACTS.get(term.strip().lower())
    if fact is None:
        return f"По запросу '{term}' ничего не найдено в справочнике."
    return fact


def build_demo_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Вычислить арифметическое выражение",
            params={"expression": "строка вида '12 * 7'"},
            fn=_calculator,
        )
    )
    registry.register(
        Tool(
            name="wiki_lookup",
            description="Найти факт в статичном мини-справочнике по Warhammer 40k (некроны, магнус)",
            params={"term": "искомый термин"},
            fn=_wiki_lookup,
        )
    )
    return registry


async def _main(query: str) -> None:
    from app.core.llm import llm_factory

    agent = MinimalAgent(build_demo_registry(), llm_factory.get_llm(temperature=0, role="agentic"))
    state = await agent.run(query)

    for t in state.thoughts:
        print(f"[раунд {t['round']}] мысль: {t['thought']}")
    for o in state.observations:
        print(f"[раунд {o['round']}] {o['tool']} -> {o['observation']}")
    print(f"stop_reason={state.stop_reason} ({STOP_REASONS.get(state.stop_reason, '?')})")
    print("ОТКАЗ" if state.refused else f"ОТВЕТ: {state.final_answer}")


if __name__ == "__main__":
    asyncio.run(_main(sys.argv[1] if len(sys.argv) > 1 else "Сколько будет 12 умножить на 7?"))
