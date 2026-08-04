"""Юнит-тесты framework-less агента (examples/minimal_agent_loop.py) — без сети/Docker/LLM,
тот же fake-LLM подход, что в tests/test_agentic_rag.py: скриптуем сырые текстовые ответы модели."""

import json

import pytest

from examples.minimal_agent_loop import (
    AgentState,
    MinimalAgent,
    Tool,
    ToolRegistry,
    _extract_json_object,
    _parse_decision,
    _is_grounded,
    MalformedDecisionError,
)


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Отдаёт по одному текстовому ответу за вызов, в заданном порядке."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return _Msg(self.responses.pop(0))


def _decision_json(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


def _registry_with_lookup(term_to_fact: dict[str, str]) -> ToolRegistry:
    registry = ToolRegistry()

    def _lookup(term: str) -> str:
        fact = term_to_fact.get(term)
        if fact is None:
            raise ValueError(f"нет данных по '{term}'")
        return fact

    registry.register(Tool(name="lookup", description="test lookup", params={"term": "..."}, fn=_lookup))
    return registry


# --------------------------------------------------------------------------- parsing


def test_extract_json_object_handles_fenced_block():
    raw = 'бла-бла\n```json\n{"action": "refuse"}\n```\nещё текст'
    assert _extract_json_object(raw) == {"action": "refuse"}


def test_extract_json_object_handles_raw_json():
    assert _extract_json_object('{"action": "refuse"}') == {"action": "refuse"}


def test_extract_json_object_rejects_non_json():
    with pytest.raises(MalformedDecisionError):
        _extract_json_object("не json вообще")


def test_parse_decision_requires_tool_for_tool_call():
    with pytest.raises(MalformedDecisionError):
        _parse_decision(_decision_json(action="tool_call"))


def test_parse_decision_rejects_unknown_action():
    with pytest.raises(MalformedDecisionError):
        _parse_decision(_decision_json(action="do_something_else"))


def test_parse_decision_fills_defaults():
    decision = _parse_decision(_decision_json(action="refuse"))
    assert decision["thought"] == ""
    assert decision["args"] == {}


# --------------------------------------------------------------------------- reflection heuristic


def test_is_grounded_false_without_observations():
    assert _is_grounded("некроны — роботы", []) is False


def test_is_grounded_true_on_shared_word():
    obs = [{"observation": "Некроны — древняя раса роботизированных скелетов."}]
    assert _is_grounded("Некроны — роботы, это точно.", obs) is True


def test_is_grounded_false_when_no_overlap():
    obs = [{"observation": "Магнус Красный — примарх Тысячи Сынов."}]
    assert _is_grounded("Гвардия Ворона летает на кораблях.", obs) is False


# --------------------------------------------------------------------------- full loop


async def test_agent_reaches_final_after_tool_call():
    registry = _registry_with_lookup({"некроны": "Некроны — роботизированная раса скелетов."})
    llm = _FakeLLM(
        [
            _decision_json(thought="ищу инфо", action="tool_call", tool="lookup", args={"term": "некроны"}),
            _decision_json(thought="хватит", action="final", answer="Некроны — раса роботизированных скелетов."),
        ]
    )
    agent = MinimalAgent(registry, llm, max_rounds=4)

    state = await agent.run("Кто такие некроны?")

    assert state.refused is False
    assert state.stop_reason == "final"
    assert state.final_answer == "Некроны — раса роботизированных скелетов."
    assert len(state.observations) == 1
    assert len(state.thoughts) == 2


async def test_agent_refuses_when_model_refuses_directly():
    registry = ToolRegistry()
    llm = _FakeLLM([_decision_json(thought="не про это", action="refuse")])
    agent = MinimalAgent(registry, llm)

    state = await agent.run("Что-то не по теме")

    assert state.refused is True
    assert state.stop_reason == "refused_by_model"


async def test_agent_retries_ungrounded_final_then_refuses():
    registry = _registry_with_lookup({"некроны": "Некроны — раса роботов."})
    llm = _FakeLLM(
        [
            _decision_json(thought="ищу", action="tool_call", tool="lookup", args={"term": "некроны"}),
            # Ответ не пересекается по словам с наблюдением -> reflection должна отклонить.
            _decision_json(thought="готово", action="final", answer="Совершенно не связанный текст."),
            _decision_json(thought="всё равно", action="final", answer="Опять другой текст."),
        ]
    )
    agent = MinimalAgent(registry, llm, max_rounds=5, max_reflection_retries=1)

    state = await agent.run("Кто такие некроны?")

    assert state.refused is True
    assert state.stop_reason == "failed_reflection"
    # Один tool_call + два final-хода, из которых один ушёл на reflection-повтор.
    assert len(llm.calls) == 3


async def test_agent_stops_on_malformed_output_after_retries():
    registry = ToolRegistry()
    llm = _FakeLLM(["это не json"] * 10)  # с запасом на все попытки парсинга
    agent = MinimalAgent(registry, llm, max_parse_retries=2)

    state = await agent.run("вопрос")

    assert state.refused is True
    assert state.stop_reason == "malformed_output"
    assert len(llm.calls) == 3  # 1 основная попытка + 2 ретрая


async def test_agent_stops_on_repeated_tool_failure():
    registry = ToolRegistry()  # инструмент 'lookup' не зарегистрирован -> всегда ошибка
    llm = _FakeLLM(
        [
            _decision_json(action="tool_call", tool="lookup", args={"term": "x"}),
            _decision_json(action="tool_call", tool="lookup", args={"term": "y"}),
        ]
    )
    agent = MinimalAgent(registry, llm, max_rounds=5)

    state = await agent.run("вопрос")

    assert state.refused is True
    assert state.stop_reason == "repeated_tool_failure"


async def test_agent_stops_on_no_new_evidence():
    registry = _registry_with_lookup({"x": "тот же самый факт без изменений"})
    llm = _FakeLLM(
        [
            _decision_json(action="tool_call", tool="lookup", args={"term": "x"}),
            _decision_json(action="tool_call", tool="lookup", args={"term": "x"}),
        ]
    )
    agent = MinimalAgent(registry, llm, max_rounds=5)

    state = await agent.run("вопрос")

    assert state.refused is True
    assert state.stop_reason == "no_new_evidence"


async def test_agent_stops_at_max_rounds_when_reflection_retry_has_no_round_left():
    # Edge case: reflection просит повторить ответ (continue), но это был последний
    # разрешённый раунд — без явного стоп-условия цикл вышел бы молча, без stop_reason.
    registry = ToolRegistry()
    llm = _FakeLLM([_decision_json(action="final", answer="ничем не подтверждённый ответ")])
    agent = MinimalAgent(registry, llm, max_rounds=1, max_reflection_retries=1)

    state = await agent.run("вопрос")

    assert state.refused is True
    assert state.stop_reason == "max_rounds"
    assert state.final_answer is None


async def test_agent_stops_at_max_rounds():
    registry = _registry_with_lookup({"x": "факт номер один", "y": "факт номер два", "z": "факт номер три"})
    llm = _FakeLLM(
        [
            _decision_json(action="tool_call", tool="lookup", args={"term": "x"}),
            _decision_json(action="tool_call", tool="lookup", args={"term": "y"}),
            _decision_json(action="tool_call", tool="lookup", args={"term": "z"}),
        ]
    )
    agent = MinimalAgent(registry, llm, max_rounds=3)

    state = await agent.run("вопрос")

    assert state.refused is True
    assert state.stop_reason == "max_rounds"
    assert state.round == 3


def test_registry_catalog_lists_registered_tools():
    registry = ToolRegistry()
    registry.register(Tool(name="calculator", description="считает", params={"expression": "выражение"}, fn=str))
    catalog = registry.catalog()
    assert "calculator" in catalog
    assert "expression" in catalog


def test_agent_state_defaults():
    state = AgentState(query="вопрос")
    assert state.round == 0
    assert state.thoughts == []
    assert state.refused is False
