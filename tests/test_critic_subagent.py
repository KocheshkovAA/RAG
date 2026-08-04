"""Юнит-тесты examples/critic_subagent.py — без сети/LLM/Docker, фейковый structured LLM."""

from langchain_core.runnables import RunnableLambda

from examples.critic_subagent import CriticSubAgent, CriticVerdict


class _FakeLLM:
    """Возвращает заранее заданный CriticVerdict вне зависимости от промпта — тот же
    паттерн, что и _FakeStructuredLLM в tests/test_guardrails.py, но проще: тут всего
    одна схема на весь класс, дифференцировать по schema не нужно."""

    def __init__(self, verdict: CriticVerdict):
        self._verdict = verdict
        self.calls: list = []

    def with_structured_output(self, schema):
        assert schema is CriticVerdict

        async def _fn(prompt_value):
            self.calls.append(prompt_value)
            return self._verdict

        return RunnableLambda(_fn)


async def test_critic_returns_approve_verdict():
    verdict = CriticVerdict(verdict="approve", concerns=[], feedback="")
    critic = CriticSubAgent(_FakeLLM(verdict))

    result = await critic.review("вопрос", "контекст", "черновик")

    assert result.verdict == "approve"
    assert result.concerns == []


async def test_critic_returns_revise_with_feedback():
    verdict = CriticVerdict(
        verdict="revise",
        concerns=["не покрыта вторая часть составного вопроса"],
        feedback="добавь ответ про Ангрона, сейчас есть только про Магнуса",
    )
    critic = CriticSubAgent(_FakeLLM(verdict))

    result = await critic.review("Кто такие Магнус и Ангрон?", "контекст", "черновик только про Магнуса")

    assert result.verdict == "revise"
    assert "Ангрона" in result.feedback
    assert len(result.concerns) == 1


async def test_critic_returns_refuse_verdict():
    verdict = CriticVerdict(verdict="refuse", concerns=["контекста недостаточно"], feedback="")
    critic = CriticSubAgent(_FakeLLM(verdict))

    result = await critic.review("вопрос", "нерелевантный контекст", "уклончивый черновик")

    assert result.verdict == "refuse"


async def test_critic_prompt_includes_question_context_and_draft():
    fake_llm = _FakeLLM(CriticVerdict(verdict="approve"))
    critic = CriticSubAgent(fake_llm)

    await critic.review("ВОПРОС123", "КОНТЕКСТ456", "ЧЕРНОВИК789")

    assert len(fake_llm.calls) == 1
    rendered = fake_llm.calls[0].to_string()
    assert "ВОПРОС123" in rendered
    assert "КОНТЕКСТ456" in rendered
    assert "ЧЕРНОВИК789" in rendered


def test_critic_verdict_defaults():
    verdict = CriticVerdict(verdict="approve")
    assert verdict.concerns == []
    assert verdict.feedback == ""
