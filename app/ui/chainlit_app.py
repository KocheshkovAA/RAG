"""
Chainlit UI для Warhammer RAG.

Не дублирует пайплайн — ходит в FastAPI `/v1/ask`.
Запуск:
  chainlit run app/ui/chainlit_app.py -w --host 0.0.0.0 --port 8501
  make ui
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import chainlit as cl

API_URL = os.getenv("RAG_API_URL", "http://api:8000").rstrip("/")
ASK_TIMEOUT = float(os.getenv("CHAINLIT_ASK_TIMEOUT", "90"))
DEBATE_PREFIX = "/debate"


WELCOME = """**Warhammer 40k Lore RAG**

Спроси про фракции, персонажей, кампании.
Оффтоп и слабый retrieval система отклонит (guardrails).

Примеры:
- Как Чернокаменные крепости связаны с Абаддоном?
- Кто такие Адептус Астартес?

Команда `/debate <вопрос>` — несколько персонажей (Орк-Варбосс, Эльдарский
Провидец, Космодесантник) обсуждают вопрос каждый в своей манере.
"""


def _format_meta(data: dict[str, Any]) -> str:
    guard = data.get("guardrail") or {}
    gate = guard.get("retrieval_gate") or {}
    faith = guard.get("faithfulness") or {}
    parts = [
        f"**mode:** `{data.get('mode', '?')}`",
        f"**latency:** `{data.get('latency_ms', '?')} ms`",
        f"**cached:** `{data.get('cached', False)}`",
    ]
    degraded = data.get("degraded") or []
    if degraded:
        parts.append(f"**degraded:** `{', '.join(degraded)}`")
    if guard.get("refused"):
        parts.append("**refused:** `true`")
    if gate:
        parts.append(
            f"**retrieval max:** `{gate.get('max_score', '—')}` "
            f"(min `{gate.get('min_score', '—')}`, {gate.get('reason', '')})"
        )
    if faith:
        if faith.get("skipped"):
            parts.append(f"**faithfulness:** skipped (`{faith.get('reason', '')}`)")
        elif faith.get("faithfulness_score") is not None:
            parts.append(
                f"**faithfulness:** `{faith.get('faithfulness_score')}` "
                f"(grounded={faith.get('is_grounded')})"
            )
    return " · ".join(parts)


def _source_elements(sources: list) -> list[cl.Text]:
    elements: list[cl.Text] = []
    for i, src in enumerate(sources[:8], 1):
        if not isinstance(src, dict):
            elements.append(cl.Text(name=f"source_{i}", content=str(src)[:800], display="side"))
            continue
        name = src.get("article_name") or src.get("title") or f"source_{i}"
        score = src.get("score")
        url = src.get("url") or ""
        lines = [f"**{name}**"]
        if score is not None:
            lines.append(f"score: `{score}`")
        if url:
            lines.append(url)
        elements.append(
            cl.Text(name=f"{i}. {name}"[:64], content="\n".join(lines), display="side")
        )
    return elements


def _format_debate_meta(data: dict[str, Any]) -> str:
    """Meta для дебата — не переиспользует _format_meta: у дебата нет top-level
    mode/cached, зато есть один общий (не по-персонажный) faithfulness-вердикт
    для черновика, на котором строятся все реплики."""
    parts = [f"**token_usage:** `{(data.get('token_usage') or {}).get('total')}`"]
    degraded = data.get("degraded") or []
    if degraded:
        parts.append(f"**degraded:** `{', '.join(degraded)}`")
    faith = (data.get("guardrail") or {}).get("faithfulness") or {}
    if faith and not faith.get("skipped"):
        parts.append(f"**draft faithfulness:** `{faith.get('faithfulness_score', '—')}`")
    return " · ".join(parts)


async def _handle_debate(question: str, session_id: Any) -> None:
    status = cl.Message(content="Персонажи готовятся к дебатам…")
    await status.send()
    status_removed = False

    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/v1/debate",
                json={"question": question, "session_id": str(session_id), "user_id": "chainlit"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)

                    if not status_removed:
                        await status.remove()
                        status_removed = True

                    event_type = event.get("type")
                    if event_type == "turn":
                        turn = event["turn"]
                        content = turn["answer"]
                        if turn.get("refused"):
                            content = f"_[отказ проверки достоверности]_ {content}"
                        await cl.Message(author=turn["persona_display_name"], content=content).send()
                    elif event_type == "refused":
                        await cl.Message(content=event.get("answer") or "Пустой ответ.").send()
                    elif event_type == "error":
                        await cl.Message(content=f"Ошибка дебата: `{event.get('message')}`").send()
                    elif event_type == "done":
                        sources = event.get("sources") or []
                        elements = _source_elements(sources)
                        if elements:
                            await cl.Message(content="Источники дебата:", elements=elements).send()
                        async with cl.Step(name="meta", type="tool") as step:
                            step.output = _format_debate_meta(event)
    except httpx.HTTPStatusError as e:
        if not status_removed:
            await status.remove()
        detail = e.response.text[:500]
        await cl.Message(content=f"API ошибка `{e.response.status_code}`:\n```\n{detail}\n```").send()
    except Exception as e:
        if not status_removed:
            await status.remove()
        await cl.Message(content=f"Не удалось достучаться до API (`{API_URL}`): `{e}`").send()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("session_id", cl.context.session.id)
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    question = (message.content or "").strip()
    if not question:
        await cl.Message(content="Пустой вопрос.").send()
        return

    session_id = cl.user_session.get("session_id") or cl.context.session.id

    if question.lower().startswith(DEBATE_PREFIX):
        debate_question = question[len(DEBATE_PREFIX):].strip()
        if not debate_question:
            await cl.Message(content="Использование: `/debate <вопрос>`").send()
            return
        await _handle_debate(debate_question, session_id)
        return

    status = cl.Message(content="Ищу в базе лора…")
    await status.send()

    try:
        async with httpx.AsyncClient(timeout=ASK_TIMEOUT) as client:
            resp = await client.post(
                f"{API_URL}/v1/ask",
                json={
                    "question": question,
                    "session_id": str(session_id),
                    "user_id": "chainlit",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500]
        await status.remove()
        await cl.Message(content=f"API ошибка `{e.response.status_code}`:\n```\n{detail}\n```").send()
        return
    except Exception as e:
        await status.remove()
        await cl.Message(content=f"Не удалось достучаться до API (`{API_URL}`): `{e}`").send()
        return

    await status.remove()

    answer = data.get("answer") or "Пустой ответ."
    sources = data.get("sources") or []
    elements = _source_elements(sources)

    async with cl.Step(name="meta", type="tool") as step:
        step.output = _format_meta(data)

    await cl.Message(
        content=answer,
        elements=elements if elements else None,
    ).send()
