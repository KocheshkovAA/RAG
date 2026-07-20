"""
Chainlit UI для Warhammer RAG.

Не дублирует пайплайн — ходит в FastAPI `/v1/ask`.
Запуск:
  chainlit run app/ui/chainlit_app.py -w --host 0.0.0.0 --port 8501
  make ui
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import chainlit as cl

API_URL = os.getenv("RAG_API_URL", "http://api:8000").rstrip("/")
ASK_TIMEOUT = float(os.getenv("CHAINLIT_ASK_TIMEOUT", "90"))


WELCOME = """**Warhammer 40k Lore RAG**

Спроси про фракции, персонажей, кампании.  
Оффтоп и слабый retrieval система отклонит (guardrails).

Примеры:
- Как Чернокаменные крепости связаны с Абаддоном?
- Кто такие Адептус Астартес?
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
