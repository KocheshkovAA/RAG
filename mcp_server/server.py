"""MCP-сервер поверх Warhammer 40k RAG API.

Свой сервис в docker-compose (см. `mcp-server`), говорит с внешним
миром по streamable-HTTP MCP-транспорту. Никакой логики пайплайна
тут нет — только тонкий HTTP-клиент к `api` по внутренней
docker-сети.
"""

import os

import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.environ.get("WARHAMMER_API_URL", "http://api:8000")
TIMEOUT_SEC = float(os.environ.get("WARHAMMER_MCP_TIMEOUT_SEC", "90"))
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8010"))

mcp = FastMCP("warhammer-lore", host=MCP_HOST, port=MCP_PORT)


def _format_answer(data: dict) -> str:
    guardrail = data.get("guardrail") or {}
    lines = [data.get("answer") or "(пустой ответ)"]

    sources = data.get("sources") or []
    if sources:
        lines.append("\nИсточники:")
        for s in sources:
            if isinstance(s, dict):
                title = s.get("article_name") or s.get("title") or "?"
                url = s.get("url") or ""
                lines.append(f"- {title}" + (f" ({url})" if url else ""))

    meta = []
    if guardrail.get("refused"):
        meta.append("отказ (недостаточно подтверждённых сведений)")
    degraded = data.get("degraded") or []
    if degraded:
        meta.append(f"деградация: {', '.join(degraded)}")
    if data.get("cached"):
        meta.append("из кэша")
    if data.get("mode"):
        meta.append(f"режим: {data['mode']}")
    if meta:
        lines.append("\n[" + "; ".join(meta) + "]")

    return "\n".join(lines)


@mcp.tool()
async def ask_warhammer_lore(question: str, session_id: str = "") -> str:
    """Задать вопрос по лору Warhammer 40k русской вики-базе знаний.

    Проходит через полный пайплайн проекта (роутинг vector/graph,
    ретрив, guardrails) и возвращает ответ вместе с источниками.
    Если в базе недостаточно надёжных данных — вернётся честный отказ,
    а не выдумка.
    """
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SEC) as client:
            resp = await client.post(f"{API_URL}/v1/ask", json=payload)
            resp.raise_for_status()
            return _format_answer(resp.json())
    except httpx.ConnectError:
        return (
            f"Не удалось подключиться к {API_URL}. "
            "Проверь, что стек поднят: `make up` в WarhammerWikiBot."
        )
    except httpx.HTTPStatusError as e:
        return f"API вернул ошибку {e.response.status_code}: {e.response.text[:300]}"


@mcp.tool()
async def warhammer_service_status() -> str:
    """Проверить, жив ли Warhammer RAG API и его зависимости (Qdrant, TEI, реранкер, LightRAG)."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{API_URL}/v1/ready")
            data = resp.json()
            return f"ready={data.get('ready')}: {data}"
    except httpx.ConnectError:
        return f"Не удалось подключиться к {API_URL}. Стек не поднят? `make up`."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
