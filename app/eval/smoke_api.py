"""
Smoke по живому API: набор кейсов из JSON → expect refuse/answer.

Запуск (стек должен быть поднят):
  make smoke
  # или
  SMOKE_API_URL=http://localhost:8000 python -m app.eval.smoke_api
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "data/eval/smoke_cases.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_case(resp: dict[str, Any], expect: dict[str, Any], latency_ms: int) -> list[str]:
    errors: list[str] = []

    if "max_latency_ms" in expect and latency_ms > expect["max_latency_ms"]:
        errors.append(f"latency {latency_ms}ms > {expect['max_latency_ms']}ms")

    refused = bool(resp.get("guardrail", {}).get("refused"))
    if "refused" in expect and refused != expect["refused"]:
        errors.append(f"refused={refused}, expected {expect['refused']}")

    stage = expect.get("refuse_stage")
    if stage == "retrieval_gate":
        gate = resp.get("guardrail", {}).get("retrieval_gate") or {}
        if gate.get("passed") is not False:
            errors.append(f"expected retrieval_gate fail, got {gate}")

    if expect.get("refused") is False:
        answer = resp.get("answer") or ""
        needles = expect.get("answer_must_contain_any") or []
        if needles and not any(n.lower() in answer.lower() for n in needles):
            errors.append(f"answer missing any of {needles}: {answer[:120]!r}")
        min_sources = expect.get("min_sources", 0)
        if len(resp.get("sources") or []) < min_sources:
            errors.append(f"sources < {min_sources}")

    return errors


def run(api_url: str, cases_path: Path, timeout: float) -> int:
    cases = load_cases(cases_path)
    ask_url = f"{api_url.rstrip('/')}/v1/ask"
    passed = failed = 0

    print(f"Smoke → {ask_url} | cases={len(cases)}\n")

    with httpx.Client(timeout=timeout) as client:
        for case in cases:
            cid = case["id"]
            question = case["question"]
            expect = case.get("expect", {})
            t0 = time.perf_counter()
            try:
                r = client.post(ask_url, json={"question": question})
                latency_ms = int((time.perf_counter() - t0) * 1000)
                r.raise_for_status()
                body = r.json()
            except Exception as e:
                failed += 1
                print(f"FAIL  {cid}: request error: {e}")
                continue

            errors = check_case(body, expect, latency_ms)
            api_latency = body.get("latency_ms", latency_ms)
            mode = body.get("mode", "?")
            if errors:
                failed += 1
                print(f"FAIL  {cid} mode={mode} ({api_latency}ms)")
                for err in errors:
                    print(f"      - {err}")
                gate = (body.get("guardrail") or {}).get("retrieval_gate")
                print(f"      gate={gate}")
                print(f"      answer={(body.get('answer') or '')[:100]!r}")
            else:
                passed += 1
                status = "REFUSE" if body.get("guardrail", {}).get("refused") else "OK"
                print(f"PASS  {cid} [{status}] mode={mode} ({api_latency}ms)")

    print(f"\n=== {passed} passed, {failed} failed ===")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG API smoke cases")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL API (default: http://localhost:8000)",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    sys.exit(run(args.api_url, args.cases, args.timeout))


if __name__ == "__main__":
    main()
