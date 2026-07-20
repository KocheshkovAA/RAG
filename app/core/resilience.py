"""Production resilience primitives: circuit breaker + timeout helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Простой in-process circuit breaker для внешних зависимостей."""

    name: str
    failure_threshold: int = 5
    recovery_timeout_sec: float = 30.0
    half_open_max_calls: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout_sec:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            return False
        # half-open: пропускаем ограниченное число пробных запросов
        if self._half_open_calls < self.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._half_open_calls = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit '%s' OPEN after %s failures",
                self.name,
                self._failures,
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self._failures,
        }


class CircuitOpenError(RuntimeError):
    def __init__(self, name: str):
        super().__init__(f"Circuit '{name}' is open")
        self.name = name


async def call_with_circuit(
    breaker: CircuitBreaker,
    coro_factory: Callable[[], Awaitable[T]],
    *,
    fallback: Optional[Callable[[], Awaitable[T] | T] | T] = None,
) -> tuple[T, bool]:
    """
    Вызов через circuit breaker.
    Returns: (result, degraded) где degraded=True если сработал fallback.
    """
    if not breaker.allow():
        if fallback is None:
            raise CircuitOpenError(breaker.name)
        result = fallback() if callable(fallback) else fallback
        if asyncio.iscoroutine(result):
            result = await result
        return result, True

    try:
        result = await coro_factory()
        breaker.record_success()
        return result, False
    except Exception:
        breaker.record_failure()
        if fallback is None:
            raise
        result = fallback() if callable(fallback) else fallback
        if asyncio.iscoroutine(result):
            result = await result
        return result, True


async def with_timeout(coro: Awaitable[T], timeout_sec: float, label: str) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"{label} timed out after {timeout_sec}s") from e
