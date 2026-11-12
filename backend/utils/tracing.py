"""Контекст трассировки запроса: request_id, текущий агент и сборщик LLM-вызовов (contextvars)."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
agent_name_var: ContextVar[str | None] = ContextVar("agent_name", default=None)
llm_trace_var: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_trace", default=None)


@contextmanager
def trace_request(request_id: str):
    token = request_id_var.set(request_id)
    try:
        yield
    finally:
        request_id_var.reset(token)


@contextmanager
def trace_agent(agent_name: str):
    """Пока агент работает, все LLM-вызовы помечаются его именем и собираются в список."""
    t1 = agent_name_var.set(agent_name)
    calls: list[dict[str, Any]] = []
    t2 = llm_trace_var.set(calls)
    try:
        yield calls
    finally:
        agent_name_var.reset(t1)
        llm_trace_var.reset(t2)


def record_llm_call(info: dict[str, Any]) -> None:
    calls = llm_trace_var.get()
    if calls is not None:
        calls.append(info)
