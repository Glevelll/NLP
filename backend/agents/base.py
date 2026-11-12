"""Абстрактный агент."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from backend.core.config import Settings, get_settings
from backend.core.models import AgentState
from backend.data.llm_base import LLMResponse
from backend.data.llm_manager import LLMManager
from backend.utils.llm_processing import clean_response


class Agent(ABC):
    name: str = "Agent"

    def __init__(self, llm: LLMManager | None = None, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()
        self.last_result: dict[str, Any] | None = None
        self.last_llm: dict[str, Any] | None = None

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Принимает состояние, возвращает обновлённое состояние."""

    def summarize_input(self, state: AgentState) -> str:
        return f"{state.product.name} | текст: {len(state.description)} симв."

    def summarize_output(self, state: AgentState) -> str:
        return str(self.last_result or {})[:1000]

    async def call_llm(self, state: AgentState, prompt: str, *, system: str | None, temperature: float,
                       task: str, payload: dict[str, Any] | None = None, json_mode: bool = False,
                       max_tokens: int | None = None) -> LLMResponse:
        if self.llm is None:
            raise RuntimeError(f"{self.name}: LLM не подключён")
        self.last_llm = None
        resp = await self.llm.generate(prompt, system=system, provider=state.provider, model=state.model,
                                       temperature=temperature, task=task, payload=payload, json_mode=json_mode,
                                       max_tokens=max_tokens)
        if not json_mode:  # бэкенды уже чистят ответ, но сторонний LLM-объект может этого не делать
            resp.text = clean_response(resp.text)
        self.last_llm = {"provider": resp.provider, "model": resp.model, "temperature": temperature,
                         "tokens": resp.total_tokens}
        tag = f"{resp.provider}:{resp.model}"
        if tag not in state.models_used:
            state.models_used.append(tag)
        return resp
