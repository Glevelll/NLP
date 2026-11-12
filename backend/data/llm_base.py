"""Общие типы для LLM-бэкендов."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class LLMBackend(Protocol):
    provider: str

    async def generate(self, prompt: str, *, model: str | None = None, system: str | None = None,
                       temperature: float = 0.7, max_tokens: int = 800, **kwargs: Any) -> LLMResponse: ...

    async def is_available(self) -> bool: ...
