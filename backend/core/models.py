"""Pydantic-модели: входные данные, состояние агентов, запросы и ответы API."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from backend.core.constants import Mode, Platform, Role, Style


def normalize_mode(value: str) -> str:
    """'rag', '+rag', 'RAG' -> '+rag'; 'baseline' остаётся без плюса."""
    v = (value or "").strip().lower()
    if v in ("baseline", "base", ""):
        return Mode.BASELINE.value
    if not v.startswith("+"):
        v = "+" + v
    if v not in {m.value for m in Mode}:
        raise ValueError(f"Неизвестный режим: {value}")
    return v


class ProductInput(BaseModel):
    """Характеристики товара, которые вводит пользователь."""

    name: str = Field(..., min_length=2, max_length=200, examples=["Беспроводные наушники SoundMax Air 3"])
    category: str = Field(..., min_length=2, max_length=100, examples=["Электроника"])
    features: list[str] = Field(default_factory=list, examples=[["активное шумоподавление", "до 30 часов работы"]])
    price: float | None = Field(default=None, ge=0)
    audience: str | None = Field(default=None, max_length=200, examples=["любители музыки и спорта"])
    brand: str | None = Field(default=None, max_length=100)
    style: Style = Style.NEUTRAL
    platform: Platform = Platform.OZON

    @field_validator("features")
    @classmethod
    def _clean_features(cls, v: list[str]) -> list[str]:
        cleaned = [f.strip() for f in v if f and f.strip()]
        if len(cleaned) > 20:
            raise ValueError("Не более 20 ключевых особенностей")
        return cleaned

    def to_text(self) -> str:
        """Текстовое представление характеристик (используется для поиска и кэша)."""
        parts = [f"Название: {self.name}", f"Категория: {self.category}"]
        if self.brand:
            parts.append(f"Бренд: {self.brand}")
        if self.features:
            parts.append("Особенности: " + "; ".join(self.features))
        if self.price is not None:
            parts.append(f"Цена: {self.price:g} ₽")
        if self.audience:
            parts.append(f"Целевая аудитория: {self.audience}")
        parts.append(f"Стиль: {self.style.value}")
        parts.append(f"Площадка: {self.platform.value}")
        return "\n".join(parts)


class RetrievedExample(BaseModel):
    id: str
    text: str
    description: str = ""
    category: str | None = None
    platform: str | None = None
    style: str | None = None
    score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None


class ReasoningStep(BaseModel):
    """Одна запись лога рассуждений."""

    step: int
    agent_name: str
    input_summary: str = ""
    output_summary: str = ""
    model_name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "ok"
    error: str | None = None
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)


class AgentState(BaseModel):
    """Общее состояние, которое агенты передают друг другу."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    product: ProductInput
    mode: str = Mode.AGENTS.value
    provider: str | None = None
    model: str | None = None

    # Контекст из базы знаний и справочника
    examples: list[RetrievedExample] = Field(default_factory=list)
    selling_phrases: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    benefit_hints: dict[str, str] = Field(default_factory=dict)  # особенность -> выгода из справочника
    use_dict: bool = True  # передавать ли справочник в промпт генератора

    # Результат
    draft: str = ""
    description: str = ""

    # Validator
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validation_passed: bool = False
    missing_features: list[str] = Field(default_factory=list)

    # Critic
    critic_scores: dict[str, float] = Field(default_factory=dict)
    critic_score: float = 0.0
    critic_reasoning: str = ""
    critic_suggestions: list[str] = Field(default_factory=list)
    critic_passed: bool = False
    critic_fallback: bool = False

    # Editor
    editor_changed: bool = False

    iteration: int = 0
    success: bool = False
    needs_review: bool = False
    reasoning_log: list[ReasoningStep] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)


# ---------- API: генерация ----------
class GenerateRequest(BaseModel):
    product: ProductInput
    mode: str = Mode.AGENTS.value
    provider: str | None = None
    model: str | None = None
    use_cache: bool = True

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return normalize_mode(v)


class MultiGenerateRequest(BaseModel):
    product: ProductInput
    platforms: list[Platform] = Field(default_factory=lambda: list(Platform))
    styles: list[Style] = Field(default_factory=list)
    mode: str = Mode.AGENTS.value
    provider: str | None = None
    model: str | None = None
    use_cache: bool = True

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return normalize_mode(v)


class BatchGenerateRequest(BaseModel):
    products: list[ProductInput] = Field(..., min_length=1, max_length=500)
    mode: str = Mode.AGENTS.value
    provider: str | None = None
    model: str | None = None
    background: bool = False

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return normalize_mode(v)


class EnsembleRequest(BaseModel):
    product: ProductInput
    models: list[str] | None = Field(default=None, examples=[["ollama:qwen2.5:7b", "openai:gpt-4o-mini"]])
    use_agents: bool = False


class GenerationResult(BaseModel):
    request_id: str
    description: str
    mode: str
    provider: str | None = None
    model: str | None = None
    validation_passed: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    critic_score: float | None = None
    critic_scores: dict[str, float] = Field(default_factory=dict)
    critic_suggestions: list[str] = Field(default_factory=list)
    iterations: int = 0
    needs_review: bool = False
    cached: bool = False
    examples_used: list[RetrievedExample] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: float = 0.0


# ---------- API: аутентификация ----------
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    username: str
    role: Role
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class RoleUpdate(BaseModel):
    role: Role


# ---------- API: справочник / база знаний ----------
class GlossaryIn(BaseModel):
    term: str = Field(..., min_length=1, max_length=500)
    definition: str = ""
    kind: str = "selling_phrase"
    category: str | None = None


class GlossaryOut(GlossaryIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class KnowledgeIn(BaseModel):
    source_text: str = Field(..., min_length=3)
    target_text: str = Field(..., min_length=10)
    category: str | None = None
    platform: str | None = None
    style: str | None = None
    product_name: str | None = None


class KnowledgeOut(KnowledgeIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    created_at: datetime


# ---------- API: оценка ----------
class EvaluateRequest(BaseModel):
    model: str = Field("mock", examples=["ollama:qwen2.5:7b", "openai:gpt-4o-mini", "mock"])
    mode: str = Mode.BASELINE.value
    sample_size: int = Field(20, ge=1, le=1000)
    seed: int = 42
    background: bool = False

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        return normalize_mode(v)


class BatchEvaluateRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["mock"])
    modes: list[str] = Field(default_factory=lambda: [Mode.BASELINE.value])
    sample_sizes: list[int] = Field(default_factory=lambda: [20])
    seed: int = 42
    background: bool = True

    @field_validator("modes")
    @classmethod
    def _modes(cls, v: list[str]) -> list[str]:
        return [normalize_mode(m) for m in v]


# ---------- API: HITL ----------
class ReviewDecision(BaseModel):
    corrected_output: str | None = None
    comment: str | None = None


class FeedbackIn(BaseModel):
    request_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None
    corrected_text: str | None = None
