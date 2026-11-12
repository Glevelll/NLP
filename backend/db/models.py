"""SQLAlchemy-модели таблиц."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class KnowledgeBase(Base):
    """Эталонные пары «характеристики -> описание»."""

    __tablename__ = "knowledge_base"
    __table_args__ = (UniqueConstraint("source_text", "platform", "style", name="uq_kb_source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_text: Mapped[str] = mapped_column(Text)
    target_text: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    platform: Mapped[str | None] = mapped_column(String(30), index=True, nullable=True)
    style: Mapped[str | None] = mapped_column(String(30), nullable=True)
    product_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(30), default="import")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Glossary(Base):
    """Справочник: продающие фразы, запрещённые формулировки, шаблоны, выгоды."""

    __tablename__ = "glossary"
    __table_args__ = (UniqueConstraint("term", "kind", "category", name="uq_glossary_term"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    term: Mapped[str] = mapped_column(String(500))
    definition: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(30), index=True)
    category: Mapped[str | None] = mapped_column(String(100), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class UncertainExample(Base):
    __tablename__ = "uncertain_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    input_text: Mapped[str] = mapped_column(Text)
    input_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_output: Mapped[str] = mapped_column(Text)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    corrected_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CacheEntry(Base):
    __tablename__ = "cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    input_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    namespace: Mapped[str] = mapped_column(String(30), default="generation", index=True)
    output: Mapped[str] = mapped_column(Text)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GenerationRequest(Base):
    """История запросов генерации (для просмотра логов и HITL)."""

    __tablename__ = "generation_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    product_name: Mapped[str] = mapped_column(String(255))
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    output: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(20))
    provider: Mapped[str | None] = mapped_column(String(30), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ReasoningLog(Base):
    __tablename__ = "reasoning_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    step: Mapped[int] = mapped_column(Integer)
    agent_name: Mapped[str] = mapped_column(String(50))
    input_summary: Mapped[str] = mapped_column(Text, default="")
    output_summary: Mapped[str] = mapped_column(Text, default="")
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model_name: Mapped[str] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(30))
    prompt_hash: Mapped[str] = mapped_column(String(32))
    response_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)
    prompt_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="success")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    mode: Mapped[str] = mapped_column(String(20), index=True)
    sample_size: Mapped[int] = mapped_column(Integer)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    evaluation_time: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ModelComparison(Base):
    __tablename__ = "model_comparisons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    models_compared: Mapped[list[Any]] = mapped_column(JSON)
    sample_size: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(String(20), default="baseline")
    results_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    winner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
