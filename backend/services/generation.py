"""Сервис генерации: режимы baseline / +dict / +rag / +agents / +ensemble, кэш, история, HITL."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.supervisor import Supervisor
from backend.core.config import Settings
from backend.core.constants import DEFAULT_FORBIDDEN, Mode
from backend.core.exceptions import LLMUnavailableError
from backend.core.models import AgentState, GenerationResult, ProductInput, ReasoningStep
from backend.data.cache import CacheService, make_cache_key
from backend.data.knowledge_base import KnowledgeService
from backend.data.llm_manager import LLMManager, parse_model_spec
from backend.db.models import GenerationRequest
from backend.hitl.active_learning import ActiveLearningQueue
from backend.utils.async_utils import gather_limited
from backend.utils.reasoning_logger import ReasoningLogStore
from backend.utils.tracing import trace_request

logger = logging.getLogger(__name__)


def ensemble_score(state: AgentState) -> float:
    """Балл кандидата ансамбля: оценка Critic + бонус за прохождение валидации - штраф за ошибки."""
    return round(state.critic_score + (1.5 if state.validation_passed else 0.0) - 0.5 * len(state.errors), 3)


class GenerationService:
    def __init__(self, settings: Settings, llm: LLMManager, knowledge: KnowledgeService, cache: CacheService,
                 session_factory: Callable[[], AsyncSession], log_store: ReasoningLogStore):
        self.settings = settings
        self.llm = llm
        self.knowledge = knowledge
        self.cache = cache
        self.session_factory = session_factory
        self.log_store = log_store

    # ---------------------------------------------------------------
    def prepare_state(self, product: ProductInput, mode: str, provider: str | None = None,
                      model: str | None = None, request_id: str | None = None) -> AgentState:
        use_dict = mode != Mode.BASELINE
        use_rag = mode in (Mode.RAG, Mode.AGENTS, Mode.ENSEMBLE)
        state = AgentState(request_id=request_id or uuid.uuid4().hex, product=product, mode=mode,
                           provider=provider, model=model, use_dict=use_dict)
        # Запрещённые формулировки валидатор проверяет всегда, а в промпт они попадают только со справочником
        state.forbidden_phrases = self.knowledge.forbidden_phrases() if self.knowledge.loaded else list(
            DEFAULT_FORBIDDEN)
        if use_dict:
            state.selling_phrases = self.knowledge.selling_phrases(product.category, k=6, seed=product.name)
            state.templates = self.knowledge.templates(product.platform.value, product.style.value)
            state.benefit_hints = self.knowledge.benefits_for(product.category, product.features)
        if use_rag:
            state.examples = self.knowledge.search_examples(product)
        return state

    async def run_state(self, product: ProductInput, mode: str, provider: str | None = None,
                        model: str | None = None, request_id: str | None = None, persist: bool = True) -> AgentState:
        state = self.prepare_state(product, mode, provider, model, request_id)
        supervisor = Supervisor(self.llm, self.settings, self.log_store if persist else None)
        return await supervisor.run(state, use_agents=(mode == Mode.AGENTS), persist=persist)

    @staticmethod
    def to_result(state: AgentState, elapsed_ms: float) -> GenerationResult:
        provider, model = (state.models_used[0].split(":", 1) if state.models_used else (state.provider, state.model))
        return GenerationResult(
            request_id=state.request_id, description=state.description, mode=state.mode, provider=provider,
            model=model, validation_passed=state.validation_passed, errors=state.errors, warnings=state.warnings,
            critic_score=state.critic_score if state.critic_scores else None, critic_scores=state.critic_scores,
            critic_suggestions=state.critic_suggestions, iterations=state.iteration,
            needs_review=state.needs_review, examples_used=state.examples, elapsed_ms=round(elapsed_ms, 1),
        )

    # ---------------------------------------------------------------
    async def generate(self, product: ProductInput, mode: str = Mode.AGENTS, provider: str | None = None,
                       model: str | None = None, use_cache: bool = True, user_id: int | None = None,
                       persist: bool = True) -> GenerationResult:
        if mode == Mode.ENSEMBLE:
            return await self.ensemble(product, None, use_agents=False, user_id=user_id, persist=persist)
        key = make_cache_key("generate", user_id, persist, product.model_dump(mode="json"), mode, provider, model)
        if use_cache:
            cached = await self.cache.get(key)
            if cached:
                return GenerationResult(**{**cached, "cached": True})

        start = time.perf_counter()
        state = await self.run_state(product, mode, provider, model, persist=persist)
        result = self.to_result(state, (time.perf_counter() - start) * 1000)
        saved = not persist or await self._persist(state, result, user_id)
        if use_cache and saved and state.description:
            await self.cache.set(key, result.model_dump(mode="json"))
        return result

    async def ensemble(self, product: ProductInput, models: list[str] | None = None, use_agents: bool = False,
                       user_id: int | None = None, persist: bool = True) -> GenerationResult:
        """Запускает несколько моделей, оценивает кандидатов Critic-ом и выбирает лучший."""
        specs = models or await self.llm.default_ensemble_specs()
        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        mode = Mode.AGENTS if use_agents else Mode.RAG

        async def candidate(idx: int, spec: str) -> tuple[str, AgentState | None, str | None]:
            prov, mdl = parse_model_spec(spec)
            if not await self.llm.model_available(prov, mdl):
                return spec, None, "модель недоступна"
            try:
                st = await self.run_state(product, mode, prov, mdl, request_id=f"{request_id}-{idx}",
                                          persist=False)
                if not st.description or not st.models_used or not self.llm.matches_model(st.models_used[0], prov, mdl):
                    return spec, None, "выбранная модель не вернула описание"
                if not use_agents:
                    sup = Supervisor(self.llm, self.settings)
                    with trace_request(request_id):
                        st = await sup.step(sup.agents["critic"], st)
                return spec, st, None
            except Exception as exc:
                return spec, None, str(exc)

        results = await gather_limited([candidate(i, s) for i, s in enumerate(specs)], limit=3)
        ok = [(spec, st) for spec, st, err in results if st is not None]
        if not ok:
            reasons = "; ".join(f"{s}: {e}" for s, _, e in results) or "нет доступных моделей"
            raise LLMUnavailableError(f"Не удалось сгенерировать описание в режиме ансамбля: {reasons}")
        ok.sort(key=lambda x: ensemble_score(x[1]), reverse=True)
        best_spec, best = ok[0]
        best.request_id = request_id
        best.mode = Mode.ENSEMBLE.value
        now = best.reasoning_log[-1].finished_at if best.reasoning_log else None
        best.reasoning_log.append(ReasoningStep(
            step=len(best.reasoning_log) + 1, agent_name="EnsembleSelector",
            input_summary=f"кандидатов: {len(specs)} ({', '.join(specs)})",
            output_summary=json.dumps({s: ensemble_score(st) for s, st in ok}, ensure_ascii=False)
            + f"; победитель: {best_spec}",
            parameters={"skipped": {s: e for s, st, e in results if st is None}},
            started_at=now or best.reasoning_log[0].started_at, finished_at=now or best.reasoning_log[0].started_at,
        ))
        result = self.to_result(best, (time.perf_counter() - start) * 1000)
        if len(ok) == 1:
            result.warnings = [*result.warnings, "Доступен один результат. Для сравнения нужны две модели."]
        result.candidates = [
            {"model": s, "score": ensemble_score(st), "critic_score": st.critic_score,
             "validation_passed": st.validation_passed, "description": st.description}
            for s, st in ok
        ]
        if persist:
            await self.log_store.save(request_id, best.reasoning_log)
            await self._persist(best, result, user_id)
        return result

    async def generate_multi(self, product: ProductInput, platforms: list, styles: list, mode: str,
                             provider: str | None, model: str | None, user_id: int | None = None,
                             *, use_cache: bool = True
                             ) -> list[GenerationResult]:
        variants = []
        for platform in platforms or [product.platform]:
            for style in styles or [product.style]:
                variants.append(ProductInput(**{**product.model_dump(), "platform": platform, "style": style}))
        return await gather_limited(
            [self.generate(v, mode, provider, model, use_cache, user_id) for v in variants], limit=3,
        )

    async def generate_batch(self, products: list[ProductInput], mode: str, provider: str | None = None,
                             model: str | None = None, user_id: int | None = None) -> list[GenerationResult]:
        return await gather_limited([self.generate(p, mode, provider, model, user_id=user_id) for p in products],
                                    limit=4)

    # ---------------------------------------------------------------
    async def _persist(self, state: AgentState, result: GenerationResult, user_id: int | None) -> bool:
        try:
            async with self.session_factory() as session:
                session.add(GenerationRequest(
                    request_id=state.request_id, user_id=user_id, product_name=state.product.name,
                    input_json=state.product.model_dump(mode="json"), output=state.description, mode=state.mode,
                    provider=result.provider, model_name=result.model, critic_score=result.critic_score,
                    validation_passed=state.validation_passed, iterations=state.iteration,
                    elapsed_ms=result.elapsed_ms,
                ))
                await session.commit()
                if state.needs_review:
                    reason = "; ".join(state.errors[:3]) or f"низкая оценка Critic ({state.critic_score})"
                    await ActiveLearningQueue(session).add(
                        request_id=state.request_id, input_text=state.product.to_text(),
                        input_json=state.product.model_dump(mode="json"), model_output=state.description,
                        critic_score=state.critic_score, reason=reason,
                    )
        except Exception as exc:
            logger.warning("Не удалось сохранить запрос %s: %s", state.request_id, exc)
            return False
        return True
