"""Supervisor — оркестратор: Generator -> (Validator -> Critic -> Editor)* с ограничением итераций
и подробным логом рассуждений."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from backend.agents.base import Agent
from backend.agents.critic import Critic
from backend.agents.editor import Editor
from backend.agents.generator import DescriptionGenerator
from backend.agents.validator import Validator
from backend.core.config import Settings
from backend.core.models import AgentState, ReasoningStep
from backend.data.llm_manager import LLMManager
from backend.utils.reasoning_logger import ReasoningLogStore
from backend.utils.tracing import trace_agent, trace_request

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Supervisor:
    def __init__(self, llm: LLMManager, settings: Settings, log_store: ReasoningLogStore | None = None,
                 agents: dict[str, Agent] | None = None):
        self.llm = llm
        self.settings = settings
        self.log_store = log_store
        self._agents = agents
        self.max_iterations = settings.max_iterations

    @property
    def agents(self) -> dict[str, Agent]:
        """Ленивая инициализация агентов."""
        if self._agents is None:
            self._agents = {
                "generator": DescriptionGenerator(self.llm, self.settings),
                "validator": Validator(self.llm, self.settings),
                "critic": Critic(self.llm, self.settings),
                "editor": Editor(self.llm, self.settings),
            }
        return self._agents

    def is_success(self, state: AgentState) -> bool:
        return state.validation_passed and state.critic_score >= self.settings.critic_threshold

    async def step(self, agent: Agent, state: AgentState, raise_errors: bool = False) -> AgentState:
        started = _now()
        input_summary = agent.summarize_input(state)
        status, error = "ok", None
        with trace_agent(agent.name) as calls:
            try:
                state = await agent.process(state)
            except Exception as exc:
                status, error = "error", str(exc)
                logger.exception("Агент %s завершился с ошибкой", agent.name)
                if raise_errors:
                    self._append(state, agent, input_summary, started, status, error, calls)
                    raise
        self._append(state, agent, input_summary, started, status, error, calls)
        return state

    def _append(self, state, agent, input_summary, started, status, error, calls) -> None:
        llm_meta = agent.last_llm or {}
        if not error and state.warnings and agent.name == "Validator":
            status = "warning"
        state.reasoning_log.append(ReasoningStep(
            step=len(state.reasoning_log) + 1,
            agent_name=agent.name,
            input_summary=input_summary,
            output_summary=agent.summarize_output(state) if not error else "",
            model_name=(f"{llm_meta['provider']}:{llm_meta['model']}" if llm_meta else None),
            parameters={"iteration": state.iteration, **({"temperature": llm_meta.get("temperature")}
                                                          if llm_meta else {})},
            started_at=started,
            finished_at=_now(),
            status=status,
            error=error,
            llm_calls=calls,
        ))

    async def run(self, state: AgentState, use_agents: bool = True, persist: bool = True) -> AgentState:
        """use_agents=False: только генерация и формальная проверка (режимы baseline/+dict/+rag)."""
        with trace_request(state.request_id):
            if not state.description:
                state = await self.step(self.agents["generator"], state, raise_errors=True)

            if not use_agents:
                state.iteration = 1
                state = await self.step(self.agents["validator"], state)
                state.success = state.validation_passed
            else:
                for iteration in range(1, self.max_iterations + 1):
                    state.iteration = iteration
                    state = await self.step(self.agents["validator"], state)
                    state = await self.step(self.agents["critic"], state)
                    if self.is_success(state):
                        state.success = True
                        break
                    if iteration == self.max_iterations:
                        break
                    state = await self.step(self.agents["editor"], state)
                    if not state.editor_changed:
                        break
                state.needs_review = (not state.success) or state.critic_fallback or \
                    state.critic_score < self.settings.uncertain_threshold

            if persist and self.log_store is not None:
                await self.log_store.save(state.request_id, state.reasoning_log)
        return state
