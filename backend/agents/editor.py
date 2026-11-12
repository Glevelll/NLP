"""Editor — исправляет описание по ошибкам Validator и рекомендациям Critic."""
from __future__ import annotations

from backend.agents import prompts
from backend.agents.base import Agent
from backend.core.models import AgentState
from backend.utils.text_processing import normalize


class Editor(Agent):
    name = "Editor"

    def summarize_input(self, state: AgentState) -> str:
        return (f"ошибок: {len(state.errors)}, предупреждений: {len(state.warnings)}, "
                f"советов: {len(state.critic_suggestions)}, оценка: {state.critic_score}")

    def summarize_output(self, state: AgentState) -> str:
        return f"changed={state.editor_changed}; текст: {state.description[:400]}"

    @staticmethod
    def needs_edit(state: AgentState) -> bool:
        return bool(state.errors or not state.critic_passed)

    async def process(self, state: AgentState) -> AgentState:
        if not self.needs_edit(state):
            state.editor_changed = False
            self.last_result = {"changed": False, "reason": "нет замечаний"}
            return state
        prompt = prompts.editor_prompt(state.product, state.description, state.errors, state.warnings,
                                       state.critic_suggestions, state.missing_features, state.forbidden_phrases)
        resp = await self.call_llm(
            state, prompt, system=prompts.EDITOR_SYSTEM, temperature=self.settings.temperature_editor, task="edit",
            payload={"description": state.description, "product": prompts.product_payload(state.product),
                     "errors": state.errors, "suggestions": state.critic_suggestions,
                     "missing_features": state.missing_features, "forbidden_phrases": state.forbidden_phrases},
        )
        new_text = resp.text.strip()
        changed = bool(new_text) and normalize(new_text) != normalize(state.description)
        if changed:
            state.description = new_text
        state.editor_changed = changed
        self.last_result = {"changed": changed, "chars": len(state.description)}
        return state
