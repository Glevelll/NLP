"""DescriptionGenerator — создаёт черновик описания по характеристикам, справочнику и примерам из RAG."""
from __future__ import annotations

from backend.agents import prompts
from backend.agents.base import Agent
from backend.core.models import AgentState


class DescriptionGenerator(Agent):
    name = "DescriptionGenerator"

    def summarize_input(self, state: AgentState) -> str:
        p = state.product
        return (f"{p.name} [{p.category}] {p.platform.value}/{p.style.value}; особенностей: {len(p.features)}; "
                f"примеров RAG: {len(state.examples)}; фраз справочника: {len(state.selling_phrases)}")

    async def process(self, state: AgentState) -> AgentState:
        p = state.product
        prompt = prompts.generator_prompt(p, state.examples, state.selling_phrases,
                                          state.forbidden_phrases if state.use_dict else [], state.templates,
                                          state.benefit_hints)
        payload = {
            "product": prompts.product_payload(p),
            "selling_phrases": state.selling_phrases,
            "benefits": state.benefit_hints,
            "examples": [{"description": e.description, "source": e.text} for e in state.examples],
            "variant": state.model or "",
        }
        resp = await self.call_llm(state, prompt, system=prompts.GENERATOR_SYSTEM,
                                   temperature=self.settings.temperature_generator, task="generate", payload=payload)
        state.draft = resp.text
        state.description = resp.text
        self.last_result = {"chars": len(resp.text), "preview": resp.text[:300]}
        return state
