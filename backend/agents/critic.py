"""Critic — LLM-as-a-Judge: оценки 0–10 по критериям, общий балл и рекомендации."""
from __future__ import annotations

import logging

from backend.agents import prompts
from backend.agents.base import Agent
from backend.core.models import AgentState
from backend.utils.llm_processing import extract_json

logger = logging.getLogger(__name__)
FALLBACK_SCORE = 5.0


def _to_float(value) -> float | None:
    try:
        return max(0.0, min(10.0, float(value)))
    except (TypeError, ValueError):
        return None


def parse_critic_response(text: str) -> tuple[dict[str, float], float, str, list[str]] | None:
    data = extract_json(text)
    if not data:
        return None
    raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {
        k: v for k, v in data.items() if k in prompts.CRITERIA
    }
    scores = {k: s for k, v in raw_scores.items() if (s := _to_float(v)) is not None}
    if not scores:
        return None
    overall = _to_float(data.get("overall"))
    if overall is None:
        overall = sum(scores.values()) / len(scores)
    suggestions = data.get("suggestions") or []
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    return scores, round(overall, 2), str(data.get("reasoning", "")), [str(s) for s in suggestions][:10]


class Critic(Agent):
    name = "Critic"

    def summarize_output(self, state: AgentState) -> str:
        return (f"score={state.critic_score} passed={state.critic_passed} fallback={state.critic_fallback}; "
                f"{state.critic_scores}; советы: {state.critic_suggestions}")

    async def process(self, state: AgentState) -> AgentState:
        parsed = None
        try:
            resp = await self.call_llm(
                state, prompts.critic_prompt(state.product, state.description), system=prompts.CRITIC_SYSTEM,
                temperature=self.settings.temperature_critic, task="critic", json_mode=True, max_tokens=400,
                payload={"description": state.description, "product": prompts.product_payload(state.product),
                         "forbidden_phrases": state.forbidden_phrases},
            )
            parsed = parse_critic_response(resp.text)
            if parsed is None:
                logger.warning("Critic: не удалось разобрать JSON, используется fallback")
        except Exception as exc:
            logger.warning("Critic: ошибка LLM (%s), используется fallback", exc)

        if parsed is None:
            state.critic_scores = {k: FALLBACK_SCORE for k in prompts.CRITERIA}
            state.critic_score = FALLBACK_SCORE
            state.critic_reasoning = "Fallback-оценка: LLM недоступна или вернула некорректный JSON"
            state.critic_suggestions = []
            state.critic_fallback = True
        else:
            state.critic_scores, state.critic_score, state.critic_reasoning, state.critic_suggestions = parsed
            state.critic_fallback = False
        state.critic_passed = state.critic_score >= self.settings.critic_threshold
        self.last_result = {"score": state.critic_score, "scores": state.critic_scores,
                            "suggestions": state.critic_suggestions, "fallback": state.critic_fallback}
        return state
