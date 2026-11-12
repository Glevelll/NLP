import json

import pytest

from backend.agents.base import Agent
from backend.agents.critic import FALLBACK_SCORE, Critic, parse_critic_response
from backend.agents.editor import Editor
from backend.agents.generator import DescriptionGenerator
from backend.agents.prompts import critic_prompt, editor_prompt, generator_prompt
from backend.agents.supervisor import Supervisor
from backend.agents.validator import Validator
from backend.core.models import AgentState, ProductInput, RetrievedExample
from backend.data.llm_base import LLMResponse
from backend.db.session import Database
from backend.utils.reasoning_logger import ReasoningLogStore

GOOD_TEXT = ("Беспроводные наушники SoundMax Air 3 — практичный выбор на каждый день. "
             "Активное шумоподавление убирает шум метро. До 30 часов автономной работы — заряжать реже. "
             "Защита от воды IPX5 позволяет не бояться дождя. Бренд SoundMax подойдёт любителям музыки. "
             "Стоимость — 4990 ₽. Оформите заказ на Ozon и получите покупку в удобном пункте выдачи уже скоро. "
             "Звук остаётся чистым и насыщенным в любой обстановке.")


class FakeLLM:
    """Возвращает заранее заданные ответы по типу задачи."""

    def __init__(self, responses: dict[str, list[str]] | None = None, fail: set[str] | None = None):
        self.responses = responses or {}
        self.fail = fail or set()
        self.calls: list[dict] = []

    async def generate(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        task = kw.get("task")
        if task in self.fail:
            raise RuntimeError("LLM упал")
        queue = self.responses.get(task, ["текст"])
        text = queue.pop(0) if len(queue) > 1 else queue[0]
        return LLMResponse(text=text, provider="fake", model="fake-1")


def critic_json(score: float, suggestions=None) -> str:
    return json.dumps({"scores": {"attractiveness": score, "persuasiveness": score, "audience_relevance": score,
                                  "factual_accuracy": score, "style_match": score},
                       "overall": score, "reasoning": "ок", "suggestions": suggestions or []}, ensure_ascii=False)


# ------------------------- Validator -------------------------
async def test_validator_passes_good_text(state, settings):
    state.description = GOOD_TEXT
    state = await Validator(None, settings).process(state)
    assert state.validation_passed and not state.errors and state.missing_features == []


async def test_validator_finds_problems(state, settings):
    state.forbidden_phrases = ["лучший в мире"]
    state.description = ("Наушники — лучший в мире выбор!!!! КРУТЕЙШИЕ ПОТРЯСАЮЩИЕ НЕВЕРОЯТНЫЕ наушники. "
                         + "Отличный звук для всех. " * 2 + "Цена всего 3990 ₽. Подробнее на https://shop.example 😀")
    v = Validator(None, settings)
    state = await v.process(state)
    text = " ".join(state.errors)
    assert not state.validation_passed
    assert "короткое" in text and "Запрещённая" in text and "Ссылки" in text and "Цена" in text
    assert len(state.missing_features) == 3
    joined = " ".join(state.warnings)
    assert "Эмодзи" in joined and "восклицательных" in joined and "КАПСОМ" in joined and "Повтор" in joined
    assert "бренд" in joined.lower()
    assert "passed=False" in v.summarize_output(state)


async def test_validator_empty_long_and_style(state, settings):
    state = await Validator(None, settings).process(state)
    assert state.errors == ["Описание пустое"]
    state.description = GOOD_TEXT + " x" * 4000
    state = await Validator(None, settings).process(state)
    assert any("длинное" in e for e in state.errors)
    state.product = ProductInput(**{**state.product.model_dump(), "style": "премиальный"})
    state.description = GOOD_TEXT + " Сейчас действует скидка."
    state = await Validator(None, settings).process(state)
    assert any("скидк" in w for w in state.warnings)


async def test_validator_with_llm_check(state, settings):
    s = settings.model_copy(update={"validator_use_llm": True})
    state.description = GOOD_TEXT
    llm = FakeLLM({"validate": ['{"issues": ["выдуман материал"]}']})
    state = await Validator(llm, s).process(state)
    assert any("выдуман" in w for w in state.warnings) and state.validation_passed
    state = await Validator(FakeLLM(fail={"validate"}), s).process(state)
    assert any("недоступна" in w for w in state.warnings)


# ------------------------- Critic -------------------------
def test_parse_critic_response():
    scores, overall, reasoning, sugg = parse_critic_response(critic_json(8, ["добавить выгоды"]))
    assert overall == 8 and len(scores) == 5 and sugg == ["добавить выгоды"]
    flat = parse_critic_response('{"attractiveness": 12, "style_match": "6", "suggestions": "одна"}')
    assert flat[0]["attractiveness"] == 10 and flat[1] == 8.0 and flat[3] == ["одна"]
    assert parse_critic_response("мусор") is None
    assert parse_critic_response('{"scores": {"a": "x"}}') is None


async def test_critic_scores_and_threshold(state, settings):
    state.description = GOOD_TEXT
    critic = Critic(FakeLLM({"critic": [critic_json(8.5)]}), settings)
    state = await critic.process(state)
    assert state.critic_score == 8.5 and state.critic_passed and not state.critic_fallback
    assert critic.last_llm["model"] == "fake-1" and "fake:fake-1" in state.models_used


@pytest.mark.parametrize("llm", [FakeLLM({"critic": ["не json"]}), FakeLLM(fail={"critic"})])
async def test_critic_fallback(state, settings, llm):
    state.description = GOOD_TEXT
    state = await Critic(llm, settings).process(state)
    assert state.critic_fallback and state.critic_score == FALLBACK_SCORE and not state.critic_passed


# ------------------------- Editor -------------------------
async def test_editor_skips_when_no_issues(state, settings):
    llm = FakeLLM()
    state.description, state.critic_passed = GOOD_TEXT, True
    state = await Editor(llm, settings).process(state)
    assert not state.editor_changed and llm.calls == []


async def test_editor_applies_fixes(state, settings):
    state.description = "Короткий текст."
    state.errors, state.critic_suggestions = ["Слишком короткое описание"], ["Добавьте выгоды"]
    llm = FakeLLM({"edit": ["Исправленный текст: " + GOOD_TEXT]})
    editor = Editor(llm, settings)
    state = await editor.process(state)
    assert state.editor_changed and state.description == GOOD_TEXT
    assert "Слишком короткое" in llm.calls[0]["prompt"] and "ошибок: 1" in editor.summarize_input(state)
    # LLM вернул тот же текст -> changed=False
    state.errors = ["ещё ошибка"]
    state = await Editor(FakeLLM({"edit": [GOOD_TEXT]}), settings).process(state)
    assert not state.editor_changed


# ------------------------- Generator и промпты -------------------------
async def test_generator_uses_context(state, settings):
    state.examples = [RetrievedExample(id="kb:1", text="Название: X", description="Пример описания")]
    state.selling_phrases, state.benefit_hints = ["надёжный гаджет"], {"активное шумоподавление": "тишина"}
    state.forbidden_phrases = ["лучший в мире"]
    llm = FakeLLM({"generate": ["Описание: " + GOOD_TEXT]})
    gen = DescriptionGenerator(llm, settings)
    state = await gen.process(state)
    prompt = llm.calls[0]["prompt"]
    assert state.description == GOOD_TEXT == state.draft
    for marker in ("ПРИМЕРЫ", "ПРОДАЮЩИЕ ФРАЗЫ", "ВЫГОДЫ", "ЗАПРЕЩЕНО", "активное шумоподавление"):
        assert marker in prompt
    assert "примеров RAG: 1" in gen.summarize_input(state)
    state.use_dict = False
    await gen.process(state)
    assert "ЗАПРЕЩЕНО" not in llm.calls[1]["prompt"]


def test_prompts_contain_requirements(product):
    assert "Ozon" in generator_prompt(product, [], [], [], ["шаблон"])
    assert "ПРИМЕР 1" in critic_prompt(product, "текст")
    p = editor_prompt(product, "текст", ["e"], ["w"], ["s"], ["особенность"], ["нельзя"])
    assert "[ошибка] e" in p and "особенность" in p and "нельзя" in p


async def test_base_agent_requires_llm(state):
    class Dummy(Agent):
        async def process(self, s):
            await self.call_llm(s, "p", system=None, temperature=0, task="x")
            return s

    with pytest.raises(RuntimeError):
        await Dummy().process(state)
    assert "текст: 0" in Dummy().summarize_input(state) and Dummy().summarize_output(state) == "{}"


# ------------------------- Supervisor -------------------------
async def test_supervisor_loop_until_success(state, settings, tmp_path):
    db = Database(f"sqlite+aiosqlite:///{(tmp_path / 'sv.db').as_posix()}")
    await db.create_all()
    store = ReasoningLogStore(db.session_factory)
    llm = FakeLLM({"generate": ["Короткий текст."], "critic": [critic_json(4, ["больше выгод"]), critic_json(9)],
                   "edit": [GOOD_TEXT]})
    sup = Supervisor(llm, settings, store)
    result = await sup.run(state)
    agents = [s.agent_name for s in result.reasoning_log]
    assert agents == ["DescriptionGenerator", "Validator", "Critic", "Editor", "Validator", "Critic"]
    assert result.success and not result.needs_review and result.iteration == 2
    saved = await store.get(state.request_id)
    assert [s["agent_name"] for s in saved] == agents and saved[1]["status"] == "warning"
    assert saved[0]["model_name"] == "fake:fake-1" and saved[0]["duration_ms"] is not None
    await db.dispose()


async def test_supervisor_max_iterations_and_review(state, settings):
    s = settings.model_copy(update={"max_iterations": 2})
    llm = FakeLLM({"generate": ["Короткий."], "critic": [critic_json(3)], "edit": ["Другой короткий.", "Третий."]})
    result = await Supervisor(llm, s).run(state, persist=False)
    assert not result.success and result.needs_review and result.iteration == 2
    assert [x.agent_name for x in result.reasoning_log].count("Editor") == 1


async def test_supervisor_stops_when_editor_does_nothing(state, settings):
    llm = FakeLLM({"generate": ["Короткий."], "critic": [critic_json(3)], "edit": ["Короткий."]})
    result = await Supervisor(llm, settings).run(state, persist=False)
    assert result.iteration == 1 and not result.editor_changed


async def test_supervisor_baseline_and_errors(state, settings):
    llm = FakeLLM({"generate": [GOOD_TEXT]})
    result = await Supervisor(llm, settings).run(state, use_agents=False, persist=False)
    assert [x.agent_name for x in result.reasoning_log] == ["DescriptionGenerator", "Validator"]
    assert result.success
    broken = AgentState(product=state.product)
    with pytest.raises(RuntimeError):
        await Supervisor(FakeLLM(fail={"generate"}), settings).run(broken, persist=False)
    assert broken.reasoning_log[0].status == "error"
