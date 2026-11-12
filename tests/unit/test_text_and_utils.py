import asyncio
import logging

import pytest

from backend.core.exceptions import AuthError
from backend.core.models import ProductInput, normalize_mode
from backend.utils.async_utils import async_retry, gather_limited
from backend.utils.llm_processing import clean_response, extract_json, text_hash
from backend.utils.security import create_access_token, decode_token, hash_password, verify_password
from backend.utils.text_processing import (
    contains_phrase,
    feature_coverage,
    normalize,
    overused_words,
    repeated_sentences,
    split_sentences,
    stems,
    tokenize,
)


def test_normalize_and_tokenize():
    assert normalize("  Ёлка   ПУШИСТАЯ ") == "елка пушистая"
    assert tokenize("Ёмкость 20000 мА·ч, вес 1,2 кг") == ["емкость", "20000", "ма", "ч", "вес", "1,2", "кг"]


def test_stems_reduce_word_forms():
    assert stems("удобные кроссовки")[0] == stems("удобная кроссовка")[0]
    assert "и" not in stems("тепло и уютно")


def test_feature_coverage_word_forms():
    text = "Активным шумоподавлением вас не удивить? Эти наушники работают до 30 часов."
    assert feature_coverage("активное шумоподавление", text)[0]
    assert feature_coverage("до 30 часов работы", text)[0]
    assert not feature_coverage("защита от воды IPX5", text)[0]
    assert feature_coverage("", text) == (True, 1.0)


def test_repeats_and_overuse():
    text = "Отличный товар для дома. Отличный товар для дома. Ещё фраза."
    assert repeated_sentences(text)
    assert overused_words("удобный " * 10 + "стол") == [stems("удобный")[0]]
    assert split_sentences("Раз. Два!\nТри") == ["Раз.", "Два!", "Три"]


def test_contains_phrase_with_forms():
    assert contains_phrase("Это лучшие в мире наушники", "лучший в мире")
    assert not contains_phrase("Хорошие наушники", "лучший в мире")
    assert not contains_phrase("текст", "")


def test_clean_response_removes_wrappers():
    raw = "<think>рассуждаю</think>Вот описание: «**Отличные** наушники»"
    assert clean_response(raw) == "Отличные наушники"
    assert clean_response("```\nТекст описания\n```") == "Текст описания"
    assert clean_response("") == ""


def test_extract_json_variants():
    assert extract_json('Ответ: {"a": 1, "b": {"c": 2}} спасибо') == {"a": 1, "b": {"c": 2}}
    assert extract_json('```json\n{"x": [1, 2,],}\n```') == {"x": [1, 2]}
    assert extract_json("{'a': 'b'}") == {"a": "b"}
    assert extract_json("нет json") is None
    assert extract_json("") is None
    assert len(text_hash("abc")) == 16


def test_password_and_jwt():
    h = hash_password("secret123")
    assert verify_password("secret123", h) and not verify_password("wrong", h)
    assert not verify_password("x", "not-a-hash")
    token = create_access_token("42", "editor")
    payload = decode_token(token)
    assert payload["sub"] == "42" and payload["role"] == "editor"
    with pytest.raises(AuthError):
        decode_token(token + "broken")
    with pytest.raises(AuthError):
        decode_token(create_access_token("1", "user", expires_minutes=-1))


async def test_async_retry_and_gather():
    calls = {"n": 0}

    @async_retry(retries=2, delay=0.001)
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    assert await flaky() == "ok" and calls["n"] == 3

    @async_retry(retries=1, delay=0.001)
    async def always_fail():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        await always_fail()

    @async_retry(retries=3, delay=0.001, no_retry=(KeyError,))
    async def no_retry():
        calls["n"] += 1
        raise KeyError("x")

    calls["n"] = 0
    with pytest.raises(KeyError):
        await no_retry()
    assert calls["n"] == 1

    async def val(x):
        await asyncio.sleep(0)
        return x * 2

    assert await gather_limited([val(i) for i in range(5)], limit=2) == [0, 2, 4, 6, 8]


def test_models_helpers():
    assert normalize_mode("RAG") == "+rag" and normalize_mode("baseline") == "baseline"
    with pytest.raises(ValueError):
        normalize_mode("magic")
    p = ProductInput(name="Товар", category="Дом и кухня", features=[" a ", "", "b"], price=100)
    assert p.features == ["a", "b"]
    assert "Цена: 100 ₽" in p.to_text()
    with pytest.raises(ValueError):
        ProductInput(name="Товар", category="Дом", features=[str(i) for i in range(21)])


def test_setup_logging_writes_files(tmp_path):
    from backend.utils.logging import JsonFormatter, setup_logging
    from backend.utils.tracing import trace_agent, trace_request

    setup_logging(tmp_path, "INFO", json_logs=True, module_levels={"httpx": "WARNING"}, force=True)
    with trace_request("req-1"), trace_agent("Critic"):
        logging.getLogger("backend.llm").info("тест", extra={"extra_data": {"k": 1}})
        logging.getLogger("x").info("app")
    for h in logging.getLogger().handlers + logging.getLogger("backend.llm").handlers:
        h.flush()
    llm_log = (tmp_path / "llm_calls.log").read_text(encoding="utf-8")
    assert '"request_id": "req-1"' in llm_log and '"agent": "Critic"' in llm_log
    assert (tmp_path / "app.log").exists()
    rec = logging.LogRecord("n", logging.ERROR, "", 0, "m", None, None)
    assert '"level": "ERROR"' in JsonFormatter().format(rec)
