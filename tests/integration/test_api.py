"""Интеграционные тесты API через TestClient (БД SQLite, LLM — rule-based mock, эмбеддинги — hashing)."""
import httpx

from backend.utils.llm_client import OllamaClient
from tests.conftest import auth_headers

API = "/api/v1"


def product_json(**kw):
    base = {"name": "Термокружка Thermo Go 450 мл", "category": "Дом и кухня",
            "features": ["двойные стенки", "держит тепло 8 часов", "герметичная крышка"], "price": 1290,
            "audience": "путешественники", "brand": "Thermo Go", "style": "нейтральный", "platform": "wildberries"}
    return {**base, **kw}


def test_health(api):
    assert api.get("/healthz").json() == {"status": "ok"}
    h = api.get(f"{API}/health").json()
    assert h["status"] == "ok" and h["services"]["database"] and h["services"]["llm"]["mock"]
    assert api.get(f"{API}/llm/providers").json()["available"]["mock"]


def test_auth_flow_and_roles(api):
    admin = auth_headers(api, "admin")
    user = auth_headers(api, "user1")
    assert api.get(f"{API}/auth/me", headers=admin).json()["role"] == "admin"
    me = api.get(f"{API}/auth/me", headers=user).json()
    assert me["role"] == "user"
    dup = api.post(f"{API}/auth/register", json={"email": "user1@example.com", "username": "user1",
                                                  "password": "secret123"})
    assert dup.status_code == 409
    assert api.post(f"{API}/auth/login", data={"username": "user1", "password": "bad"}).status_code == 401
    assert api.get(f"{API}/auth/me").status_code == 401
    assert api.get(f"{API}/auth/me", headers={"Authorization": "Bearer x"}).status_code == 401
    assert api.get(f"{API}/admin/glossary", headers=user).status_code == 403
    assert api.get(f"{API}/auth/users", headers=user).status_code == 403
    assert len(api.get(f"{API}/auth/users", headers=admin).json()) == 2
    r = api.patch(f"{API}/auth/users/{me['id']}/role", json={"role": "editor"}, headers=admin)
    assert r.json()["role"] == "editor"
    assert api.patch(f"{API}/auth/users/999/role", json={"role": "editor"}, headers=admin).status_code == 404
    assert api.get(f"{API}/admin/glossary", headers=user).status_code == 200


def test_registration_errors_do_not_expose_input(api):
    response = api.post(f"{API}/auth/register", json={
        "email": "invalid-email-sample", "username": "ab", "password": "abcde",
    })

    assert response.status_code == 422
    details = {error["loc"][-1]: error for error in response.json()["detail"]}
    assert set(details) == {"email", "username", "password"}
    assert details["password"]["type"] == "string_too_short"
    assert details["password"]["ctx"]["min_length"] == 6
    assert all("input" not in error for error in details.values())
    assert "abcde" not in response.text and "invalid-email-sample" not in response.text

    invalid_body = api.post(f"{API}/auth/register", content='{"password":"malformed-secret",')
    assert invalid_body.status_code == 422
    assert "malformed-secret" not in invalid_body.text


def test_generate_endpoints(api, admin_headers):
    r = api.post(f"{API}/generate", json={"product": product_json(), "mode": "agents"}, headers=admin_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "+agents" and body["critic_score"] and body["examples_used"]
    cached = api.post(f"{API}/generate", json={"product": product_json(), "mode": "agents"}, headers=admin_headers)
    assert cached.json()["cached"]
    assert api.post(f"{API}/generate", json={"product": product_json(), "mode": "x"},
                    headers=admin_headers).status_code == 422
    bad_provider = api.post(f"{API}/generate", json={"product": product_json(), "provider": "nope",
                                                     "use_cache": False}, headers=admin_headers)
    assert bad_provider.status_code == 400

    multi = api.post(f"{API}/generate/multi", json={"product": product_json(), "platforms": ["ozon", "site"],
                                                    "styles": ["нейтральный", "премиальный"], "mode": "dict"},
                     headers=admin_headers)
    assert len(multi.json()) == 4
    multi_payload = {"product": product_json(), "platforms": ["ozon", "site"], "mode": "baseline",
                     "provider": "mock", "model": "custom", "use_cache": False}
    first_multi = api.post(f"{API}/generate/multi", json=multi_payload, headers=admin_headers).json()
    second_multi = api.post(f"{API}/generate/multi", json=multi_payload, headers=admin_headers).json()
    assert all(result["model"] == "custom" and not result["cached"] for result in second_multi)
    assert {r["request_id"] for r in first_multi}.isdisjoint(r["request_id"] for r in second_multi)
    too_many = api.post(f"{API}/generate/multi", json={"product": product_json(), "platforms": ["ozon"] * 4,
                                                       "styles": ["нейтральный"] * 4}, headers=admin_headers)
    assert too_many.status_code == 422

    batch = api.post(f"{API}/generate/batch", json={"products": [product_json(), product_json(name="Плед Uyut")],
                                                    "mode": "baseline"}, headers=admin_headers)
    assert batch.json()["count"] == 2
    bg = api.post(f"{API}/generate/batch", json={"products": [product_json()], "mode": "baseline",
                                                 "background": True}, headers=admin_headers).json()
    assert bg["status"] == "SUCCESS" and bg["result"]["count"] == 1

    csv = ("name,category,features,price,audience,brand,style,platform\n"
           "Плед Uyut,Дом и кухня,мягкий;тёплый,990,семьи,Uyut,нейтральный,ozon\n")
    up = api.post(f"{API}/generate/batch/upload", files={"file": ("p.csv", csv.encode(), "text/csv")},
                  data={"mode": "baseline", "background": "false"}, headers=admin_headers)
    assert up.json()["count"] == 1
    bad_csv = api.post(f"{API}/generate/batch/upload", files={"file": ("p.csv", b"name,category\nX,\n", "text/csv")},
                       data={"mode": "baseline", "background": "false"}, headers=admin_headers)
    assert bad_csv.status_code == 422

    ens = api.post(f"{API}/generate/ensemble", json={"product": product_json(), "models": ["mock:a", "mock:b"]},
                   headers=admin_headers)
    assert len(ens.json()["candidates"]) == 2
    fail = api.post(f"{API}/generate/ensemble", json={"product": product_json(), "models": ["ollama:x"]},
                    headers=admin_headers)
    assert fail.status_code == 503


def test_generation_cache_and_feedback_respect_request_owner(api, admin_headers):
    owner = auth_headers(api, "owner")
    other = auth_headers(api, "other")
    payload = {"product": product_json(), "mode": "baseline"}
    first = api.post(f"{API}/generate", json=payload, headers=owner).json()
    own_repeat = api.post(f"{API}/generate", json=payload, headers=owner).json()
    second = api.post(f"{API}/generate", json=payload, headers=other).json()
    assert own_repeat["cached"] and own_repeat["request_id"] == first["request_id"]
    assert not second["cached"] and second["request_id"] != first["request_id"]
    for result, headers in [(first, owner), (second, other)]:
        log = api.get(f"{API}/requests/{result['request_id']}/reasoning_log", headers=headers)
        assert log.status_code == 200 and log.json()["steps"]

    feedback = {"request_id": first["request_id"], "rating": 1}
    assert api.post(f"{API}/hitl/feedback", json=feedback, headers=other).status_code == 403
    assert api.post(f"{API}/hitl/feedback", json={**feedback, "request_id": "unknown"},
                    headers=admin_headers).status_code == 404
    assert api.get(f"{API}/hitl/uncertain", headers=admin_headers).json() == []
    assert api.post(f"{API}/hitl/feedback", json=feedback, headers=owner).status_code == 201


def test_ollama_models_for_generation_and_evaluation(api, admin_headers):
    installed = "qwen2.5:1.5b"

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": installed}]})
        return httpx.Response(200, json={"response": "Описание термокружки на 450 мл."})

    manager = api.container.llm
    manager.ollama = OllamaClient("http://ollama", transport=httpx.MockTransport(handler))
    manager.settings.llm_fallback_order = ["ollama", "mock"]
    manager.settings.ensemble_models = ["ollama:qwen2.5:7b", "mock"]

    providers = api.get(f"{API}/llm/providers").json()
    assert providers["ollama_models"] == [installed]
    assert set(providers["ensemble_models"]) == {f"ollama:{installed}", "mock"}
    body = {"product": product_json(), "mode": "baseline"}
    assert api.post(f"{API}/generate", json=body, headers=admin_headers).json()["model"] == installed
    ensemble = api.post(f"{API}/generate/ensemble", json=body, headers=admin_headers).json()
    assert {c["model"] for c in ensemble["candidates"]} == {f"ollama:{installed}", "mock"}
    missing = api.post(f"{API}/evaluate", json={"model": "ollama:qwen2.5:7b", "sample_size": 1},
                       headers=admin_headers)
    assert missing.status_code == 503
    evaluation = api.post(f"{API}/evaluate", json={"model": f"ollama:{installed}", "sample_size": 1},
                          headers=admin_headers)
    assert evaluation.json()["models_actually_used"] == [f"ollama:{installed}"]


def test_requests_and_reasoning_log(api, admin_headers):
    rid = api.post(f"{API}/generate", json={"product": product_json(), "mode": "+agents", "use_cache": False},
                   headers=admin_headers).json()["request_id"]
    listing = api.get(f"{API}/requests", headers=admin_headers).json()
    assert listing[0]["request_id"] == rid
    assert api.get(f"{API}/requests/{rid}", headers=admin_headers).json()["output"]
    log = api.get(f"{API}/requests/{rid}/reasoning_log", headers=admin_headers).json()
    assert [s["agent_name"] for s in log["steps"]][:3] == ["DescriptionGenerator", "Validator", "Critic"]
    assert log["steps"][0]["model_name"] == "mock:rule-based-v1" and log["steps"][0]["started_at"]
    calls = api.get(f"{API}/requests/{rid}/llm_calls", headers=admin_headers).json()["calls"]
    assert {c["agent_name"] for c in calls} >= {"DescriptionGenerator", "Critic"}
    assert all(c["prompt_hash"] and c["latency_ms"] >= 0 for c in calls)
    assert api.get(f"{API}/requests/unknown", headers=admin_headers).status_code == 404
    other = auth_headers(api, "stranger")
    assert api.get(f"{API}/requests/{rid}/reasoning_log", headers=other).status_code == 403
    assert api.get(f"{API}/requests", headers=other).json() == []


def test_admin_crud_and_import(api, admin_headers):
    h = admin_headers
    g = api.post(f"{API}/admin/glossary", json={"term": "тёплый уют", "kind": "selling_phrase",
                                                "category": "Дом и кухня"}, headers=h).json()
    assert api.get(f"{API}/admin/glossary", params={"q": "тёплый уют"}, headers=h).json()[0]["id"] == g["id"]
    assert api.get(f"{API}/admin/glossary", params={"kind": "forbidden", "category": "x"}, headers=h).json() == []
    upd = api.put(f"{API}/admin/glossary/{g['id']}", json={"term": "тёплый уют дома", "kind": "forbidden"},
                  headers=h)
    assert upd.json()["term"] == "тёплый уют дома"
    assert "тёплый уют дома" in api.container.knowledge.forbidden_phrases()
    assert "тёплый уют" not in api.container.knowledge.selling_phrases("Дом и кухня", k=100)
    assert api.put(f"{API}/admin/glossary/0", json={"term": "x"}, headers=h).status_code == 404
    assert api.delete(f"{API}/admin/glossary/{g['id']}", headers=h).status_code == 204
    assert "тёплый уют дома" not in api.container.knowledge.forbidden_phrases()
    assert api.delete(f"{API}/admin/glossary/{g['id']}", headers=h).status_code == 404

    k = api.post(f"{API}/admin/knowledge", json={"source_text": "Название: Новый товар", "category": "Обувь",
                                                 "target_text": "Отличное описание нового товара"}, headers=h).json()
    assert api.get(f"{API}/admin/knowledge", params={"category": "Обувь", "q": "Новый"}, headers=h).json()
    assert api.get(f"{API}/admin/knowledge", params={"platform": "site"}, headers=h).status_code == 200
    put = api.put(f"{API}/admin/knowledge/{k['id']}", json={"source_text": "Название: Новый товар 2",
                                                           "target_text": "Обновлённое описание товара"}, headers=h)
    assert put.json()["source_text"].endswith("2")
    assert api.put(f"{API}/admin/knowledge/0", json={"source_text": "abc", "target_text": "abcdefghijk"},
                   headers=h).status_code == 404
    assert api.delete(f"{API}/admin/knowledge/{k['id']}", headers=h).status_code == 204
    assert api.delete(f"{API}/admin/knowledge/{k['id']}", headers=h).status_code == 404

    csv = "term,definition,kind,category\nновая выгода,описание,selling_phrase,Обувь\n"
    imp = api.post(f"{API}/admin/import/glossary", files={"file": ("g.csv", csv.encode())}, headers=h).json()
    assert imp["added"] == 1
    js = '{"records": [{"source_text": "Название: Импорт", "target_text": "Описание импортированного товара"}]}'
    assert api.post(f"{API}/admin/import/knowledge", files={"file": ("k.json", js.encode())},
                    headers=h).json()["added"] == 1
    assert api.post(f"{API}/admin/import/other", files={"file": ("k.json", b"[]")}, headers=h).status_code == 422
    for filename, data in [
        ("bad.json", b"{"),
        ("bad.json", b"[1]"),
        ("bad.json", b'[{"term": 123}]'),
        ("bad.csv", b"term,kind\n\"unclosed"),
        ("bad.csv", b"term,kind\n\xff"),
    ]:
        bad = api.post(f"{API}/admin/import/glossary", files={"file": (filename, data)}, headers=h)
        assert bad.status_code == 422 and isinstance(bad.json()["detail"], str)
    assert api.post(f"{API}/admin/reindex", headers=h).json()["stats"]["examples"] == 121
    assert api.post(f"{API}/admin/reindex", params={"background": True}, headers=h).json()["status"] == "SUCCESS"
    stats = api.get(f"{API}/admin/stats", headers=h).json()
    assert stats["rows"]["knowledge_base"] == 121 and stats["glossary_by_kind"]["forbidden"] == 50


def test_evaluate_endpoints(api, admin_headers):
    h = admin_headers
    r = api.post(f"{API}/evaluate", json={"model": "mock", "mode": "rag", "sample_size": 4}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["bleu"] >= 0 and r.json()["id"]
    assert api.post(f"{API}/evaluate", json={"model": "ollama:qwen2.5:7b", "sample_size": 2},
                    headers=h).status_code == 503
    bg = api.post(f"{API}/evaluate", json={"model": "mock", "sample_size": 2, "background": True}, headers=h).json()
    assert bg["status"] == "SUCCESS"
    cmp = api.get(f"{API}/evaluate/compare", params={"models": "mock:a,mock:b", "sample_size": 3}, headers=h).json()
    assert cmp["winner"] in ("mock:a", "mock:b")
    cmp_bg = api.get(f"{API}/evaluate/compare", params={"models": "mock", "sample_size": 2, "background": True},
                     headers=h).json()
    assert cmp_bg["status"] == "SUCCESS"
    batch = api.post(f"{API}/evaluate/batch", json={"models": ["mock", "openai:gpt"], "modes": ["baseline"],
                                                    "sample_sizes": [2], "background": False}, headers=h).json()
    assert len(batch["results"]) == 2 and "error" in batch["results"][1]
    batch_bg = api.post(f"{API}/evaluate/batch", json={"models": ["mock"], "sample_sizes": [2]}, headers=h).json()
    assert batch_bg["status"] == "SUCCESS"
    hist = api.get(f"{API}/evaluate/history", headers=h).json()
    assert len(hist) >= 5 and api.get(f"{API}/evaluate/history", params={"model": "mock:a"}, headers=h).json()
    assert len(api.get(f"{API}/evaluate/comparisons", headers=h).json()) == 2


def test_hitl_endpoints(api, admin_headers):
    h = admin_headers
    rid = api.post(f"{API}/generate", json={"product": product_json(), "mode": "baseline"}, headers=h
                   ).json()["request_id"]
    assert api.post(f"{API}/hitl/feedback", json={"request_id": rid, "rating": 2, "comment": "слабо"},
                    headers=h).status_code == 201
    items = api.get(f"{API}/hitl/uncertain", headers=h).json()
    assert items and items[0]["request_id"] == rid
    ok = api.post(f"{API}/hitl/uncertain/{items[0]['id']}/approve",
                  json={"corrected_output": "Отредактированное описание термокружки"}, headers=h)
    assert ok.json()["status"] == "approved" and ok.json()["knowledge_id"]
    again = api.post(f"{API}/hitl/uncertain/{items[0]['id']}/approve", json={}, headers=h)
    assert again.status_code == 422
    api.post(f"{API}/hitl/feedback", json={"request_id": rid, "rating": 1}, headers=h)
    second = api.get(f"{API}/hitl/uncertain", headers=h).json()[0]
    assert api.post(f"{API}/hitl/uncertain/{second['id']}/reject", headers=h).json()["status"] == "rejected"
    assert api.post(f"{API}/hitl/uncertain/999/reject", headers=h).status_code == 404
    assert api.get(f"{API}/hitl/uncertain", params={"status": ""}, headers=h).json()
    stats = api.get(f"{API}/hitl/stats", headers=h).json()
    assert stats["queue"] == {"approved": 1, "rejected": 1} and stats["average_rating"] == 1.5


def test_monitoring(api, admin_headers):
    api.post(f"{API}/generate", json={"product": product_json(), "mode": "+agents"}, headers=admin_headers)
    stats = api.get(f"{API}/monitoring/stats", headers=admin_headers).json()
    assert stats["llm_calls"] and stats["generations"][0]["mode"] == "+agents"
    assert stats["cache"]["entries"] == 1
    assert api.post(f"{API}/monitoring/cache/clear", headers=admin_headers).json()["deleted"] == 1
    task = api.get(f"{API}/tasks/some-id", headers=admin_headers).json()
    assert task["task_id"] == "some-id"
