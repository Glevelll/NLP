"""HTTP-клиент Streamlit-интерфейса к FastAPI."""
from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")


class APIError(Exception):
    def __init__(self, message: str, field_errors: dict[str, str] | None = None):
        super().__init__(message)
        self.field_errors = field_errors or {}


def _auth_field_errors(path: str, details: list) -> dict[str, str]:
    fields = {"username", "password"}
    if path == "/auth/register":
        fields.add("email")
    elif path != "/auth/login":
        return {}

    messages = {}
    for error in details:
        if not isinstance(error, dict):
            continue
        location = error.get("loc")
        if not isinstance(location, (list, tuple)) or len(location) != 2 or location[0] != "body":
            continue
        field = location[1]
        if not isinstance(field, str) or field not in fields:
            continue
        error_type = error.get("type")
        bounds = error.get("ctx")
        if not isinstance(bounds, dict):
            bounds = {}
        if error_type == "missing":
            message = "Заполните это поле."
        elif field == "email":
            message = "Введите email в формате name@example.com."
        elif error_type == "string_too_short" and isinstance(bounds.get("min_length"), int):
            message = f"Введите не менее {bounds['min_length']} символов."
        elif error_type == "string_too_long" and isinstance(bounds.get("max_length"), int):
            message = f"Введите не более {bounds['max_length']} символов."
        else:
            message = "Проверьте значение в этом поле."
        messages.setdefault(field, message)
    return messages


class APIClient:
    def __init__(self, token: str | None = None, base_url: str = API_URL, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            r = httpx.request(method, f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout,
                              **kwargs)
        except httpx.HTTPError as exc:
            raise APIError("Не удалось связаться с сервером. Попробуйте ещё раз.") from exc
        if r.status_code >= 400:
            if r.status_code >= 500:
                raise APIError("Сервер не смог обработать запрос. Попробуйте ещё раз.")
            try:
                body = r.json()
            except ValueError:
                body = {}
            detail = body.get("detail") if isinstance(body, dict) else None
            if isinstance(detail, list):
                raise APIError("Проверьте введённые данные.", _auth_field_errors(path, detail))
            message = detail if isinstance(detail, str) and detail.strip() else "Не удалось выполнить запрос."
            raise APIError(message)
        return r.json() if r.content else None

    # auth
    def login(self, username: str, password: str) -> dict:
        return self._request("POST", "/auth/login", data={"username": username, "password": password})

    def register(self, email: str, username: str, password: str) -> dict:
        return self._request("POST", "/auth/register", json={"email": email, "username": username,
                                                               "password": password})

    def me(self) -> dict:
        return self._request("GET", "/auth/me")

    # generation
    def generate(self, product: dict, mode: str, provider: str | None = None, model: str | None = None,
                 use_cache: bool = True) -> dict:
        return self._request("POST", "/generate", json={"product": product, "mode": mode, "provider": provider,
                                                        "model": model, "use_cache": use_cache})

    def generate_multi(self, product: dict, platforms: list[str], styles: list[str], mode: str) -> list[dict]:
        return self._request("POST", "/generate/multi", json={"product": product, "platforms": platforms,
                                                              "styles": styles, "mode": mode})

    def ensemble(self, product: dict, models: list[str] | None, use_agents: bool) -> dict:
        return self._request("POST", "/generate/ensemble", json={"product": product, "models": models,
                                                                 "use_agents": use_agents})

    def batch_upload(self, filename: str, content: bytes, mode: str, background: bool) -> dict:
        return self._request("POST", "/generate/batch/upload", files={"file": (filename, content, "text/csv")},
                             data={"mode": mode, "background": str(background).lower()})

    def task(self, task_id: str) -> dict:
        return self._request("GET", f"/tasks/{task_id}")

    # admin
    def glossary(self, **params) -> list[dict]:
        return self._request("GET", "/admin/glossary", params=params)

    def add_glossary(self, item: dict) -> dict:
        return self._request("POST", "/admin/glossary", json=item)

    def delete_glossary(self, item_id: int) -> None:
        return self._request("DELETE", f"/admin/glossary/{item_id}")

    def knowledge(self, **params) -> list[dict]:
        return self._request("GET", "/admin/knowledge", params=params)

    def add_knowledge(self, item: dict) -> dict:
        return self._request("POST", "/admin/knowledge", json=item)

    def delete_knowledge(self, item_id: int) -> None:
        return self._request("DELETE", f"/admin/knowledge/{item_id}")

    def import_file(self, kind: str, filename: str, content: bytes) -> dict:
        return self._request("POST", f"/admin/import/{kind}", files={"file": (filename, content)})

    def reindex(self) -> dict:
        return self._request("POST", "/admin/reindex")

    def admin_stats(self) -> dict:
        return self._request("GET", "/admin/stats")

    # hitl
    def uncertain(self, status: str = "pending") -> list[dict]:
        return self._request("GET", "/hitl/uncertain", params={"status": status})

    def approve(self, item_id: int, corrected: str | None) -> dict:
        return self._request("POST", f"/hitl/uncertain/{item_id}/approve", json={"corrected_output": corrected})

    def reject(self, item_id: int) -> dict:
        return self._request("POST", f"/hitl/uncertain/{item_id}/reject")

    def feedback(self, request_id: str, rating: int, comment: str | None = None) -> dict:
        return self._request("POST", "/hitl/feedback", json={"request_id": request_id, "rating": rating,
                                                             "comment": comment})

    # evaluation
    def evaluate(self, model: str, mode: str, sample_size: int, background: bool = False) -> dict:
        return self._request("POST", "/evaluate", json={"model": model, "mode": mode, "sample_size": sample_size,
                                                        "background": background})

    def evaluate_batch(self, models: list[str], modes: list[str], sizes: list[int], background: bool) -> dict:
        return self._request("POST", "/evaluate/batch", json={"models": models, "modes": modes,
                                                              "sample_sizes": sizes, "background": background})

    def compare(self, models: list[str], sample_size: int, mode: str) -> dict:
        return self._request("GET", "/evaluate/compare", params={"models": ",".join(models),
                                                                 "sample_size": sample_size, "mode": mode})

    def eval_history(self, limit: int = 200) -> list[dict]:
        return self._request("GET", "/evaluate/history", params={"limit": limit})

    # logs & monitoring
    def requests(self, limit: int = 50) -> list[dict]:
        return self._request("GET", "/requests", params={"limit": limit})

    def request(self, request_id: str) -> dict:
        return self._request("GET", f"/requests/{request_id}")

    def reasoning_log(self, request_id: str) -> dict:
        return self._request("GET", f"/requests/{request_id}/reasoning_log")

    def llm_calls(self, request_id: str) -> dict:
        return self._request("GET", f"/requests/{request_id}/llm_calls")

    def health(self) -> dict:
        return self._request("GET", "/health")

    def stats(self) -> dict:
        return self._request("GET", "/monitoring/stats")

    def providers(self) -> dict:
        return self._request("GET", "/llm/providers")

    def clear_cache(self) -> dict:
        return self._request("POST", "/monitoring/cache/clear")
