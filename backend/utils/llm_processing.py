"""Очистка ответов LLM и извлечение JSON."""
from __future__ import annotations

import json
import re
from typing import Any

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json|text|markdown)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PREFIX_RE = re.compile(
    r"^\s*(?:вот\s+)?(?:исправленн\w+\s+|улучшенн\w+\s+|готов\w+\s+)?(?:описание|текст|ответ)(?:\s+товара)?\s*[:：]\s*",
    re.IGNORECASE,
)


def clean_response(text: str) -> str:
    """Удаляет служебные теги, обёртки кода, вступления вида «Вот описание:» и лишние кавычки."""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    fence = FENCE_RE.search(text)
    if fence and len(fence.group(1).strip()) > 0.5 * len(text.strip().strip("`")):
        text = fence.group(1)
    text = text.strip()
    text = PREFIX_RE.sub("", text, count=1)
    if len(text) >= 2 and text[0] in "\"«'" and text[-1] in "\"»'":
        text = text[1:-1]
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # markdown-жирный не нужен в карточке
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """Находит первый JSON-объект в ответе модели. Возвращает None, если разобрать не удалось."""
    if not text:
        return None
    text = THINK_RE.sub("", text)
    candidates: list[str] = []
    fence = FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        for variant in (cand, re.sub(r",\s*([}\]])", r"\1", cand), cand.replace("'", '"')):
            try:
                data = json.loads(variant)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(data, dict):
                return data
    return None


def text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
