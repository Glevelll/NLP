"""Общие структуры для поиска."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    rank: int


def match_filters(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            if metadata.get(key) not in value:
                return False
        elif metadata.get(key) != value:
            return False
    return True
