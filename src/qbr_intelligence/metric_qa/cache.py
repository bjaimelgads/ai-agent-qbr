"""In-memory caches for metric QA."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class LruCache:
    max_size: int = 128

    def __post_init__(self) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.pop(key)
        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)
