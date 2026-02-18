"""Stub catalog utilities."""

from __future__ import annotations

def build_catalog(*_args, **_kwargs):
    return []


def tool(*_args, **_kwargs):
    def decorator(func):
        return func
    return decorator
