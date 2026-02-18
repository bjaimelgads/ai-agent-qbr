"""Stub node module."""

class Node:
    def __init__(self, func, name: str | None = None):
        self.func = func
        self.name = name or getattr(func, "__name__", "node")
