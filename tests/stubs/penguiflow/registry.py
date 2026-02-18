"""Stub registry module."""

class ModelRegistry:
    def __init__(self):
        self._items = {}

    def register(self, name, _input, _output):
        self._items[name] = (_input, _output)
