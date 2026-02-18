"""Stub rich_output module."""

DEFAULT_ALLOWLIST = []

class RichOutputConfig:
    def __init__(self, *args, **kwargs):
        self.enabled = False


def attach_rich_output_nodes(*args, **kwargs):
    return []


def get_runtime(*args, **kwargs):
    class _Runtime:
        def prompt_section(self):
            return ""

    return _Runtime()
