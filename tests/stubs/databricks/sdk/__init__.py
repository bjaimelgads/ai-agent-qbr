"""Stub databricks sdk."""

class _Config:
    auth_type = ""
    token = ""

    def authenticate(self):
        return {}


class WorkspaceClient:
    def __init__(self, *args, **kwargs):
        self.config = _Config()
