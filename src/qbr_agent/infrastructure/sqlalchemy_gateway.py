"""SQLAlchemy async gateway for QBR storage."""

from __future__ import annotations

from contextlib import asynccontextmanager

import os
import time
import uuid
from dataclasses import dataclass
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


class DatabaseGateway:
    def __init__(self, *, database_url: str, echo: bool = False) -> None:
        self._database_url = database_url
        self._echo = echo
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._connect_args = _database_connect_args(database_url)

    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                connect_args=self._connect_args,
            )
            _attach_lakebase_password_injector(self._engine, self._database_url)
        return self._engine

    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        if self._sessionmaker is None:
            self._sessionmaker = async_sessionmaker(self.engine(), expire_on_commit=False)
        return self._sessionmaker

    @asynccontextmanager
    async def session_scope(self) -> AsyncSession:
        async_session = self.sessionmaker()
        async with async_session() as session:
            yield session


def _database_connect_args(database_url: str) -> dict:
    if not database_url.startswith("postgresql+asyncpg://"):
        return {}
    if os.getenv("QBR_LAKEBASE_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
        return {}
    return {"ssl": True}


@dataclass
class _LakebaseCredentialInjector:
    workspace_client: WorkspaceClient
    db_instance_name: str
    refresh_seconds: int = 600
    _cached_password: str | None = None
    _cached_host: str | None = None
    _last_refresh_monotonic: float = 0.0

    def get_connect_params(self) -> tuple[str | None, str]:
        now = time.monotonic()
        if (
            self._cached_password
            and self._cached_host
            and (now - self._last_refresh_monotonic) < self.refresh_seconds
        ):
            return self._cached_host, self._cached_password

        instance = self.workspace_client.database.get_database_instance(name=self.db_instance_name)
        host = getattr(instance, "read_write_dns", None)
        if not host:
            raise RuntimeError(
                f"Lakebase instance {self.db_instance_name!r} did not return read_write_dns."
            )

        credential = self.workspace_client.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[self.db_instance_name],
        )
        password = getattr(credential, "token", None)
        if not password:
            raise RuntimeError("Lakebase credential generation returned empty token.")

        self._cached_host = host
        self._cached_password = password
        self._last_refresh_monotonic = now
        return host, password


def _build_lakebase_workspace_client() -> WorkspaceClient:
    app_auth_mode = os.getenv("QBR_DATABRICKS_APP_AUTH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if app_auth_mode:
        # In Databricks Apps, rely on injected env and enforce OAuth M2M.
        return WorkspaceClient(auth_type="oauth-m2m")

    host = (
        os.getenv("QBR_LAKEBASE_DATABRICKS_HOST", "").strip()
        or os.getenv("DATABRICKS_HOST", "").strip()
    )
    if not host:
        raise RuntimeError("QBR_LAKEBASE_DATABRICKS_HOST or DATABRICKS_HOST is required.")
    if not host.startswith(("https://", "http://")):
        host = f"https://{host}"

    client_id = os.getenv("DATABRICKS_CLIENT_ID", "").strip()
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return WorkspaceClient(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
            auth_type="oauth-m2m",
        )

    token = (
        os.getenv("QBR_LAKEBASE_TOKEN", "").strip()
        or os.getenv("DATABRICKS_TOKEN", "").strip()
        or os.getenv("DATABRICKS_API_KEY", "").strip()
    )
    if token:
        return WorkspaceClient(host=host, token=token, auth_type="pat")
    raise RuntimeError(
        "Missing Databricks auth for Lakebase: provide OAuth "
        "(DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET) or token "
        "(QBR_LAKEBASE_TOKEN/DATABRICKS_TOKEN/DATABRICKS_API_KEY)."
    )


def _attach_lakebase_password_injector(engine: AsyncEngine, database_url: str) -> None:
    if not database_url.startswith("postgresql+asyncpg://"):
        return
    if os.getenv("QBR_LAKEBASE_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
        return

    db_instance_name = os.getenv("QBR_LAKEBASE_DB_INSTANCE", "").strip()
    if not db_instance_name:
        raise RuntimeError("QBR_LAKEBASE_DB_INSTANCE is required when QBR_LAKEBASE_ENABLED=true.")

    parsed = urlparse(database_url)
    expected_host = parsed.hostname or ""
    refresh_minutes_raw = os.getenv("QBR_LAKEBASE_REFRESH_MINUTES", "10").strip()
    try:
        refresh_minutes = max(1, int(refresh_minutes_raw))
    except ValueError:
        refresh_minutes = 10

    injector = _LakebaseCredentialInjector(
        workspace_client=_build_lakebase_workspace_client(),
        db_instance_name=db_instance_name,
        refresh_seconds=refresh_minutes * 60,
    )

    @event.listens_for(engine.sync_engine, "do_connect")
    def _inject_password(
        _dialect: object,
        _conn_rec: object,
        _cargs: object,
        cparams: dict,
    ) -> None:
        host, password = injector.get_connect_params()
        cparams["password"] = password
        if host:
            cparams["host"] = host
        elif expected_host:
            cparams["host"] = expected_host
