"""SQLAlchemy async gateway for QBR storage."""

from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


class DatabaseGateway:
    def __init__(self, *, database_url: str, echo: bool = False) -> None:
        self._database_url = database_url
        self._echo = echo
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    def engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(self._database_url, echo=self._echo)
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
