"""Pydantic models for WebSocket protocol DTOs."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMetadata(BaseModel):
    document_id: str | None = None
    client_name: str | None = None
    period: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None


class ChatRequest(BaseModel):
    message: str
    metadata: ChatMetadata | None = None


class OutputReady(BaseModel):
    status: Literal["ready"] = "ready"
    session_id: str


class OutputThinking(BaseModel):
    status: Literal["thinking"] = "thinking"
    message: str | None = None
    step: int | None = None


class OutputUserMessage(BaseModel):
    status: Literal["user_message"] = "user_message"
    data: dict[str, Any] = Field(default_factory=dict)


class OutputFinal(BaseModel):
    status: Literal["final"] = "final"
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)


class OutputPartial(BaseModel):
    status: Literal["partial"] = "partial"
    data: dict[str, Any] = Field(default_factory=dict)


class OutputError(BaseModel):
    status: Literal["error"] = "error"
    message: str


class OutputPing(BaseModel):
    type: Literal["ping"] = "ping"


class OutputPong(BaseModel):
    type: Literal["pong"] = "pong"


ServerMessage = (
    OutputReady
    | OutputThinking
    | OutputUserMessage
    | OutputPartial
    | OutputFinal
    | OutputError
    | OutputPing
    | OutputPong
)


class PlannerEventPayload(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
