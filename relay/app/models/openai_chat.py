"""Pydantic models for the relay's OpenAI-compatible chat surface."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(default="")

class ChatCompletionsRequest(BaseModel):
    """Supported subset of an OpenAI chat-completions request."""

    model: str = "local-mlx"
    messages: list[ChatMessage]

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False  # Streaming is rejected explicitly by the route.

class ChatCompletionsChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionsResponse(BaseModel):
    """Response returned to clients and recorded in request traces."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionsChoice]
    usage: Usage
