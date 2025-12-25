from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(default="")


class ChatCompletionsRequest(BaseModel):
    model: str = "local-mlx"
    messages: list[ChatMessage]

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False  # we will reject streaming for now


class ChatCompletionsChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionsChoice]
    usage: Usage
