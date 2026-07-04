"""Canonicalize chat messages before exact-cache hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.models.openai_chat import ChatMessage


@dataclass(frozen=True)
class NormalizedRequest:
    """Trimmed messages and their deterministic cache identity."""

    messages: tuple[ChatMessage, ...]
    canonical_text: str
    request_hash: str

def normalize_messages(messages: list[ChatMessage]) -> NormalizedRequest:
    """Trim messages and hash their role-prefixed canonical text."""
    parts: list[str] = []
    canon_msgs: list[ChatMessage] = []

    for m in messages:
        role = (m.role or "").strip()
        content = (m.content or "").strip()
        canon_msgs.append(ChatMessage(role=role, content=content))
        parts.append(f"{role}:{content}")

    canonical_text = "\n".join(parts)
    request_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()

    return NormalizedRequest(messages=tuple(canon_msgs), canonical_text=canonical_text, request_hash=request_hash)
