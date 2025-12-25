from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Dict, Any

@dataclass(frozen=True)
class GenerationResult:
    text : str
    prompt_tokens : Optional[int] = None
    completion_tokens : Optional[int] = None
    total_tokens : Optional[int] = None

    backend_latency_ms : Optional[int] = None
    backend_ttft_ms : Optional[int] = None
    backend_name : Optional[str] = None
    backend_meta : Optional[Dict[str,Any]] = None


class BackendAdapter(Protocol):
    name : str
    async def generate(self, *, model:str, prompt : str, temperature:float, max_tokens : int)-> GenerationResult:
        raise NotImplementedError
