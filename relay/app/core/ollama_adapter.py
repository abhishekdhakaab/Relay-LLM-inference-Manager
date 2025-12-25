from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.core.backend import GenerationResult


@dataclass(frozen=True)
class OllamaAdapter:
    base_url: str
    name: str = "ollama"

    async def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        t0 = time.perf_counter()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{self.base_url}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()

        latency_ms = int((time.perf_counter() - t0) * 1000)

        text = (data.get("response") or "").strip()

        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens,
            backend_latency_ms=latency_ms,
            backend_ttft_ms=None,
            backend_name=self.name,
            backend_meta={"endpoint": "/api/generate"},
        )
