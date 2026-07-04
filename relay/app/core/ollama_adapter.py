"""Ollama backend adapter with circuit-breaker protection."""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from app.core.backend import GenerationResult
from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.core.settings import settings

# Module-level singleton — shared across all requests in the same process.
_circuit_breaker: CircuitBreaker | None = None


def get_circuit_breaker() -> CircuitBreaker:
    global _circuit_breaker
    if _circuit_breaker is None:
        policy = settings.load_policy()
        _circuit_breaker = CircuitBreaker(policy.circuit_breaker)
    return _circuit_breaker


@dataclass(frozen=True)
class OllamaAdapter:
    """Generate text through Ollama's non-streaming HTTP endpoint."""

    base_url: str
    name: str = "ollama"
    async def generate(self,*,model: str,prompt: str,temperature: float,max_tokens: int,) -> GenerationResult:
        """Run one protected backend call and update breaker state."""
        cb = get_circuit_breaker()

        await cb.before_call()

        t0 = time.perf_counter()
        try:
            result = await self._do_generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            await cb.on_failure()
            raise
        except Exception:
            # Parse errors also indicate an unusable backend response.
            await cb.on_failure()
            raise

        await cb.on_success()
        return result
    async def _do_generate(self,*,model: str,prompt: str,temperature: float,max_tokens: int,) -> GenerationResult:
        """Call Ollama and normalize its provider-specific response."""
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
        # Long generations need a wider timeout than ordinary API traffic.
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
