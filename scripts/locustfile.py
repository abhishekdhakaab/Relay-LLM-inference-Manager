from __future__ import annotations

import json
import random
import time

from locust import HttpUser, task, between

TENANT = "default"


def make_payload(prompt: str) -> dict:
    return {"model": "local-ollama", "messages": [{"role": "user", "content": prompt}]}


SHORT_PROMPTS = [
    "Explain caching in one sentence.",
    "What is an API gateway? One sentence.",
    "Write a short definition of rate limiting.",
    "What is tail latency?",
]

LONG_BASE = "Explain distributed systems like I'm an engineer. Include tradeoffs, examples, and pitfalls. "
LONG_PROMPTS = [
    LONG_BASE * 40,
    ("Summarize what a policy engine does in an LLM relay. " * 50),
    ("Explain semantic caching and failure cases. " * 50),
]

SEMANTIC_PAIRS = [
    ("Explain what an API gateway does in 2 lines.", "In 2 lines, what does an API gateway do?"),
    ("What is a cache key? Give 2 examples.", "Give 2 examples of cache keys and explain what they are."),
    ("Define admission control in 2 lines.", "In 2 lines, define admission control."),
]


class RelayUser(HttpUser):
    wait_time = between(0.05, 0.3)

    def _post(self, prompt: str) -> None:
        payload = make_payload(prompt)
        self.client.post(
            "/v1/chat/completions",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json", "X-Tenant-Id": TENANT},
            name="/v1/chat/completions",
            timeout=120,
        )

    @task(6)
    def short(self) -> None:
        self._post(random.choice(SHORT_PROMPTS))

    @task(2)
    def long(self) -> None:
        self._post(random.choice(LONG_PROMPTS))

    @task(2)
    def semantic(self) -> None:
        a, b = random.choice(SEMANTIC_PAIRS)
        prompt = a if int(time.time() * 10) % 2 == 0 else b
        self._post(prompt)
