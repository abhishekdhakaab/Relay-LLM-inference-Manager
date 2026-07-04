"""Typed policy schema and environment-backed application settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SemanticCaching(BaseModel):
    """Per-tenant semantic-cache policy."""

    enabled : bool = False
    threshold :float = 0.90
    ttl_seconds : int = 18000
    verifier : str ='off'
class TenantCaching(BaseModel):
    exact_enabled : bool = True
    semantic : SemanticCaching  = Field(default_factory = SemanticCaching)

class SchedulerAdmissionComputeMs(BaseModel):
    short : int = 1200
    long : int = 3500
class SchedulerDegrade(BaseModel):
    enabled : bool = True
    max_tokens_floor : int = 128
    max_tokens_scale : float = 0.5
class SchedulerReject(BaseModel):
    enabled : bool = True
    retry_after_seconds : int = 2
class SchedulerAdmission(BaseModel):
    enabled : bool = True
    default_compute_ms : SchedulerAdmissionComputeMs = Field(default_factory = SchedulerAdmissionComputeMs)
    degrade : SchedulerDegrade = Field(default_factory = SchedulerDegrade)
    reject : SchedulerReject = Field(default_factory = SchedulerReject)
class SchedulerConfig(BaseModel):
    """Queue layout and admission policy."""

    short_max_prompt_chars : int = 1200
    workers : int =2
    max_queue_depth_per_lane : int = 200
    admission: SchedulerAdmission = Field(default_factory = SchedulerAdmission)


class ClassifierWeights(BaseModel):
    structural: float = 0.35
    semantic: float = 0.40
    intent: float = 0.25
class TierConfig(BaseModel):
    """Upper complexity bound and generation settings for one tier."""

    complexity_max: float
    model: str
    max_tokens: int
    cost_per_1k_tokens: float
class ToolFlagOverrides(BaseModel):
    needs_computation: str = "medium"
    needs_code: str = "complex"
    needs_personal_context: str = "medium"
    is_multi_hop: str = "medium"
class EscalationConfig(BaseModel):
    """Signals that trigger a retry on a more capable tier."""

    enabled: bool = True
    min_response_length_ratio: float = 0.3
    hedge_phrases: list[str] = Field(default_factory=lambda: [
        "I'm not sure", "I don't know", "I cannot answer",
        "I don't have enough information", "As an AI",
    ])
    escalation_cost_multiplier: float = 1.5
class CostRouterConfig(BaseModel):
    classifier_weights: ClassifierWeights = Field(default_factory=ClassifierWeights)
    tiers: dict[str, TierConfig] = Field(default_factory=dict)
    tool_flag_overrides: ToolFlagOverrides = Field(default_factory=ToolFlagOverrides)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)

class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout_seconds: int = 30
    timeout_counts_as_failure: bool = True

class TokenBudgetConfig(BaseModel):
    """Fixed-window token quota for a tenant."""

    limit: int = 500000
    window_seconds: int = 3600
    hard_reject: bool = True

class SLOConfig(BaseModel):
    p50_ms: int = 12000
    p95_ms: int = 45000
    p99_ms: int = 60000
    error_rate_max_pct: float = 1.0

class TenantPolicy(BaseModel):
    latency_slo_ms: int = 8000
    caching: TenantCaching = Field(default_factory=TenantCaching)
    token_budget: Optional[TokenBudgetConfig] = None
    slo: SLOConfig = Field(default_factory=SLOConfig)
class PolicyConfig(BaseModel):
    """Root object loaded from a policy YAML file."""

    policy_version: str
    tenants: dict[str, TenantPolicy]
    routing: dict[str, Any]
    plans: dict[str, Any]
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    cost_router: CostRouterConfig = Field(default_factory=CostRouterConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
REPO_ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    relay_host: str = Field(default="0.0.0.0", alias="RELAY_HOST")
    relay_port: int = Field(default=8000, alias="RELAY_PORT")
    relay_log_level: str = Field(default="info", alias="RELAY_LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+asyncpg://relay:relay@localhost:5433/relay",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Relative policy paths are always interpreted from the repository root.
    policy_path: str = Field(default="policies/policy.dev.yaml", alias="POLICY_PATH")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"


    exact_cache_ttl_seconds : int = 3600

    backend_mode : str ="mock"  # CI uses mock mode because no Ollama daemon is available.


    semantic_cache_ttl_seconds : int =1800
    semantic_cache_max_entries: int =200
    semantic_cache_threshold : float = 0.90
    embedding_model : str = 'BAAI/bge-small-en-v1.5'
    def load_policy(self) -> PolicyConfig:
        """Load and validate the configured policy file."""
        p = Path(self.policy_path)

        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()

        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {p}")

        raw = yaml.safe_load(p.read_text())
        return PolicyConfig.model_validate(raw)


settings = Settings()
