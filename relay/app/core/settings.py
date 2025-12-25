from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# -------------------------
# Policy schema
# -------------------------
class TenantCaching(BaseModel):
    exact_enabled: bool = True
    semantic_enabled: bool = False


class TenantPolicy(BaseModel):
    latency_slo_ms: int = 8000
    caching: TenantCaching = Field(default_factory=TenantCaching)




class SemanticCaching(BaseModel):
    enabled : bool = False
    threshold :float = 0.90
    ttl_seconds : int = 1800
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
    short_max_prompt_chars : int = 1200
    workers : int =2
    max_queue_depth_per_lane : int = 200
    admission: SchedulerAdmission = Field(default_factory = SchedulerAdmission)



class PolicyConfig(BaseModel):
    policy_version: str
    tenants: dict[str, TenantPolicy]
    routing: dict[str, Any]
    plans: dict[str, Any]
    scheduler : SchedulerConfig = Field(default_factory = SchedulerConfig)
# -------------------------
# Settings
# -------------------------
# settings.py lives at: relay/app/core/settings.py
# repo root is:        llm-relay/
REPO_ROOT = Path(__file__).resolve().parents[3]  # -> llm-relay


class Settings(BaseSettings):
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

    # repo-root relative path by default
    policy_path: str = Field(default="policies/policy.dev.yaml", alias="POLICY_PATH")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"


    exact_cache_ttl_seconds : int = 300

    backend_mode : str ="mock" ## added for github action CI, as we dont have ollama over github action


    semantic_cache_ttl_seconds : int =1800
    semantic_cache_max_entries: int =200
    semantic_cache_threshold : float = 0.90
    embedding_model : str = 'BAAI/bge-small-en-v1.5'
    def load_policy(self) -> PolicyConfig:
        p = Path(self.policy_path)

        # If user provided a relative path, interpret it from repo root
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()

        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {p}")

        raw = yaml.safe_load(p.read_text())
        return PolicyConfig.model_validate(raw)


settings = Settings()
