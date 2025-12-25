from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from app.core.settings import PolicyConfig, TenantPolicy

@dataclass(frozen = True)
class ExecutionPlan : 
    tier : str
    decoding_profile : str
    max_tokens : int
    temperature: float
    cache : dict[str,Any]
    plan_name : str


@dataclass(frozen=True)
class DecisionTrace : 
    reasons : list[str]
    bucket : str
    tenant_id : str
    policy_version: str


def _pick_length_bucket(policy:PolicyConfig, prompt_chars : int) -> str:
    buckets = policy.routing.get('length_buckets',{})
    ordered = ['short','medium','long']
    ## let get a name(ordered) which is just greater than the prompt_char and for fallback we return 'long'
    for name in ordered :
        cfg = buckets.get(name)
        if not cfg:
            continue
        max_chars = int(cfg.get('max_chars',0))
        if prompt_chars <= max_chars:
            return name
    return 'long'


def build_plan(*,policy: PolicyConfig, tenant_id : str, prompt_chars : int, override_temperature:float|None, override_max_tokens: int|None)-> tuple[ExecutionPlan, DecisionTrace]:
    ## becuase user can request for explicit temperature and max_token that means they can be overriden by defualt which we are getting from yaml
    tenant : TenantPolicy = policy.tenants.get(tenant_id, policy.tenants['default'])
    bucket = _pick_length_bucket(policy, prompt_chars)
    plan_cfg = policy.plans.get(bucket) or policy.plans.get('short') ## defualt set ot short
    if not plan_cfg :
        plan_cfg = {"tier":"standard", "decoding_profile":"standard","max_tokens":256,"temperature":0.7}

    temperature = float(override_temperature) if override_temperature is not None else float(plan_cfg.get("temperature",0.7))

    max_tokens = int(override_max_tokens) if override_max_tokens is not None else int(plan_cfg.get('max_tokens',256))

    plan = ExecutionPlan(tier = str(plan_cfg.get('tier','standard')), 
        decoding_profile=str(plan_cfg.get("decoding_profile", "standard")),
        max_tokens=max_tokens,
        temperature=temperature,
        cache=tenant.caching.model_dump(),
        plan_name=bucket,)
    trace = DecisionTrace(
        reasons=[
            f"bucket={bucket} (prompt_chars={prompt_chars})",
            f"tenant={tenant_id}",
            "plan selected from policy.plans[bucket]",
        ],
        bucket=bucket,
        tenant_id=tenant_id,
        policy_version=policy.policy_version,
    )

    return plan, trace