"""Translate classifier output into model, token, and cost decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.classifier import CapabilityVector
from app.core.settings import CostRouterConfig

# Rank comparisons guarantee that capability flags can only upgrade a request.
_TIER_RANK: dict[str, int] = {"simple": 0, "medium": 1, "complex": 2}

@dataclass
class RoutingDecision:
    """Routing values needed by generation, accounting, and tracing."""

    tier_name: str                      # "simple" | "medium" | "complex"
    model: str
    max_tokens: int
    cost_per_1k_tokens: float
    forced_by_tool_flag: Optional[str]  # name of the flag that upgraded tier, or None
    estimated_cost_usd: float           # pre-generation estimate using max_tokens
    would_cost_usd: float               # what it would cost routed to complex tier


def route(vector: CapabilityVector, policy: CostRouterConfig) -> RoutingDecision:
    """Choose a base tier, apply capability floors, and estimate its cost."""


    tiers = policy.tiers
    if not tiers:
        # A conservative fallback keeps malformed or legacy policies usable.
        return RoutingDecision(
            tier_name="complex",
            model="llama3.1:8b",
            max_tokens=1024,
            cost_per_1k_tokens=0.002,
            forced_by_tool_flag=None,
            estimated_cost_usd=(1024 / 1000) * 0.002,
            would_cost_usd=(1024 / 1000) * 0.002,
        )

    tier_name = _pick_tier_by_score(vector.complexity_score, tiers)

    overrides = policy.tool_flag_overrides
    forced_by: Optional[str] = None
    flag_override_map = {
        "needs_computation": overrides.needs_computation,
        "needs_code": overrides.needs_code,
        "needs_personal_context": overrides.needs_personal_context,
        "is_multi_hop": overrides.is_multi_hop,
    }
    flag_values = {
        "needs_computation": vector.needs_computation,
        "needs_code": vector.needs_code,
        "needs_personal_context": vector.needs_personal_context,
        "is_multi_hop": vector.is_multi_hop,
    }

    for flag_name, min_tier in flag_override_map.items():
        if flag_values[flag_name] and _TIER_RANK.get(min_tier, 0) > _TIER_RANK.get(tier_name, 0):
            tier_name = min_tier
            forced_by = flag_name

    tier_cfg = tiers[tier_name]
    complex_cfg = tiers.get("complex", tier_cfg)

    # Compare every route with the complex tier to make savings visible in traces.
    estimated_cost = (tier_cfg.max_tokens / 1000) * tier_cfg.cost_per_1k_tokens
    would_cost = (complex_cfg.max_tokens / 1000) * complex_cfg.cost_per_1k_tokens

    return RoutingDecision(
        tier_name=tier_name,
        model=tier_cfg.model,
        max_tokens=tier_cfg.max_tokens,
        cost_per_1k_tokens=tier_cfg.cost_per_1k_tokens,
        forced_by_tool_flag=forced_by,
        estimated_cost_usd=estimated_cost,
        would_cost_usd=would_cost,
    )


def _pick_tier_by_score(score: float, tiers: dict) -> str:
    """Return the first configured tier whose upper bound contains the score."""
    ordered = sorted(tiers.items(), key=lambda kv: kv[1].complexity_max)
    for name, cfg in ordered:
        if score <= cfg.complexity_max:
            return name
    return ordered[-1][0] if ordered else "complex"

def check_escalation(
    response_text: str,
    decision: RoutingDecision,
    policy: CostRouterConfig,
) -> bool:
    """Flag short or hedged responses for retry on the next tier."""
    esc = policy.escalation
    if not esc.enabled:
        return False
    if decision.tier_name == "complex":
        return False  # The complex tier has no fallback above it.

    # Tokens average roughly 0.75 English words in the benchmark workload.
    word_count = len(response_text.split())
    min_words = esc.min_response_length_ratio * decision.max_tokens * 0.75
    if word_count < min_words:
        return True

    lower_text = response_text.lower()
    for phrase in esc.hedge_phrases:
        if phrase.lower() in lower_text:
            return True

    return False


def next_tier(tier_name: str, tiers: dict) -> Optional[str]:
    """Return the next tier up from tier_name, or None if already at top."""
    ranked = sorted(tiers.keys(), key=lambda t: _TIER_RANK.get(t, 0))
    try:
        idx = ranked.index(tier_name)
        return ranked[idx + 1] if idx + 1 < len(ranked) else None
    except ValueError:
        return None
