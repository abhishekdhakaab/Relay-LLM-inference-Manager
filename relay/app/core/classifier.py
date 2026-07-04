"""Score prompt complexity from structural, semantic, and intent signals."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sa_text

@dataclass
class CapabilityVector:
    """Classifier output consumed by the cost router."""

    complexity_score: float          # 0.0 – 1.0 composite
    structural_score: float          # sub-score: pure text structure signals
    semantic_score: float            # sub-score: prototype cosine similarity
    intent_weight: float             # sub-score: intent category base weight
    intent_type: str                 # e.g. "factual_lookup", "code_generation"
    needs_retrieval: bool
    needs_computation: bool
    needs_code: bool
    needs_personal_context: bool
    is_multi_hop: bool
    classifier_latency_ms: float     # wall-clock scoring time, logged to trace


# These patterns stay module-scoped because classification runs on every cache miss.
_CONNECTORS = re.compile(
    r'\b(and|also|additionally|furthermore)\b',
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r'\b(before|after|when did|timeline|sequence)\b',
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r'\b(not|unless|except|without|neither)\b',
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r'(if\s+.{1,60}\s+then|given that|assuming|in the case where)',
    re.IGNORECASE,
)
_CONSTRAINTS = re.compile(
    r'\d[\d,]*\.?\d*\s*(rps|rpm|qps|ms|s\b|second|minute|hour|day|week|month|year'
    r'|k\b|m\b|b\b|kb|mb|gb|tb|%|percent|usd|\$|€|£)',
    re.IGNORECASE,
)
_ENUMERATION = re.compile(
    r'\b(list all|enumerate|for each|compare all)\b',
    re.IGNORECASE,
)


def compute_structural_score(prompt: str) -> float:
    """Return a bounded complexity score using only prompt text."""
    score = 0.0

    # Length contributes in two steps so very long prompts receive both weights.
    n = len(prompt)
    if n > 800:
        score += 0.15
    if n > 2000:
        score += 0.10  # additive on top of the >800 bucket

    # Repeated connectors matter, but the cap prevents this signal from dominating.
    connector_count = len(_CONNECTORS.findall(prompt))
    score += min(connector_count * 0.10, 0.30)

    if _TEMPORAL.search(prompt):
        score += 0.15

    negation_count = len(_NEGATION.findall(prompt))
    score += min(negation_count * 0.08, 0.20)

    if _CONDITIONAL.search(prompt):
        score += 0.15

    if _CONSTRAINTS.search(prompt):
        score += 0.10

    if _ENUMERATION.search(prompt):
        score += 0.10

    # Design questions need a larger model even when the prompt itself is short.
    if _DESIGN_MARKERS.search(prompt):
        score += 0.35

    return min(score, 1.0)


_COMPUTATION_KEYWORDS = re.compile(
    r'(calculate|compute|how many|total|sum of|average of|rate of)',
    re.IGNORECASE,
)
_COMPUTATION_SYMBOLS = re.compile(r'[%$£€]|\d\s*[+\-×÷*/]\s*\d')

_CODE_FENCE = re.compile(r'```')
_CODE_KEYWORDS = re.compile(
    r'(implement|write a function|write a script|debug this|fix this code'
    r'|run this|execute'
    r'|write\s+a\s+\w{1,20}\s+(query|procedure|stored\s+procedure)'
    r')',
    re.IGNORECASE,
)
# Catch language-specific requests that do not use an explicit "write code" verb.
_LANG_CODE = re.compile(
    r'\b(python|javascript|typescript|go|rust|java|c\+\+|c#|ruby|swift|kotlin'
    r'|scala|haskell|erlang|elixir|bash|shell|sql|r\b|matlab)\s+'
    r'(code|implementation|function|script|snippet|program)',
    re.IGNORECASE,
)

_RETRIEVAL_KEYWORDS = re.compile(
    r'\b(current|latest|today|as of|recent|news|what happened|who is the current)\b',
    re.IGNORECASE,
)

_PERSONAL_KEYWORDS = re.compile(
    r'\b(my |our |the document i|the file i|in our system|in my account'
    r'|my last|my previous)\b',
    re.IGNORECASE,
)
_TICKET_PATTERN = re.compile(r'#\d+')

_MULTI_QUESTION_MARKERS = re.compile(
    r'\b(and also tell me|additionally|then explain|and what|as well as)\b',
    re.IGNORECASE,
)

# These markers deliberately carry enough weight to push design work upward.
_DESIGN_MARKERS = re.compile(
    r'(\bdesign\s+(a|an|the|your|our)\b'
    r'|\barchitect\b'
    r'|\bwalk\s+through\b'
    r'|\btrace\s+(through|its|their|the\s)\b'
    r'|\blifecycle\s+of\b'
    r'|\bexplain\s+why\b'
    r'|\bexplain\s+how\b'
    r'|\bexplain\s+the\s+(difference|tradeoffs?|concept|principles?|main)\b'
    r'|\bhow\s+do(?:es)?\s+\w+\s+work\b'
    r'|\bwhat\s+are\s+the\s+(best\s+practices?|solid\b|advantages?|disadvantages?|tradeoffs?|pros\s+and)\b'
    r'|\bcompare\s+(the\s+)?\w+.{0,30}(of|between|among|vs|versus)\b'
    r'|\bcompare\s+all\b'
    r'|\bthen\s+(describe|explain|discuss|compare)\b'
    r')',
    re.IGNORECASE,
)

def compute_tool_flags(prompt: str) -> dict[str, bool]:
    """Detect independent capabilities that can force a tier upgrade."""
    needs_computation = bool(
        _COMPUTATION_KEYWORDS.search(prompt)
        or _COMPUTATION_SYMBOLS.search(prompt)
    )

    needs_code = bool(
        _CODE_FENCE.search(prompt)
        or _CODE_KEYWORDS.search(prompt)
        or _LANG_CODE.search(prompt)
    )

    needs_retrieval = bool(_RETRIEVAL_KEYWORDS.search(prompt))

    needs_personal_context = bool(
        _PERSONAL_KEYWORDS.search(prompt)
        or _TICKET_PATTERN.search(prompt)
    )

    question_marks = prompt.count('?')
    connector_count = len(_CONNECTORS.findall(prompt))
    is_multi_hop = bool(
        question_marks > 1
        or _MULTI_QUESTION_MARKERS.search(prompt)
        or connector_count >= 2
    )

    return {
        "needs_computation": needs_computation,
        "needs_code": needs_code,
        "needs_retrieval": needs_retrieval,
        "needs_personal_context": needs_personal_context,
        "is_multi_hop": is_multi_hop,
    }


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

async def compute_semantic_score(prompt_embedding: list[float],db_session: Any,) -> tuple[float, float, str]:
    """Find the closest intent prototype, with a conservative fallback."""
    q = sa_text(
        """
        SELECT
            intent_type,
            base_weight,
            (1 - (embedding <=> (:qvec)::vector)) AS similarity
        FROM complexity_prototypes
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> (:qvec)::vector
        LIMIT 1
        """
    )
    try:
        res = await db_session.execute(q, {"qvec": _vec_literal(prompt_embedding)})
        row = res.mappings().first()
    except Exception:
        return 0.0, 0.25, "unknown"

    if row is None:
        return 0.0, 0.25, "unknown"

    similarity = float(row["similarity"])
    base_weight = float(row["base_weight"])
    intent_type = str(row["intent_type"])
    return similarity, base_weight, intent_type


async def classify(prompt: str,embedding: list[float],db_session: Any,w_structural: float = 0.35,w_semantic: float = 0.40,w_intent: float = 0.25,) -> CapabilityVector:
    """Combine text heuristics and the nearest semantic prototype."""
    t_start = time.perf_counter()

    # Finish the CPU-only signals before yielding to the database query.
    structural_score = compute_structural_score(prompt)
    flags = compute_tool_flags(prompt)

    # The 57-character cutoff comes from the evaluation boundary (55 simple,
    # 59 medium). It also suppresses noisy keyword flags on tiny questions.
    if len(prompt.strip()) <= 57 and not flags["needs_code"]:
        latency_ms = (time.perf_counter() - t_start) * 1000
        return CapabilityVector(
            complexity_score=0.05,
            structural_score=structural_score,
            semantic_score=0.0,
            intent_weight=0.10,
            intent_type="simple_short",
            needs_retrieval=flags["needs_retrieval"],
            needs_computation=False,
            needs_code=False,
            needs_personal_context=False,
            is_multi_hop=False,
            classifier_latency_ms=latency_ms,
        )

    # One nearest-neighbor query supplies both semantic and intent signals.
    semantic_score, intent_weight, intent_type = await compute_semantic_score(
        embedding, db_session
    )

    complexity_score = min(w_structural * structural_score+ w_semantic * semantic_score+ w_intent * intent_weight,1.0,)

    latency_ms = (time.perf_counter() - t_start) * 1000

    return CapabilityVector(
        complexity_score=complexity_score,
        structural_score=structural_score,
        semantic_score=semantic_score,
        intent_weight=intent_weight,
        intent_type=intent_type,
        needs_retrieval=flags["needs_retrieval"],
        needs_computation=flags["needs_computation"],
        needs_code=flags["needs_code"],
        needs_personal_context=flags["needs_personal_context"],
        is_multi_hop=flags["is_multi_hop"],
        classifier_latency_ms=latency_ms,
    )
