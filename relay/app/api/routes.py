"""Health and OpenAI-compatible chat-completions routes."""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import orjson
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.settings import settings
from app.db.postgres import insert_trace, get_sessionmaker
from app.models.openai_chat import (
    ChatCompletionsChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    Usage,
)
from app.utils.normalize import normalize_messages
from app.core.ollama_adapter import OllamaAdapter
from app.core.circuit_breaker import CircuitOpenError
from app.core.policy_engine import build_plan

from app.db.redis_client import get_redis
from app.utils.cache_keys import exact_cache_key, plan_signature

from app.core.embeddings import embed_text
from app.db.semantic_cache_pg import semantic_lookup, semantic_store

import asyncio
from app.core.runtime import get_scheduler
from app.core.scheduler import QueueFullError
from app.core.policy_engine import ExecutionPlan
from app.core.backend import GenerationResult

from app.core.classifier import classify as run_classifier
from app.core.cost_router import route as run_router, check_escalation, next_tier, RoutingDecision
from app.core.budget import check_and_reserve, record_actual_usage
from app.core.telemetry import get_tracer, extract_trace_id

from app.auth import resolve_user
from app.core.user_budget import check_user_budget

router = APIRouter()
log = get_logger(component="api")


@router.get("/health")
async def health() -> dict[str, str]:
    """Return a dependency-free liveness response."""
    return {"status": "ok"}


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    ctx: dict = Depends(resolve_user),
) -> ChatCompletionsResponse:
    """Run one authenticated chat request through cache, policy, and backend."""

    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported yet")

    x_tenant_id = ctx["tenant_id"]
    x_user_id   = ctx["user_id"]
    user_row     = ctx.get("user_row")
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    tracer = get_tracer()

    with tracer.start_as_current_span("relay.request") as root_span:
        root_span.set_attribute("tenant_id", x_tenant_id)
        root_span.set_attribute("request_id", request_id)

        policy = settings.load_policy()
        tenant_policy = policy.tenants.get(x_tenant_id, policy.tenants["default"])

        # ── Normalise ────────────────────────────────────────────────────────
        normalized = normalize_messages(req.messages)
        prompt_chars = len(normalized.canonical_text)

        root_span.set_attribute("prompt_length", prompt_chars)

        plan_obj, trace_obj = build_plan(
            policy=policy,
            tenant_id=x_tenant_id,
            prompt_chars=prompt_chars,
            override_temperature=req.temperature,
            override_max_tokens=req.max_tokens,
        )

        plan: dict[str, Any] = {
            "plan_name": plan_obj.plan_name,
            "tier": plan_obj.tier,
            "decoding_profile": plan_obj.decoding_profile,
            "max_tokens": plan_obj.max_tokens,
            "temperature": plan_obj.temperature,
            "cache": plan_obj.cache,
        }

        # Exclude "cache" from the plan signature — cache config (semantic threshold,
        # ttl_seconds, etc.) doesn't affect what the LLM generates. Including it causes
        # key mismatches whenever the policy is reloaded between warm and test passes.
        sig = plan_signature({k: v for k, v in plan.items() if k != "cache"})
        decision_trace: dict[str, Any] = {
            "reasons": trace_obj.reasons,
            "bucket": trace_obj.bucket,
            "tenant_id": trace_obj.tenant_id,
            "policy_version": trace_obj.policy_version,
        }

        # ── Exact cache ───────────────────────────────────────────────────────
        redis = get_redis()
        cache_info: dict[str, Any] = {
            "exact": {"enabled": bool(plan["cache"].get("exact_enabled", True))}
        }

        with tracer.start_as_current_span("relay.cache.exact") as exact_span:
            if plan["cache"].get("exact_enabled", True):
                key = exact_cache_key(
                    tenant_id=x_tenant_id,
                    request_hash=normalized.request_hash,
                    plan_sig=sig,
                )
                cached = await redis.get(key)

                if cached is not None:
                    await redis.incr(f"metrics:cache_exact_hit:{x_tenant_id}")
                    cached_obj = orjson.loads(cached)
                    if "body" in cached_obj:
                        cached_tier = cached_obj.get("tier", "")
                        resp = ChatCompletionsResponse.model_validate(cached_obj["body"])
                    else:
                        cached_tier = ""
                        resp = ChatCompletionsResponse.model_validate(cached_obj)
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    cache_info["exact"].update({"hit": True, "key": key, "plan_sig": sig})
                    exact_span.set_attribute("hit", True)
                    exact_span.set_attribute("latency_ms", latency_ms)
                    otel_trace_id = extract_trace_id()

                    await insert_trace({
                        "request_id": request_id,
                        "tenant_id": x_tenant_id,
                        "user_id": x_user_id,
                        "endpoint": "/v1/chat/completions",
                        "model": req.model,
                        "status_code": 200,
                        "request_hash": normalized.request_hash,
                        "latency_ms": latency_ms,
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                        "request_json": orjson.dumps(req.model_dump()).decode(),
                        "response_json": orjson.dumps(resp.model_dump()).decode(),
                        "error_json": "null",
                        "policy_version": policy.policy_version,
                        "plan_json": orjson.dumps(plan).decode(),
                        "decision_trace_json": orjson.dumps(decision_trace).decode(),
                        "cache_json": orjson.dumps(cache_info).decode(),
                        "otel_trace_id": otel_trace_id,
                    })

                    log.info("cache_hit", request_id=request_id, tenant_id=x_tenant_id,
                             latency_ms=latency_ms, request_hash=normalized.request_hash)
                    return JSONResponse(
                        content=resp.model_dump(),
                        headers={
                            "X-Trace-Id": otel_trace_id or "",
                            "X-Budget-Remaining": "",
                            "X-Tier-Selected": cached_tier,
                        },
                    )

                await redis.incr(f"metrics:cache_exact_miss:{x_tenant_id}")
                cache_info["exact"].update({"hit": False, "key": key, "plan_sig": sig})
                exact_span.set_attribute("hit", False)

        # ── Semantic cache ────────────────────────────────────────────────────
        sem_cfg = plan["cache"].get("semantic", {})
        cache_info["semantic"] = {
            "enabled": bool(sem_cfg.get("enabled", False)),
            "plan_sig": sig,
        }

        with tracer.start_as_current_span("relay.cache.semantic") as sem_span:
            if sem_cfg.get("enabled", False):
                qvec = embed_text(normalized.canonical_text)
                row = await semantic_lookup(
                    tenant_id=x_tenant_id, plan_sig=sig, query_vec=qvec
                )
                if row is not None:
                    similarity = float(row.get("similarity", 0.0))
                    threshold = float(sem_cfg.get("threshold", 0.90))
                    sem_span.set_attribute("similarity_score", similarity)
                    if similarity >= threshold:
                        resp = ChatCompletionsResponse.model_validate(row["response_json"])
                        latency_ms = int((time.perf_counter() - t0) * 1000)
                        cache_info["semantic"].update({
                            "hit": True, "similarity": similarity,
                            "threshold": threshold, "entry_id": row.get("id"),
                        })
                        sem_span.set_attribute("hit", True)
                        sem_span.set_attribute("latency_ms", latency_ms)
                        otel_trace_id = extract_trace_id()

                        await insert_trace({
                            "request_id": request_id,
                            "tenant_id": x_tenant_id,
                            "user_id": x_user_id,
                            "endpoint": "/v1/chat/completions",
                            "model": req.model,
                            "status_code": 200,
                            "request_hash": normalized.request_hash,
                            "latency_ms": latency_ms,
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "total_tokens": resp.usage.total_tokens,
                            "request_json": orjson.dumps(req.model_dump()).decode(),
                            "response_json": orjson.dumps(resp.model_dump()).decode(),
                            "error_json": "null",
                            "policy_version": policy.policy_version,
                            "plan_json": orjson.dumps(plan).decode(),
                            "decision_trace_json": orjson.dumps(decision_trace).decode(),
                            "cache_json": orjson.dumps(cache_info).decode(),
                            "otel_trace_id": otel_trace_id,
                        })

                        log.info("semantic_cache_hit_pgvector", request_id=request_id,
                                 similarity=similarity)
                        return JSONResponse(
                            content=resp.model_dump(),
                            headers={
                                "X-Trace-Id": otel_trace_id or "",
                                "X-Budget-Remaining": "",
                                "X-Tier-Selected": "",
                            },
                        )

                    cache_info["semantic"].update({
                        "hit": False, "best_similarity": similarity,
                        "threshold": threshold, "best_entry_id": row.get("id"),
                    })
                    sem_span.set_attribute("hit", False)
                else:
                    cache_info["semantic"].update({"hit": False, "best_similarity": None})
                    sem_span.set_attribute("hit", False)

        # ── Classifier + Cost Router (cache miss path only) ───────────────────
        cap_vector = None
        routing_decision: Optional[RoutingDecision] = None
        classifier_ran = False

        with tracer.start_as_current_span("relay.classifier") as clf_span:
            if policy.cost_router.tiers:
                try:
                    async with get_sessionmaker()() as db_sess:
                        embedding = embed_text(normalized.canonical_text)
                        cap_vector = await run_classifier(
                            prompt=normalized.canonical_text,
                            embedding=embedding,
                            db_session=db_sess,
                            w_structural=policy.cost_router.classifier_weights.structural,
                            w_semantic=policy.cost_router.classifier_weights.semantic,
                            w_intent=policy.cost_router.classifier_weights.intent,
                        )
                    routing_decision = run_router(cap_vector, policy.cost_router)
                    classifier_ran = True

                    clf_span.set_attribute("complexity_score", cap_vector.complexity_score)
                    clf_span.set_attribute("intent_type", cap_vector.intent_type)
                    clf_span.set_attribute("tier_selected", routing_decision.tier_name)
                    clf_span.set_attribute("needs_code", cap_vector.needs_code)
                    clf_span.set_attribute("needs_computation", cap_vector.needs_computation)
                    clf_span.set_attribute("needs_retrieval", cap_vector.needs_retrieval)
                    clf_span.set_attribute("needs_personal_context", cap_vector.needs_personal_context)
                    clf_span.set_attribute("is_multi_hop", cap_vector.is_multi_hop)
                    clf_span.set_attribute("latency_ms", cap_vector.classifier_latency_ms)

                    plan["max_tokens"] = routing_decision.max_tokens

                except Exception as exc:
                    log.warning("classifier_error", error=str(exc), request_id=request_id)

        # ── Token Budget Check ────────────────────────────────────────────────
        budget_remaining: Optional[int] = None
        if tenant_policy.token_budget is not None:
            estimated_tokens = plan["max_tokens"]
            budget_result = await check_and_reserve(
                tenant_id=x_tenant_id,
                estimated_tokens=estimated_tokens,
                config=tenant_policy.token_budget,
            )
            if not budget_result.allowed:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                await insert_trace({
                    "request_id": request_id,
                    "tenant_id": x_tenant_id,
                    "user_id": x_user_id,
                    "endpoint": "/v1/chat/completions",
                    "model": req.model,
                    "status_code": 429,
                    "request_hash": normalized.request_hash,
                    "latency_ms": latency_ms,
                    "request_json": orjson.dumps(req.model_dump()).decode(),
                    "error_json": orjson.dumps({
                        "type": "budget_exhausted",
                        "reset_in_seconds": budget_result.reset_in_seconds,
                    }).decode(),
                    "policy_version": policy.policy_version,
                    "plan_json": orjson.dumps(plan).decode(),
                    "decision_trace_json": orjson.dumps(decision_trace).decode(),
                    "cache_json": orjson.dumps(cache_info).decode(),
                    "error": True,
                    "error_type": "budget_exhausted",
                })
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "token_budget_exhausted",
                        "tenant_id": x_tenant_id,
                        "reset_in_seconds": budget_result.reset_in_seconds,
                    },
                )
            if budget_result.degraded and budget_result.allowed_tokens is not None:
                plan["max_tokens"] = budget_result.allowed_tokens
            budget_remaining = budget_result.remaining

        # ── Per-user budget (soft cap) ────────────────────────────────────────
        if user_row is not None and user_row.get("per_hour_token_limit"):
            user_limit = int(user_row["per_hour_token_limit"])
            estimated = plan["max_tokens"]
            u_result = await check_user_budget(
                tenant_id=x_tenant_id,
                user_id=x_user_id,
                estimated_tokens=estimated,
                limit=user_limit,
            )
            if not u_result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "user_token_budget_exhausted",
                        "user_id": x_user_id,
                        "reset_in_seconds": u_result.reset_in_seconds,
                    },
                )

        # ── Scheduler ─────────────────────────────────────────────────────────
        scheduler = get_scheduler()
        lane = scheduler.lane_for_prompt_chars(prompt_chars)

        admission, predicted_wait_ms = await scheduler.admission_check(
            lane=lane,
            tenant_slo_ms=tenant_policy.latency_slo_ms,
            prompt_chars=prompt_chars,
        )

        degraded = False
        effective_max_tokens = plan["max_tokens"]

        if admission.degraded:
            degraded = True
            adm = policy.scheduler.admission.degrade
            scaled = int(effective_max_tokens * float(adm.max_tokens_scale))
            effective_max_tokens = max(int(adm.max_tokens_floor), scaled)
            plan["max_tokens"] = effective_max_tokens
            decision_trace["reasons"].append(
                f"degraded max_tokens to {effective_max_tokens} due to admission control"
            )

        if admission.rejected:
            retry_after = admission.retry_after_seconds or 1
            cache_info["scheduler"] = {
                "lane": lane, "admission": admission.reason,
                "predicted_wait_ms": predicted_wait_ms,
                "degraded": degraded, "rejected": True,
            }
            latency_ms = int((time.perf_counter() - t0) * 1000)
            await insert_trace({
                "request_id": request_id,
                "tenant_id": x_tenant_id,
                "user_id": x_user_id,
                "endpoint": "/v1/chat/completions",
                "model": req.model,
                "status_code": 429,
                "request_hash": normalized.request_hash,
                "latency_ms": latency_ms,
                "queue_wait_ms": predicted_wait_ms,
                "request_json": orjson.dumps(req.model_dump()).decode(),
                "error_json": orjson.dumps({
                    "type": "rate_limited",
                    "detail": "Predicted SLO miss; retry later",
                    "retry_after_seconds": retry_after,
                }).decode(),
                "policy_version": policy.policy_version,
                "plan_json": orjson.dumps(plan).decode(),
                "decision_trace_json": orjson.dumps(decision_trace).decode(),
                "cache_json": orjson.dumps(cache_info).decode(),
                "error": True,
                "error_type": "scheduler_rejected",
            })
            raise HTTPException(status_code=429, detail={"retry_after_seconds": retry_after})

        # ── Backend ───────────────────────────────────────────────────────────
        prompt = normalized.canonical_text + "\n assistance:"
        model_name = routing_decision.model if routing_decision else settings.ollama_model
        adapter = OllamaAdapter(base_url=settings.ollama_base_url)

        async def run_backend() -> object:
            if settings.backend_mode == "mock":
                return GenerationResult(
                    text="(mock) " + normalized.canonical_text[:120].replace("\n", " "),
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    backend_latency_ms=50,
                    backend_ttft_ms=None,
                )
            return await adapter.generate(
                model=model_name,
                prompt=prompt,
                temperature=float(plan["temperature"]),
                max_tokens=int(plan["max_tokens"]),
            )

        fut: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        queue_entered = time.perf_counter()

        from app.core.scheduler import ScheduledJob

        job = ScheduledJob(
            request_id=request_id,
            tenant_id=x_tenant_id,
            lane=lane,
            created_at=time.time(),
            slo_ms=tenant_policy.latency_slo_ms,
            plan=plan_obj,
            run=run_backend,
            fut=fut,
            queue_entered_at=queue_entered,
        )

        with tracer.start_as_current_span("relay.scheduler") as sched_span:
            sched_span.set_attribute("lane", lane)
            sched_span.set_attribute("admission_result", admission.reason)
            sched_span.set_attribute("queue_depth", sum(
                q.qsize()
                for tmap in scheduler._queues.values()
                for q in tmap.values()
            ))

            try:
                await scheduler.submit(job)
            except QueueFullError:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                cache_info["scheduler"] = {
                    "lane": lane, "admission": "queue_full",
                    "predicted_wait_ms": predicted_wait_ms,
                    "degraded": degraded, "rejected": True,
                }
                await insert_trace({
                    "request_id": request_id,
                    "tenant_id": x_tenant_id,
                    "user_id": x_user_id,
                    "endpoint": "/v1/chat/completions",
                    "model": req.model,
                    "status_code": 503,
                    "request_hash": normalized.request_hash,
                    "latency_ms": latency_ms,
                    "queue_wait_ms": predicted_wait_ms,
                    "request_json": orjson.dumps(req.model_dump()).decode(),
                    "error_json": orjson.dumps({"type": "queue_full"}).decode(),
                    "policy_version": policy.policy_version,
                    "plan_json": orjson.dumps(plan).decode(),
                    "decision_trace_json": orjson.dumps(decision_trace).decode(),
                    "cache_json": orjson.dumps(cache_info).decode(),
                    "error": True,
                    "error_type": "scheduler_rejected",
                })
                raise HTTPException(status_code=503, detail="Queue full, try later")

        with tracer.start_as_current_span("relay.backend") as backend_span:
            backend_span.set_attribute("model", model_name)
            backend_span.set_attribute("tier", routing_decision.tier_name if routing_decision else "n/a")

            try:
                result_obj = await fut
            except CircuitOpenError as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                await insert_trace({
                    "request_id": request_id,
                    "tenant_id": x_tenant_id,
                    "user_id": x_user_id,
                    "endpoint": "/v1/chat/completions",
                    "model": req.model,
                    "status_code": 503,
                    "request_hash": normalized.request_hash,
                    "latency_ms": latency_ms,
                    "request_json": orjson.dumps(req.model_dump()).decode(),
                    "error_json": orjson.dumps({
                        "type": "circuit_open",
                        "retry_after_seconds": e.retry_after_seconds,
                    }).decode(),
                    "policy_version": policy.policy_version,
                    "plan_json": orjson.dumps(plan).decode(),
                    "decision_trace_json": orjson.dumps(decision_trace).decode(),
                    "cache_json": orjson.dumps(cache_info).decode(),
                    "error": True,
                    "error_type": "circuit_open",
                })
                raise HTTPException(
                    status_code=503,
                    detail={"error": "backend_unavailable", "retry_after_seconds": e.retry_after_seconds},
                )

            result = result_obj
            queue_wait_ms = int((time.perf_counter() - queue_entered) * 1000) - (result.backend_latency_ms or 0)
            if queue_wait_ms < 0:
                queue_wait_ms = int((time.perf_counter() - queue_entered) * 1000)

            backend_span.set_attribute("prompt_tokens", result.prompt_tokens or 0)
            backend_span.set_attribute("completion_tokens", result.completion_tokens or 0)
            backend_span.set_attribute("latency_ms", result.backend_latency_ms or 0)

        cache_info["scheduler"] = {
            "lane": lane, "admission": admission.reason,
            "predicted_wait_ms": predicted_wait_ms,
            "queue_wait_ms": queue_wait_ms,
            "degraded": degraded, "rejected": False,
        }

        # ── Quality Guardrail / Escalation ────────────────────────────────────
        assistant_text = result.text or "(empty response)"
        escalated = False
        escalated_from_tier: Optional[str] = None
        actual_cost_usd: Optional[float] = None
        would_cost_usd: Optional[float] = None

        if classifier_ran and routing_decision is not None and policy.cost_router.tiers:
            would_cost_usd = routing_decision.would_cost_usd

            with tracer.start_as_current_span("relay.escalation") as esc_span:
                if check_escalation(assistant_text, routing_decision, policy.cost_router):
                    up_tier = next_tier(routing_decision.tier_name, policy.cost_router.tiers)
                    if up_tier:
                        escalated_from_tier = routing_decision.tier_name
                        log.info("escalating", request_id=request_id,
                                 from_tier=escalated_from_tier, to_tier=up_tier)
                        esc_span.set_attribute("escalated_from", escalated_from_tier)
                        esc_span.set_attribute("escalated_to", up_tier)

                        up_cfg = policy.cost_router.tiers[up_tier]
                        try:
                            esc_result = await adapter.generate(
                                model=up_cfg.model,
                                prompt=prompt,
                                temperature=float(plan["temperature"]),
                                max_tokens=up_cfg.max_tokens,
                            )
                            assistant_text = esc_result.text or assistant_text
                            result = esc_result
                            escalated = True
                            actual_cost_usd = (
                                (result.completion_tokens or up_cfg.max_tokens) / 1000
                            ) * up_cfg.cost_per_1k_tokens * policy.cost_router.escalation.escalation_cost_multiplier
                        except Exception as exc:
                            log.warning("escalation_backend_error", error=str(exc))
                else:
                    esc_span.set_attribute("escalated_from", "none")

            if actual_cost_usd is None and routing_decision is not None:
                actual_cost_usd = (
                    (result.completion_tokens or routing_decision.max_tokens) / 1000
                ) * routing_decision.cost_per_1k_tokens

        # ── Token Budget Correction ───────────────────────────────────────────
        if tenant_policy.token_budget is not None:
            actual_toks = result.total_tokens or plan["max_tokens"]
            estimated_toks = plan["max_tokens"]
            await record_actual_usage(x_tenant_id, actual_toks, estimated_toks,
                                      tenant_policy.token_budget)
            budget_remaining = max(0, tenant_policy.token_budget.limit - actual_toks)

        # ── Cost span ─────────────────────────────────────────────────────────
        if routing_decision is not None:
            with tracer.start_as_current_span("relay.cost") as cost_span:
                cost_span.set_attribute("actual_cost_usd", actual_cost_usd or 0.0)
                cost_span.set_attribute("would_cost_usd", would_cost_usd or 0.0)
                savings = (would_cost_usd or 0.0) - (actual_cost_usd or 0.0)
                cost_span.set_attribute("savings_usd", savings)

        # ── Assemble response ─────────────────────────────────────────────────
        created = int(time.time())
        resp = ChatCompletionsResponse(
            id=request_id,
            created=created,
            model=req.model,
            choices=[
                ChatCompletionsChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=assistant_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens or 0,
                completion_tokens=result.completion_tokens or 0,
                total_tokens=result.total_tokens or 0,
            ),
        )

        # ── Store caches ──────────────────────────────────────────────────────
        if sem_cfg.get("enabled", False):
            ttl_seconds = int(sem_cfg.get("ttl_seconds", 1800))
            entry_id = await semantic_store(
                tenant_id=x_tenant_id,
                plan_sig=sig,
                request_hash=normalized.request_hash,
                prompt_text=normalized.canonical_text,
                embedding=embed_text(normalized.canonical_text),
                response_obj=resp.model_dump(),
                ttl_seconds=ttl_seconds,
            )
            cache_info["semantic"].update({
                "stored": True, "entry_id": entry_id,
                "ttl_seconds": ttl_seconds,
                "threshold": float(sem_cfg.get("threshold", 0.90)),
            })
        else:
            cache_info["semantic"].update({"stored": False})

        if plan["cache"].get("exact_enabled", True):
            store_key = exact_cache_key(
                tenant_id=x_tenant_id,
                request_hash=normalized.request_hash,
                plan_sig=sig,
            )
            await redis.setex(
                store_key,
                settings.exact_cache_ttl_seconds,
                orjson.dumps({
                    "tier": routing_decision.tier_name if routing_decision else "",
                    "body": resp.model_dump(),
                }),
            )
            cache_info["exact"].update({
                "stored": True,
                "ttl_s": settings.exact_cache_ttl_seconds,
                "key": store_key,
                "plan_sig": sig,
            })
        else:
            cache_info["exact"].update({"stored": False})

        # ── Write trace ───────────────────────────────────────────────────────
        latency_ms = int((time.perf_counter() - t0) * 1000)
        otel_trace_id = extract_trace_id()

        trace_payload: dict[str, Any] = {
            "request_id": request_id,
            "tenant_id": x_tenant_id,
            "user_id": x_user_id,
            "endpoint": "/v1/chat/completions",
            "model": req.model,
            "status_code": 200,
            "request_hash": normalized.request_hash,
            "latency_ms": latency_ms,
            "backend_latency_ms": result.backend_latency_ms,
            "queue_wait_ms": queue_wait_ms,
            "backend_ttft_ms": result.backend_ttft_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "request_json": orjson.dumps(req.model_dump()).decode(),
            "response_json": orjson.dumps(resp.model_dump()).decode(),
            "error_json": "null",
            "policy_version": policy.policy_version,
            "plan_json": orjson.dumps(plan).decode(),
            "decision_trace_json": orjson.dumps(decision_trace).decode(),
            "cache_json": orjson.dumps(cache_info).decode(),
            "otel_trace_id": otel_trace_id,
            "error": False,
        }

        if cap_vector is not None:
            trace_payload.update({
                "complexity_score": cap_vector.complexity_score,
                "intent_type": cap_vector.intent_type,
                "needs_computation": cap_vector.needs_computation,
                "needs_code": cap_vector.needs_code,
                "needs_retrieval": cap_vector.needs_retrieval,
                "needs_personal_context": cap_vector.needs_personal_context,
                "is_multi_hop": cap_vector.is_multi_hop,
                "classifier_latency_ms": cap_vector.classifier_latency_ms,
            })

        if routing_decision is not None:
            trace_payload.update({
                "tier_selected": routing_decision.tier_name,
                "forced_by_tool_flag": routing_decision.forced_by_tool_flag,
                "actual_cost_usd": actual_cost_usd,
                "would_cost_usd": would_cost_usd,
                "escalated": escalated,
                "escalated_from_tier": escalated_from_tier,
            })

        await insert_trace(trace_payload)

        log.info(
            "request_complete",
            request_id=request_id,
            tenant_id=x_tenant_id,
            latency_ms=latency_ms,
            tier=routing_decision.tier_name if routing_decision else None,
            escalated=escalated,
            otel_trace_id=otel_trace_id,
        )

        # ── Build response with X- headers ───────────────────────────────────
        response = JSONResponse(
            content=resp.model_dump(),
            headers={
                "X-Trace-Id": otel_trace_id or "",
                "X-Budget-Remaining": str(budget_remaining) if budget_remaining is not None else "",
                "X-Tier-Selected": routing_decision.tier_name if routing_decision else "",
            },
        )
        return response
