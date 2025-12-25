from __future__ import annotations

import time
import uuid
from typing import Any

import orjson
from fastapi import APIRouter, Header, HTTPException

from app.core.logging import get_logger
from app.core.settings import settings
from app.db.postgres import insert_trace
from app.models.openai_chat import (
    ChatCompletionsChoice,
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    ChatMessage,
    Usage,
)
from app.utils.normalize import normalize_messages
from app.core.ollama_adapter import OllamaAdapter
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
router = APIRouter()
log = get_logger(component="api")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    x_tenant_id: str = Header(default="default"),
) -> ChatCompletionsResponse:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported yet")

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    policy = settings.load_policy()
    tenant_policy = policy.tenants.get(x_tenant_id, policy.tenants["default"])

    # Normalize request (used for caching later)
    normalized = normalize_messages(req.messages)

    prompt_chars = len(normalized.canonical_text)

    plan_obj, trace_obj = build_plan(
        policy=policy,
        tenant_id=x_tenant_id,
        prompt_chars=prompt_chars,
        override_temperature=req.temperature,
        override_max_tokens=req.max_tokens,
    )

    plan = {
        "plan_name": plan_obj.plan_name,
        "tier": plan_obj.tier,
        "decoding_profile": plan_obj.decoding_profile,
        "max_tokens": plan_obj.max_tokens,
        "temperature": plan_obj.temperature,
        "cache": plan_obj.cache,
    }

    sig = plan_signature(plan)
    decision_trace = {
        "reasons": trace_obj.reasons,
        "bucket": trace_obj.bucket,
        "tenant_id": trace_obj.tenant_id,
        "policy_version": trace_obj.policy_version,
    }

    # Getting cachce 
    redis = get_redis()
    cache_info : dict[str,Any] = {'exact':{'enabled':bool(plan['cache'].get('exact_enabled',True))}}
    if plan['cache'].get('exact_enabled',True):
        sig = plan_signature(plan)
        key = exact_cache_key(tenant_id=x_tenant_id,request_hash = normalized.request_hash,plan_sig=sig )
        cached = await redis.get(key)

        if cached is not None:
            await redis.incr(f'metrics:cache_exact_hit:{x_tenant_id}')

            cached_obj = orjson.loads(cached)

            resp = ChatCompletionsResponse.model_validate(cached_obj)

            latency_ms = int((time.perf_counter()-t0)*1000)
            cache_info['exact'].update({'hit':True,'key':key,'plan_sig':sig})

            await insert_trace(
                {
                    "request_id": request_id,
                    "tenant_id": x_tenant_id,
                    "endpoint": "/v1/chat/completions",
                    "model": req.model,
                    "status_code": 200,
                    "request_hash": normalized.request_hash,
                    "latency_ms": latency_ms,
                    "backend_latency_ms": None,
                    "queue_wait_ms": None,
                    "backend_ttft_ms": None,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                    "request_json": orjson.dumps(req.model_dump()).decode("utf-8"),
                    "response_json": orjson.dumps(resp.model_dump()).decode("utf-8"),
                    "error_json": "null",
                    "policy_version": policy.policy_version,
                    "plan_json": orjson.dumps(plan).decode("utf-8"),
                    "decision_trace_json": orjson.dumps(decision_trace).decode("utf-8"),
                    "cache_json": orjson.dumps(cache_info).decode("utf-8"),
                }
            )

            log.info(
                "cache_hit",
                request_id=request_id,
                tenant_id=x_tenant_id,
                latency_ms=latency_ms,
                request_hash=normalized.request_hash,
            )
            return resp
        else:
            await redis.incr(f'metrics:cache_exact_miss:{x_tenant_id}')
            cache_info['exact'].update({'hit':False,'key':key,'plan_sig':sig})


    # Now let's check if we can save time with semantic caching
    sem_cfg = plan['cache'].get('semantic',{})
    cache_info['semantic'] = {'enabled':bool(sem_cfg.get('enabled',False)), 'plan_sig':sig}

    if sem_cfg.get('enabled',False):
        qvec = embed_text(normalized.canonical_text)
        row = await semantic_lookup(tenant_id=x_tenant_id, plan_sig = sig, query_vec = qvec)
        if row is not None:
            similarity = float(row.get('similarity',0.0))
            threshold = float(sem_cfg.get('threshold',0.90))
            if similarity>=threshold : 
                resp = ChatCompletionsResponse.model_validate(row['response_json'])

                latency_ms = int((time.perft_counter()-t0)*1000)
                cache_info['semantic'].update(
                                        {
                        "hit": True,
                        "similarity": similarity,
                        "threshold": threshold,
                        "entry_id": row.get("id"),
                        "verifier": sem_cfg.get("verifier", "off"),
                    }

                )
                await insert_trace(
                    {
                        "request_id": request_id,
                        "tenant_id": x_tenant_id,
                        "endpoint": "/v1/chat/completions",
                        "model": req.model,
                        "status_code": 200,
                        "request_hash": normalized.request_hash,
                        "latency_ms": latency_ms,
                        "backend_latency_ms": None,
                        "queue_wait_ms": None,
                        "backend_ttft_ms": None,
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                        "request_json": orjson.dumps(req.model_dump()).decode("utf-8"),
                        "response_json": orjson.dumps(resp.model_dump()).decode("utf-8"),
                        "error_json": "null",
                        "policy_version": policy.policy_version,
                        "plan_json": orjson.dumps(plan).decode("utf-8"),
                        "decision_trace_json": orjson.dumps(decision_trace).decode("utf-8"),
                        "cache_json": orjson.dumps(cache_info).decode("utf-8"),
                    }
                )

                log.info("semantic_cache_hit_pgvector", request_id=request_id, similarity=similarity)
                return resp

            cache_info["semantic"].update(
                {
                    "hit": False,
                    "best_similarity": similarity,
                    "threshold": threshold,
                    "best_entry_id": row.get("id"),
                    "verifier": sem_cfg.get("verifier", "off"),
                }
            )
        else:
            cache_info["semantic"].update({"hit": False, "best_similarity": None})


    scheduler = get_scheduler()
    lane  = scheduler.lane_for_prompt_chars(prompt_chars)

    admission, predicted_wait_ms = scheduler.admission_check(lane=lane, tenant_slo_ms= tenant_policy.latency_slo_ms, prompt_chars=prompt_chars)


    degraded = False
    rejected = False
    rejected_retry_after = False

    effective_max_tokens = plan_obj.max_tokens
    if admission.degraded :
        degraded = True
        adm = policy.scheduler.admission.degrade
        scaled = int(effective_max_tokens*float(adm.max_tokens_scale)) # e.g. currently max_token_scale =0.5 so we reduce the effective max token to half
        effective_max_tokens = max(int(adm.max_tokens_floor),scaled)
        plan['max_tokens'] = effective_max_tokens
        decision_trace['reasons'].append(f'degraded max_tokens to {effective_max_tokens} due to admission control')

    if admission.rejected : 
        rejected = True
        rejected_retry_after = admission.retry_after_seconds or 1 
        cache_info['scheduler'] = {
            'lane':lane,
            'admission':admission.reason,
            'predicted_wait_ms':predicted_wait_ms,
            'degraded':degraded,
            'rejected':True
        }
        latency_ms = int((time.perf_counter()-t0)*1000)
        await insert_trace(
            {
                "request_id": request_id,
                "tenant_id": x_tenant_id,
                "endpoint": "/v1/chat/completions",
                "model": req.model,
                "status_code": 429,
                "request_hash": normalized.request_hash,
                "latency_ms": latency_ms,
                "backend_latency_ms": None,
                "queue_wait_ms": predicted_wait_ms,
                "backend_ttft_ms": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "request_json": orjson.dumps(req.model_dump()).decode("utf-8"),
                "response_json": "null",
                "error_json": orjson.dumps(
                    {"type": "rate_limited", "detail": "Predicted SLO miss; retry later", "retry_after_seconds": reject_retry_after}
                ).decode("utf-8"),
                "policy_version": policy.policy_version,
                "plan_json": orjson.dumps(plan).decode("utf-8"),
                "decision_trace_json": orjson.dumps(decision_trace).decode("utf-8"),
                "cache_json": orjson.dumps(cache_info).decode("utf-8"),
            }
        )
        raise HTTPException(status_code=429, detail={"retry_after_seconds": reject_retry_after})






    
    prompt = normalized.canonical_text + '\n assitance:'

    adapter = OllamaAdapter(base_url = settings.ollama_base_url)

    async def run_backend()-> object:
        ## for github CI
        if settings.backend_mode == 'mock':
            return GenerationResult(
                    text="(mock) " + (normalized.canonical_text[:120].replace("\n", " ")),
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    backend_latency_ms=50,
                    backend_ttft_ms=None,
            )
        return await adapter.generate(model=settings.ollama_model,
                                      prompt=prompt,
                                      temperature=float(plan['temperature']),
                                      max_tokens = int(plan['max_tokens']))
    ## lets use asyncio out event loop to set a future return value

    fut : asyncio.Future[object]  = asyncio.get_running_loop().create_future()
    queue_entered = time.perf_counter()

    from app.core.scheduler import ScheduledJob

    job = ScheduledJob(
        request_id = request_id, 
        tenant_id  = x_tenant_id,
        lane=lane,
        created_at = time.time(), 
        slo_ms = tenant_policy.latency_slo_ms,
        plan = plan_obj,
        run = run_backend, 
        fut=fut, 
        queue_entered_at= queue_entered
    )

    try:
        await scheduler.submit(job)
    except QueueFullError:
        latency_ms = int((time.perf_counter()-t0)*1000)
        cache_info["scheduler"] = {
            "lane": lane,
            "admission": "queue_full",
            "predicted_wait_ms": predicted_wait_ms,
            "degraded": degraded,
            "rejected": True,
        }
        await insert_trace(
            {
                "request_id": request_id,
                "tenant_id": x_tenant_id,
                "endpoint": "/v1/chat/completions",
                "model": req.model,
                "status_code": 503,
                "request_hash": normalized.request_hash,
                "latency_ms": latency_ms,
                "backend_latency_ms": None,
                "queue_wait_ms": predicted_wait_ms,
                "backend_ttft_ms": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "request_json": orjson.dumps(req.model_dump()).decode("utf-8"),
                "response_json": "null",
                "error_json": orjson.dumps({"type": "queue_full", "detail": "Queue full, try later"}).decode("utf-8"),
                "policy_version": policy.policy_version,
                "plan_json": orjson.dumps(plan).decode("utf-8"),
                "decision_trace_json": orjson.dumps(decision_trace).decode("utf-8"),
                "cache_json": orjson.dumps(cache_info).decode("utf-8"),
            }
        )
        raise HTTPException(status_code=503, detail="Queue full, try later")
    
    result_obj = await fut
    result = result_obj
    queue_wait_ms = int((time.perf_counter()-queue_entered)*1000)- (result.backend_latency_ms or 0)
    if queue_wait_ms < 0:
        queue_wait_ms = int((time.perf_counter() - queue_entered)*1000)

    cache_info["scheduler"] = {
        "lane": lane,
        "admission": admission.reason,
        "predicted_wait_ms": predicted_wait_ms,
        "queue_wait_ms": queue_wait_ms,
        "degraded": degraded,
        "rejected": False,
    }




    
    #result = await adapter.generate(model=settings.ollama_model, prompt= prompt, temperature=plan['temperature'], max_tokens = plan['max_tokens'])
    assistant_text = result.text or "(empty response)"
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

    ## let's store the respo (pgvector)
    if sem_cfg.get('enabled',False):
        ttl_seconds = int(sem_cfg.get('ttl_seconds',1800))
        entry_id = await semantic_store(
            tenant_id = x_tenant_id,
            plan_sig = sig,
            request_hash = normalized.request_hash,
            prompt_text=normalized.canonical_text,
            embedding = embed_text(normalized.canonical_text),
            response_obj=resp.model_dump(),
            ttl_seconds = ttl_seconds,
        )
        cache_info['semantic'].update(
            {
                                "stored": True,
                "entry_id": entry_id,
                "ttl_seconds": ttl_seconds,
                "threshold": float(sem_cfg.get("threshold", 0.90)),
                "verifier": sem_cfg.get("verifier", "off"),

            }
        )
    else :
        cache_info['semantic'].update({'stored':False})

    if plan['cache'].get('exact_enabled',True):
        sig = plan_signature(plan)
        key = exact_cache_key(tenant_id=x_tenant_id, request_hash=normalized.request_hash,plan_sig=sig)
        await redis.setex(key,settings.exact_cache_ttl_seconds,orjson.dumps(resp.model_dump()))
        cache_info['exact'].update({'store':True,'ttl_s':settings.exact_cache_ttl_seconds, 'key':key,'plan_sig':sig})
    else : 
        cache_info['exact'].update({'stored':False})


    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Store trace (minimal for now)
    await insert_trace(
        {
            "request_id": request_id,
            "tenant_id": x_tenant_id,
            "endpoint": "/v1/chat/completions",
            "model": req.model,
            "status_code": 200,
            "request_hash": normalized.request_hash,
            "latency_ms": latency_ms,
            "backend_latency_ms": result.backend_latency_ms,
            "queue_wait_ms": None,
            "backend_ttft_ms": result.backend_ttft_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "request_json": orjson.dumps(req.model_dump()).decode("utf-8"),
            "response_json": orjson.dumps(resp.model_dump()).decode("utf-8"),
            "error_json": "null",
            "policy_version": policy.policy_version,
            "plan_json": orjson.dumps(plan).decode("utf-8"),
            "cache_json": orjson.dumps(cache_info).decode("utf-8"),
            "decision_trace_json": orjson.dumps(decision_trace).decode("utf-8"),
            "queue_wait_ms": queue_wait_ms,

            "backend_latency_ms": result.backend_latency_ms

        }
    )

    log.info(
        "request_complete",
        request_id=request_id,
        tenant_id=x_tenant_id,
        latency_ms=latency_ms,
        request_hash=normalized.request_hash,
        policy_version=policy.policy_version,
    )

    return resp
