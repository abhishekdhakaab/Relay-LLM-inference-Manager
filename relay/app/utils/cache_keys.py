from __future__ import annotations
import hashlib
from typing import Any
import orjson

def plan_signature(plan:dict[str,Any]) ->str:
    ## because we want to hash the plan as well : for same input query with different plan should be responded differently
    ## hence we will hash the plan along with query and if both matches then only cache hit

    b = orjson.dumps(plan, option=orjson.OPT_SORT_KEYS)
    ## we have to converst string json to byte format for hashing to work and because the order of key pair doesn't matter so we use ojson.OPT_SORT_KEYS option of orjson instead of using json

    return hashlib.sha256(b).hexdigest()[:16]

def exact_cache_key(*,tenant_id:str, request_hash : str, plan_sig : str) -> str:
    # to have user privacy we will only cache hit when all 3 matches 
    return f"exact:{tenant_id}:{plan_sig}:{request_hash}"