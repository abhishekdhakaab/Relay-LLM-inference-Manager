from __future__ import annotations
from typing import Optional
from app.core.scheduler import Scheduler
from app.core.settings import PolicyConfig 

_scheduler : Optional[Scheduler] = None

def init_scheduler(policy:PolicyConfig)->Scheduler:
    global _scheduler
    _scheduler = Scheduler(policy)
    _scheduler.start()
    return _scheduler

def get_scheduler()->Scheduler : 
    assert _scheduler is not None, "Scheduler not initalized"
    return _scheduler