from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Observation(BaseModel):
    task_id: str
    step: int
    alerts: List[str]
    logs: List[str]
    metrics: Dict[str, float]
    services: List[str]
    history: List[str]
    hint: Optional[str]
    valid_actions: List[str]


class Action(BaseModel):
    action_type: str
    target: str


class Reward(BaseModel):
    value: float
    done: bool
    info: Dict[str, Any]
