from __future__ import annotations

import time
import uuid
from typing import Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from env.environment import IncidentEnv
from env.grader import grade
from env.models import Action, Observation
from env.tasks import TASKS, normalize_task_id

app = FastAPI(title="Incident Triage Environment", version="1.0.0")

# ── Session store ─────────────────────────────────────────────────────────────

SESSION_TTL = 1800  # 30 minutes

class EpisodeState:
    def __init__(self, seed: int = 42) -> None:
        self.env = IncidentEnv(seed=seed)
        self.last_access = time.time()

    def touch(self) -> None:
        self.last_access = time.time()

    def is_expired(self) -> bool:
        return time.time() - self.last_access > SESSION_TTL


sessions: Dict[str, EpisodeState] = {}


def _evict_expired() -> None:
    expired = [sid for sid, s in sessions.items() if s.is_expired()]
    for sid in expired:
        del sessions[sid]


def _get_session(session_id: str) -> EpisodeState:
    _evict_expired()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    s = sessions[session_id]
    s.touch()
    return s


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    session_id: str
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict


class GradeRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    history: Optional[list] = None


class GradeResponse(BaseModel):
    task_id: str
    score: float
    history: list


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "incident-triage-env",
        "description": (
            "SRE Incident Triage — a multi-task reinforcement learning environment "
            "where an agent diagnoses and resolves production incidents across three "
            "difficulty tiers (easy / medium / hard)."
        ),
    }


@app.get("/tasks")
def tasks() -> dict:
    return {
        "tasks": [
            {
                "id": t["id"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "has_grader": True,
            }
            for t in TASKS
        ]
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "inspect_logs", "check_metrics", "restart_service",
                        "scale_up", "rollback_deploy", "escalate",
                        "silence_alert", "correlate_services",
                    ],
                },
                "target": {"type": "string"},
            },
            "required": ["action_type", "target"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "alerts": {"type": "array", "items": {"type": "string"}},
                "logs": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "object"},
                "services": {"type": "array", "items": {"type": "string"}},
                "history": {"type": "array", "items": {"type": "string"}},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "action_history": {"type": "array", "items": {"type": "string"}},
                "current_observation": {"type": "object"},
            },
        },
    }


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[object] = None
    method: Optional[str] = None
    params: Optional[dict] = None


@app.post("/mcp")
def mcp(req: MCPRequest = Body(default=None)) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req.id if req else None,
        "result": {"capabilities": {}},
    }


@app.get("/state")
def get_state_query(session_id: str) -> dict:
    state = _get_session(session_id)
    return state.env.get_state()


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = Body(default=None)) -> ResetResponse:
    if req is None:
        req = ResetRequest()
    session_id = str(uuid.uuid4())
    state = EpisodeState(seed=req.seed)
    obs = state.env.reset(task_id=req.task_id)
    sessions[session_id] = state
    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    state = _get_session(req.session_id)
    try:
        obs, reward, done, info = state.env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # Clamp reward to safe interval (0.01, 0.99)
    reward = max(0.01, min(0.99, float(reward)))
    reward = float(f"{reward:.4f}")
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state/{session_id}")
def get_state(session_id: str) -> dict:
    state = _get_session(session_id)
    return state.env.get_state()


@app.post("/grade", response_model=GradeResponse)
def grader(req: GradeRequest) -> GradeResponse:
    if req.session_id is not None:
        state = _get_session(req.session_id)
        env_state = state.env.get_state()
        task_id = env_state["task_id"]
        history = env_state["action_history"]
        score = grade(task_id, history)
    elif req.task_id is not None:
        # Echo the requested ID but use normalized for grading
        task_id = req.task_id
        normalized_id = normalize_task_id(task_id)
        history = req.history or []
        score = grade(normalized_id, history)
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either session_id or task_id",
        )
    score = max(0.01, min(0.99, float(score)))
    score = float(f"{score:.4f}")
    return GradeResponse(task_id=task_id, score=score, history=history)


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
