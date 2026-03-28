---
title: Incident Triage Env
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sre
  - agent
license: mit
---

# Incident Triage Environment

An OpenEnv-compatible reinforcement learning environment that simulates **SRE incident triage**. An agent must diagnose and resolve production incidents across three difficulty tiers by choosing from a discrete action space.

## Overview

| Task | Difficulty | Solution |
|------|-----------|---------|
| `single_service_down` | easy | `restart_service` → `api_gateway` |
| `memory_leak_cascade` | medium | `restart_service` → `worker_pool` |
| `ghost_deploy_regression` | hard | `rollback_deploy` → `payment_service` |

The environment exposes a REST API that follows the OpenEnv protocol. Each episode starts with a call to `/reset` and proceeds via `/step` until the episode terminates (terminal action or max 8 steps).

## Action Space

```
inspect_logs       check_metrics      correlate_services
restart_service    scale_up           rollback_deploy
escalate           silence_alert
```

## Reward Structure

| Event | Reward |
|-------|--------|
| `inspect_logs` (first 2 uses) | +0.15 each |
| `check_metrics` (first 2 uses) | +0.10 each |
| `correlate_services` (first use) | +0.10 |
| Perfect terminal action | +1.00 |
| Efficiency bonus (≤ 3 steps) | +0.20 |
| Correct action type, wrong target | +0.40 |
| Destructive wrong action | −0.30 |
| Loop penalty (same action twice) | −0.10 |
| Step penalty (after step 4) | −0.05/step |
| Max-steps penalty (≥ 6 non-terminal) | −0.30 |

Final score is clipped to **[0.0, 1.0]**.

## API Reference

### `POST /reset`

Start a new episode.

```json
// Request
{ "task_id": "single_service_down", "seed": 42 }

// Response
{
  "session_id": "uuid",
  "observation": {
    "task_id": "single_service_down",
    "step": 0,
    "alerts": ["CRITICAL: api_gateway is not responding"],
    "logs": ["ERROR: Connection refused on port 8080", ...],
    "metrics": {"api_gateway.cpu": 0.0, ...},
    "services": ["api_gateway", "auth_service", "database"],
    "history": [],
    "hint": "The api_gateway service is not responding",
    "valid_actions": ["inspect_logs", "check_metrics", ...]
  }
}
```

### `POST /step`

Apply an action to the current episode.

```json
// Request
{
  "session_id": "uuid",
  "action": { "action_type": "restart_service", "target": "api_gateway" }
}

// Response
{
  "observation": { ... },
  "reward": 1.2,
  "done": true,
  "info": { "task_id": "single_service_down", "step": 1, "action": "restart_service:api_gateway", "reward": 1.2 }
}
```

### `GET /state/{session_id}`

Returns full internal state (step count, done flag, action history, current observation).

### `POST /grade`

Grade a trajectory. Accepts either `session_id` or `task_id` + `history`.

```json
// By session
{ "session_id": "uuid" }

// By trajectory
{
  "task_id": "single_service_down",
  "history": ["inspect_logs:api_gateway", "restart_service:api_gateway"]
}

// Response
{ "task_id": "single_service_down", "score": 1.0, "history": [...] }
```

## Quick Start

### Local

```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 7860
```

### Docker

```bash
docker build -t incident-triage-env .
docker run -p 7860:7860 incident-triage-env
```

### Run the heuristic agent

```bash
# Single episode (random task)
python inference.py

# Pin a task
python inference.py --task-id ghost_deploy_regression

# Baseline scores across all tasks
python inference.py --baseline --base-url http://localhost:7860
```

### Run with an LLM

```bash
export OPENAI_API_KEY=sk-...
python inference.py --task-id memory_leak_cascade --model gpt-4o-mini
```

## Baseline Scores (heuristic agent)

| Task | Score |
|------|-------|
| `single_service_down` | 1.00 |
| `memory_leak_cascade` | 1.00 |
| `ghost_deploy_regression` | 1.00 |
| **Mean** | **1.00** |

## Development

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/test_grader.py -v
```

## Project Structure

```
incident-triage-env/
├── env/
│   ├── __init__.py
│   ├── environment.py   # IncidentEnv class
│   ├── grader.py        # compute_step_reward(), grade()
│   ├── models.py        # Pydantic models (Observation, Action, Reward)
│   └── tasks.py         # TASKS, VALID_ACTIONS, get_task()
├── tests/
│   ├── test_environment.py
│   ├── test_grader.py
│   ├── test_models.py
│   ├── test_server.py
│   └── test_tasks.py
├── server.py            # FastAPI server
├── inference.py         # LLM + heuristic agent loop
├── openenv.yaml         # OpenEnv manifest
├── Dockerfile
├── requirements.txt
└── README.md
```
