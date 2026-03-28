"""
inference.py — LLM agent loop with heuristic fallback.

Usage:
    python inference.py [--task-id TASK_ID] [--model MODEL] [--base-url URL]

The agent calls /reset, then loops calling /step until done or MAX_STEPS.
If an OpenAI-compatible API key is available it uses the LLM; otherwise it
falls back to a deterministic heuristic agent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import requests

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
MAX_STEPS = 8

SYSTEM_PROMPT = """\
You are an expert SRE agent performing incident triage.
You will receive a JSON observation describing an ongoing incident.
Pick ONE action from the valid_actions list and ONE target from the services list.
Reply with ONLY valid JSON in the form: {"action_type": "...", "target": "..."}.
No explanation, no markdown, just the JSON object.
"""


# ── Heuristic fallback agent ──────────────────────────────────────────────────

_INVESTIGATION_TYPES = {"inspect_logs", "check_metrics", "correlate_services"}


def _heuristic_action(obs: dict) -> dict:
    """
    Simple rule-based agent that achieves reasonable scores without an LLM.
    Rules (in priority order):
    1. Deployment signal in logs + ≥1 investigation done → rollback_deploy the deployed service.
    2. Memory-leak signal in logs ("unbounded"/"leak") + ≥1 investigation done → restart leak source.
    3. Single CRITICAL alert (no deploy signal) + ≥1 investigation done → restart the CRITICAL service.
    4. Inspect the CRITICAL service if not yet inspected.
    5. Inspect any uninspected service.
    6. Check metrics for unchecked services.
    7. Default → escalate.
    """
    history: List[str] = obs.get("history", [])
    alerts: List[str] = obs.get("alerts", [])
    logs: List[str] = obs.get("logs", [])
    services: List[str] = obs.get("services", [])

    investigated = {e.split(":")[0] for e in history if e.split(":")[0] in _INVESTIGATION_TYPES}
    inspected_services = {e.split(":")[1] for e in history if e.startswith("inspect_logs:")}
    checked_services = {e.split(":")[1] for e in history if e.startswith("check_metrics:")}

    # Find CRITICAL service from alerts
    critical_service: Optional[str] = None
    for alert in alerts:
        if "CRITICAL" in alert:
            for svc in services:
                if svc in alert:
                    critical_service = svc
                    break
            break

    # Check for deployment mentions → rollback candidate
    deploy_service: Optional[str] = None
    for log in logs:
        if "deployment" in log.lower() or "deploy" in log.lower():
            for svc in services:
                if svc in log:
                    deploy_service = svc
                    break

    # Check for memory-leak signal → restart candidate
    leak_service: Optional[str] = None
    for log in logs:
        if "leak" in log.lower() or "unbounded" in log.lower():
            for svc in services:
                if svc in log:
                    leak_service = svc
                    break

    uninspected = [s for s in services if s not in inspected_services]
    unchecked = [s for s in services if s not in checked_services]

    # P1: deploy signal + any investigation done → rollback
    if deploy_service and investigated:
        return {"action_type": "rollback_deploy", "target": deploy_service}

    # P2: leak signal + any investigation done → restart the leak source
    if leak_service and investigated:
        return {"action_type": "restart_service", "target": leak_service}

    # P3: single CRITICAL (no deploy) + investigation done → restart CRITICAL service
    critical_alerts = [a for a in alerts if "CRITICAL" in a]
    if critical_service and len(critical_alerts) == 1 and investigated and not deploy_service:
        return {"action_type": "restart_service", "target": critical_service}

    # P4: inspect CRITICAL service first
    if critical_service and critical_service not in inspected_services:
        return {"action_type": "inspect_logs", "target": critical_service}

    # P5: inspect any uninspected service
    if uninspected:
        return {"action_type": "inspect_logs", "target": uninspected[0]}

    # P6: check metrics for unchecked services
    if unchecked:
        return {"action_type": "check_metrics", "target": unchecked[0]}

    return {"action_type": "escalate", "target": "ops_team"}


# ── LLM agent ─────────────────────────────────────────────────────────────────

def _llm_action(obs: dict, model: str, api_key: str, base_url: str) -> dict:
    try:
        from openai import OpenAI
        llm_base_url = os.environ.get("API_BASE_URL") or (base_url if base_url else None)
        client = OpenAI(api_key=api_key, base_url=llm_base_url)
        user_msg = json.dumps(obs, indent=2)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[LLM error] {e} — falling back to heuristic", file=sys.stderr)
        return _heuristic_action(obs)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    task_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    verbose: bool = True,
) -> Dict:
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    use_llm = bool(api_key)

    if verbose:
        mode = f"LLM ({model})" if use_llm else "heuristic"
        print(f"[inference] mode={mode}  task_id={task_id or 'random'}")

    # Reset
    reset_payload: Dict = {}
    if task_id:
        reset_payload["task_id"] = task_id
    r = requests.post(f"{base_url}/reset", json=reset_payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    session_id = data["session_id"]
    obs = data["observation"]

    if verbose:
        print(f"  task={obs['task_id']}  session={session_id}")

    total_reward = 0.0

    for step in range(MAX_STEPS):
        # Choose action
        if use_llm:
            action = _llm_action(obs, model, api_key, None)
        else:
            action = _heuristic_action(obs)

        if verbose:
            print(f"  step {step+1}: {action['action_type']} -> {action['target']}")

        # Step
        r = requests.post(
            f"{base_url}/step",
            json={"session_id": session_id, "action": action},
            timeout=10,
        )
        r.raise_for_status()
        step_data = r.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        total_reward += reward

        if verbose:
            print(f"    reward={reward:.3f}  done={done}")

        if done:
            break

    # Grade
    r = requests.post(f"{base_url}/grade", json={"session_id": session_id}, timeout=10)
    r.raise_for_status()
    grade_data = r.json()
    score = grade_data["score"]

    if verbose:
        print(f"  final score={score:.3f}")

    return {
        "session_id": session_id,
        "task_id": obs["task_id"],
        "score": score,
        "steps": obs["step"],
        "history": obs["history"],
    }


def run_baseline(base_url: str = DEFAULT_BASE_URL) -> Dict:
    """Run heuristic agent on all 3 tasks, return per-task and mean scores."""
    task_ids = ["single_service_down", "memory_leak_cascade", "ghost_deploy_regression"]
    results = {}
    for task_id in task_ids:
        result = run_episode(task_id=task_id, base_url=base_url, verbose=False)
        results[task_id] = result["score"]
    results["mean"] = sum(results.values()) / len(task_ids)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM or heuristic agent on incident triage env")
    parser.add_argument("--task-id", default=None, help="Pin task ID (default: random)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--baseline", action="store_true", help="Run baseline on all tasks")
    args = parser.parse_args()

    if args.baseline:
        scores = run_baseline(base_url=args.base_url)
        print(json.dumps(scores, indent=2))
    else:
        result = run_episode(task_id=args.task_id, model=args.model, base_url=args.base_url)
        print(json.dumps(result, indent=2))
