from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from .grader import compute_step_reward
from .models import Action, Observation
from .tasks import TASKS, VALID_ACTIONS, get_task


class IncidentEnv:
    MAX_STEPS = 8

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self._current_task: Optional[Dict] = None
        self._episode_state: Dict = {}
        self._step_count: int = 0
        self._action_history: List[str] = []
        self._done: bool = False

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Return clean initial observation. Optionally pin task by ID."""
        if task_id is not None:
            self._current_task = get_task(task_id)
        else:
            self._current_task = self.rng.choice(TASKS)

        self._episode_state = copy.deepcopy(self._current_task["initial_state"])
        self._step_count = 0
        self._action_history = []
        self._done = False

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Apply action, compute step reward, check termination."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if action.action_type not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action_type {action.action_type!r}. "
                f"Valid: {VALID_ACTIONS}"
            )

        task_id = self._current_task["id"]
        entry = f"{action.action_type}:{action.target}"

        step_reward, done = compute_step_reward(
            task_id, action.action_type, action.target, self._action_history
        )
        # Clamp to safe interval (0.01, 0.99)
        step_reward = max(0.01, min(0.99, float(step_reward)))
        step_reward = float(f"{step_reward:.4f}")

        self._action_history.append(entry)
        self._step_count += 1

        # Force done if max steps reached
        if self._step_count >= self.MAX_STEPS:
            done = True

        self._done = done
        obs = self._build_observation()

        info = {
            "task_id": task_id,
            "step": self._step_count,
            "action": entry,
            "reward": step_reward,
        }

        return obs, step_reward, done, info

    def get_state(self) -> dict:
        """Return full internal state (for /state endpoint)."""
        task_id = self._current_task["id"] if self._current_task else None
        return {
            "task_id": task_id,
            "step": self._step_count,
            "done": self._done,
            "action_history": list(self._action_history),
            "current_observation": self._build_observation().model_dump() if self._current_task else None,
        }

    def _build_observation(self) -> Observation:
        task = self._current_task
        state = self._episode_state
        return Observation(
            task_id=task["id"],
            step=self._step_count,
            alerts=list(state["alerts"]),
            logs=list(state["logs"]),
            metrics=dict(state["metrics"]),
            services=list(task["services"]),
            history=list(self._action_history),
            hint=task.get("hint"),
            valid_actions=list(VALID_ACTIONS),
        )
