from __future__ import annotations

from typing import Dict, List, Tuple

from .tasks import get_task

# Actions that terminate an episode
TERMINAL_ACTIONS = frozenset({
    "restart_service",
    "scale_up",
    "rollback_deploy",
    "escalate",
    "silence_alert",
})

# Exploration rewards: (max_times, reward_per_use)
EXPLORATION_REWARDS: Dict[str, Tuple[int, float]] = {
    "inspect_logs": (2, 0.15),
    "check_metrics": (2, 0.10),
    "correlate_services": (1, 0.10),
}


def _count_action_type(history: List[str], action_type: str) -> int:
    """Count how many times action_type appears in the history so far."""
    return sum(1 for entry in history if entry.split(":")[0] == action_type)


def _investigation_count(history: List[str]) -> int:
    """Count total investigation steps (inspect_logs + check_metrics + correlate_services)."""
    investigation_types = {"inspect_logs", "check_metrics", "correlate_services"}
    return sum(1 for entry in history if entry.split(":")[0] in investigation_types)


def compute_step_reward(
    task_id: str,
    action_type: str,
    target: str,
    prior_history: List[str],
) -> Tuple[float, bool]:
    """
    Compute the reward for a single step.

    Args:
        task_id: The current task identifier.
        action_type: The action type being taken.
        target: The target service/component.
        prior_history: List of "action_type:target" strings BEFORE this step.

    Returns:
        (reward, done): reward for this step and whether episode ends.
    """
    task = get_task(task_id)
    solution = task["solution"]
    reward = 0.0
    done = False

    # Loop penalty: same action_type as immediately prior
    if prior_history and prior_history[-1].split(":")[0] == action_type:
        reward -= 0.10

    # Exploration rewards (capped)
    if action_type in EXPLORATION_REWARDS:
        max_times, step_reward = EXPLORATION_REWARDS[action_type]
        prior_count = _count_action_type(prior_history, action_type)
        if prior_count < max_times:
            reward += step_reward
        # Step penalty for non-terminal after step 4
        step_num = len(prior_history) + 1  # 1-based
        if step_num > 4:
            reward -= 0.05
        return reward, done

    # Terminal action
    if action_type in TERMINAL_ACTIONS:
        done = True
        step_num = len(prior_history) + 1

        if action_type == solution["action_type"] and target == solution["target"]:
            # Perfect resolution
            reward += 1.00
            # Efficiency bonus: ≤ 3 steps
            if step_num <= 3:
                # For T2 (memory_leak_cascade), only if ≥1 investigation step was taken
                if task_id == "memory_leak_cascade":
                    if _investigation_count(prior_history) >= 1:
                        reward += 0.20
                else:
                    reward += 0.20
        elif action_type == solution["action_type"] and target != solution["target"]:
            # Correct action type, wrong target
            reward += 0.40
        elif action_type in {"restart_service", "scale_up", "rollback_deploy"}:
            # Destructive wrong action
            reward -= 0.30
        elif action_type == "escalate":
            # Escalate after ≥2 investigation steps gives partial credit for T3
            if task_id == "ghost_deploy_regression" and _investigation_count(prior_history) >= 2:
                reward += 0.10

        return reward, done

    # Non-investigation, non-terminal (silence_alert already handled above as terminal)
    # Step penalty for late steps
    step_num = len(prior_history) + 1
    if step_num > 4:
        reward -= 0.05

    return reward, done


def grade(task_id: str, history: List[str]) -> float:
    """
    Grade a full episode trajectory.

    Args:
        task_id: The task identifier.
        history: List of "action_type:target" strings in order.

    Returns:
        Final score clipped to [0.0, 1.0].
    """
    total_reward = 0.0
    terminal_reached = False

    for i, entry in enumerate(history):
        parts = entry.split(":", 1)
        action_type = parts[0]
        target = parts[1] if len(parts) > 1 else ""
        prior = history[:i]

        step_reward, done = compute_step_reward(task_id, action_type, target, prior)
        total_reward += step_reward

        if done:
            terminal_reached = True
            break

    # Max-steps penalty: no terminal resolution and ≥6 steps taken
    if not terminal_reached and len(history) >= 6:
        total_reward -= 0.30

    return max(0.0, min(1.0, total_reward))
