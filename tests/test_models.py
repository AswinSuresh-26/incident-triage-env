import pytest
from pydantic import ValidationError

from env.models import Action, Observation, Reward


def make_obs(**kwargs):
    defaults = dict(
        task_id="test",
        step=0,
        alerts=[],
        logs=[],
        metrics={},
        services=[],
        history=[],
        hint=None,
        valid_actions=[],
    )
    defaults.update(kwargs)
    return Observation(**defaults)


def test_observation_basic():
    obs = make_obs(task_id="single_service_down", step=1, hint="Try restarting")
    assert obs.task_id == "single_service_down"
    assert obs.step == 1
    assert obs.hint == "Try restarting"


def test_observation_hint_optional():
    obs = make_obs()
    assert obs.hint is None


def test_observation_valid_actions():
    obs = make_obs(valid_actions=["inspect_logs", "check_metrics"])
    assert "inspect_logs" in obs.valid_actions


def test_action_fields():
    action = Action(action_type="restart_service", target="api_gateway")
    assert action.action_type == "restart_service"
    assert action.target == "api_gateway"


def test_reward_fields():
    reward = Reward(value=1.0, done=True, info={"reason": "perfect"})
    assert reward.value == 1.0
    assert reward.done is True
    assert reward.info["reason"] == "perfect"
