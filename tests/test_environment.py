import pytest

from env.environment import IncidentEnv
from env.models import Action, Observation


# ── reset ────────────────────────────────────────────────────────────────────

def test_reset_returns_observation():
    env = IncidentEnv()
    obs = env.reset()
    assert isinstance(obs, Observation)


def test_reset_with_task_id():
    env = IncidentEnv()
    obs = env.reset(task_id="single_service_down")
    assert obs.task_id == "single_service_down"


def test_reset_without_task_id_uses_rng():
    env = IncidentEnv(seed=42)
    obs = env.reset()
    assert obs.task_id in {"single_service_down", "memory_leak_cascade", "ghost_deploy_regression"}


def test_reset_step_zero():
    env = IncidentEnv()
    obs = env.reset(task_id="single_service_down")
    assert obs.step == 0


def test_reset_history_empty():
    env = IncidentEnv()
    obs = env.reset(task_id="single_service_down")
    assert obs.history == []


def test_reset_clears_previous_episode():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    env.step(Action(action_type="inspect_logs", target="api_gateway"))
    obs = env.reset(task_id="single_service_down")
    assert obs.step == 0
    assert obs.history == []


# ── step ─────────────────────────────────────────────────────────────────────

def test_step_returns_tuple():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    result = env.step(Action(action_type="inspect_logs", target="api_gateway"))
    obs, reward, done, info = result
    assert isinstance(obs, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_invalid_action_raises():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    with pytest.raises(ValueError, match="Invalid action_type"):
        env.step(Action(action_type="not_a_real_action", target="api_gateway"))


def test_step_on_done_raises():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    env.step(Action(action_type="restart_service", target="api_gateway"))  # terminal
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(Action(action_type="inspect_logs", target="api_gateway"))


def test_step_increments_step_count():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    _, _, _, info = env.step(Action(action_type="inspect_logs", target="api_gateway"))
    assert info["step"] == 1


def test_step_action_in_history():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    env.step(Action(action_type="inspect_logs", target="api_gateway"))
    obs, _, _, _ = env.step(Action(action_type="check_metrics", target="api_gateway"))
    assert "inspect_logs:api_gateway" in obs.history
    assert "check_metrics:api_gateway" in obs.history


# ── max steps ────────────────────────────────────────────────────────────────

def test_max_steps_forces_done():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    done = False
    for i in range(IncidentEnv.MAX_STEPS):
        # alternate between two non-terminal actions
        action_type = "inspect_logs" if i % 2 == 0 else "check_metrics"
        _, _, done, _ = env.step(Action(action_type=action_type, target="api_gateway"))
    assert done is True


# ── seeded reproducibility ───────────────────────────────────────────────────

def test_seeded_task_selection_reproducible():
    env1 = IncidentEnv(seed=99)
    env2 = IncidentEnv(seed=99)
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert obs1.task_id == obs2.task_id


# ── get_state ────────────────────────────────────────────────────────────────

def test_get_state_structure():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    state = env.get_state()
    assert "task_id" in state
    assert "step" in state
    assert "done" in state
    assert "action_history" in state
    assert "current_observation" in state


# ── full episode ─────────────────────────────────────────────────────────────

def test_full_episode_t1_correct():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    _, r1, done1, _ = env.step(Action(action_type="inspect_logs", target="api_gateway"))
    assert not done1
    assert r1 == pytest.approx(0.15)
    _, r2, done2, _ = env.step(Action(action_type="restart_service", target="api_gateway"))
    assert done2
    assert r2 == pytest.approx(1.0 + 0.20)  # perfect + efficiency bonus


def test_full_episode_t1_wrong_target():
    env = IncidentEnv()
    env.reset(task_id="single_service_down")
    _, reward, done, _ = env.step(Action(action_type="restart_service", target="auth_service"))
    assert done
    assert reward == pytest.approx(0.40)
