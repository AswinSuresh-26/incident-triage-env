import pytest

from env.tasks import TASKS, VALID_ACTIONS, get_task


def test_valid_actions_count():
    assert len(VALID_ACTIONS) == 8


def test_valid_actions_contains_expected():
    expected = {
        "inspect_logs", "check_metrics", "restart_service",
        "scale_up", "rollback_deploy", "escalate",
        "silence_alert", "correlate_services",
    }
    assert set(VALID_ACTIONS) == expected


def test_tasks_count():
    assert len(TASKS) == 3


def test_task_ids():
    ids = [t["id"] for t in TASKS]
    assert "single_service_down" in ids
    assert "memory_leak_cascade" in ids
    assert "ghost_deploy_regression" in ids


def test_task_difficulties():
    by_id = {t["id"]: t["difficulty"] for t in TASKS}
    assert by_id["single_service_down"] == "easy"
    assert by_id["memory_leak_cascade"] == "medium"
    assert by_id["ghost_deploy_regression"] == "hard"


def test_get_task_returns_correct():
    task = get_task("single_service_down")
    assert task["solution"]["action_type"] == "restart_service"
    assert task["solution"]["target"] == "api_gateway"
    assert task["hint"] is not None


def test_get_task_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown task_id"):
        get_task("nonexistent_task")
