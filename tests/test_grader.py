import pytest

from env.grader import compute_step_reward, grade


# ── compute_step_reward ──────────────────────────────────────────────────────

def test_exploration_inspect_logs_reward():
    reward, done = compute_step_reward("single_service_down", "inspect_logs", "api_gateway", [])
    assert reward == pytest.approx(0.15)
    assert done is False


def test_exploration_check_metrics_reward():
    reward, done = compute_step_reward("single_service_down", "check_metrics", "api_gateway", [])
    assert reward == pytest.approx(0.10)
    assert done is False


def test_exploration_capped_at_max():
    history = ["inspect_logs:api_gateway", "check_metrics:api_gateway", "inspect_logs:api_gateway"]
    # Third inspect_logs: capped (0 exploration) + loop penalty (-0.10) = -0.10
    reward, done = compute_step_reward("single_service_down", "inspect_logs", "api_gateway", history)
    assert reward == pytest.approx(-0.10)
    assert done is False


def test_loop_penalty():
    history = ["inspect_logs:api_gateway"]
    reward, done = compute_step_reward("single_service_down", "inspect_logs", "api_gateway", history)
    # Capped (2nd use) so +0.15, but loop penalty -0.10 → net 0.05
    assert reward == pytest.approx(0.05)


def test_perfect_terminal_reward():
    reward, done = compute_step_reward("single_service_down", "restart_service", "api_gateway", [])
    assert reward == pytest.approx(1.0 + 0.20)  # perfect + efficiency (≤3 steps)
    assert done is True


def test_wrong_target_reward():
    reward, done = compute_step_reward("single_service_down", "restart_service", "auth_service", [])
    assert reward == pytest.approx(0.40)
    assert done is True


def test_destructive_wrong_action_penalty():
    # Restarting wrong critical service on T3
    reward, done = compute_step_reward(
        "ghost_deploy_regression", "restart_service", "order_service", []
    )
    assert reward == pytest.approx(-0.30)
    assert done is True


def test_t2_efficiency_bonus_requires_investigation():
    # T2: perfect answer in step 1 (no investigation) → no efficiency bonus
    reward, done = compute_step_reward(
        "memory_leak_cascade", "restart_service", "worker_pool", []
    )
    assert reward == pytest.approx(1.0)  # perfect but NO bonus (no investigation)
    assert done is True


def test_t2_efficiency_bonus_with_investigation():
    history = ["inspect_logs:worker_pool"]
    reward, done = compute_step_reward(
        "memory_leak_cascade", "restart_service", "worker_pool", history
    )
    assert reward == pytest.approx(1.0 + 0.20)  # perfect + efficiency bonus
    assert done is True


def test_t3_escalate_partial_credit():
    history = ["inspect_logs:payment_service", "correlate_services:payment_service"]
    reward, done = compute_step_reward("ghost_deploy_regression", "escalate", "ops_team", history)
    assert reward == pytest.approx(0.10)
    assert done is True


# ── grade ────────────────────────────────────────────────────────────────────

def test_grade_perfect_t1():
    history = ["inspect_logs:api_gateway", "restart_service:api_gateway"]
    score = grade("single_service_down", history)
    assert score == pytest.approx(min(1.0, 0.15 + 1.0))


def test_grade_clipped_to_one():
    # Stacking exploration + efficiency could exceed 1.0 — must be clipped
    history = ["inspect_logs:api_gateway", "check_metrics:api_gateway", "restart_service:api_gateway"]
    score = grade("single_service_down", history)
    assert score <= 1.0


def test_grade_no_resolution_penalty():
    # 6 non-terminal steps → max-steps penalty
    history = [
        "inspect_logs:api_gateway",
        "check_metrics:api_gateway",
        "correlate_services:api_gateway",
        "inspect_logs:auth_service",
        "check_metrics:database",
        "correlate_services:database",
    ]
    score = grade("single_service_down", history)
    # Should be clipped to 0.0 (all rewards minus 0.30 penalty)
    assert 0.0 <= score <= 1.0
