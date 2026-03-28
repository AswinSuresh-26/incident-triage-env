import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /reset ────────────────────────────────────────────────────────────────────

def test_reset_returns_session_id():
    r = client.post("/reset", json={"task_id": "single_service_down"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36  # UUID


def test_reset_returns_observation():
    r = client.post("/reset", json={"task_id": "single_service_down"})
    obs = r.json()["observation"]
    assert obs["task_id"] == "single_service_down"
    assert obs["step"] == 0
    assert obs["history"] == []


def test_reset_random_task():
    r = client.post("/reset", json={})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs["task_id"] in {"single_service_down", "memory_leak_cascade", "ghost_deploy_regression"}


# ── /step ─────────────────────────────────────────────────────────────────────

def _new_session(task_id: str = "single_service_down") -> str:
    r = client.post("/reset", json={"task_id": task_id})
    return r.json()["session_id"]


def test_step_valid():
    sid = _new_session()
    r = client.post("/step", json={"session_id": sid, "action": {"action_type": "inspect_logs", "target": "api_gateway"}})
    assert r.status_code == 200
    data = r.json()
    assert data["reward"] == pytest.approx(0.15)
    assert data["done"] is False


def test_step_invalid_action_type():
    sid = _new_session()
    r = client.post("/step", json={"session_id": sid, "action": {"action_type": "not_real", "target": "api_gateway"}})
    assert r.status_code == 422


def test_step_unknown_session():
    r = client.post("/step", json={"session_id": "00000000-0000-0000-0000-000000000000", "action": {"action_type": "inspect_logs", "target": "api_gateway"}})
    assert r.status_code == 404


def test_step_on_done_session():
    sid = _new_session()
    client.post("/step", json={"session_id": sid, "action": {"action_type": "restart_service", "target": "api_gateway"}})
    r = client.post("/step", json={"session_id": sid, "action": {"action_type": "inspect_logs", "target": "api_gateway"}})
    assert r.status_code == 400


# ── /state ────────────────────────────────────────────────────────────────────

def test_get_state():
    sid = _new_session()
    r = client.get(f"/state/{sid}")
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == "single_service_down"
    assert data["step"] == 0


def test_get_state_unknown():
    r = client.get("/state/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


# ── /grade ────────────────────────────────────────────────────────────────────

def test_grade_via_session_id():
    sid = _new_session()
    client.post("/step", json={"session_id": sid, "action": {"action_type": "inspect_logs", "target": "api_gateway"}})
    client.post("/step", json={"session_id": sid, "action": {"action_type": "restart_service", "target": "api_gateway"}})
    r = client.post("/grade", json={"session_id": sid})
    assert r.status_code == 200
    data = r.json()
    assert data["score"] <= 1.0
    assert data["score"] > 0.0


def test_grade_via_task_id_and_history():
    r = client.post("/grade", json={
        "task_id": "single_service_down",
        "history": ["inspect_logs:api_gateway", "restart_service:api_gateway"],
    })
    assert r.status_code == 200
    data = r.json()
    assert data["score"] == pytest.approx(min(1.0, 0.15 + 1.0 + 0.20))


def test_grade_missing_params():
    r = client.post("/grade", json={})
    assert r.status_code == 422
