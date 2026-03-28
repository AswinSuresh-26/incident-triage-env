from __future__ import annotations

from typing import Dict, List

VALID_ACTIONS: List[str] = [
    "inspect_logs",
    "check_metrics",
    "restart_service",
    "scale_up",
    "rollback_deploy",
    "escalate",
    "silence_alert",
    "correlate_services",
]

TASKS: List[Dict] = [
    {
        "id": "single_service_down",
        "difficulty": "easy",
        "description": "One microservice is down. Single alert. Logs show connection refused. CPU/memory normal.",
        "hint": "The api_gateway service is not responding",
        "solution": {"action_type": "restart_service", "target": "api_gateway"},
        "services": ["api_gateway", "auth_service", "database"],
        "initial_state": {
            "alerts": ["CRITICAL: api_gateway is not responding"],
            "logs": [
                "ERROR: Connection refused on port 8080",
                "ERROR: api_gateway health check failed",
                "INFO: auth_service is running normally",
                "INFO: database connections are stable",
            ],
            "metrics": {
                "api_gateway.cpu": 0.0,
                "api_gateway.memory": 0.0,
                "auth_service.cpu": 0.32,
                "auth_service.memory": 0.45,
                "database.cpu": 0.21,
                "database.memory": 0.60,
            },
        },
    },
    {
        "id": "memory_leak_cascade",
        "difficulty": "medium",
        "description": (
            "Two services showing high memory. One is the actual leak source; the other is a symptom. "
            "Must check metrics AND inspect logs before finding the correct target."
        ),
        "hint": None,
        "solution": {"action_type": "restart_service", "target": "worker_pool"},
        "services": ["web_server", "worker_pool", "cache", "database"],
        "initial_state": {
            "alerts": [
                "WARNING: web_server memory usage at 92%",
                "WARNING: worker_pool memory usage at 88%",
            ],
            "logs": [
                "ERROR: web_server OOM warning — memory pressure from upstream",
                "ERROR: worker_pool memory growing unbounded — possible leak in job processor",
                "INFO: cache evictions increasing",
                "INFO: database query times elevated",
            ],
            "metrics": {
                "web_server.cpu": 0.55,
                "web_server.memory": 0.92,
                "worker_pool.cpu": 0.78,
                "worker_pool.memory": 0.88,
                "cache.cpu": 0.30,
                "cache.memory": 0.40,
                "database.cpu": 0.45,
                "database.memory": 0.55,
            },
        },
    },
    {
        "id": "ghost_deploy_regression",
        "difficulty": "hard",
        "description": (
            "Recent deployment completed successfully (logs say so). DB connection pool is exhausted. "
            "Multiple services timing out. CPU/memory look normal. "
            "Standard instinct is to restart — but the correct fix is rollback_deploy on payment_service "
            "because the new code introduced a connection pool misconfiguration."
        ),
        "hint": None,
        "solution": {"action_type": "rollback_deploy", "target": "payment_service"},
        "services": ["payment_service", "order_service", "inventory_service", "database"],
        "initial_state": {
            "alerts": [
                "WARNING: order_service request timeout rate at 45%",
                "WARNING: inventory_service request timeout rate at 38%",
                "CRITICAL: database connection pool exhausted",
            ],
            "logs": [
                "INFO: deployment completed successfully for payment_service v2.3.1",
                "WARNING: retry succeeded after 3 attempts",
                "ERROR: database connection pool exhausted — max 100 connections reached",
                "ERROR: order_service: upstream timeout from payment_service",
                "ERROR: inventory_service: upstream timeout from payment_service",
            ],
            "metrics": {
                "payment_service.cpu": 0.85,
                "payment_service.memory": 0.50,
                "order_service.cpu": 0.40,
                "order_service.memory": 0.45,
                "inventory_service.cpu": 0.38,
                "inventory_service.memory": 0.42,
                "database.cpu": 0.65,
                "database.memory": 0.70,
                "database.connection_pool_used": 1.0,
            },
        },
    },
]


def get_task(task_id: str) -> Dict:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise ValueError(f"Unknown task_id: {task_id!r}")
