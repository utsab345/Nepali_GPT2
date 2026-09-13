"""Lightweight FastAPI app tests — no torch, no checkpoint required."""

from api.main import app


def test_routes_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/generate" in paths
    assert "/next_token" in paths
    assert "/health" in paths
    assert "/metrics" in paths


def test_metrics_counters_start_at_zero() -> None:
    from api import main as api_module

    snapshot = api_module.metrics.snapshot()
    assert snapshot["requests"] == 0
    assert snapshot["errors"] == 0
