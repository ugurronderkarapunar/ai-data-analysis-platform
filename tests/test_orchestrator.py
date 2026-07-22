"""Tests for orchestrator pipeline smoke path."""

from __future__ import annotations

from pathlib import Path

from core.orchestrator import OrchestratorAgent


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_sales.csv"


def test_orchestrator_end_to_end() -> None:
    """Full pipeline should enrich context without raising."""
    orch = OrchestratorAgent()
    context = orch.run_end_to_end(
        SAMPLE,
        user_query="Satış analizi yap",
        run_forecast=True,
    )
    assert context.dataframe is not None
    assert context.profile
    assert context.intent
    assert context.dashboard_spec
    assert context.report.get("markdown")
    assert context.ml_advice.get("trained") is False
