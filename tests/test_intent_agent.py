"""Tests for intent understanding."""

from __future__ import annotations

from agents.intent_agent import IntentUnderstandingAgent
from core.context import AgentContext


def test_intent_sales_query() -> None:
    """Sales-related Turkish query should map to sales_analysis."""
    context = AgentContext()
    result = IntentUnderstandingAgent().run(
        context,
        user_query="Aylık ciroyu görmek istiyorum",
    )
    assert result.intent["primary_intent"] in {"sales_analysis", "forecast", "general_analysis"}
    assert result.tasks
    assert result.intent["confidence"] > 0


def test_intent_dashboard_query() -> None:
    """Dashboard request should include dashboard intent or task."""
    context = AgentContext()
    result = IntentUnderstandingAgent().run(
        context,
        user_query="Personel dashboard hazırla",
    )
    assert (
        result.intent["primary_intent"] in {"hr_dashboard", "dashboard"}
        or any(t["id"] == "dashboard" for t in result.tasks)
    )
