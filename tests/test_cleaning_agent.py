"""Tests for cleaning agent approval behavior."""

from __future__ import annotations

import pandas as pd

from agents.cleaning_agent import DataCleaningAgent
from core.context import AgentContext


def test_cleaning_does_not_mutate_without_approval() -> None:
    """Cleaning plan must not alter dataframe unless apply=True."""
    df = pd.DataFrame(
        {
            "a": [1, 1, None, 4],
            "b": ["x", "x", "y", None],
        }
    )
    context = AgentContext(dataframe=df.copy(), raw_dataframe=df.copy())
    result = DataCleaningAgent().run(context, apply=False)
    assert result.cleaning_plan["applied"] is False
    assert result.dataframe is not None
    assert result.dataframe.equals(df)
