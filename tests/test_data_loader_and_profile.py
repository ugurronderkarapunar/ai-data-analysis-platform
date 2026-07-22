"""Tests for data loader and profiling agents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.data_loader_agent import DataLoaderAgent
from agents.profiling_agent import DataProfilingAgent
from core.context import AgentContext


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_sales.csv"


def test_data_loader_reads_sample_csv() -> None:
    """Loader should read the sample CSV into a non-empty dataframe."""
    context = AgentContext()
    agent = DataLoaderAgent()
    result = agent.run(context, source=SAMPLE)
    assert result.dataframe is not None
    assert len(result.dataframe) > 20
    assert result.file_name == "sample_sales.csv"


def test_profiler_produces_core_fields() -> None:
    """Profiler should expose rows/columns/missing/duplicates."""
    df = pd.read_csv(SAMPLE, parse_dates=["date"])
    context = AgentContext(dataframe=df, raw_dataframe=df.copy())
    result = DataProfilingAgent().run(context)
    assert result.profile["rows"] == len(df)
    assert result.profile["columns"] == df.shape[1]
    assert "missing_counts" in result.profile
    assert "target_candidates" in result.profile
