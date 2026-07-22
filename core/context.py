"""Shared typed context passed between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class AgentContext:
    """Mutable shared state flowing through the multi-agent pipeline.

    Attributes:
        dataframe: Primary working dataframe.
        raw_dataframe: Immutable copy of the originally loaded dataframe.
        file_name: Source file name if loaded from disk/upload.
        profile: Output of the profiling agent.
        intent: Parsed user intent structure.
        tasks: Task list derived from user intent.
        cleaning_plan: Proposed cleaning operations (not applied yet).
        statistics: Statistical analysis results.
        forecast: Forecast outputs when applicable.
        ml_advice: ML advisor recommendations.
        viz_advice: Visualization recommendations.
        dashboard_spec: Dashboard construction specification.
        report: Auto-generated report sections.
        messages: Human-readable status / insight messages.
        errors: Captured non-fatal errors.
        metadata: Free-form metadata bag.
    """

    dataframe: Optional[pd.DataFrame] = None
    raw_dataframe: Optional[pd.DataFrame] = None
    file_name: Optional[str] = None
    profile: Dict[str, Any] = field(default_factory=dict)
    intent: Dict[str, Any] = field(default_factory=dict)
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    cleaning_plan: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    forecast: Dict[str, Any] = field(default_factory=dict)
    ml_advice: Dict[str, Any] = field(default_factory=dict)
    viz_advice: Dict[str, Any] = field(default_factory=dict)
    dashboard_spec: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: str) -> None:
        """Append a user-facing message.

        Args:
            message: Message text.
        """
        self.messages.append(message)

    def add_error(self, error: str) -> None:
        """Append a non-fatal error message.

        Args:
            error: Error text.
        """
        self.errors.append(error)
