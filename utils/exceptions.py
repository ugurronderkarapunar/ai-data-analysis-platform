"""Shared exception hierarchy for the platform."""

from __future__ import annotations


class PlatformError(Exception):
    """Base exception for all platform errors."""


class DataLoadError(PlatformError):
    """Raised when a dataset cannot be loaded or parsed."""


class AgentExecutionError(PlatformError):
    """Raised when an agent fails during execution."""


class ValidationError(PlatformError):
    """Raised when input or data validation fails."""


class ForecastError(PlatformError):
    """Raised when forecasting cannot be completed."""


class ReportError(PlatformError):
    """Raised when report generation fails."""
