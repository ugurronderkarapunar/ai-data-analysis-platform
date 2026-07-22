"""Utility helpers for dataframe inspection and validation."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("dataframe_helpers")


def is_datetime_series(series: pd.Series) -> bool:
    """Return True if a series is datetime-like or parseable as dates.

    Args:
        series: Pandas series to inspect.

    Returns:
        bool: Whether the series looks like a date column.
    """
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            sample = series.dropna().astype(str).head(20)
            if sample.empty:
                return False
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            return bool(parsed.notna().mean() >= 0.8)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Datetime detection failed for %s: %s", series.name, exc)
        return False


def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain datetime values.

    Args:
        df: Input dataframe.

    Returns:
        List[str]: Names of datetime-like columns.
    """
    try:
        return [col for col in df.columns if is_datetime_series(df[col])]
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to detect datetime columns: %s", exc)
        return []


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """Heuristically detect identifier columns.

    Args:
        df: Input dataframe.

    Returns:
        List[str]: Candidate ID column names.
    """
    candidates: List[str] = []
    try:
        n_rows = len(df)
        if n_rows == 0:
            return candidates
        for col in df.columns:
            name = str(col).lower()
            series = df[col]
            unique_ratio = series.nunique(dropna=True) / max(n_rows, 1)
            name_hit = any(token in name for token in ("id", "kod", "code", "key", "uuid"))
            if name_hit and unique_ratio >= 0.9:
                candidates.append(col)
            elif unique_ratio == 1.0 and pd.api.types.is_integer_dtype(series):
                candidates.append(col)
        logger.info("Detected ID columns: %s", candidates)
        return candidates
    except Exception as exc:  # noqa: BLE001
        logger.error("ID column detection failed: %s", exc)
        return candidates


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    """Suggest possible target / dependent variable columns.

    Args:
        df: Input dataframe.

    Returns:
        List[str]: Ordered list of candidate target columns.
    """
    keywords = (
        "target",
        "label",
        "y",
        "ciro",
        "satis",
        "sales",
        "revenue",
        "amount",
        "tutar",
        "profit",
        "kar",
        "price",
        "fiyat",
        "score",
        "churn",
    )
    candidates: List[str] = []
    try:
        for col in df.columns:
            name = str(col).lower()
            if any(k in name for k in keywords):
                candidates.append(col)
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric:
            if col not in candidates:
                candidates.append(col)
        logger.info("Target candidates: %s", candidates[:5])
        return candidates[:10]
    except Exception as exc:  # noqa: BLE001
        logger.error("Target detection failed: %s", exc)
        return candidates


def safe_memory_usage_mb(df: pd.DataFrame) -> float:
    """Estimate dataframe memory usage in megabytes.

    Args:
        df: Input dataframe.

    Returns:
        float: Approximate memory usage in MB.
    """
    try:
        return float(df.memory_usage(deep=True).sum() / (1024**2))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Memory usage calculation failed: %s", exc)
        return 0.0


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that required columns exist.

    Args:
        df: Input dataframe.
        columns: Required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")


def to_serializable(value: Any) -> Any:
    """Convert numpy / pandas scalars to JSON-serializable Python types.

    Args:
        value: Arbitrary value.

    Returns:
        Any: Serializable representation.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if pd.isna(value):
        return None
    return value
