"""Automatic data profiling agent."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from utils.dataframe_helpers import (
    detect_datetime_columns,
    detect_id_columns,
    detect_target_candidates,
    safe_memory_usage_mb,
    to_serializable,
)
from utils.exceptions import ValidationError


class DataProfilingAgent(BaseAgent):
    """Produce a human-readable statistical profile of the dataset."""

    name = "data_profiler"
    description = "Veri setinin otomatik profilini çıkarır"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Profile the dataframe in context.

        Args:
            context: Shared context with a loaded dataframe.
            **kwargs: Unused.

        Returns:
            AgentContext: Context with ``profile`` populated.

        Raises:
            ValidationError: If dataframe is missing.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("Profil için dataframe gerekli.")

        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            datetime_cols = context.metadata.get("datetime_columns") or detect_datetime_columns(df)
            missing = df.isna().sum()
            missing_pct = (missing / max(len(df), 1) * 100).round(2)

            outliers = self._detect_outliers(df, numeric_cols)
            corr = self._correlation(df, numeric_cols)

            profile: Dict[str, Any] = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "missing_counts": {str(k): int(v) for k, v in missing.items()},
                "missing_percent": {str(k): float(v) for k, v in missing_pct.items()},
                "duplicate_rows": int(df.duplicated().sum()),
                "memory_mb": round(safe_memory_usage_mb(df), 3),
                "numeric_summary": self._numeric_summary(df, numeric_cols),
                "categorical_summary": self._categorical_summary(df, categorical_cols),
                "outliers": outliers,
                "correlations": corr,
                "datetime_columns": list(datetime_cols),
                "id_columns": detect_id_columns(df),
                "target_candidates": detect_target_candidates(df),
                "column_names": [str(c) for c in df.columns],
            }
            context.profile = profile
            context.add_message(
                "Veri profili hazır: "
                f"{profile['rows']} satır, {profile['columns']} sütun, "
                f"{profile['duplicate_rows']} yinelenen kayıt."
            )
            self.logger.info(
                "Profile complete rows=%s cols=%s missing_total=%s",
                profile["rows"],
                profile["columns"],
                int(missing.sum()),
            )
            return context
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Profiling failed")
            raise ValidationError(f"Profil oluşturulamadı: {exc}") from exc

    def _numeric_summary(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Build descriptive stats for numeric columns.

        Args:
            df: Dataframe.
            columns: Numeric column names.

        Returns:
            Dict[str, Dict[str, Any]]: Summary per column.
        """
        result: Dict[str, Dict[str, Any]] = {}
        try:
            if not columns:
                return result
            desc = df[columns].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            for col, stats in desc.items():
                result[str(col)] = {str(k): to_serializable(v) for k, v in stats.items()}
            return result
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Numeric summary failed: %s", exc)
            return result

    def _categorical_summary(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Build frequency summaries for categorical columns.

        Args:
            df: Dataframe.
            columns: Categorical column names.

        Returns:
            Dict[str, Dict[str, Any]]: Summary per column.
        """
        result: Dict[str, Dict[str, Any]] = {}
        try:
            for col in columns:
                series = df[col]
                top = series.value_counts(dropna=True).head(10)
                result[str(col)] = {
                    "unique": int(series.nunique(dropna=True)),
                    "top_values": {str(k): int(v) for k, v in top.items()},
                }
            return result
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Categorical summary failed: %s", exc)
            return result

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Detect IQR-based outliers for numeric columns.

        Args:
            df: Dataframe.
            columns: Numeric columns.

        Returns:
            Dict[str, Dict[str, Any]]: Outlier counts and bounds.
        """
        result: Dict[str, Dict[str, Any]] = {}
        try:
            for col in columns:
                series = df[col].dropna()
                if series.empty:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                result[str(col)] = {
                    "count": int(mask.sum()),
                    "lower": to_serializable(lower),
                    "upper": to_serializable(upper),
                }
            return result
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Outlier detection failed: %s", exc)
            return result

    def _correlation(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Dict[str, Any]:
        """Compute pairwise Pearson correlations for numeric columns.

        Args:
            df: Dataframe.
            columns: Numeric columns.

        Returns:
            Dict[str, Any]: Correlation matrix and strong pairs.
        """
        try:
            if len(columns) < 2:
                return {"matrix": {}, "strong_pairs": []}
            corr = df[columns].corr(numeric_only=True)
            matrix = {
                str(r): {str(c): to_serializable(corr.loc[r, c]) for c in corr.columns}
                for r in corr.index
            }
            strong_pairs = []
            for i, a in enumerate(corr.columns):
                for b in corr.columns[i + 1 :]:
                    value = corr.loc[a, b]
                    if pd.notna(value) and abs(value) >= 0.7:
                        strong_pairs.append(
                            {
                                "a": str(a),
                                "b": str(b),
                                "corr": to_serializable(value),
                            }
                        )
            strong_pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
            return {"matrix": matrix, "strong_pairs": strong_pairs[:20]}
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Correlation failed: %s", exc)
            return {"matrix": {}, "strong_pairs": []}
