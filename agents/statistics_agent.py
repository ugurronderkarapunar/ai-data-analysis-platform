"""Statistics agent wrapping the statistics engine."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from stat_analysis.engine import (
    confidence_intervals,
    correlation_analysis,
    descriptive_stats,
    hypothesis_tests,
    normality_tests,
    simple_regression,
    trend_and_stationarity,
)
from utils.exceptions import ValidationError


class StatisticsAgent(BaseAgent):
    """Select and run suitable statistical analyses automatically."""

    name = "statistics"
    description = "Veriye uygun istatistiksel analizleri otomatik uygular"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Execute statistical suite on the dataframe.

        Args:
            context: Shared context.
            **kwargs: Optional ``target``.

        Returns:
            AgentContext: Context with statistics results.

        Raises:
            ValidationError: If dataframe is missing.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("İstatistik analizi için dataframe gerekli.")

        target: Optional[str] = kwargs.get("target")
        if not target:
            candidates = context.profile.get("target_candidates") or []
            target = candidates[0] if candidates else None

        try:
            results: dict[str, Any] = {
                "descriptive": descriptive_stats(df),
                "normality": normality_tests(df),
                "correlation": correlation_analysis(df),
                "regression": simple_regression(df, target=target),
                "hypothesis": hypothesis_tests(df),
                "confidence_intervals": confidence_intervals(df),
                "plain_language": [],
            }

            datetime_cols = context.metadata.get("datetime_columns") or context.profile.get(
                "datetime_columns", []
            )
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if datetime_cols and numeric_cols:
                ts = (
                    df[[datetime_cols[0], numeric_cols[0]]]
                    .dropna()
                    .sort_values(datetime_cols[0])
                )
                results["trend"] = trend_and_stationarity(ts[numeric_cols[0]])
            else:
                results["trend"] = {
                    "available": False,
                    "message": "Trend için tarih + sayısal sütun bulunamadı.",
                }

            results["plain_language"] = self._plain_language(results)
            context.statistics = results
            context.add_message("İstatistiksel analiz tamamlandı.")
            self.logger.info("Statistics agent completed")
            return context
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Statistics agent failed")
            raise ValidationError(f"İstatistik analizi başarısız: {exc}") from exc

    def _plain_language(self, results: dict[str, Any]) -> list[str]:
        """Convert technical results into short Turkish explanations.

        Args:
            results: Statistics payload.

        Returns:
            list[str]: Bullet explanations.
        """
        notes: list[str] = []
        try:
            desc = results.get("descriptive", {})
            if desc.get("available"):
                notes.append("Sayısal alanlar için tanımlayıcı istatistikler üretildi.")
            corr = results.get("correlation", {})
            if corr.get("available"):
                notes.append("Pearson ve Spearman korelasyonları hesaplandı.")
            reg = results.get("regression", {})
            if reg.get("available"):
                notes.append(reg.get("explanation", "Regresyon özeti hazır."))
            for test in results.get("hypothesis", {}).get("t_tests", []):
                notes.append(test.get("explanation", ""))
            for test in results.get("hypothesis", {}).get("anova", []):
                notes.append(test.get("explanation", ""))
            trend = results.get("trend", {})
            if trend.get("available"):
                notes.append(trend.get("explanation", ""))
            return [n for n in notes if n]
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Plain language summary failed: %s", exc)
            return notes
