"""Forecast agent that auto-selects and runs suitable models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.base_agent import BaseAgent
from core.context import AgentContext
from forecasting.engine import (
    evaluate_candidate_models,
    fit_and_forecast,
    prepare_series,
)
from utils.exceptions import ForecastError, ValidationError


class ForecastAgent(BaseAgent):
    """Create multi-horizon forecasts when a time series is suitable."""

    name = "forecast"
    description = "Uygun zaman serilerinde otomatik tahmin üretir"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Evaluate models and produce 1W/1M/3M/6M/1Y forecasts.

        Args:
            context: Shared context.
            **kwargs: Optional date_col, value_col.

        Returns:
            AgentContext: Context with forecast payload.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("Tahmin için dataframe gerekli.")

        date_col: Optional[str] = kwargs.get("date_col")
        value_col: Optional[str] = kwargs.get("value_col")
        if not date_col:
            date_candidates: List[str] = context.metadata.get("datetime_columns") or context.profile.get(
                "datetime_columns", []
            )
            date_col = date_candidates[0] if date_candidates else None
        if not value_col:
            targets = context.profile.get("target_candidates") or []
            value_col = targets[0] if targets else None

        try:
            prepared = prepare_series(df, date_col=date_col, value_col=value_col)
            values = prepared.frame[prepared.value_col].to_numpy(dtype=float)
            evaluation = evaluate_candidate_models(values)
            selected = evaluation["selected"]
            selected_meta = evaluation["candidates"][selected]
            forecasts = fit_and_forecast(prepared, selected)

            unavailable = [
                "SARIMA",
                "SARIMAX",
                "Prophet",
                "LSTM",
                "GRU",
                "TemporalFusionTransformer",
            ]
            payload: Dict[str, Any] = {
                "available": True,
                "frequency": prepared.frequency,
                "date_col": prepared.date_col,
                "value_col": prepared.value_col,
                "selected_model": selected,
                "selection_reason": selected_meta.get("reason"),
                "metrics": selected_meta.get("metrics"),
                "candidates": {
                    name: {
                        "metrics": meta["metrics"],
                        "family": meta["family"],
                        "reason": meta["reason"],
                    }
                    for name, meta in evaluation["candidates"].items()
                },
                "horizons": forecasts,
                "validation": {
                    "strategy": "TimeSeriesSplit",
                    "metrics": ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "R2"],
                },
                "deferred_models": {
                    "names": unavailable,
                    "note": (
                        "SARIMA/Prophet/deep learning modelleri opsiyonel bağımlılıklar "
                        "ve daha yüksek kaynak ihtiyacı nedeniyle değerlendirme kuyruğunda "
                        "bırakıldı; mimari genişletmeye açıktır."
                    ),
                },
                "explanation": (
                    f"Veri sıklığı '{prepared.frequency}' olarak algılandı. "
                    f"Adaylar TimeSeriesSplit ile kıyaslandı; en düşük RMSE ile "
                    f"'{selected}' seçildi. {selected_meta.get('reason', '')}"
                ),
            }
            context.forecast = payload
            context.add_message(
                f"Tahmin hazır ({selected}): 1 hafta / 1 ay / 3 ay / 6 ay / 1 yıl."
            )
            self.logger.info("Forecast completed with model=%s", selected)
            return context
        except ForecastError as exc:
            self.logger.warning("Forecast skipped: %s", exc)
            context.forecast = {
                "available": False,
                "reason": str(exc),
                "explanation": (
                    "Otomatik tahmin üretilemedi. Veride düzenli bir zaman serisi "
                    "veya yeterli gözlem olmayabilir."
                ),
            }
            context.add_message(f"Tahmin atlandı: {exc}")
            return context
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Forecast agent failed")
            context.forecast = {"available": False, "reason": str(exc)}
            context.add_error(f"Forecast agent hatası: {exc}")
            return context
