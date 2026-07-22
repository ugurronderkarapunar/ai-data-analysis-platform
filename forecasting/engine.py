"""Forecasting utilities: frequency detection, metrics, model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from utils.dataframe_helpers import to_serializable
from utils.exceptions import ForecastError
from utils.logging_config import get_logger

logger = get_logger("forecasting")

HORIZON_MAP: Dict[str, int] = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
}


@dataclass
class PreparedSeries:
    """Prepared univariate time series for forecasting."""

    frame: pd.DataFrame
    date_col: str
    value_col: str
    frequency: str
    freq_pandas: str


def detect_frequency(dates: pd.Series) -> Tuple[str, str]:
    """Infer series frequency from datetime gaps.

    Args:
        dates: Sorted datetime series.

    Returns:
        Tuple[str, str]: Human label and pandas offset alias.
    """
    try:
        ordered = pd.to_datetime(dates).sort_values().dropna().drop_duplicates()
        if len(ordered) < 3:
            return "unknown", "D"
        deltas = ordered.diff().dropna().dt.days
        median_gap = float(deltas.median())
        if median_gap <= 1.5:
            return "daily", "D"
        if median_gap <= 8:
            return "weekly", "W"
        if median_gap <= 35:
            return "monthly", "MS"
        if median_gap <= 400:
            return "yearly", "YS"
        return "irregular", "D"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Frequency detection failed: %s", exc)
        return "unknown", "D"


def prepare_series(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> PreparedSeries:
    """Prepare a clean univariate time series frame.

    Args:
        df: Source dataframe.
        date_col: Optional datetime column.
        value_col: Optional numeric value column.

    Returns:
        PreparedSeries: Prepared series object.

    Raises:
        ForecastError: If series cannot be prepared.
    """
    try:
        datetime_cols = [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        dcol = date_col or (datetime_cols[0] if datetime_cols else None)
        vcol = value_col or (numeric_cols[0] if numeric_cols else None)
        if dcol is None or vcol is None:
            raise ForecastError("Tahmin için tarih ve sayısal sütun gerekli.")

        frame = (
            df[[dcol, vcol]]
            .dropna()
            .assign(**{dcol: lambda x: pd.to_datetime(x[dcol], errors="coerce")})
            .dropna()
            .sort_values(dcol)
            .drop_duplicates(subset=[dcol], keep="last")
        )
        if len(frame) < 12:
            raise ForecastError("Tahmin için en az 12 zaman noktası gerekir.")

        freq_label, freq_pandas = detect_frequency(frame[dcol])
        frame = frame.set_index(dcol).asfreq(freq_pandas)
        frame[vcol] = frame[vcol].interpolate(limit_direction="both")
        frame = frame.reset_index()
        return PreparedSeries(frame, dcol, vcol, freq_label, freq_pandas)
    except ForecastError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ForecastError(f"Zaman serisi hazırlanamadı: {exc}") from exc


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: MAPE score.
    """
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        float: SMAPE score.
    """
    denom = np.clip((np.abs(y_true) + np.abs(y_pred)) / 2.0, 1e-8, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, MSE, RMSE, MAPE, SMAPE, R².

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict[str, float]: Metric dictionary.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": _mape(y_true, y_pred),
        "SMAPE": _smape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _make_lag_features(values: np.ndarray, lags: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Build supervised lag matrix from a univariate series.

    Args:
        values: Numeric series values.
        lags: Number of lag features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and targets.
    """
    x_rows, y_rows = [], []
    for i in range(lags, len(values)):
        x_rows.append(values[i - lags : i])
        y_rows.append(values[i])
    return np.asarray(x_rows), np.asarray(y_rows)


def evaluate_candidate_models(values: np.ndarray) -> Dict[str, Any]:
    """Evaluate classical and ML forecasting candidates with TimeSeriesSplit.

    Args:
        values: Univariate target array.

    Returns:
        Dict[str, Any]: Candidate metrics and selected model.
    """
    candidates: Dict[str, Any] = {}
    n_splits = min(3, max(2, len(values) // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Baseline: seasonal naive / last value
    try:
        preds, trues = [], []
        for train_idx, test_idx in tscv.split(values):
            last = values[train_idx][-1]
            preds.extend([last] * len(test_idx))
            trues.extend(values[test_idx])
        candidates["NaiveLast"] = {
            "metrics": regression_metrics(np.asarray(trues), np.asarray(preds)),
            "family": "baseline",
            "reason": "Basit son gözlem baseline modeli.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Naive model failed: %s", exc)

    # ARIMA (optional dependency)
    try:
        from statsmodels.tsa.arima.model import ARIMA

        preds, trues = [], []
        for train_idx, test_idx in tscv.split(values):
            model = ARIMA(values[train_idx], order=(1, 1, 1)).fit()
            forecast = model.forecast(steps=len(test_idx))
            preds.extend(forecast)
            trues.extend(values[test_idx])
        candidates["ARIMA"] = {
            "metrics": regression_metrics(np.asarray(trues), np.asarray(preds)),
            "family": "classical",
            "reason": "Kısa/orta vadeli doğrusal zaman serileri için klasik seçenek.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("ARIMA evaluation skipped: %s", exc)

    # Random Forest on lags
    try:
        x, y = _make_lag_features(values, lags=min(7, max(3, len(values) // 10)))
        preds, trues = [], []
        for train_idx, test_idx in tscv.split(x):
            model = RandomForestRegressor(
                n_estimators=120,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(x[train_idx], y[train_idx])
            preds.extend(model.predict(x[test_idx]))
            trues.extend(y[test_idx])
        candidates["RandomForest"] = {
            "metrics": regression_metrics(np.asarray(trues), np.asarray(preds)),
            "family": "ml",
            "reason": "Doğrusal olmayan örüntüler ve lag etkileşimleri için güçlü aday.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("RandomForest evaluation skipped: %s", exc)

    # Optional boosters
    for model_name, builder in (
        ("XGBoost", _try_xgb),
        ("LightGBM", _try_lgbm),
    ):
        try:
            metrics = builder(values, tscv)
            if metrics:
                candidates[model_name] = metrics
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s evaluation skipped: %s", model_name, exc)

    if not candidates:
        raise ForecastError("Hiçbir tahmin adayı değerlendirilemedi.")

    selected = min(
        candidates.items(),
        key=lambda item: item[1]["metrics"]["RMSE"],
    )[0]
    return {"candidates": candidates, "selected": selected}


def _try_xgb(values: np.ndarray, tscv: TimeSeriesSplit) -> Optional[Dict[str, Any]]:
    """Evaluate XGBoost if installed.

    Args:
        values: Series values.
        tscv: Time series splitter.

    Returns:
        Optional[Dict[str, Any]]: Candidate payload or None.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.info("XGBoost not installed; skipped.")
        return None
    x, y = _make_lag_features(values, lags=min(7, max(3, len(values) // 10)))
    preds, trues = [], []
    for train_idx, test_idx in tscv.split(x):
        model = XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(x[train_idx], y[train_idx])
        preds.extend(model.predict(x[test_idx]))
        trues.extend(y[test_idx])
    return {
        "metrics": regression_metrics(np.asarray(trues), np.asarray(preds)),
        "family": "ml",
        "reason": "Gradient boosting; karmaşık tablosal zaman örüntüleri için uygundur.",
    }


def _try_lgbm(values: np.ndarray, tscv: TimeSeriesSplit) -> Optional[Dict[str, Any]]:
    """Evaluate LightGBM if installed.

    Args:
        values: Series values.
        tscv: Time series splitter.

    Returns:
        Optional[Dict[str, Any]]: Candidate payload or None.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.info("LightGBM not installed; skipped.")
        return None
    x, y = _make_lag_features(values, lags=min(7, max(3, len(values) // 10)))
    preds, trues = [], []
    for train_idx, test_idx in tscv.split(x):
        model = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.08,
            random_state=42,
        )
        model.fit(x[train_idx], y[train_idx])
        preds.extend(model.predict(x[test_idx]))
        trues.extend(y[test_idx])
    return {
        "metrics": regression_metrics(np.asarray(trues), np.asarray(preds)),
        "family": "ml",
        "reason": "Hızlı gradient boosting; büyük serilerde ölçeklenebilir.",
    }


def fit_and_forecast(
    prepared: PreparedSeries,
    model_name: str,
    horizons: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Fit selected model on full history and produce multi-horizon forecasts.

    Args:
        prepared: Prepared series.
        model_name: Selected model key.
        horizons: Horizon day map.

    Returns:
        Dict[str, Any]: Forecast values per horizon.
    """
    horizons = horizons or HORIZON_MAP
    values = prepared.frame[prepared.value_col].to_numpy(dtype=float)
    last_date = pd.to_datetime(prepared.frame[prepared.date_col].iloc[-1])
    forecasts: Dict[str, Any] = {}

    try:
        if model_name == "ARIMA":
            from statsmodels.tsa.arima.model import ARIMA

            fitted = ARIMA(values, order=(1, 1, 1)).fit()
            for label, steps in horizons.items():
                scaled_steps = _scale_horizon(steps, prepared.frequency)
                preds = fitted.forecast(steps=scaled_steps)
                forecasts[label] = _format_forecast(last_date, preds, prepared.freq_pandas)
        else:
            lags = min(7, max(3, len(values) // 10))
            x, y = _make_lag_features(values, lags=lags)
            model = _build_ml_model(model_name)
            model.fit(x, y)
            history = list(values[-lags:])
            for label, steps in horizons.items():
                scaled_steps = _scale_horizon(steps, prepared.frequency)
                preds = []
                window = history.copy()
                for _ in range(scaled_steps):
                    pred = float(model.predict(np.asarray(window[-lags:]).reshape(1, -1))[0])
                    preds.append(pred)
                    window.append(pred)
                forecasts[label] = _format_forecast(
                    last_date,
                    np.asarray(preds),
                    prepared.freq_pandas,
                )
        return forecasts
    except Exception as exc:  # noqa: BLE001
        raise ForecastError(f"Tahmin üretilemedi ({model_name}): {exc}") from exc


def _build_ml_model(model_name: str) -> Any:
    """Instantiate an ML regressor by name.

    Args:
        model_name: Model key.

    Returns:
        Any: Estimator instance.
    """
    if model_name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=42,
        )
    if model_name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=200, learning_rate=0.08, random_state=42)
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def _scale_horizon(days: int, frequency: str) -> int:
    """Convert a day-based horizon to steps for the detected frequency.

    Args:
        days: Horizon in days.
        frequency: Detected frequency label.

    Returns:
        int: Number of forecast steps.
    """
    mapping = {"daily": 1, "weekly": 7, "monthly": 30, "yearly": 365}
    unit = mapping.get(frequency, 1)
    return max(1, int(round(days / unit)))


def _format_forecast(
    last_date: pd.Timestamp,
    preds: np.ndarray,
    freq_pandas: str,
) -> List[Dict[str, Any]]:
    """Attach future timestamps to predictions.

    Args:
        last_date: Last observed timestamp.
        preds: Predicted values.
        freq_pandas: Pandas frequency alias.

    Returns:
        List[Dict[str, Any]]: Date/value pairs.
    """
    idx = pd.date_range(start=last_date, periods=len(preds) + 1, freq=freq_pandas)[1:]
    return [
        {"date": ts.isoformat(), "value": to_serializable(val)}
        for ts, val in zip(idx, preds)
    ]
