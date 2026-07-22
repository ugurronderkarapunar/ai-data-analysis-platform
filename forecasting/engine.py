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

# Çok uzun serilerde (örn. yanlışlıkla toplanmamış ham işlem verisi ya da
# yıllarca süren günlük veri) model eğitimini makul sürede tutmak için
# kullanılan üst sınır. Aşılırsa serinin en GÜNCEL kısmı kullanılır.
MAX_POINTS_FOR_FORECAST = 3000

# Ciro (revenue) sütunu otomatik türetilirken aranan miktar/fiyat
# sütunu adları (küçük harfe çevrilerek karşılaştırılır).
_QUANTITY_COLUMN_NAMES = {"quantity", "miktar", "adet", "qty"}
_PRICE_COLUMN_NAMES = {"unitprice", "price", "fiyat", "birim_fiyat", "birimfiyat"}

# Hedef değişken adayı seçilirken dışlanacak ID benzeri sütun kalıpları.
_ID_LIKE_NAMES = {"id", "index"}


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


def _maybe_build_revenue_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Derive a revenue column (quantity * price) when possible.

    E-ticaret / satış verilerinde genellikle doğrudan bir "ciro" sütunu
    bulunmaz; bunun yerine "Quantity" ve "UnitPrice" gibi ayrı sütunlar
    yer alır. Her ikisi de tespit edilirse, tahmin için çok daha anlamlı
    olan bir ciro sütunu otomatik olarak türetilir.

    Args:
        df: Kaynak DataFrame.

    Returns:
        Tuple[pd.DataFrame, Optional[str]]: (Gerekirse ciro sütunu eklenmiş
        DataFrame kopyası, türetilen sütunun adı ya da None).
    """
    quantity_col = next(
        (c for c in df.columns if str(c).strip().lower() in _QUANTITY_COLUMN_NAMES),
        None,
    )
    price_col = next(
        (c for c in df.columns if str(c).strip().lower() in _PRICE_COLUMN_NAMES),
        None,
    )

    if quantity_col is None or price_col is None:
        return df, None

    try:
        quantity = pd.to_numeric(df[quantity_col], errors="coerce")
        price = pd.to_numeric(df[price_col], errors="coerce")
        revenue = quantity * price

        working = df.copy()
        working["_computed_revenue"] = revenue
        logger.info(
            "Ciro sütunu türetildi: '%s' x '%s' -> '_computed_revenue'",
            quantity_col,
            price_col,
        )
        return working, "_computed_revenue"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ciro sütunu türetilemedi: %s", exc)
        return df, None


def _resolve_value_column(numeric_cols: List[str]) -> Optional[str]:
    """Pick the most sensible numeric column to forecast.

    Sütun adına göre satış/ciro anahtar kelimelerini tercih eder; bulamazsa
    ID benzeri (yüksek olasılıkla kimlik) sütunları eleyerek ilk uygun
    sayısal sütunu döndürür. Bu, örn. "CustomerID" gibi bir kimlik
    sütununun yanlışlıkla tahmin hedefi olarak seçilmesini engeller.

    Args:
        numeric_cols: Sayısal sütun adları.

    Returns:
        Optional[str]: Seçilen sütun adı ya da uygun sütun yoksa None.
    """
    if not numeric_cols:
        return None

    target_keywords = (
        "ciro", "revenue", "sales", "satis", "satış", "tutar", "amount", "total",
    )
    for col in numeric_cols:
        col_lower = str(col).strip().lower()
        if any(keyword in col_lower for keyword in target_keywords):
            return col

    non_id_cols = [
        col for col in numeric_cols
        if not (
            str(col).strip().lower() in _ID_LIKE_NAMES
            or str(col).strip().lower().endswith(("_id", "id"))
        )
    ]
    return (non_id_cols or numeric_cols)[0]


def _limit_series_length(
    values: np.ndarray, max_points: int = MAX_POINTS_FOR_FORECAST
) -> np.ndarray:
    """Trim a series to its most recent `max_points` observations.

    Args:
        values: Tam zaman serisi değerleri.
        max_points: İzin verilen azami nokta sayısı.

    Returns:
        np.ndarray: Gerekirse kısaltılmış seri (en güncel kısım korunur).
    """
    if len(values) > max_points:
        logger.info(
            "Seri %d noktadan, performans amacıyla son %d noktaya kısaltıldı.",
            len(values),
            max_points,
        )
        return values[-max_points:]
    return values


def prepare_series(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> PreparedSeries:
    """Prepare a clean, properly aggregated univariate time series frame.

    Önemli: Bu fonksiyon artık ham (satır bazlı) işlem verisini, tespit
    edilen frekansa göre TOPLAYARAK (`resample(...).sum()`) gerçek bir
    zaman serisine dönüştürür. Önceki sürüm yalnızca aynı zaman damgasına
    sahip satırları `drop_duplicates` ile eleyip sonucu `asfreq` ile
    zorluyordu; bu, yüz binlerce satırlık ham e-ticaret verisinde gerçek
    satış toplamlarının kaybolup büyük ölçüde interpolasyonla üretilmiş
    (uydurma) bir seriye dönüşmesine yol açıyordu.

    Ayrıca, bir "Quantity" ve "UnitPrice" benzeri sütun çifti tespit
    edilirse otomatik olarak bir ciro (revenue) sütunu türetilip tahmin
    hedefi olarak tercih edilir; bu da rastgele bir sayısal sütunun
    (örn. bir kimlik alanının) yanlışlıkla hedef seçilmesini önler.

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
        working_df, computed_revenue_col = _maybe_build_revenue_column(df)

        datetime_cols = [
            c for c in working_df.columns
            if pd.api.types.is_datetime64_any_dtype(working_df[c])
        ]
        numeric_cols = working_df.select_dtypes(include=[np.number]).columns.tolist()

        dcol = date_col or (datetime_cols[0] if datetime_cols else None)
        if dcol is None:
            raise ForecastError("Tahmin için tarih sütunu bulunamadı.")

        if value_col and value_col in working_df.columns:
            vcol = value_col
        elif computed_revenue_col:
            vcol = computed_revenue_col
        else:
            vcol = _resolve_value_column(numeric_cols)

        if vcol is None:
            raise ForecastError("Tahmin için uygun sayısal sütun bulunamadı.")

        raw = working_df[[dcol, vcol]].dropna().copy()
        raw[dcol] = pd.to_datetime(raw[dcol], errors="coerce")
        raw = raw.dropna(subset=[dcol])

        if len(raw) < 12:
            raise ForecastError("Tahmin için en az 12 zaman noktası gerekir.")

        freq_label, freq_pandas = detect_frequency(raw[dcol])

        aggregated = (
            raw.set_index(dcol)[vcol]
            .resample(freq_pandas)
            .sum()
            .to_frame()
            .reset_index()
        )
        aggregated[vcol] = aggregated[vcol].interpolate(limit_direction="both")

        if len(aggregated) < 12:
            raise ForecastError("Tahmin için en az 12 zaman noktası gerekir.")

        return PreparedSeries(aggregated, dcol, vcol, freq_label, freq_pandas)
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
    values = _limit_series_length(values)

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

    frame = prepared.frame
    if len(frame) > MAX_POINTS_FOR_FORECAST:
        logger.info(
            "Eğitim verisi %d noktadan, performans amacıyla son %d noktaya kısaltıldı.",
            len(frame),
            MAX_POINTS_FOR_FORECAST,
        )
        frame = frame.tail(MAX_POINTS_FOR_FORECAST).reset_index(drop=True)

    values = frame[prepared.value_col].to_numpy(dtype=float)
    last_date = pd.to_datetime(frame[prepared.date_col].iloc[-1])
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
