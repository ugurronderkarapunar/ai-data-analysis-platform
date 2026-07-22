"""Statistical analysis helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from utils.dataframe_helpers import to_serializable
from utils.logging_config import get_logger

logger = get_logger("statistics_engine")


def descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute descriptive statistics for numeric columns.

    Args:
        df: Input dataframe.

    Returns:
        Dict[str, Any]: Descriptive statistics payload.
    """
    try:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return {"available": False, "message": "Sayısal sütun yok."}
        summary = numeric.describe().T
        summary["skew"] = numeric.skew()
        summary["kurtosis"] = numeric.kurtosis()
        payload = {
            str(idx): {str(k): to_serializable(v) for k, v in row.items()}
            for idx, row in summary.iterrows()
        }
        return {"available": True, "summary": payload}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Descriptive stats failed")
        return {"available": False, "error": str(exc)}


def normality_tests(df: pd.DataFrame, max_cols: int = 8) -> Dict[str, Any]:
    """Run Shapiro-Wilk normality tests on numeric columns.

    Args:
        df: Input dataframe.
        max_cols: Maximum columns to test.

    Returns:
        Dict[str, Any]: Test results with plain-language notes.
    """
    results: Dict[str, Any] = {}
    try:
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns[:max_cols]:
            series = numeric[col].dropna()
            if len(series) < 8:
                results[str(col)] = {
                    "skipped": True,
                    "reason": "Yeterli gözlem yok (min 8).",
                }
                continue
            sample = series.sample(n=min(len(series), 500), random_state=42)
            stat, p_value = stats.shapiro(sample)
            normal = bool(p_value >= 0.05)
            results[str(col)] = {
                "statistic": to_serializable(stat),
                "p_value": to_serializable(p_value),
                "is_normal": normal,
                "explanation": (
                    "Dağılım normale yakın görünüyor."
                    if normal
                    else "Dağılım normalden sapıyor; parametrik olmayan testler tercih edilebilir."
                ),
            }
        return {"tests": results}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Normality tests failed")
        return {"tests": {}, "error": str(exc)}


def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Pearson and Spearman correlation summaries.

    Args:
        df: Input dataframe.

    Returns:
        Dict[str, Any]: Correlation payload.
    """
    try:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return {"available": False, "message": "Korelasyon için en az 2 sayısal sütun gerekir."}
        pearson = numeric.corr(method="pearson")
        spearman = numeric.corr(method="spearman")
        cov = numeric.cov()
        return {
            "available": True,
            "pearson": pearson.round(4).replace({np.nan: None}).to_dict(),
            "spearman": spearman.round(4).replace({np.nan: None}).to_dict(),
            "covariance": cov.round(4).replace({np.nan: None}).to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Correlation analysis failed")
        return {"available": False, "error": str(exc)}


def simple_regression(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit a simple ordinary-least-squares style linear regression via numpy.

    Args:
        df: Input dataframe.
        target: Optional target column.

    Returns:
        Dict[str, Any]: Regression summary in plain language.
    """
    try:
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if numeric.shape[1] < 2 or numeric.empty:
            return {"available": False, "message": "Regresyon için yeterli sayısal veri yok."}

        y_col = target if target in numeric.columns else numeric.columns[-1]
        x_cols = [c for c in numeric.columns if c != y_col][:5]
        if not x_cols:
            return {"available": False, "message": "Bağımsız değişken bulunamadı."}

        y = numeric[y_col].to_numpy(dtype=float)
        x = numeric[x_cols].to_numpy(dtype=float)
        x_design = np.column_stack([np.ones(len(x)), x])
        coeffs, residuals, rank, _ = np.linalg.lstsq(x_design, y, rcond=None)
        y_hat = x_design @ coeffs
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        return {
            "available": True,
            "target": str(y_col),
            "features": [str(c) for c in x_cols],
            "coefficients": {
                "intercept": to_serializable(coeffs[0]),
                **{str(c): to_serializable(coeffs[i + 1]) for i, c in enumerate(x_cols)},
            },
            "r2": to_serializable(r2),
            "explanation": (
                f"{y_col} değişkeni seçilen özelliklerle modellendi. "
                f"R²={r2:.3f}. Bu keşifsel bir doğrusal ilişkidir; nedensellik iddiası değildir."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Regression failed")
        return {"available": False, "error": str(exc)}


def hypothesis_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run basic hypothesis tests when suitable columns exist.

    Args:
        df: Input dataframe.

    Returns:
        Dict[str, Any]: Test results.
    """
    results: Dict[str, Any] = {"t_tests": [], "anova": []}
    try:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [
            c
            for c in df.columns
            if c not in numeric and df[c].nunique(dropna=True) <= 8
        ]
        if len(numeric) >= 2:
            a, b = numeric[0], numeric[1]
            s1, s2 = df[a].dropna(), df[b].dropna()
            if len(s1) > 5 and len(s2) > 5:
                stat, p_value = stats.ttest_ind(s1, s2, equal_var=False)
                results["t_tests"].append(
                    {
                        "a": str(a),
                        "b": str(b),
                        "statistic": to_serializable(stat),
                        "p_value": to_serializable(p_value),
                        "explanation": (
                            "Ortalamalar arasında anlamlı fark olabilir."
                            if p_value < 0.05
                            else "Ortalamalar arasında anlamlı fark bulunamadı."
                        ),
                    }
                )

        if categorical and numeric:
            cat = categorical[0]
            num = numeric[0]
            groups = [
                g[num].dropna().to_numpy()
                for _, g in df[[cat, num]].dropna().groupby(cat)
                if len(g) >= 3
            ]
            if len(groups) >= 2:
                stat, p_value = stats.f_oneway(*groups)
                results["anova"].append(
                    {
                        "group": str(cat),
                        "value": str(num),
                        "statistic": to_serializable(stat),
                        "p_value": to_serializable(p_value),
                        "explanation": (
                            f"{cat} grupları arasında {num} ortalaması anlamlı farklı olabilir."
                            if p_value < 0.05
                            else f"{cat} grupları arasında anlamlı ortalama farkı yok."
                        ),
                    }
                )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.exception("Hypothesis tests failed")
        return {"t_tests": [], "anova": [], "error": str(exc)}


def confidence_intervals(
    df: pd.DataFrame,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute mean confidence intervals for numeric columns.

    Args:
        df: Input dataframe.
        confidence: Confidence level.

    Returns:
        Dict[str, Any]: Intervals per column.
    """
    output: Dict[str, Any] = {}
    try:
        alpha = 1 - confidence
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            mean = float(series.mean())
            sem = float(stats.sem(series))
            ci = stats.t.interval(confidence, len(series) - 1, loc=mean, scale=sem)
            output[str(col)] = {
                "mean": mean,
                "lower": to_serializable(ci[0]),
                "upper": to_serializable(ci[1]),
                "confidence": confidence,
            }
        return {"level": confidence, "intervals": output}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Confidence intervals failed")
        return {"intervals": {}, "error": str(exc)}


def trend_and_stationarity(
    series: pd.Series,
) -> Dict[str, Any]:
    """Assess trend and Augmented Dickey-Fuller stationarity when possible.

    Büyük serilerde (örn. yüz binlerce satırlık ham işlem verisi)
    performans ve bellek sorunlarını önlemek için:

    * Seri, belirli bir üst sınırın (MAX_POINTS_FOR_TREND) üzerindeyse
      zaman sırası korunarak eşit aralıklarla downsample edilir.
    * ADF (Augmented Dickey-Fuller) testinde `autolag` KAPALI tutulur
      ve `maxlag` sabit/makul bir üst sınıra çekilir. Aksi halde
      statsmodels'in varsayılan davranışı (`maxlag ≈ 12*(n/100)^0.25`,
      `autolag='AIC'`) büyük n değerlerinde onlarca ayrı OLS
      regresyonu kurar; bu da işlemi dakikalarca sürebilecek hale
      getirir ve düşük kaynaklı ortamlarda (örn. Streamlit Community
      Cloud) uygulamanın donmasına/çökmesine yol açar.

    Args:
        series: Time-ordered numeric series.

    Returns:
        Dict[str, Any]: Trend / stationarity summary.
    """
    try:
        clean = series.dropna().astype(float)
        if len(clean) < 10:
            return {"available": False, "message": "Trend analizi için yeterli nokta yok."}

        original_length = len(clean)
        MAX_POINTS_FOR_TREND = 5000
        if original_length > MAX_POINTS_FOR_TREND:
            step = original_length // MAX_POINTS_FOR_TREND
            clean = clean.iloc[::step]
            logger.info(
                "Seri %d noktadan %d noktaya düşürüldü (performans amaçlı).",
                original_length,
                len(clean),
            )

        x = np.arange(len(clean))
        slope, intercept, r_value, p_value, _ = stats.linregress(x, clean.to_numpy())
        trend = "yukarı" if slope > 0 else "aşağı" if slope < 0 else "yatay"

        adf_result: Dict[str, Any] = {"available": False}
        try:
            from statsmodels.tsa.stattools import adfuller

            # maxlag sabit ve makul bir üst sınıra çekiliyor; autolag
            # kapatılıyor ki statsmodels onlarca regresyon denemesin.
            safe_maxlag = min(20, max(1, len(clean) // 20))
            adf_stat, adf_p, *_ = adfuller(clean, maxlag=safe_maxlag, autolag=None)
            adf_result = {
                "available": True,
                "adf_stat": to_serializable(adf_stat),
                "p_value": to_serializable(adf_p),
                "is_stationary": bool(adf_p < 0.05),
                "explanation": (
                    "Seri durağan görünüyor."
                    if adf_p < 0.05
                    else "Seri durağan değil; fark alma / dönüşüm gerekebilir."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("ADF test unavailable: %s", exc)
            adf_result = {"available": False, "error": str(exc)}

        return {
            "available": True,
            "trend_direction": trend,
            "slope": to_serializable(slope),
            "r_value": to_serializable(r_value),
            "p_value": to_serializable(p_value),
            "stationarity": adf_result,
            "explanation": f"Zaman içinde genel eğilim {trend} yönünde.",
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Trend analysis failed")
        return {"available": False, "error": str(exc)}
