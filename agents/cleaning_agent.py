"""Data cleaning advisor agent (does not mutate without approval)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from utils.exceptions import ValidationError


class DataCleaningAgent(BaseAgent):
    """Analyze cleaning needs and optionally apply an approved plan."""

    name = "data_cleaner"
    description = "Eksik veri, duplicate ve outlier için temizleme planı üretir"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Build a cleaning plan; apply only when ``apply=True``.

        Args:
            context: Shared context.
            **kwargs: ``apply`` (bool), ``operations`` (list of op ids).

        Returns:
            AgentContext: Context with cleaning_plan and optionally updated df.

        Raises:
            ValidationError: If dataframe is missing.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("Temizleme analizi için dataframe gerekli.")

        apply = bool(kwargs.get("apply", False))
        selected_ops = kwargs.get("operations")

        try:
            plan = self._build_plan(df)
            context.cleaning_plan = plan
            context.add_message(
                "Temizleme planı hazır. Onayınız olmadan veri değiştirilmez."
            )
            self.logger.info(
                "Cleaning plan built with %s recommendations",
                len(plan.get("recommendations", [])),
            )

            if apply:
                ops = selected_ops or [
                    r["id"] for r in plan["recommendations"] if r.get("safe_default")
                ]
                cleaned, applied = self._apply_plan(df, ops)
                context.dataframe = cleaned
                context.cleaning_plan["applied_operations"] = applied
                context.cleaning_plan["applied"] = True
                context.add_message(
                    f"Onaylanan temizleme uygulandı: {', '.join(applied) or 'yok'}"
                )
                self.logger.info("Applied cleaning operations: %s", applied)
            else:
                context.cleaning_plan["applied"] = False
            return context
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Cleaning agent failed")
            raise ValidationError(f"Temizleme planı oluşturulamadı: {exc}") from exc

    def _build_plan(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect dataframe quality issues.

        Args:
            df: Input dataframe.

        Returns:
            Dict[str, Any]: Structured cleaning plan.
        """
        recommendations: List[Dict[str, Any]] = []
        missing = df.isna().sum()
        missing_total = int(missing.sum())
        dupes = int(df.duplicated().sum())

        if missing_total > 0:
            recommendations.append(
                {
                    "id": "drop_high_missing_columns",
                    "title": "Yüksek eksik oranlı sütunları kaldır",
                    "detail": "Eksik oranı %60 üzerinde olan sütunlar adaydır.",
                    "safe_default": False,
                    "impact": {
                        "columns": [
                            str(c)
                            for c, v in missing.items()
                            if v / max(len(df), 1) >= 0.6
                        ]
                    },
                }
            )
            recommendations.append(
                {
                    "id": "impute_numeric_median",
                    "title": "Sayısal eksikleri medyan ile doldur",
                    "detail": "Sayısal sütunlardaki NaN değerler medyan ile doldurulur.",
                    "safe_default": False,
                }
            )
            recommendations.append(
                {
                    "id": "impute_categorical_mode",
                    "title": "Kategorik eksikleri mod ile doldur",
                    "detail": "Kategorik sütunlardaki NaN değerler en sık değer ile doldurulur.",
                    "safe_default": False,
                }
            )

        if dupes > 0:
            recommendations.append(
                {
                    "id": "drop_duplicates",
                    "title": "Yinelenen satırları kaldır",
                    "detail": f"{dupes} yinelenen satır tespit edildi.",
                    "safe_default": False,
                    "impact": {"duplicate_rows": dupes},
                }
            )

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            recommendations.append(
                {
                    "id": "winsorize_outliers",
                    "title": "Aykırı değerleri IQR sınırına çek (winsorize)",
                    "detail": "Aykırı değerler silinmez; IQR alt/üst sınırına kırpılır.",
                    "safe_default": False,
                }
            )
            recommendations.append(
                {
                    "id": "scale_standard",
                    "title": "Sayısal özellikleri standartlaştır",
                    "detail": "Z-score scaling önerilir (model eğitimi öncesi).",
                    "safe_default": False,
                }
            )

        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            recommendations.append(
                {
                    "id": "encode_onehot",
                    "title": "Kategorik alanları one-hot encode et",
                    "detail": "Düşük kardinaliteli kategoriler için uygundur.",
                    "safe_default": False,
                }
            )

        recommendations.append(
            {
                "id": "feature_datetime_parts",
                "title": "Tarih alanlarından yıl/ay/gün özellikleri üret",
                "detail": "Tarih sütunları varsa feature engineering önerilir.",
                "safe_default": False,
            }
        )

        return {
            "missing_total": missing_total,
            "duplicate_rows": dupes,
            "validation": {
                "empty_dataframe": df.empty,
                "all_null_columns": [
                    str(c) for c in df.columns if df[c].isna().all()
                ],
            },
            "recommendations": recommendations,
            "note": "Kullanıcı onayı olmadan hiçbir dönüşüm uygulanmaz.",
        }

    def _apply_plan(
        self,
        df: pd.DataFrame,
        operations: List[str],
    ) -> tuple[pd.DataFrame, List[str]]:
        """Apply selected cleaning operations.

        Args:
            df: Input dataframe.
            operations: Operation identifiers.

        Returns:
            tuple: Cleaned dataframe and list of applied operation ids.
        """
        cleaned = df.copy()
        applied: List[str] = []
        try:
            if "drop_high_missing_columns" in operations:
                keep = [
                    c
                    for c in cleaned.columns
                    if cleaned[c].isna().mean() < 0.6
                ]
                cleaned = cleaned[keep]
                applied.append("drop_high_missing_columns")

            if "drop_duplicates" in operations:
                cleaned = cleaned.drop_duplicates()
                applied.append("drop_duplicates")

            if "impute_numeric_median" in operations:
                nums = cleaned.select_dtypes(include=[np.number]).columns
                for col in nums:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                applied.append("impute_numeric_median")

            if "impute_categorical_mode" in operations:
                cats = cleaned.select_dtypes(exclude=[np.number]).columns
                for col in cats:
                    mode = cleaned[col].mode(dropna=True)
                    if not mode.empty:
                        cleaned[col] = cleaned[col].fillna(mode.iloc[0])
                applied.append("impute_categorical_mode")

            if "winsorize_outliers" in operations:
                for col in cleaned.select_dtypes(include=[np.number]).columns:
                    series = cleaned[col]
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    cleaned[col] = series.clip(lower, upper)
                applied.append("winsorize_outliers")

            return cleaned, applied
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Failed applying cleaning plan")
            raise ValidationError(f"Temizleme uygulanamadı: {exc}") from exc
