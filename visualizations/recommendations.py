"""Chart recommendation rules."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.context import AgentContext
from utils.logging_config import get_logger

logger = get_logger("viz_recommendations")


def recommend_charts(df: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
    """Recommend chart types based on data shape and column types.

    Args:
        df: Input dataframe.
        context: Shared context for metadata.

    Returns:
        Dict[str, Any]: Recommendations and anti-patterns.
    """
    try:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [c for c in df.columns if c not in numeric]
        datetime_cols = context.metadata.get("datetime_columns") or context.profile.get(
            "datetime_columns", []
        )
        recommendations: List[Dict[str, Any]] = []
        avoid: List[Dict[str, str]] = []

        if datetime_cols and numeric:
            recommendations.append(
                {
                    "chart": "line",
                    "title": "Trend çizgi grafiği",
                    "x": str(datetime_cols[0]),
                    "y": str(numeric[0]),
                    "why": "Zaman serisi değişimini en doğru line chart gösterir.",
                }
            )
            recommendations.append(
                {
                    "chart": "area",
                    "title": "Alan trend grafiği",
                    "x": str(datetime_cols[0]),
                    "y": str(numeric[0]),
                    "why": "Kümülatif/hacim hissi için alan grafiği uygundur.",
                }
            )

        if categorical and numeric:
            cat = categorical[0]
            recommendations.append(
                {
                    "chart": "bar",
                    "title": "Kategori bazlı bar grafik",
                    "x": str(cat),
                    "y": str(numeric[0]),
                    "why": "Kategoriler arası büyüklük karşılaştırması için bar en güvenlisidir.",
                }
            )
            if df[cat].nunique(dropna=True) <= 8:
                recommendations.append(
                    {
                        "chart": "pie",
                        "title": "Pay dilimi (pie)",
                        "names": str(cat),
                        "values": str(numeric[0]),
                        "why": "Az kategoride oran göstermek için pie kullanılabilir.",
                    }
                )
            else:
                avoid.append(
                    {
                        "chart": "pie",
                        "why": f"'{cat}' çok fazla kategori içeriyor; pie okunaksız olur. Bar tercih edin.",
                    }
                )

        if len(numeric) >= 2:
            recommendations.append(
                {
                    "chart": "scatter",
                    "title": "Saçılım grafiği",
                    "x": str(numeric[0]),
                    "y": str(numeric[1]),
                    "why": "İki sayısal değişken ilişkisini scatter net gösterir.",
                }
            )
            recommendations.append(
                {
                    "chart": "heatmap",
                    "title": "Korelasyon ısı haritası",
                    "why": "Çoklu sayısal korelasyonları heatmap ile okumak kolaydır.",
                }
            )

        if numeric:
            recommendations.append(
                {
                    "chart": "histogram",
                    "title": "Dağılım histogramı",
                    "x": str(numeric[0]),
                    "why": "Tek değişken dağılımı için histogram uygundur.",
                }
            )

        if not recommendations:
            recommendations.append(
                {
                    "chart": "table",
                    "title": "Veri tablosu",
                    "why": "Grafik için uygun tip bulunamadı; tablo ile başlayın.",
                }
            )

        avoid.append(
            {
                "chart": "3d_pie",
                "why": "3D pie algıyı bozar; kurumsal raporlarda önerilmez.",
            }
        )

        return {
            "recommendations": recommendations,
            "avoid": avoid,
            "notes": [
                "Yanlış grafik kullanımı karar kalitesini düşürür; öneriler veri tipine göre seçildi.",
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chart recommendation failed")
        return {"recommendations": [], "avoid": [], "error": str(exc)}
