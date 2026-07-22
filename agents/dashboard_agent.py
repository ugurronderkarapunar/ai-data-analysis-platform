"""Dashboard specification agent for Streamlit rendering."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from utils.exceptions import ValidationError


class DashboardAgent(BaseAgent):
    """Build a Streamlit-compatible dashboard specification from data."""

    name = "dashboard_builder"
    description = "KPI, filtre ve grafik içeren dashboard spesifikasyonu üretir"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Create dashboard spec (cards, filters, charts, tables).

        Args:
            context: Shared context.
            **kwargs: Unused.

        Returns:
            AgentContext: Context with dashboard_spec.

        Raises:
            ValidationError: If dataframe is missing.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("Dashboard için dataframe gerekli.")

        try:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical = [c for c in df.columns if c not in numeric]
            datetime_cols = context.metadata.get("datetime_columns") or context.profile.get(
                "datetime_columns", []
            )
            intent = (context.intent or {}).get("primary_intent", "general_analysis")

            kpis = self._build_kpis(df, numeric, intent)
            filters = self._build_filters(df, categorical, datetime_cols)
            charts = self._build_charts(df, numeric, categorical, datetime_cols, context)
            tables = [
                {
                    "id": "preview",
                    "title": "Veri Önizleme",
                    "columns": [str(c) for c in df.columns[:20]],
                    "limit": 100,
                }
            ]

            context.dashboard_spec = {
                "title": self._title(intent, context.file_name),
                "theme": {"default_dark": True},
                "kpis": kpis,
                "filters": filters,
                "charts": charts,
                "tables": tables,
                "maps": self._maybe_map(df),
                "layout": {
                    "kpi_columns": min(4, max(1, len(kpis))),
                    "responsive": True,
                },
            }
            context.add_message(
                f"Dashboard spesifikasyonu hazır: {len(kpis)} KPI, {len(charts)} grafik."
            )
            self.logger.info("Dashboard spec generated")
            return context
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Dashboard agent failed")
            raise ValidationError(f"Dashboard oluşturulamadı: {exc}") from exc

    def _title(self, intent: str, file_name: str | None) -> str:
        """Build dashboard title.

        Args:
            intent: Primary intent.
            file_name: Source file name.

        Returns:
            str: Title text.
        """
        labels = {
            "sales_analysis": "Satış Analiz Dashboard",
            "hr_dashboard": "Personel Dashboard",
            "profit_loss": "Kâr-Zarar Dashboard",
            "document_tracking": "Belge Takip Dashboard",
            "forecast": "Tahmin Dashboard",
            "dashboard": "Yönetici Dashboard",
        }
        base = labels.get(intent, "Veri Analiz Dashboard")
        return f"{base} — {file_name}" if file_name else base

    def _build_kpis(
        self,
        df: pd.DataFrame,
        numeric: List[str],
        intent: str,
    ) -> List[Dict[str, Any]]:
        """Create KPI card definitions.

        Args:
            df: Dataframe.
            numeric: Numeric columns.
            intent: Primary intent.

        Returns:
            List[Dict[str, Any]]: KPI specs.
        """
        kpis: List[Dict[str, Any]] = [
            {
                "id": "row_count",
                "label": "Kayıt Sayısı",
                "value": int(len(df)),
                "format": "number",
            }
        ]
        for col in numeric[:3]:
            series = df[col].dropna()
            if series.empty:
                continue
            kpis.append(
                {
                    "id": f"sum_{col}",
                    "label": f"Toplam {col}",
                    "value": float(series.sum()),
                    "format": "number",
                }
            )
            kpis.append(
                {
                    "id": f"mean_{col}",
                    "label": f"Ortalama {col}",
                    "value": float(series.mean()),
                    "format": "number",
                }
            )
            if len(kpis) >= 6:
                break
        if intent == "profit_loss" and len(numeric) >= 2:
            kpis.append(
                {
                    "id": "margin_proxy",
                    "label": f"Fark ({numeric[0]} - {numeric[1]})",
                    "value": float(df[numeric[0]].fillna(0).sum() - df[numeric[1]].fillna(0).sum()),
                    "format": "number",
                }
            )
        return kpis[:8]

    def _build_filters(
        self,
        df: pd.DataFrame,
        categorical: List[str],
        datetime_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Create filter control definitions.

        Args:
            df: Dataframe.
            categorical: Categorical columns.
            datetime_cols: Datetime columns.

        Returns:
            List[Dict[str, Any]]: Filter specs.
        """
        filters: List[Dict[str, Any]] = []
        for col in categorical[:4]:
            values = [str(v) for v in df[col].dropna().astype(str).unique()[:50]]
            filters.append(
                {
                    "id": f"filter_{col}",
                    "column": str(col),
                    "type": "multiselect",
                    "options": values,
                }
            )
        for col in datetime_cols[:1]:
            series = pd.to_datetime(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            filters.append(
                {
                    "id": f"filter_{col}",
                    "column": str(col),
                    "type": "date_range",
                    "min": series.min().date().isoformat(),
                    "max": series.max().date().isoformat(),
                }
            )
        return filters

    def _build_charts(
        self,
        df: pd.DataFrame,
        numeric: List[str],
        categorical: List[str],
        datetime_cols: List[str],
        context: AgentContext,
    ) -> List[Dict[str, Any]]:
        """Create chart definitions, preferring viz advisor output.

        Args:
            df: Dataframe.
            numeric: Numeric columns.
            categorical: Categorical columns.
            datetime_cols: Datetime columns.
            context: Shared context.

        Returns:
            List[Dict[str, Any]]: Chart specs.
        """
        charts: List[Dict[str, Any]] = []
        advice = (context.viz_advice or {}).get("recommendations") or []
        for item in advice:
            charts.append(
                {
                    "id": f"chart_{item.get('chart')}_{len(charts)}",
                    "type": item.get("chart"),
                    "title": item.get("title"),
                    "encoding": {
                        k: item.get(k)
                        for k in ("x", "y", "names", "values")
                        if item.get(k)
                    },
                    "why": item.get("why"),
                }
            )

        if not charts:
            if datetime_cols and numeric:
                charts.append(
                    {
                        "id": "trend_line",
                        "type": "line",
                        "title": "Trend",
                        "encoding": {"x": str(datetime_cols[0]), "y": str(numeric[0])},
                    }
                )
            if categorical and numeric:
                charts.append(
                    {
                        "id": "category_bar",
                        "type": "bar",
                        "title": "Kategori Karşılaştırma",
                        "encoding": {"x": str(categorical[0]), "y": str(numeric[0])},
                    }
                )
            if len(numeric) >= 2:
                charts.append(
                    {
                        "id": "scatter",
                        "type": "scatter",
                        "title": "İlişki",
                        "encoding": {"x": str(numeric[0]), "y": str(numeric[1])},
                    }
                )
                charts.append(
                    {
                        "id": "corr_heatmap",
                        "type": "heatmap",
                        "title": "Korelasyon Heatmap",
                        "encoding": {},
                    }
                )
        return charts[:8]

    def _maybe_map(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect possible geo columns for map widgets.

        Args:
            df: Dataframe.

        Returns:
            List[Dict[str, Any]]: Map specs if lat/lon found.
        """
        cols = {str(c).lower(): c for c in df.columns}
        lat = next((cols[k] for k in cols if k in {"lat", "latitude", "enlem"}), None)
        lon = next((cols[k] for k in cols if k in {"lon", "lng", "longitude", "boylam"}), None)
        if lat and lon:
            return [
                {
                    "id": "geo_map",
                    "title": "Coğrafi Dağılım",
                    "lat": str(lat),
                    "lon": str(lon),
                }
            ]
        return []
