"""Streamlit rendering helpers for dashboard specs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.logging_config import get_logger

logger = get_logger("dashboard_render")


def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    """Render sidebar filters and return filtered dataframe.

    Args:
        df: Source dataframe.
        filters: Filter specifications.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    filtered = df.copy()
    try:
        st.sidebar.markdown("### Filtreler")
        for spec in filters:
            col = spec["column"]
            if col not in filtered.columns:
                continue
            if spec["type"] == "multiselect":
                options = spec.get("options") or sorted(
                    filtered[col].dropna().astype(str).unique().tolist()
                )
                selected = st.sidebar.multiselect(
                    f"{col}",
                    options=options,
                    default=[],
                    key=spec["id"],
                )
                if selected:
                    filtered = filtered[filtered[col].astype(str).isin(selected)]
            elif spec["type"] == "date_range":
                series = pd.to_datetime(filtered[col], errors="coerce")
                min_d = pd.to_datetime(spec.get("min")).date()
                max_d = pd.to_datetime(spec.get("max")).date()
                start, end = st.sidebar.date_input(
                    f"{col} aralığı",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                    key=spec["id"],
                )
                if isinstance(start, tuple) or not hasattr(start, "isoformat"):
                    continue
                mask = (series.dt.date >= start) & (series.dt.date <= end)
                filtered = filtered.loc[mask]
        return filtered
    except Exception as exc:  # noqa: BLE001
        logger.exception("Filter application failed")
        st.sidebar.warning(f"Filtre uygulanamadı: {exc}")
        return df


def render_kpis(kpis: List[Dict[str, Any]]) -> None:
    """Render KPI metric cards.

    Args:
        kpis: KPI specifications.
    """
    if not kpis:
        return
    try:
        cols = st.columns(min(4, len(kpis)))
        for i, kpi in enumerate(kpis):
            with cols[i % len(cols)]:
                value = kpi.get("value")
                if isinstance(value, float):
                    display = f"{value:,.2f}"
                else:
                    display = f"{value:,}" if isinstance(value, (int,)) else str(value)
                st.markdown(
                    f"""
                    <div class="kpi-card">
                      <div class="kpi-label">{kpi.get('label')}</div>
                      <div class="kpi-value">{display}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("KPI render failed")
        st.error(f"KPI render hatası: {exc}")


def render_charts(df: pd.DataFrame, charts: List[Dict[str, Any]]) -> None:
    """Render interactive Plotly charts from dashboard specs.

    Args:
        df: Filtered dataframe.
        charts: Chart specifications.
    """
    if df.empty:
        st.info("Filtrelenmiş veri boş; grafik çizilemedi.")
        return
    try:
        for chart in charts:
            chart_type = chart.get("type")
            title = chart.get("title") or chart_type
            enc = chart.get("encoding") or {}
            st.subheader(title)
            if chart.get("why"):
                st.caption(chart["why"])

            fig = None
            if chart_type in {"line", "area"} and enc.get("x") and enc.get("y"):
                plot_df = df[[enc["x"], enc["y"]]].dropna()
                if chart_type == "line":
                    fig = px.line(plot_df, x=enc["x"], y=enc["y"], template="plotly_dark")
                else:
                    fig = px.area(plot_df, x=enc["x"], y=enc["y"], template="plotly_dark")
            elif chart_type == "bar" and enc.get("x") and enc.get("y"):
                grouped = (
                    df.groupby(enc["x"], dropna=False)[enc["y"]]
                    .sum(numeric_only=True)
                    .reset_index()
                    .sort_values(enc["y"], ascending=False)
                    .head(25)
                )
                fig = px.bar(grouped, x=enc["x"], y=enc["y"], template="plotly_dark")
            elif chart_type == "pie" and enc.get("names") and enc.get("values"):
                grouped = (
                    df.groupby(enc["names"], dropna=False)[enc["values"]]
                    .sum(numeric_only=True)
                    .reset_index()
                    .head(8)
                )
                fig = px.pie(
                    grouped,
                    names=enc["names"],
                    values=enc["values"],
                    template="plotly_dark",
                )
            elif chart_type == "scatter" and enc.get("x") and enc.get("y"):
                fig = px.scatter(
                    df,
                    x=enc["x"],
                    y=enc["y"],
                    template="plotly_dark",
                    opacity=0.75,
                )
            elif chart_type == "histogram" and enc.get("x"):
                fig = px.histogram(df, x=enc["x"], template="plotly_dark")
            elif chart_type == "heatmap":
                numeric = df.select_dtypes(include=[np.number])
                if numeric.shape[1] >= 2:
                    corr = numeric.corr(numeric_only=True)
                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="Teal",
                        template="plotly_dark",
                    )
            if fig is not None:
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "table":
                st.dataframe(df.head(100), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chart render failed")
        st.error(f"Grafik render hatası: {exc}")


def render_map(df: pd.DataFrame, maps: List[Dict[str, Any]]) -> None:
    """Render map charts when lat/lon exist.

    Args:
        df: Dataframe.
        maps: Map specifications.
    """
    try:
        for spec in maps:
            lat, lon = spec.get("lat"), spec.get("lon")
            if lat in df.columns and lon in df.columns:
                st.subheader(spec.get("title", "Harita"))
                st.map(df[[lat, lon]].dropna().rename(columns={lat: "lat", lon: "lon"}))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Map render failed: %s", exc)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialize dataframe to Excel bytes for download.

    Args:
        df: Dataframe.

    Returns:
        bytes: XLSX payload.
    """
    from io import BytesIO

    buffer = BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Excel export failed")
        raise RuntimeError(f"Excel indirme hazırlanamadı: {exc}") from exc
