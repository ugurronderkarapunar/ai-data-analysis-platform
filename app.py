"""Aether Analytics — Multi-Agent AI Data Analysis Platform (Streamlit entrypoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when launched via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.reporting_agent import export_pdf_from_markdown, export_pptx_from_report
from config.settings import get_settings
from core.orchestrator import OrchestratorAgent
from dashboard.components import (
    apply_filters,
    dataframe_to_excel_bytes,
    render_charts,
    render_kpis,
    render_map,
)
from dashboard.theme import apply_theme, render_hero
from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("app")


def _init_state() -> None:
    """Initialize Streamlit session state defaults."""
    defaults = {
        "context": None,
        "orchestrator": None,
        "dark_mode": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_orchestrator() -> OrchestratorAgent:
    """Return a cached orchestrator instance.

    Returns:
        OrchestratorAgent: Shared orchestrator.
    """
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = OrchestratorAgent()
        logger.info("Orchestrator initialized for Streamlit session")
    return st.session_state.orchestrator


def main() -> None:
    """Run the Streamlit application."""
    settings = get_settings()
    st.set_page_config(
        page_title="Aether Analytics",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    with st.sidebar:
        st.markdown("### Aether Analytics")
        st.caption("Kurumsal Multi-Agent Veri Analiz Platformu")
        st.session_state.dark_mode = st.toggle(
            "Dark Mode",
            value=st.session_state.dark_mode,
        )
        uploaded = st.file_uploader(
            "CSV veya Excel yükleyin",
            type=["csv", "xlsx", "xls", "xlsm", "tsv", "txt"],
        )
        user_query = st.text_area(
            "Ne yapmak istiyorsunuz?",
            placeholder="Örn: Aylık ciroyu görmek istiyorum / Satış analizi yap",
            height=100,
        )
        run_forecast = st.checkbox("Uygunsa otomatik tahmin üret", value=True)
        analyze_clicked = st.button("Analizi Başlat", use_container_width=True)

    apply_theme(dark_mode=st.session_state.dark_mode)
    render_hero(
        title="Kurumsal ölçeklenebilir yapay zekâ veri analiz platformu",
        subtitle=(
            "Veriyi yükleyin, niyetinizi yazın; profil, istatistik, dashboard, "
            "tahmin ve rapor ajanları birlikte çalışsın."
        ),
    )

    orchestrator = _get_orchestrator()

    if analyze_clicked:
        if uploaded is None:
            st.warning("Lütfen önce bir veri dosyası yükleyin.")
        else:
            with st.spinner("Ajanlar çalışıyor..."):
                try:
                    context = orchestrator.load_data(uploaded)
                    context = orchestrator.analyze(
                        context,
                        user_query=user_query,
                        run_forecast=run_forecast,
                    )
                    st.session_state.context = context
                    logger.info("Analysis finished for %s", uploaded.name)
                    st.success("Analiz tamamlandı.")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("UI analysis failed")
                    st.error(f"Analiz başarısız: {exc}")

    context = st.session_state.context
    if context is None or context.dataframe is None:
        st.info(
            "Başlamak için sol panelden dosya yükleyip **Analizi Başlat** düğmesine basın. "
            f"Örnek veri: `{settings.data_dir / 'sample_sales.csv'}`"
        )
        with st.expander("Kayıtlı ajanlar"):
            st.json(orchestrator.status())
        return

    # Messages / errors
    if context.messages:
        for msg in context.messages[-8:]:
            st.caption(f"• {msg}")
    if context.errors:
        for err in context.errors:
            st.warning(err)

    spec = context.dashboard_spec or {}
    filtered = apply_filters(context.dataframe, spec.get("filters") or [])

    st.markdown(f"### {spec.get('title', 'Dashboard')}")
    render_kpis(spec.get("kpis") or [])

    tab_dash, tab_profile, tab_stats, tab_forecast, tab_ml, tab_clean, tab_report = st.tabs(
        [
            "Dashboard",
            "Profil",
            "İstatistik",
            "Tahmin",
            "ML Danışman",
            "Temizleme",
            "Rapor",
        ]
    )

    with tab_dash:
        render_charts(filtered, spec.get("charts") or [])
        render_map(filtered, spec.get("maps") or [])
        st.subheader("Tablo")
        st.dataframe(filtered.head(200), use_container_width=True)
        try:
            st.download_button(
                "Excel indir",
                data=dataframe_to_excel_bytes(filtered),
                file_name="aether_filtered.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(str(exc))

    with tab_profile:
        st.json(context.profile)

    with tab_stats:
        for note in (context.statistics or {}).get("plain_language", []):
            st.write(f"- {note}")
        st.json(context.statistics)

    with tab_forecast:
        forecast = context.forecast or {}
        if forecast.get("available"):
            st.write(forecast.get("explanation"))
            st.write(
                f"Seçilen model: **{forecast.get('selected_model')}** | "
                f"Sıklık: **{forecast.get('frequency')}**"
            )
            st.json(
                {
                    "metrics": forecast.get("metrics"),
                    "candidates": forecast.get("candidates"),
                    "horizons": {
                        k: v[:5] for k, v in (forecast.get("horizons") or {}).items()
                    },
                }
            )
            for horizon, points in (forecast.get("horizons") or {}).items():
                if not points:
                    continue
                hdf = pd.DataFrame(points)
                hdf["date"] = pd.to_datetime(hdf["date"])
                st.line_chart(hdf.set_index("date")["value"], height=220)
                st.caption(f"Ufuk: {horizon}")
        else:
            st.info(forecast.get("explanation") or forecast.get("reason") or "Tahmin yok.")

    with tab_ml:
        st.info("Model eğitimi yapılmaz; yalnızca öneri üretilir.")
        st.json(context.ml_advice)

    with tab_clean:
        plan = context.cleaning_plan or {}
        st.write(plan.get("note", ""))
        for rec in plan.get("recommendations", []):
            st.markdown(f"**{rec.get('title')}** — {rec.get('detail')}")
        apply_ops = st.multiselect(
            "Uygulanacak işlemler (onay gerekir)",
            options=[r["id"] for r in plan.get("recommendations", [])],
        )
        if st.button("Seçili temizlemeyi uygula"):
            try:
                updated = orchestrator.run_agent(
                    "data_cleaner",
                    context,
                    apply=True,
                    operations=apply_ops,
                )
                updated = orchestrator.run_agent("data_profiler", updated)
                updated = orchestrator.run_agent("dashboard_builder", updated)
                st.session_state.context = updated
                st.success("Temizleme uygulandı ve profil yenilendi.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Cleaning apply failed")
                st.error(str(exc))

    with tab_report:
        report = context.report or {}
        st.markdown(report.get("markdown") or "_Rapor henüz yok._")
        if report.get("markdown"):
            st.download_button(
                "Markdown indir",
                data=report["markdown"],
                file_name="aether_report.md",
                mime="text/markdown",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("PDF üret"):
                    try:
                        out = settings.outputs_dir / "report_export.pdf"
                        export_pdf_from_markdown(report["markdown"], out)
                        st.success(f"PDF hazır: {out}")
                        st.download_button(
                            "PDF indir",
                            data=out.read_bytes(),
                            file_name="aether_report.pdf",
                            mime="application/pdf",
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(str(exc))
            with col_b:
                if st.button("PowerPoint üret"):
                    try:
                        out = settings.outputs_dir / "report_export.pptx"
                        export_pptx_from_report(report, out)
                        st.success(f"PPTX hazır: {out}")
                        st.download_button(
                            "PPTX indir",
                            data=out.read_bytes(),
                            file_name="aether_report.pptx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument"
                                ".presentationml.presentation"
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(str(exc))


if __name__ == "__main__":
    main()
