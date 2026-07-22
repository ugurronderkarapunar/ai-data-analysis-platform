"""Streamlit theme and layout helpers."""

from __future__ import annotations

import streamlit as st

from utils.logging_config import get_logger

logger = get_logger("dashboard_theme")


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-0: #0b1220;
  --bg-1: #121a2b;
  --bg-2: #1a2438;
  --line: rgba(148, 163, 184, 0.22);
  --text: #e8eef8;
  --muted: #9aa8c0;
  --accent: #2dd4bf;
  --accent-2: #38bdf8;
  --warn: #f59e0b;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(45, 212, 191, 0.16), transparent 55%),
    radial-gradient(900px 500px at 90% 0%, rgba(56, 189, 248, 0.12), transparent 50%),
    linear-gradient(180deg, var(--bg-0), #070b14 70%);
  color: var(--text);
}

.block-container {
  padding-top: 1.2rem;
  max-width: 1280px;
}

h1, h2, h3 {
  letter-spacing: -0.02em;
}

.hero-brand {
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.35rem;
}

.hero-title {
  font-size: clamp(1.8rem, 3vw, 2.6rem);
  font-weight: 700;
  line-height: 1.15;
  margin: 0 0 0.5rem 0;
}

.hero-sub {
  color: var(--muted);
  font-size: 1.02rem;
  max-width: 52ch;
  margin-bottom: 1.2rem;
}

.kpi-card {
  background: linear-gradient(180deg, rgba(26,36,56,0.95), rgba(18,26,43,0.95));
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  min-height: 96px;
}

.kpi-label {
  color: var(--muted);
  font-size: 0.82rem;
  margin-bottom: 0.35rem;
}

.kpi-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--text);
}

.section-rule {
  height: 1px;
  background: var(--line);
  margin: 0.4rem 0 1rem 0;
}

.stButton > button {
  border-radius: 10px;
  border: 1px solid rgba(45, 212, 191, 0.35);
  background: linear-gradient(135deg, rgba(45,212,191,0.18), rgba(56,189,248,0.12));
  color: var(--text);
  font-weight: 600;
}

div[data-testid="stMetric"] {
  background: rgba(18,26,43,0.8);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.6rem 0.8rem;
}
</style>
"""


def apply_theme(dark_mode: bool = True) -> None:
    """Inject custom CSS for the Streamlit app.

    Args:
        dark_mode: Whether dark styling is active.
    """
    try:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        if not dark_mode:
            st.markdown(
                """
                <style>
                :root {
                  --bg-0: #f5f7fb;
                  --bg-1: #ffffff;
                  --bg-2: #eef2f8;
                  --text: #0f172a;
                  --muted: #475569;
                }
                .stApp {
                  background:
                    radial-gradient(1000px 500px at 0% 0%, rgba(45,212,191,0.12), transparent 50%),
                    linear-gradient(180deg, #f8fafc, #eef2ff);
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        logger.debug("Theme applied dark_mode=%s", dark_mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Theme injection failed: %s", exc)


def render_hero(title: str, subtitle: str) -> None:
    """Render the first-viewport brand composition.

    Args:
        title: Hero headline.
        subtitle: Supporting sentence.
    """
    try:
        st.markdown(
            f"""
            <div class="hero-brand">Aether Analytics</div>
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
            <div class="section-rule"></div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Hero render failed: %s", exc)
        st.title(title)
        st.caption(subtitle)
