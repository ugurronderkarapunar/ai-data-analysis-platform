"""Optional FastAPI integration layer for future cloud deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.orchestrator import OrchestratorAgent
from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("api")

app = FastAPI(
    title="Aether Analytics API",
    description="Multi-agent AI data analysis platform API surface",
    version="1.0.0",
)
orchestrator = OrchestratorAgent()


class AnalyzeRequest(BaseModel):
    """Request body for path-based analysis."""

    file_path: str = Field(..., description="Server-side path to CSV/Excel file")
    user_query: str = Field("", description="Natural language analysis request")
    run_forecast: bool = True


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe.

    Returns:
        Dict[str, str]: Status payload.
    """
    return {"status": "ok"}


@app.get("/agents")
def list_agents() -> Dict[str, Any]:
    """List registered agents.

    Returns:
        Dict[str, Any]: Orchestrator status.
    """
    return orchestrator.status()


@app.post("/analyze/path")
def analyze_path(payload: AnalyzeRequest) -> Dict[str, Any]:
    """Run end-to-end analysis for a local file path.

    Args:
        payload: Analysis request.

    Returns:
        Dict[str, Any]: Serializable analysis summary.

    Raises:
        HTTPException: On validation or processing errors.
    """
    path = Path(payload.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        context = orchestrator.run_end_to_end(
            path,
            user_query=payload.user_query,
            run_forecast=payload.run_forecast,
        )
        return _serialize_context(context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("API analyze_path failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    user_query: str = "",
    run_forecast: bool = True,
) -> Dict[str, Any]:
    """Run analysis for an uploaded file.

    Args:
        file: Uploaded CSV/Excel.
        user_query: Natural language query.
        run_forecast: Whether to attempt forecasting.

    Returns:
        Dict[str, Any]: Serializable analysis summary.

    Raises:
        HTTPException: On processing errors.
    """
    try:
        context = orchestrator.run_end_to_end(
            file,
            user_query=user_query,
            run_forecast=run_forecast,
        )
        return _serialize_context(context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("API analyze_upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _serialize_context(context: Any) -> Dict[str, Any]:
    """Convert AgentContext into a JSON-friendly summary.

    Args:
        context: Agent context.

    Returns:
        Dict[str, Any]: Summary payload.
    """
    return {
        "file_name": context.file_name,
        "shape": None
        if context.dataframe is None
        else list(context.dataframe.shape),
        "intent": context.intent,
        "tasks": context.tasks,
        "profile": context.profile,
        "statistics_plain": (context.statistics or {}).get("plain_language"),
        "forecast": {
            "available": (context.forecast or {}).get("available"),
            "selected_model": (context.forecast or {}).get("selected_model"),
            "explanation": (context.forecast or {}).get("explanation"),
        },
        "ml_advice": context.ml_advice,
        "dashboard_title": (context.dashboard_spec or {}).get("title"),
        "report_markdown": (context.report or {}).get("markdown"),
        "messages": context.messages,
        "errors": context.errors,
    }
