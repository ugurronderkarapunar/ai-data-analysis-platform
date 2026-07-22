"""Orchestrator that coordinates independent agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from core.context import AgentContext
from core.registry import AgentRegistry, build_default_registry
from utils.exceptions import AgentExecutionError, ValidationError
from utils.logging_config import get_logger

logger = get_logger("orchestrator")


class OrchestratorAgent:
    """Central coordinator for the multi-agent analysis pipeline.

    Agents remain independent; the orchestrator only sequences calls and
    shares :class:`AgentContext`.
    """

    name: str = "orchestrator"
    description: str = "Coordinates all analysis agents"

    def __init__(self, registry: Optional[AgentRegistry] = None) -> None:
        """Initialize orchestrator with an agent registry.

        Args:
            registry: Optional custom registry. Defaults to built-in agents.
        """
        self.registry = registry or build_default_registry()
        self.logger = get_logger(self.name)

    def create_context(self) -> AgentContext:
        """Create an empty shared context.

        Returns:
            AgentContext: Fresh context instance.
        """
        return AgentContext()

    def run_agent(
        self,
        agent_name: str,
        context: AgentContext,
        **kwargs: Any,
    ) -> AgentContext:
        """Run a single registered agent.

        Args:
            agent_name: Registered agent name.
            context: Shared context.
            **kwargs: Agent-specific options.

        Returns:
            AgentContext: Updated context.

        Raises:
            ValidationError: If the agent is unknown.
            AgentExecutionError: If ``raise_on_error`` is True and agent fails.
        """
        self.logger.info("Orchestrator dispatching agent: %s", agent_name)
        agent = self.registry.get(agent_name)
        return agent.safe_run(context, **kwargs)

    def load_data(
        self,
        source: Union[str, Path, Any],
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentContext:
        """Load data via the data loader agent.

        Args:
            source: File path or Streamlit uploaded file-like object.
            context: Optional existing context.
            **kwargs: Loader options (sheet_name, encoding, etc.).

        Returns:
            AgentContext: Context with loaded dataframe.
        """
        ctx = context or self.create_context()
        return self.run_agent("data_loader", ctx, source=source, **kwargs)

    def analyze(
        self,
        context: AgentContext,
        user_query: str = "",
        run_forecast: bool = True,
        raise_on_error: bool = False,
    ) -> AgentContext:
        """Run the full analysis pipeline on an already-loaded context.

        Args:
            context: Context containing a dataframe.
            user_query: Natural language user request.
            run_forecast: Whether to attempt forecasting when suitable.
            raise_on_error: Propagate agent failures.

        Returns:
            AgentContext: Fully enriched context.

        Raises:
            ValidationError: If no dataframe is present.
            AgentExecutionError: On critical failure when raise_on_error=True.
        """
        if context.dataframe is None:
            raise ValidationError("No dataframe loaded. Call load_data first.")

        pipeline: List[str] = [
            "data_profiler",
            "intent_understanding",
            "data_cleaner",
            "statistics",
            "ml_advisor",
            "visualization_advisor",
            "dashboard_builder",
        ]

        self.logger.info("Starting analysis pipeline for query=%r", user_query)
        try:
            context = self.run_agent(
                "intent_understanding",
                context,
                user_query=user_query,
                raise_on_error=raise_on_error,
            )
            # Profiling should run even without a query.
            context = self.run_agent(
                "data_profiler",
                context,
                raise_on_error=raise_on_error,
            )
            context = self.run_agent(
                "data_cleaner",
                context,
                raise_on_error=raise_on_error,
            )
            context = self.run_agent(
                "statistics",
                context,
                raise_on_error=raise_on_error,
            )
            context = self.run_agent(
                "ml_advisor",
                context,
                raise_on_error=raise_on_error,
            )
            context = self.run_agent(
                "visualization_advisor",
                context,
                raise_on_error=raise_on_error,
            )

            if run_forecast:
                context = self.run_agent(
                    "forecast",
                    context,
                    raise_on_error=False,
                )

            context = self.run_agent(
                "dashboard_builder",
                context,
                raise_on_error=raise_on_error,
            )
            context = self.run_agent(
                "reporting",
                context,
                user_query=user_query,
                raise_on_error=raise_on_error,
            )
            context.add_message("Analiz pipeline'ı tamamlandı.")
            self.logger.info("Analysis pipeline completed")
            return context
        except (ValidationError, AgentExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001
            message = f"Orchestrator pipeline failed: {exc}"
            self.logger.exception(message)
            context.add_error(message)
            if raise_on_error:
                raise AgentExecutionError(message) from exc
            return context

    def run_end_to_end(
        self,
        source: Union[str, Path, Any],
        user_query: str = "",
        **kwargs: Any,
    ) -> AgentContext:
        """Load data and run the full analysis pipeline.

        Args:
            source: Data source.
            user_query: Natural language request.
            **kwargs: Forwarded to load_data / analyze.

        Returns:
            AgentContext: Completed analysis context.
        """
        load_kwargs = {
            k: kwargs.pop(k)
            for k in list(kwargs.keys())
            if k in {"sheet_name", "encoding", "delimiter"}
        }
        context = self.load_data(source, **load_kwargs)
        return self.analyze(context, user_query=user_query, **kwargs)

    def status(self) -> Dict[str, Any]:
        """Return orchestrator / registry status.

        Returns:
            Dict[str, Any]: Status payload.
        """
        return {
            "orchestrator": self.name,
            "agents": self.registry.list_agents(),
        }
