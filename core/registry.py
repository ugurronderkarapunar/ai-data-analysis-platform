"""Agent registry enabling plug-and-play extensibility."""

from __future__ import annotations

from typing import Dict, List, Type

from core.base_agent import BaseAgent
from utils.exceptions import ValidationError
from utils.logging_config import get_logger

logger = get_logger("registry")


class AgentRegistry:
    """Registry that maps agent names to concrete agent classes.

    Future agents (SQL, OCR, RAG, MCP, etc.) can be registered without
    modifying orchestrator internals.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._agents: Dict[str, Type[BaseAgent]] = {}

    def register(self, agent_cls: Type[BaseAgent], *, replace: bool = True) -> None:
        """Register an agent class (idempotent for Streamlit reruns).

        Args:
            agent_cls: Concrete :class:`BaseAgent` subclass.
            replace: When True, overwrite an existing registration with the
                same name. When False, keep the existing registration.

        Raises:
            ValidationError: If the agent name is missing, or if ``replace`` is
                False and the name is already registered to a different class.
        """
        name = getattr(agent_cls, "name", None)
        if not name:
            raise ValidationError("Agent class must define a non-empty 'name'.")
        if name in self._agents:
            existing = self._agents[name]
            if existing is agent_cls or existing.__name__ == agent_cls.__name__:
                logger.debug("Agent already registered, skipping: %s", name)
                return
            if not replace:
                raise ValidationError(f"Agent already registered: {name}")
            logger.warning("Replacing registered agent: %s", name)
        self._agents[name] = agent_cls
        logger.debug("Registered agent: %s", name)

    def get(self, name: str) -> BaseAgent:
        """Instantiate a registered agent by name.

        Args:
            name: Registered agent name.

        Returns:
            BaseAgent: Fresh agent instance.

        Raises:
            ValidationError: If the agent is not registered.
        """
        if name not in self._agents:
            raise ValidationError(f"Unknown agent: {name}")
        try:
            return self._agents[name]()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to instantiate agent %s", name)
            raise ValidationError(f"Cannot create agent '{name}': {exc}") from exc

    def count(self) -> int:
        """Return the number of registered agents.

        Returns:
            int: Agent count.
        """
        return len(self._agents)

    def list_agents(self) -> List[Dict[str, str]]:
        """List registered agents without instantiating them.

        Returns:
            List[Dict[str, str]]: Agent metadata dictionaries.
        """
        return [
            {
                "name": str(getattr(cls, "name", cls.__name__)),
                "description": str(getattr(cls, "description", "")),
            }
            for cls in self._agents.values()
        ]

    def has(self, name: str) -> bool:
        """Check whether an agent is registered.

        Args:
            name: Agent name.

        Returns:
            bool: True if registered.
        """
        return name in self._agents


def build_default_registry() -> AgentRegistry:
    """Build the default registry with all built-in agents.

    Returns:
        AgentRegistry: Populated registry.
    """
    # Local imports avoid circular dependencies at module import time.
    from agents.cleaning_agent import DataCleaningAgent
    from agents.dashboard_agent import DashboardAgent
    from agents.data_loader_agent import DataLoaderAgent
    from agents.forecast_agent import ForecastAgent
    from agents.intent_agent import IntentUnderstandingAgent
    from agents.ml_advisor_agent import MLAdvisorAgent
    from agents.profiling_agent import DataProfilingAgent
    from agents.reporting_agent import ReportingAgent
    from agents.statistics_agent import StatisticsAgent
    from agents.visualization_advisor_agent import VisualizationAdvisorAgent

    registry = AgentRegistry()
    for agent_cls in (
        DataLoaderAgent,
        DataProfilingAgent,
        IntentUnderstandingAgent,
        DashboardAgent,
        ForecastAgent,
        StatisticsAgent,
        MLAdvisorAgent,
        DataCleaningAgent,
        VisualizationAdvisorAgent,
        ReportingAgent,
    ):
        registry.register(agent_cls)
    logger.info("Default registry ready (%s agents)", registry.count())
    return registry
