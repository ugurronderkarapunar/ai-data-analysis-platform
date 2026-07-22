"""Abstract base agent definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from core.context import AgentContext
from utils.exceptions import AgentExecutionError
from utils.logging_config import get_logger


class BaseAgent(ABC):
    """Single-responsibility agent contract.

    Every concrete agent must implement :meth:`run` and should only mutate
    the parts of :class:`AgentContext` it owns.
    """

    name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self) -> None:
        """Initialize agent logger."""
        self.logger = get_logger(self.name)

    @abstractmethod
    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Execute the agent against the shared context.

        Args:
            context: Shared pipeline context.
            **kwargs: Agent-specific options.

        Returns:
            AgentContext: Updated context.
        """

    def safe_run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Execute :meth:`run` with logging and standardized error handling.

        Args:
            context: Shared pipeline context.
            **kwargs: Agent-specific options.

        Returns:
            AgentContext: Updated context (errors recorded on failure).

        Raises:
            AgentExecutionError: When the agent fails critically and
                ``raise_on_error`` is True.
        """
        raise_on_error = bool(kwargs.pop("raise_on_error", False))
        self.logger.info("Starting agent '%s'", self.name)
        try:
            result = self.run(context, **kwargs)
            self.logger.info("Completed agent '%s'", self.name)
            return result
        except Exception as exc:  # noqa: BLE001
            message = f"Agent '{self.name}' failed: {exc}"
            self.logger.exception(message)
            context.add_error(message)
            if raise_on_error:
                raise AgentExecutionError(message) from exc
            return context

    def info(self) -> Dict[str, str]:
        """Return agent metadata for registry / UI display.

        Returns:
            Dict[str, str]: Name and description.
        """
        return {"name": self.name, "description": self.description}
