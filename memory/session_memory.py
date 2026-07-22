"""In-memory session store for analysis contexts."""

from __future__ import annotations

from typing import Dict, Optional

from core.context import AgentContext
from utils.logging_config import get_logger

logger = get_logger("memory")


class SessionMemory:
    """Simple process-local memory for Streamlit / API sessions."""

    def __init__(self) -> None:
        """Initialize empty memory."""
        self._store: Dict[str, AgentContext] = {}

    def save(self, session_id: str, context: AgentContext) -> None:
        """Persist a context for a session.

        Args:
            session_id: Session key.
            context: Agent context.
        """
        self._store[session_id] = context
        logger.info("Saved context for session=%s", session_id)

    def get(self, session_id: str) -> Optional[AgentContext]:
        """Retrieve a context by session id.

        Args:
            session_id: Session key.

        Returns:
            Optional[AgentContext]: Stored context if present.
        """
        return self._store.get(session_id)

    def clear(self, session_id: str) -> None:
        """Remove a session context.

        Args:
            session_id: Session key.
        """
        if session_id in self._store:
            del self._store[session_id]
            logger.info("Cleared session=%s", session_id)


GLOBAL_MEMORY = SessionMemory()
