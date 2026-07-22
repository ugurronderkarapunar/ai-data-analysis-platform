"""Visualization advisor agent."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from utils.exceptions import ValidationError
from visualizations.recommendations import recommend_charts


class VisualizationAdvisorAgent(BaseAgent):
    """Recommend suitable chart types and discourage misuse."""

    name = "visualization_advisor"
    description = "Veriye en uygun grafikleri önerir"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Produce visualization recommendations.

        Args:
            context: Shared context.
            **kwargs: Unused.

        Returns:
            AgentContext: Context with viz_advice.

        Raises:
            ValidationError: If dataframe is missing.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("Görselleştirme önerisi için dataframe gerekli.")

        try:
            advice = recommend_charts(df, context)
            context.viz_advice = advice
            context.add_message(
                f"{len(advice.get('recommendations', []))} grafik önerisi hazırlandı."
            )
            self.logger.info("Visualization advice ready")
            return context
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Visualization advisor failed")
            raise ValidationError(f"Görselleştirme önerisi başarısız: {exc}") from exc
