"""Machine learning advisor agent (recommends, does not train)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.base_agent import BaseAgent
from core.context import AgentContext
from utils.exceptions import ValidationError


class MLAdvisorAgent(BaseAgent):
    """Recommend suitable ML problem types without training models."""

    name = "ml_advisor"
    description = "Veriye uygun ML yaklaşımını önerir; model eğitmez"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Analyze dataframe and recommend ML approaches.

        Args:
            context: Shared context.
            **kwargs: Unused.

        Returns:
            AgentContext: Context with ml_advice.

        Raises:
            ValidationError: If dataframe is missing.
        """
        df = context.dataframe
        if df is None:
            raise ValidationError("ML danışmanı için dataframe gerekli.")

        try:
            advice = self._analyze(df, context)
            context.ml_advice = advice
            top = advice["recommendations"][0]["type"] if advice["recommendations"] else "N/A"
            context.add_message(
                f"ML önerisi hazır. Birincil yaklaşım: {top}. Model eğitimi yapılmadı."
            )
            self.logger.info("ML advice generated: %s", [r["type"] for r in advice["recommendations"]])
            return context
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("ML advisor failed")
            raise ValidationError(f"ML önerisi üretilemedi: {exc}") from exc

    def _analyze(self, df: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """Build ranked ML recommendations.

        Args:
            df: Input dataframe.
            context: Shared context for profile hints.

        Returns:
            Dict[str, Any]: Advice payload.
        """
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [c for c in df.columns if c not in numeric]
        datetime_cols = context.metadata.get("datetime_columns") or context.profile.get(
            "datetime_columns", []
        )
        targets = context.profile.get("target_candidates") or []
        recommendations: List[Dict[str, Any]] = []

        if datetime_cols and numeric:
            recommendations.append(
                {
                    "type": "Forecasting",
                    "score": 0.9,
                    "why": "Tarih + sayısal ölçüm bulundu; zaman serisi tahmini uygundur.",
                }
            )

        if targets:
            target = targets[0]
            if target in numeric:
                nunique = int(df[target].nunique(dropna=True))
                if nunique <= max(15, int(len(df) * 0.05)):
                    recommendations.append(
                        {
                            "type": "Sınıflandırma",
                            "score": 0.85,
                            "why": (
                                f"Hedef adayı '{target}' sınırlı sayıda benzersiz değer içeriyor; "
                                "sınıflandırma uygun olabilir."
                            ),
                        }
                    )
                else:
                    recommendations.append(
                        {
                            "type": "Regresyon",
                            "score": 0.88,
                            "why": (
                                f"Hedef adayı '{target}' sürekli sayısal; regresyon uygundur."
                            ),
                        }
                    )

        if len(numeric) >= 2:
            recommendations.append(
                {
                    "type": "Kümeleme",
                    "score": 0.7,
                    "why": "Birden fazla sayısal özellik var; segmentasyon / kümeleme denenebilir.",
                }
            )
            recommendations.append(
                {
                    "type": "Anomali Tespiti",
                    "score": 0.65,
                    "why": "Sayısal dağılımlarda aykırı davranış / fraud-kalite kontrolü için uygundur.",
                }
            )

        if categorical and (targets or numeric):
            recommendations.append(
                {
                    "type": "Recommendation System",
                    "score": 0.55,
                    "why": (
                        "Kullanıcı/ürün benzeri kategorik kimlik alanları varsa "
                        "öneri sistemleri değerlendirilebilir."
                    ),
                }
            )

        if not recommendations:
            recommendations.append(
                {
                    "type": "Keşifsel Analiz",
                    "score": 0.5,
                    "why": "Net bir ML hedefi çıkarılamadı; önce EDA ve iş hedefi netleştirilmeli.",
                }
            )

        recommendations.sort(key=lambda r: r["score"], reverse=True)
        return {
            "trained": False,
            "note": "Kullanıcı istemeden model eğitimi yapılmaz.",
            "data_signals": {
                "rows": int(df.shape[0]),
                "numeric_columns": len(numeric),
                "categorical_columns": len(categorical),
                "datetime_columns": len(datetime_cols),
                "target_candidates": targets[:5],
            },
            "recommendations": recommendations,
        }
