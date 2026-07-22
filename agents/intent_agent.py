"""Natural-language intent understanding agent."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from core.base_agent import BaseAgent
from core.context import AgentContext
from prompts.intent_patterns import INTENT_PATTERNS


class IntentUnderstandingAgent(BaseAgent):
    """Parse user natural language into structured analysis tasks."""

    name = "intent_understanding"
    description = "Kullanıcı niyetini doğal dilden görev listesine çevirir"

    def run(self, context: AgentContext, **kwargs: Any) -> AgentContext:
        """Interpret a user query into intents and tasks.

        Args:
            context: Shared context.
            **kwargs: Expects optional ``user_query``.

        Returns:
            AgentContext: Context with ``intent`` and ``tasks``.
        """
        user_query = str(kwargs.get("user_query") or "").strip()
        try:
            if not user_query:
                intent = {
                    "raw_query": "",
                    "primary_intent": "general_analysis",
                    "confidence": 0.5,
                    "matched_intents": ["general_analysis"],
                    "explanation": "Sorgu verilmedi; genel analiz pipeline'ı çalıştırılacak.",
                }
                tasks = self._tasks_for_intents(["general_analysis"])
            else:
                matched = self._match_intents(user_query)
                primary = matched[0][0] if matched else "general_analysis"
                confidence = matched[0][1] if matched else 0.4
                intent_names = [m[0] for m in matched] or ["general_analysis"]
                intent = {
                    "raw_query": user_query,
                    "primary_intent": primary,
                    "confidence": confidence,
                    "matched_intents": intent_names,
                    "explanation": self._explain(primary, user_query),
                }
                tasks = self._tasks_for_intents(intent_names)

            context.intent = intent
            context.tasks = tasks
            context.add_message(
                f"Niyet anlaşıldı: {intent['primary_intent']} "
                f"(güven: {intent['confidence']:.0%})"
            )
            self.logger.info("Intent parsed: %s", intent)
            return context
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Intent parsing failed")
            context.intent = {
                "raw_query": user_query,
                "primary_intent": "general_analysis",
                "confidence": 0.3,
                "matched_intents": ["general_analysis"],
                "explanation": f"Niyet ayrıştırılamadı, genel analize düşüldü: {exc}",
            }
            context.tasks = self._tasks_for_intents(["general_analysis"])
            context.add_error(f"Intent agent uyarısı: {exc}")
            return context

    def _match_intents(self, query: str) -> List[Tuple[str, float]]:
        """Score intents against the query using keyword patterns.

        Args:
            query: User text.

        Returns:
            List[Tuple[str, float]]: Sorted (intent, score) pairs.
        """
        normalized = query.lower()
        scores: Dict[str, float] = {}
        for intent_name, pattern in INTENT_PATTERNS.items():
            hits = 0
            for keyword in pattern["keywords"]:
                if keyword.lower() in normalized:
                    hits += 1
            for regex in pattern.get("regex", []):
                if re.search(regex, normalized, flags=re.IGNORECASE):
                    hits += 1
            if hits:
                score = min(1.0, 0.35 + hits * 0.2)
                scores[intent_name] = score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self.logger.debug("Intent scores: %s", ranked)
        return ranked

    def _tasks_for_intents(self, intents: List[str]) -> List[Dict[str, Any]]:
        """Expand intents into an ordered task checklist.

        Args:
            intents: Matched intent names.

        Returns:
            List[Dict[str, Any]]: Task dictionaries.
        """
        tasks: List[Dict[str, Any]] = []
        seen = set()
        for intent_name in intents:
            pattern = INTENT_PATTERNS.get(intent_name, {})
            for task in pattern.get("tasks", []):
                key = task["id"]
                if key not in seen:
                    tasks.append(dict(task))
                    seen.add(key)
        if not tasks:
            tasks = [
                {"id": "profile", "title": "Veri profili çıkar", "agent": "data_profiler"},
                {"id": "stats", "title": "İstatistiksel analiz yap", "agent": "statistics"},
                {"id": "dashboard", "title": "Dashboard oluştur", "agent": "dashboard_builder"},
                {"id": "report", "title": "Rapor hazırla", "agent": "reporting"},
            ]
        return tasks

    def _explain(self, primary: str, query: str) -> str:
        """Build a short Turkish explanation of the detected intent.

        Args:
            primary: Primary intent name.
            query: Original query.

        Returns:
            str: Explanation text.
        """
        labels = {
            "sales_analysis": "Satış / ciro odaklı analiz",
            "dashboard": "Dashboard / KPI görünümü",
            "forecast": "Tahmin / projeksiyon",
            "profit_loss": "Kâr-zarar analizi",
            "hr_dashboard": "Personel / İK dashboard",
            "document_tracking": "Belge takip görünümü",
            "cleaning": "Veri temizleme planı",
            "statistics": "İstatistiksel derinlik analizi",
            "general_analysis": "Genel keşifsel veri analizi",
        }
        label = labels.get(primary, primary)
        return f"\"{query}\" isteği '{label}' olarak yorumlandı."
