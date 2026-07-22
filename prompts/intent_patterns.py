"""Intent keyword patterns used by the intent understanding agent."""

from __future__ import annotations

from typing import Any, Dict

INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "sales_analysis": {
        "keywords": [
            "satış",
            "satis",
            "ciro",
            "revenue",
            "sales",
            "sipariş",
            "order",
        ],
        "regex": [r"ayl[iı]k\s+ciro", r"sat[iı][sş]\s+analiz"],
        "tasks": [
            {"id": "profile", "title": "Veriyi profille", "agent": "data_profiler"},
            {"id": "stats", "title": "Satış istatistikleri", "agent": "statistics"},
            {"id": "dashboard", "title": "Satış dashboard", "agent": "dashboard_builder"},
            {"id": "forecast", "title": "Ciro tahmini", "agent": "forecast"},
            {"id": "report", "title": "Satış raporu", "agent": "reporting"},
        ],
    },
    "dashboard": {
        "keywords": ["dashboard", "panel", "kpi", "gösterge", "gosterge", "ekran"],
        "regex": [r"dashboard\s+haz[iı]rla"],
        "tasks": [
            {"id": "profile", "title": "Veriyi profille", "agent": "data_profiler"},
            {"id": "viz", "title": "Grafik öner", "agent": "visualization_advisor"},
            {"id": "dashboard", "title": "Dashboard oluştur", "agent": "dashboard_builder"},
        ],
    },
    "forecast": {
        "keywords": [
            "tahmin",
            "forecast",
            "projeksiyon",
            "gelecek",
            "öngörü",
            "ongoru",
        ],
        "regex": [r"\d+\s*(hafta|ay|y[iı]l)"],
        "tasks": [
            {"id": "profile", "title": "Veriyi profille", "agent": "data_profiler"},
            {"id": "forecast", "title": "Tahmin üret", "agent": "forecast"},
            {"id": "report", "title": "Tahmin raporu", "agent": "reporting"},
        ],
    },
    "profit_loss": {
        "keywords": ["kâr", "kar", "zarar", "profit", "loss", "gelir", "gider", "pnl"],
        "regex": [r"k[aâ]r[- ]?zarar"],
        "tasks": [
            {"id": "profile", "title": "Veriyi profille", "agent": "data_profiler"},
            {"id": "stats", "title": "Kâr-zarar istatistikleri", "agent": "statistics"},
            {"id": "dashboard", "title": "Kâr-zarar dashboard", "agent": "dashboard_builder"},
            {"id": "report", "title": "Finansal özet", "agent": "reporting"},
        ],
    },
    "hr_dashboard": {
        "keywords": [
            "personel",
            "ik",
            "hr",
            "çalışan",
            "calisan",
            "maaş",
            "maas",
            "izin",
        ],
        "regex": [r"personel\s+dashboard"],
        "tasks": [
            {"id": "profile", "title": "Personel verisini profille", "agent": "data_profiler"},
            {"id": "dashboard", "title": "Personel dashboard", "agent": "dashboard_builder"},
            {"id": "report", "title": "İK özeti", "agent": "reporting"},
        ],
    },
    "document_tracking": {
        "keywords": ["belge", "doküman", "dokuman", "document", "takip", "tracking"],
        "regex": [r"belge\s+takip"],
        "tasks": [
            {"id": "profile", "title": "Belge verisini profille", "agent": "data_profiler"},
            {"id": "dashboard", "title": "Belge takip paneli", "agent": "dashboard_builder"},
            {"id": "report", "title": "Belge durum raporu", "agent": "reporting"},
        ],
    },
    "cleaning": {
        "keywords": ["temizle", "eksik", "duplicate", "outlier", "cleaning", "doğrula"],
        "regex": [r"veri\s+temiz"],
        "tasks": [
            {"id": "clean", "title": "Temizleme planı çıkar", "agent": "data_cleaner"},
            {"id": "profile", "title": "Temizlik sonrası profil", "agent": "data_profiler"},
        ],
    },
    "statistics": {
        "keywords": [
            "istatistik",
            "korelasyon",
            "regresyon",
            "anova",
            "hipotez",
            "normalite",
        ],
        "regex": [r"istatistiksel\s+analiz"],
        "tasks": [
            {"id": "stats", "title": "İstatistiksel analiz", "agent": "statistics"},
            {"id": "report", "title": "İstatistik raporu", "agent": "reporting"},
        ],
    },
    "general_analysis": {
        "keywords": ["analiz", "incele", "özet", "ozet", "keşfet", "kesfet"],
        "regex": [],
        "tasks": [
            {"id": "profile", "title": "Veri profili çıkar", "agent": "data_profiler"},
            {"id": "stats", "title": "İstatistiksel analiz yap", "agent": "statistics"},
            {"id": "ml", "title": "ML yaklaşımı öner", "agent": "ml_advisor"},
            {"id": "dashboard", "title": "Dashboard oluştur", "agent": "dashboard_builder"},
            {"id": "report", "title": "Rapor hazırla", "agent": "reporting"},
        ],
    },
}
