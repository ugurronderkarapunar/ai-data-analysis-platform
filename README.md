# Aether Analytics

Kurumsal ölçeklenebilir, çok ajanlı (multi-agent) Yapay Zekâ Veri Analiz Platformu.

Streamlit arayüzü, FastAPI entegrasyon katmanı, Docker paketleme ve genişletilebilir ajan registry mimarisi ile gelir.

## Özellikler

- **10 bağımsız ajan** + Orchestrator
- Excel / CSV otomatik yükleme (sheet, encoding, delimiter algılama)
- Otomatik veri profili, istatistik, görselleştirme önerisi
- Doğal dil niyet anlama → görev listesi
- Streamlit dashboard (KPI, filtre, Plotly grafikler, dark mode)
- Uygun zaman serilerinde otomatik forecast (1H / 1A / 3A / 6A / 1Y)
- ML danışmanı (önerir, kullanıcı istemeden model eğitmez)
- Temizleme planı (onay olmadan veri değiştirmez)
- Markdown / PDF / PowerPoint rapor export
- pytest ile test edilebilir, logging ile debug edilebilir

## Kurulum

### Gereksinimler

- Python **3.12+**
- pip
- (Opsiyonel) Docker / Docker Compose

```bash
git clone https://github.com/ugurronderkarapunar/ai-data-analysis-platform.git
cd ai-data-analysis-platform

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Kullanım

### Streamlit

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır.

1. Sol panelden CSV/Excel yükleyin (örnek: `data/sample_sales.csv`)
2. Doğal dil isteği yazın: `Aylık ciroyu görmek istiyorum`
3. **Analizi Başlat**

### FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

- Health: `GET /health`
- Ajanlar: `GET /agents`
- Path analizi: `POST /analyze/path`
- Upload analizi: `POST /analyze/upload`

### Docker

```bash
docker compose up --build
```

- Streamlit: `http://localhost:8501`
- API: `http://localhost:8000`

## Klasör Yapısı

```text
ai-data-analysis-platform/
├── agents/              # Bağımsız ajanlar (SRP)
├── api/                 # FastAPI yüzeyi
├── config/              # Ayarlar
├── core/                # BaseAgent, Registry, Orchestrator, Context
├── dashboard/           # Streamlit tema & bileşenler
├── data/                # Örnek / yüklenen veriler
├── forecasting/         # Tahmin motoru
├── logs/                # Log dosyaları
├── memory/              # Oturum belleği
├── models/              # Model artifact alanı
├── outputs/             # Rapor çıktıları
├── prompts/             # Niyet kalıpları
├── stat_analysis/       # İstatistik motoru
├── tests/               # pytest
├── utils/               # Logging, exception, helpers
├── visualizations/      # Grafik önerileri
├── app.py               # Streamlit giriş noktası
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Mimari

```text
                ┌─────────────────────┐
                │  OrchestratorAgent  │
                └─────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  DataLoader        Intent / Profile     Statistics
  Cleaning          Viz Advisor          ML Advisor
  Forecast          Dashboard            Reporting
```

- Her ajan yalnızca kendi bağlam alanını günceller.
- Yeni ajanlar (`SQL`, `OCR`, `RAG`, `MCP`, …) `AgentRegistry.register(...)` ile eklenir; orchestrator bozulmaz.
- Paylaşılan durum: `AgentContext`

### Built-in ajanlar

| Ajan | name | Görev |
|------|------|-------|
| Data Loader | `data_loader` | CSV/Excel okuma |
| Data Profiler | `data_profiler` | Otomatik profil |
| Intent | `intent_understanding` | NL → görevler |
| Dashboard | `dashboard_builder` | KPI/filtre/grafik spec |
| Forecast | `forecast` | Multi-horizon tahmin |
| Statistics | `statistics` | İstatistik suite |
| ML Advisor | `ml_advisor` | Problem tipi önerisi |
| Cleaning | `data_cleaner` | Temizleme planı |
| Viz Advisor | `visualization_advisor` | Grafik önerisi |
| Reporting | `reporting` | Yönetici/teknik rapor |

## Ekran Görüntüleri

Streamlit arayüzü dark mode destekler; KPI kartları, filtreler, etkileşimli Plotly grafikleri ve rapor indirme sekmeleri içerir.

> İlk çalıştırmadan sonra kendi ekran görüntülerinizi `docs/screenshots/` altına ekleyebilirsiniz.

## Test

```bash
pytest -q
```

## Roadmap

- [ ] SQL Agent
- [ ] Power BI Agent
- [ ] OCR Agent
- [ ] RAG Agent
- [ ] MCP Agent
- [ ] API / Web Scraping Agent
- [ ] LLM Agent (provider-agnostic)
- [ ] E-posta / WhatsApp / Teams / Telegram Agent
- [ ] Prophet / SARIMAX / LSTM / GRU / TFT opsiyonel forecast paketleri
- [ ] Cloud deploy şablonları (Azure / AWS / GCP)

## Lisans

MIT
