# AI Trader

A high-performance, AI-driven trading bot for MetaTrader 5, combining:

- **Event-Driven Streaming** via Kafka for low-latency market data ingestion
- **MLOps Governance** with Feast feature store and MLflow/DVC model versioning
- **Real-Time Monitoring** using Prometheus & Grafana
- **Modular, Containerized Architecture** for scalability and resilience

## Quickstart

1. Clone: `git clone <repo-url> && cd AI tradeer`
2. Create env: `python -m venv venv && source venv/bin/activate`
3. Install deps: `pip install -r requirements.txt`
4. Copy & edit config: `cp .env.example .env`
5. Run: `python run_bot.py`
