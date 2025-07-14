# Risk Guardrails & Scenario Planning

This document outlines risk guardrails and scenario planning for AI Trader.

## Risk Guardrails

- ⚠️ **Position Sizing**: Enforced via `calculate_position_size`; max risk per trade = `RISK_PERCENT_PER_TRADE`.
- ⚠️ **Max Open Positions**: No more than `MAX_OPEN_POSITIONS` active trades.
- ⚠️ **Daily Drawdown**: Halt trading if drawdown exceeds `MAX_DAILY_DRAWDOWN_PERCENT`.
- ✅ **Mitigation**: Automated checks in `run_bot.py` before order execution.

## Scenario Planning

1. **Tick Surge Test**: Simulate high-frequency ticks to validate consumer throughput and CPU usage. See `examples/scenario_planning.py`.
2. **Model Drift Scenario**: Introduce concept drift in input features; monitor drift detectors and trigger rollback.
3. **Service Outage**: Kill Kafka broker; ensure consumer reconnects and logs errors.
4. **Latency Spike**: Throttle network; measure end-to-end latency in metrics dashboard.

Ensure all scenarios are included in CI or run periodically in a staging environment.
