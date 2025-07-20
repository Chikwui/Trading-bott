# Trading Configuration

This document provides detailed information about the trading configuration options available in the trading bot.

## Table of Contents
- [Trading Modes](#trading-modes)
- [Environment Variables](#environment-variables)
- [Risk Management](#risk-management)
- [Order Execution](#order-execution)
- [Monitoring & Alerts](#monitoring--alerts)

## Trading Modes

The bot supports several trading modes that can be configured via environment variables:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRADING_MODE` | string | `paper` | Trading mode: `paper` or `live` |
| `AUTO_TRADE` | boolean | `true` | Enable/disable automatic trading |
| `SIMULATION_MODE` | boolean | `true` | Run in simulation mode (no real trades) |
| `PAPER_TRADING` | boolean | `true` | Use paper trading account |
| `REQUIRE_CONFIRMATION` | boolean | `false` | Require manual confirmation for trades |

## Environment Variables

### Core Trading Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_OPEN_TRADES` | integer | `10` | Maximum number of concurrent open trades |
| `DEFAULT_SLIPPAGE` | float | `0.0005` | Default slippage (0.05%) |
| `DEFAULT_COMMISSION` | float | `0.001` | Default commission (0.1%) |
| `ORDER_EXECUTION_TIMEOUT` | integer | `30` | Order execution timeout in seconds |
| `MAX_ORDER_RETRIES` | integer | `3` | Maximum number of order retry attempts |

### Risk Management

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RISK_PER_TRADE` | float | `0.01` | Maximum risk per trade (1% of balance) |
| `MAX_DAILY_DRAWDOWN` | float | `0.1` | Maximum daily drawdown (10%) |
| `MAX_LEVERAGE` | float | `10.0` | Maximum leverage (10x) |
| `DAILY_LOSS_LIMIT` | float | `0.05` | Daily loss limit (5%) |
| `MAX_DAILY_TRADES` | integer | `100` | Maximum number of trades per day |
| `STOP_LOSS_PIPS` | integer | `20` | Default stop loss in pips |
| `TAKE_PROFIT_PIPS` | integer | `40` | Default take profit in pips |

### Order Execution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_ORDER_TYPE` | string | `market` | Default order type (`market` or `limit`) |
| `POST_ONLY` | boolean | `true` | Use post-only orders when possible |
| `IMMEDIATE_OR_CANCEL` | boolean | `false` | Use immediate-or-cancel orders |
| `FILL_OR_KILL` | boolean | `false` | Use fill-or-kill orders |

### Monitoring & Alerts

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MONITORING_ENABLED` | boolean | `true` | Enable/disable monitoring |
| `METRICS_PORT` | integer | `9090` | Port for metrics server |
| `ALERT_MANAGER_URL` | string | `http://localhost:9093` | Alert manager URL |
| `ALERT_THRESHOLD_CRITICAL` | integer | `5` | Critical alert threshold |
| `ALERT_THRESHOLD_WARNING` | integer | `10` | Warning alert threshold |

## Configuration Examples

### Basic Paper Trading
```ini
TRADING_MODE=paper
AUTO_TRADE=true
SIMULATION_MODE=true
PAPER_TRADING=true
REQUIRE_CONFIRMATION=false
```

### Live Trading with Risk Controls
```ini
TRADING_MODE=live
AUTO_TRADE=true
SIMULATION_MODE=false
PAPER_TRADING=false
RISK_PER_TRADE=0.02
MAX_DAILY_DRAWDOWN=0.05
MAX_LEVERAGE=5.0
```

### Development with Manual Confirmation
```ini
TRADING_MODE=paper
AUTO_TRADE=true
SIMULATION_MODE=true
PAPER_TRADING=true
REQUIRE_CONFIRMATION=true
```

## Best Practices

1. **Start in Simulation Mode**: Always test new strategies in simulation mode first.
2. **Use Paper Trading**: Verify your strategy with paper trading before going live.
3. **Set Conservative Limits**: Start with conservative risk parameters and adjust as needed.
4. **Monitor Alerts**: Pay attention to alerts and adjust your strategy accordingly.
5. **Regularly Review Logs**: Check logs for any unusual activity or errors.

## Troubleshooting

### Common Issues

1. **Trades Not Executing**
   - Check `AUTO_TRADE` is set to `true`
   - Verify `SIMULATION_MODE` and `PAPER_TRADING` settings
   - Check logs for any error messages

2. **Orders Being Rejected**
   - Verify your account has sufficient balance
   - Check if you've reached position limits
   - Review exchange-specific requirements

3. **Unexpected Slippage**
   - Adjust `DEFAULT_SLIPPAGE` if needed
   - Consider using limit orders instead of market orders
   - Check market liquidity

For additional help, please refer to the [Troubleshooting Guide](../TROUBLESHOOTING.md) or open an issue in the repository.
