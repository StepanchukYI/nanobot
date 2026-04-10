---
name: trading
description: Market analysis and trade execution using Trading Core and Market Data services.
metadata: {"nanobot": {"emoji": "📈"}}
---

# Trading Skill

Use this skill when the user wants to check positions, analyze markets, submit orders, or review trading performance.

## Available Tools

| Tool | Purpose |
|------|---------|
| `trading_positions` | Get all open positions from Trading Core |
| `trading_orders` | List open orders or submit new orders |
| `trading_balance` | Get account equity, available balance, PnL |
| `market_data` | Get OHLCV candles for technical analysis |
| `exchange_balance` | Get exchange account balances |
| `market_screener` | Screen pairs (top gainers, volume, etc.) |

## Workflow

### 1. Market Analysis
```
market_data(symbol="BTCUSDT", interval="1h", limit=100)
market_screener(filter="top_gainers", limit=20)
```

### 2. Position Review
```
trading_positions()
trading_balance()
```

### 3. Order Submission
```
# Market order
trading_orders(action="submit", symbol="BTCUSDT", side="buy", qty=0.01)

# Limit order
trading_orders(action="submit", symbol="BTCUSDT", side="buy", qty=0.01, order_type="limit", price=95000)

# Stop-loss
trading_orders(action="submit", symbol="BTCUSDT", side="sell", qty=0.01, order_type="stop", stop_price=90000)
```

### 4. Open Orders
```
trading_orders(action="list", symbol="BTCUSDT")
```

## Gerchik Strategy Notes

When applying Gerchik strategy rules:
- Check 5M, 15M, 1H, 4H timeframes for confluence
- ATR-based position sizing: risk max 1-2% per trade
- Look for level tests, retracements, and candle patterns
- Always confirm with trading_balance before large orders

## Risk Rules

- Never risk more than 2% of account on a single trade
- Check margin usage before submitting orders
- Review open positions before new entries
- Log all trading decisions for review
