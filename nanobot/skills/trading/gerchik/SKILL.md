---
name: gerchik
description: Gerchik strategy implementation for market analysis and trade execution.
metadata: {"nanobot": {"emoji": "🎯"}}
---

# Gerchik Trading Strategy

Gerchik strategy focuses on:
- Multi-timeframe analysis (5M, 15M, 1H, 4H, 1D)
- ATR-based position sizing and risk management
- Key level identification (support, resistance, round numbers)
- Candle pattern recognition

## Core Rules

### Position Sizing
```
Risk Amount = Account × Risk%
Position Size = Risk Amount ÷ ATR
```

### Entry Conditions
1. Trend confirmation on 4H/1D
2. Pullback to key level on 1H
3. Bullish/bearish candle on 5M/15M
4. Minimum 3:1 reward-to-risk ratio

### Exit Rules
- Stop-loss: Below/above key level + ATR buffer
- Take-profit: 1.5-3× risk
- Time exit: Close after 2-4 candles if no progress

## Tools

Use these to gather data:
- `market_data` with intervals: 5m, 15m, 1h, 4h, 1d
- `trading_positions` to check current exposure
- `trading_balance` to check account equity
- `market_screener` to find rotation candidates

## Workflow

1. **Scan** — `market_screener(filter="top_gainers", limit=20)` for rotation candidates
2. **Analyze** — Get 4H, 1H, 5M data for top candidates
3. **Assess** — Check key levels, ATR, trend direction
4. **Plan** — Calculate position size, entry, stop, target
5. **Execute** — Submit order with `trading_orders`
6. **Log** — Note decision rationale

## Rotation Logic

Rotation = moving from one pair to another based on signal strength.
- Compare ATR-normalized signals across pairs
- Rotate when current pair signal weakens AND another strengthens
- Close existing position before opening new one

## References

- Uses `trading_shared` package for GerchikStrategy implementation
- Requires Gerchik candles and levels calculation
