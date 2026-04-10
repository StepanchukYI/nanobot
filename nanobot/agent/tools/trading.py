"""Trading tools: native nanobot tools that call trading REST APIs."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool


class TradingPositionsTool(Tool):
    """Get current positions from Trading Core service."""

    name = "trading_positions"
    description = "Get all current open positions from the Trading Core service."
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Optional: filter by trading symbol (e.g. BTCUSDT)",
            },
        },
    }

    def __init__(self, trading_core_url: str = "http://localhost:8010"):
        self.trading_core_url = trading_core_url.rstrip("/")

    async def execute(self, symbol: str | None = None, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.trading_core_url}/positions"
                params = {"symbol": symbol} if symbol else {}
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                if not data:
                    return "No open positions."
                lines = ["Open Positions:"]
                for pos in data:
                    lines.append(
                        f"  {pos.get('symbol')} |Qty: {pos.get('qty')} |Side: {pos.get('side')} |PnL: {pos.get('unrealized_pnl', 'N/A')}"
                    )
                return "\n".join(lines)
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            logger.warning("trading_positions failed: {}", e)
            return f"Error fetching positions: {e}"


class TradingOrdersTool(Tool):
    """Get or submit orders via Trading Core service."""

    name = "trading_orders"
    description = (
        "Get open orders or submit a new order. "
        "Use action='list' to get open orders, action='submit' to place a new order."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "submit"],
                "description": "Action: 'list' (get open orders) or 'submit' (place new order)",
            },
            "symbol": {
                "type": "string",
                "description": "Trading symbol (e.g. BTCUSDT) — required for submit",
            },
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "Order side — required for submit",
            },
            "qty": {
                "type": "number",
                "description": "Order quantity — required for submit",
            },
            "order_type": {
                "type": "string",
                "enum": ["market", "limit", "stop", "take_profit"],
                "description": "Order type (default: market)",
            },
            "price": {
                "type": "number",
                "description": "Limit price (required for limit orders)",
            },
            "stop_price": {
                "type": "number",
                "description": "Stop price (required for stop/take_profit orders)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, trading_core_url: str = "http://localhost:8010"):
        self.trading_core_url = trading_core_url.rstrip("/")

    async def execute(
        self,
        action: str,
        symbol: str | None = None,
        side: str | None = None,
        qty: float | None = None,
        order_type: str = "market",
        price: float | None = None,
        stop_price: float | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                if action == "list":
                    url = f"{self.trading_core_url}/orders"
                    params = {"symbol": symbol} if symbol else {}
                    r = await client.get(url, params=params)
                    r.raise_for_status()
                    data = r.json()
                    if not data:
                        return "No open orders."
                    lines = ["Open Orders:"]
                    for order in data:
                        lines.append(
                            f"  {order.get('symbol')} |{order.get('side')} |{order.get('qty')} @{order.get('price', 'MARKET')} |{order.get('status')} |ID: {order.get('id')}"
                        )
                    return "\n".join(lines)

                elif action == "submit":
                    if not symbol or not side or qty is None:
                        return "Error: symbol, side, and qty are required for submit action."
                    payload: dict[str, Any] = {
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "type": order_type,
                    }
                    if price is not None:
                        payload["price"] = price
                    if stop_price is not None:
                        payload["stop_price"] = stop_price
                    r = await client.post(f"{self.trading_core_url}/orders", json=payload)
                    r.raise_for_status()
                    data = r.json()
                    return f"Order submitted: {data.get('id', 'unknown')} |{data.get('symbol')} |{data.get('side')} |Qty: {data.get('qty')} @{data.get('price', 'MARKET')}"

                return f"Unknown action: {action}"
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            logger.warning("trading_orders failed: {}", e)
            return f"Error with trading orders: {e}"


class TradingBalanceTool(Tool):
    """Get account balance from Trading Core service."""

    name = "trading_balance"
    description = "Get account balance from Trading Core service (total equity, available, PnL)."
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, trading_core_url: str = "http://localhost:8010"):
        self.trading_core_url = trading_core_url.rstrip("/")

    async def execute(self, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.trading_core_url}/balance")
                r.raise_for_status()
                data = r.json()
                return (
                    f"Balance:\n"
                    f"  Total Equity: {data.get('total_equity', 'N/A')}\n"
                    f"  Available: {data.get('available', 'N/A')}\n"
                    f"  Unrealized PnL: {data.get('unrealized_pnl', 'N/A')}\n"
                    f"  Margin Used: {data.get('margin_used', 'N/A')}"
                )
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            logger.warning("trading_balance failed: {}", e)
            return f"Error fetching balance: {e}"


class MarketDataTool(Tool):
    """Get OHLCV market data from Market Data service."""

    name = "market_data"
    description = (
        "Get OHLCV candlestick data from Market Data service. "
        "Returns recent candles for analysis."
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Trading symbol (e.g. BTCUSDT)",
            },
            "interval": {
                "type": "string",
                "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "description": "Candle interval (default: 1h)",
            },
            "limit": {
                "type": "integer",
                "description": "Number of candles (default: 100, max: 1000)",
            },
        },
        "required": ["symbol"],
    }

    def __init__(self, market_data_url: str = "http://localhost:8020"):
        self.market_data_url = market_data_url.rstrip("/")

    async def execute(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.market_data_url}/ohlcv/{symbol}"
                params = {"interval": interval, "limit": min(limit, 1000)}
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                if not data:
                    return f"No market data for {symbol}."
                lines = [f"OHLCV ({interval}) for {symbol}:"]
                for candle in data[-10:]:  # Show last 10 candles
                    ts = candle.get("timestamp", "")
                    o = candle.get("open", "")
                    h = candle.get("high", "")
                    l = candle.get("low", "")
                    c = candle.get("close", "")
                    v = candle.get("volume", "")
                    lines.append(f"  {ts} O:{o} H:{h} L:{l} C:{c} V:{v}")
                return "\n".join(lines)
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            logger.warning("market_data failed: {}", e)
            return f"Error fetching market data: {e}"


class ExchangeBalanceTool(Tool):
    """Get exchange account balance from Exchange Connectors service."""

    name = "exchange_balance"
    description = (
        "Get balance from Exchange Connectors service for a specific exchange. "
        "Use exchange='binance', 'okx', or 'bybit'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "exchange": {
                "type": "string",
                "enum": ["binance", "okx", "bybit"],
                "description": "Exchange name",
            },
            "symbol": {
                "type": "string",
                "description": "Optional: filter by asset symbol (e.g. USDT)",
            },
        },
        "required": ["exchange"],
    }

    def __init__(self, exchange_url: str = "http://localhost:8050"):
        self.exchange_url = exchange_url.rstrip("/")

    async def execute(
        self,
        exchange: str,
        symbol: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.exchange_url}/balance/{exchange}"
                params = {"symbol": symbol} if symbol else {}
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                if not data:
                    return f"No balance data for {exchange}."
                lines = [f"{exchange.capitalize()} Balance:"]
                for asset, balance in data.items():
                    free = balance.get("free", "N/A")
                    locked = balance.get("locked", "N/A")
                    lines.append(f"  {asset}: Free={free} Locked={locked}")
                return "\n".join(lines)
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            logger.warning("exchange_balance failed: {}", e)
            return f"Error fetching exchange balance: {e}"


class MarketScreenerTool(Tool):
    """Screen trading pairs by criteria from Market Data service."""

    name = "market_screener"
    description = (
        "Screen trading pairs by performance criteria. "
        "Returns pairs matching filters like top gainers, top volume, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["top_gainers", "top_volume", "oversold", "overbought"],
                "description": "Screening filter",
            },
            "limit": {
                "type": "integer",
                "description": "Number of results (default: 20)",
            },
        },
        "required": ["filter"],
    }

    def __init__(self, market_data_url: str = "http://localhost:8020"):
        self.market_data_url = market_data_url.rstrip("/")

    async def execute(
        self,
        filter: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.market_data_url}/screener"
                params = {"filter": filter, "limit": limit}
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                if not data:
                    return f"No pairs match filter: {filter}"
                lines = [f"Top {limit} pairs for '{filter}':"]
                for pair in data:
                    sym = pair.get("symbol", "")
                    price = pair.get("price", "")
                    change = pair.get("change_24h", "")
                    volume = pair.get("volume_24h", "")
                    lines.append(f"  {sym} |Price: {price} |24h: {change}% |Vol: {volume}")
                return "\n".join(lines)
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            logger.warning("market_screener failed: {}", e)
            return f"Error screening market: {e}"
