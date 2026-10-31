from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from websocket import create_connection

from .config import get_settings
from .utils import ensure_price_frame, symbol_to_api_code


API_URL = "https://api.deriv.com/api-explorer/#ticks-history"


@dataclass(frozen=True)
class MarketDataStatus:
    source: str
    message: str
    symbol_code: str | None = None


def _generate_demo_data(symbol: str, count: int) -> pd.DataFrame:
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    count = max(count, 250)

    base_price = 100 + (seed % 30)
    trend_strength = 0.00015 if "Jump" in symbol else 0.00005
    volatility = 0.006 if "Jump" in symbol else 0.003
    jump_scale = 0.018 if "Jump" in symbol else 0.006

    returns = rng.normal(loc=trend_strength, scale=volatility, size=count)
    jump_mask = rng.random(count) < (0.08 if "Jump" in symbol else 0.03)
    returns = returns + jump_mask * rng.normal(loc=0.0, scale=jump_scale, size=count)
    prices = base_price * np.exp(np.cumsum(returns))

    close = pd.Series(prices)
    open_ = close.shift(1).fillna(close.iloc[0] * (1 - volatility))
    candle_spread = np.maximum(np.abs(close - open_), close * volatility * 0.4)
    high = np.maximum(open_, close) + candle_spread
    low = np.minimum(open_, close) - candle_spread
    volume = rng.integers(100, 1000, size=count)
    timestamp = pd.date_range(end=pd.Timestamp.utcnow(), periods=count, freq="min")

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "source": "demo",
        }
    )


def _timeframe_to_granularity(timeframe: str) -> int:
    return {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
    }.get(timeframe, 60)


def _request_public_market_data(payload: dict) -> dict:
    settings = get_settings()
    connection = create_connection(settings.public_market_ws_url, timeout=15)
    try:
        connection.send(json.dumps(payload))
        while True:
            message = json.loads(connection.recv())
            if message.get("req_id") == payload.get("req_id"):
                if "error" in message:
                    raise ValueError(message["error"].get("message", "Unknown Deriv API error."))
                return message
    finally:
        connection.close()


def _normalize_symbol_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _resolve_deriv_symbol(display_name: str) -> tuple[str, str]:
    message = _request_public_market_data(
        {
            "active_symbols": "brief",
            "req_id": 100,
        }
    )
    active_symbols = message.get("active_symbols") or []
    if not active_symbols:
        raise ValueError("Deriv returned no active symbols.")

    requested = _normalize_symbol_name(display_name)
    fallback_code = symbol_to_api_code(display_name)
    fallback_normalized = _normalize_symbol_name(fallback_code)
    candidates: list[tuple[int, dict]] = []

    for item in active_symbols:
        code = str(item.get("symbol") or item.get("underlying_symbol") or "")
        name = str(
            item.get("display_name")
            or item.get("underlying_symbol_name")
            or code
        )
        code_key = _normalize_symbol_name(code)
        name_key = _normalize_symbol_name(name)
        score = 0
        if name_key == requested:
            score = 100
        elif requested == "step" and name_key == "stepindex100":
            score = 95
        elif name_key.startswith(requested) or requested in name_key:
            score = 80
        elif code_key == fallback_normalized:
            score = 70
        elif fallback_normalized and fallback_normalized in code_key:
            score = 50
        if score:
            candidates.append((score, item))

    if not candidates:
        examples = ", ".join(
            str(
                item.get("display_name")
                or item.get("underlying_symbol_name")
                or item.get("symbol")
                or item.get("underlying_symbol")
            )
            for item in active_symbols[:8]
        )
        raise ValueError(
            f"Deriv did not return a matching symbol for '{display_name}'. "
            f"Sample available symbols: {examples}"
        )

    _, selected = max(candidates, key=lambda candidate: candidate[0])
    selected_code = selected.get("symbol") or selected.get("underlying_symbol")
    selected_name = (
        selected.get("display_name")
        or selected.get("underlying_symbol_name")
        or selected_code
    )
    return str(selected_code), str(selected_name)


def _fetch_deriv_public_data(symbol: str, timeframe: str, count: int) -> tuple[pd.DataFrame, str]:
    symbol_code, resolved_name = _resolve_deriv_symbol(symbol)
    payload = {
        "ticks_history": symbol_code,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": _timeframe_to_granularity(timeframe),
        "style": "candles",
        "req_id": 101,
    }
    message = _request_public_market_data(payload)
    candles = message.get("candles") or message.get("history", {}).get("candles") or []

    if not candles:
        raise ValueError("Deriv public market data returned no candles.")

    frame = pd.DataFrame(candles)
    frame["source"] = "deriv_public"
    return ensure_price_frame(frame), f"{resolved_name} ({symbol_code})"


def fetch_historical_data_detailed(
    symbol: str,
    timeframe: str = "1m",
    count: int = 1000,
) -> tuple[pd.DataFrame, MarketDataStatus]:
    try:
        data, resolved_symbol = _fetch_deriv_public_data(symbol, timeframe, count)
        return data, MarketDataStatus(
            source="deriv_public",
            message=f"Live public Deriv candles loaded for {resolved_symbol}.",
            symbol_code=resolved_symbol,
        )
    except Exception as error:
        error_message = f"Deriv public data unavailable: {error}"
        return ensure_price_frame(_generate_demo_data(symbol, count)), MarketDataStatus(
            source="demo",
            message=error_message,
        )


def fetch_historical_data(symbol: str, timeframe: str = "1m", count: int = 1000) -> tuple[pd.DataFrame, str]:
    data, status = fetch_historical_data_detailed(symbol, timeframe, count)
    return data, status.source


def trading_auth_configured() -> bool:
    return get_settings().account_auth_ready
