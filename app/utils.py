from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SYMBOLS = {
    "Jump 10": "JD10",
    "Jump 25": "JD25",
    "Jump 50": "JD50",
    "Jump 100": "JD100",
    "Step": "STEP",
    "Step 200": "STEP200",
    "Step 300": "STEP300",
    "Step 400": "STEP400",
    "Step 500": "STEP500",
}


@dataclass(frozen=True)
class FeatureConfig:
    lookback: int = 20
    prediction_horizon: int = 1


def symbol_to_api_code(symbol: str) -> str:
    return SYMBOLS.get(symbol, symbol)


def ensure_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    rename_map = {}
    for candidate in ("epoch", "time", "timestamp", "date"):
        if candidate in frame.columns:
            rename_map[candidate] = "timestamp"
            break
    for candidate in ("quote", "price", "last", "close"):
        if candidate in frame.columns:
            rename_map[candidate] = "close"
            break
    if rename_map:
        frame = frame.rename(columns=rename_map)

    if "close" not in frame.columns:
        raise ValueError("Input data must include a close/price column.")

    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).copy()

    if "open" not in frame.columns:
        frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    if "high" not in frame.columns:
        frame["high"] = frame[["open", "close"]].max(axis=1)
    if "low" not in frame.columns:
        frame["low"] = frame[["open", "close"]].min(axis=1)
    if "volume" not in frame.columns:
        frame["volume"] = 1.0

    for column in ("open", "high", "low", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()

    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", errors="coerce")
    else:
        frame["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(frame), freq="min")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def add_features(data: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    config = config or FeatureConfig()
    frame = ensure_price_frame(data)
    frame["returns_1"] = frame["close"].pct_change()
    frame["returns_5"] = frame["close"].pct_change(5)
    frame["returns_10"] = frame["close"].pct_change(10)
    frame["ema_fast"] = frame["close"].ewm(span=10, adjust=False).mean()
    frame["ema_slow"] = frame["close"].ewm(span=30, adjust=False).mean()
    frame["trend_strength"] = (frame["ema_fast"] / frame["ema_slow"]) - 1
    frame["volatility_10"] = frame["returns_1"].rolling(10, min_periods=3).std()
    frame["volatility_20"] = frame["returns_1"].rolling(20, min_periods=5).std()
    frame["rsi_14"] = calculate_rsi(frame["close"], 14)
    frame["atr"] = calculate_atr(frame, 14)
    frame["atr_pct"] = frame["atr"] / frame["close"]
    frame["momentum_3"] = frame["close"] / frame["close"].shift(3) - 1
    frame["momentum_8"] = frame["close"] / frame["close"].shift(8) - 1
    frame["rolling_min_20"] = frame["low"].rolling(20, min_periods=5).min()
    frame["rolling_max_20"] = frame["high"].rolling(20, min_periods=5).max()
    frame["distance_to_low"] = frame["close"] / frame["rolling_min_20"] - 1
    frame["distance_to_high"] = frame["close"] / frame["rolling_max_20"] - 1
    frame["target_price"] = frame["close"].shift(-config.prediction_horizon)
    frame["target_return"] = frame["target_price"] / frame["close"] - 1
    return frame.dropna().reset_index(drop=True)


def summarize_market_state(frame: pd.DataFrame) -> dict[str, float | str]:
    latest = frame.iloc[-1]
    trend = "Bullish" if latest["ema_fast"] > latest["ema_slow"] else "Bearish"
    if abs(float(latest["trend_strength"])) < 0.001:
        trend = "Sideways"
    return {
        "trend": trend,
        "close": float(latest["close"]),
        "rsi": float(latest["rsi_14"]),
        "atr_pct": float(latest["atr_pct"]),
        "volatility_20": float(latest["volatility_20"]),
    }

