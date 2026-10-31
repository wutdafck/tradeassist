from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ModelBundle, predict_prices
from .risk_management import build_trade_plan
from .utils import add_features


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (np.array(equity_curve) - peaks) / peaks
    return float(abs(drawdowns.min()))


def backtest_strategy(
    raw_data: pd.DataFrame,
    model_bundle: ModelBundle,
    risk_percentage: float,
    starting_balance: float = 10_000.0,
    confidence_threshold: float = 0.18,
    transaction_cost: float = 0.0008,
) -> tuple[pd.DataFrame, dict[str, float]]:
    feature_frame = add_features(raw_data)
    # Keep the cloud request fast by evaluating the already-prepared features
    # directly instead of recomputing indicators for every historical bar.
    max_bars = 700
    if len(feature_frame) > max_bars:
        feature_frame = feature_frame.tail(max_bars).reset_index(drop=True)
    balance = starting_balance
    equity_curve = [balance]
    open_position = None
    trades: list[dict[str, float | str]] = []

    for i in range(80, len(feature_frame) - 1):
        latest = feature_frame.iloc[i]

        if open_position is not None:
            bar_high = float(latest["high"])
            bar_low = float(latest["low"])
            exit_price = None
            exit_reason = None

            if open_position["signal"] == "BUY":
                if bar_low <= open_position["stop_loss"]:
                    exit_price = open_position["stop_loss"]
                    exit_reason = "stop_loss"
                elif bar_high >= open_position["take_profit"]:
                    exit_price = open_position["take_profit"]
                    exit_reason = "take_profit"
            else:
                if bar_high >= open_position["stop_loss"]:
                    exit_price = open_position["stop_loss"]
                    exit_reason = "stop_loss"
                elif bar_low <= open_position["take_profit"]:
                    exit_price = open_position["take_profit"]
                    exit_reason = "take_profit"

            if exit_price is None and i == len(feature_frame) - 2:
                exit_price = float(latest["close"])
                exit_reason = "end_of_test"

            if exit_price is not None:
                direction = 1 if open_position["signal"] == "BUY" else -1
                gross_pnl = (exit_price - open_position["entry_price"]) * direction * open_position["size"]
                fees = transaction_cost * open_position["entry_price"] * open_position["size"]
                net_pnl = gross_pnl - fees
                balance += net_pnl
                trades.append(
                    {
                        "timestamp": latest["timestamp"],
                        "signal": open_position["signal"],
                        "entry_price": open_position["entry_price"],
                        "exit_price": exit_price,
                        "size": open_position["size"],
                        "pnl": net_pnl,
                        "return_pct": net_pnl / starting_balance,
                        "exit_reason": exit_reason,
                    }
                )
                equity_curve.append(balance)
                open_position = None
                continue

        current_price = float(latest["close"])
        predicted_price = float(
            predict_prices(
                model_bundle.models,
                model_bundle.weights,
                feature_frame.loc[[i], model_bundle.feature_columns],
            )[0]
        )
        predicted_return = (predicted_price / current_price) - 1
        confidence = min(
            abs(predicted_return) / max(float(latest["atr_pct"]), 1e-6),
            3.0,
        ) / 3.0
        signal_strength = abs(predicted_return)
        if signal_strength < float(latest["atr_pct"]) * 0.35 or confidence < confidence_threshold:
            continue

        if open_position is None:
            trade_plan = build_trade_plan(
                current_price=float(latest["close"]),
                predicted_price=predicted_price,
                atr=float(latest["atr"]),
                account_balance=balance,
                risk_percentage=risk_percentage,
                confidence=confidence,
            )
            open_position = {
                "signal": trade_plan.signal,
                "entry_price": trade_plan.entry_price,
                "stop_loss": trade_plan.stop_loss,
                "take_profit": trade_plan.take_profit,
                "size": trade_plan.position_size_units,
            }

    trades_frame = pd.DataFrame(trades)
    wins = float((trades_frame["pnl"] > 0).mean()) if not trades_frame.empty else 0.0
    gross_profit = float(trades_frame.loc[trades_frame["pnl"] > 0, "pnl"].sum()) if not trades_frame.empty else 0.0
    gross_loss = abs(float(trades_frame.loc[trades_frame["pnl"] < 0, "pnl"].sum())) if not trades_frame.empty else 0.0

    metrics = {
        "starting_balance": float(starting_balance),
        "ending_balance": float(balance),
        "total_return_pct": float((balance / starting_balance - 1) * 100),
        "trades": float(len(trades_frame)),
        "win_rate_pct": float(wins * 100),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf" if gross_profit > 0 else 0.0),
        "max_drawdown_pct": float(_max_drawdown(equity_curve) * 100),
    }
    return trades_frame, metrics

