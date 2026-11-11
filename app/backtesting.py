from __future__ import annotations

import numpy as np
import pandas as pd

from .models import fit_ensemble, predict_returns
from .risk_management import MIN_CONFIDENCE, build_trade_plan, trade_filter
from .utils import FEATURE_COLUMNS, FeatureConfig, add_features, labeled_rows


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (np.array(equity_curve) - peaks) / peaks
    return float(abs(drawdowns.min()))


def walk_forward_predictions(
    feature_frame: pd.DataFrame,
    min_train_rows: int,
    retrain_every: int,
    max_train_rows: int = 2000,
    prediction_horizon: int = FeatureConfig().prediction_horizon,
) -> np.ndarray:
    """Predicted returns for each bar using only models trained on earlier bars.

    At bar `start` the newest row whose target is known is
    `start - prediction_horizon`, so training rows stop there. Each model is
    then used for the next `retrain_every` bars before being refit.
    """
    predictions = np.full(len(feature_frame), np.nan)
    for start in range(min_train_rows, len(feature_frame), retrain_every):
        end = min(start + retrain_every, len(feature_frame))
        history_end = start - prediction_horizon + 1
        history = labeled_rows(feature_frame.iloc[max(0, history_end - max_train_rows):history_end])
        split = int(len(history) * 0.8)
        models, weights = fit_ensemble(history.iloc[:split], history.iloc[split:])
        predictions[start:end] = predict_returns(models, weights, feature_frame.iloc[start:end][FEATURE_COLUMNS])
    return predictions


def backtest_strategy(
    raw_data: pd.DataFrame,
    risk_percentage: float,
    starting_balance: float = 10_000.0,
    confidence_threshold: float = MIN_CONFIDENCE,
    transaction_cost: float = 0.0008,
    min_train_rows: int = 300,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Walk-forward backtest: every traded bar is predicted by a model that never saw it."""
    feature_frame = add_features(raw_data)
    min_train_rows = max(120, min(min_train_rows, len(feature_frame) // 2))
    retrain_every = max(100, (len(feature_frame) - min_train_rows) // 6)
    predicted_returns = walk_forward_predictions(feature_frame, min_train_rows, retrain_every)

    balance = starting_balance
    equity_curve = [balance]
    open_position = None
    trades: list[dict[str, float | str]] = []

    for i in range(min_train_rows, len(feature_frame) - 1):
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
        predicted_return = float(predicted_returns[i])
        predicted_price = current_price * (1 + predicted_return)
        confidence = min(
            abs(predicted_return) / max(float(latest["atr_pct"]), 1e-6),
            3.0,
        ) / 3.0
        should_trade, _ = trade_filter(
            predicted_return,
            float(latest["atr_pct"]),
            confidence,
            min_confidence=confidence_threshold,
        )
        if not should_trade:
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
        "out_of_sample_bars": float(len(feature_frame) - 1 - min_train_rows),
        "retrain_every": float(retrain_every),
    }
    return trades_frame, metrics

