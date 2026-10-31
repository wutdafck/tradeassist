from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradePlan:
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_units: float
    capital_at_risk: float
    risk_reward_ratio: float


def calculate_position_size(
    account_balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss: float,
) -> tuple[float, float]:
    capital_at_risk = account_balance * risk_percentage
    risk_per_unit = max(abs(entry_price - stop_loss), 1e-6)
    position_size_units = capital_at_risk / risk_per_unit
    return position_size_units, capital_at_risk


def build_trade_plan(
    current_price: float,
    predicted_price: float,
    atr: float,
    account_balance: float,
    risk_percentage: float,
    confidence: float,
) -> TradePlan:
    atr = max(float(atr), current_price * 0.0025)
    direction = "BUY" if predicted_price >= current_price else "SELL"
    confidence_boost = 1 + max(confidence - 0.5, 0) * 1.5
    stop_distance = atr * 1.5
    target_distance = atr * 2.5 * confidence_boost

    if direction == "BUY":
        stop_loss = current_price - stop_distance
        take_profit = current_price + target_distance
    else:
        stop_loss = current_price + stop_distance
        take_profit = current_price - target_distance

    position_size_units, capital_at_risk = calculate_position_size(
        account_balance,
        risk_percentage,
        current_price,
        stop_loss,
    )

    reward = abs(take_profit - current_price)
    risk = abs(current_price - stop_loss)
    risk_reward_ratio = reward / max(risk, 1e-6)

    return TradePlan(
        signal=direction,
        entry_price=float(current_price),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        position_size_units=float(position_size_units),
        capital_at_risk=float(capital_at_risk),
        risk_reward_ratio=float(risk_reward_ratio),
    )

