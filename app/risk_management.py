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


MIN_CONFIDENCE = 0.18
MIN_MOVE_ATR_FRACTION = 0.35


def trade_filter(
    predicted_return: float,
    atr_pct: float,
    confidence: float,
    min_confidence: float = MIN_CONFIDENCE,
    min_move_atr_fraction: float = MIN_MOVE_ATR_FRACTION,
) -> tuple[bool, str]:
    """Decide whether a forecast is strong enough to act on.

    Shared by the live signal and the backtest so both follow the same rule.
    Returns (should_trade, reason).
    """
    min_move = atr_pct * min_move_atr_fraction
    if abs(predicted_return) < min_move:
        return False, (
            f"Forecast move {abs(predicted_return) * 100:.3f}% is below "
            f"{min_move_atr_fraction:.2f}× ATR ({min_move * 100:.3f}%)"
        )
    if confidence < min_confidence:
        return False, f"Signal strength {confidence * 100:.1f}% is below {min_confidence * 100:.0f}%"
    return True, "Forecast clears the move and strength filters"


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

