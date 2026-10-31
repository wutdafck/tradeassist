from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .risk_management import TradePlan


def build_candlestick_chart(
    frame: pd.DataFrame,
    symbol: str,
    trade_plan: TradePlan | None = None,
) -> go.Figure:
    chart = go.Figure()

    chart.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["ema_fast"],
            name="EMA 10",
            line={"color": "#60a5fa", "width": 1.5},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["ema_slow"],
            name="EMA 30",
            line={"color": "#f59e0b", "width": 1.5},
        )
    )

    if trade_plan is not None and not frame.empty:
        x0 = frame["timestamp"].iloc[0]
        x1 = frame["timestamp"].iloc[-1]
        for label, price, color in (
            ("Entry", trade_plan.entry_price, "#e5e7eb"),
            ("Stop", trade_plan.stop_loss, "#ef4444"),
            ("Target", trade_plan.take_profit, "#22c55e"),
        ):
            chart.add_hline(
                y=price,
                line_dash="dot",
                line_color=color,
                annotation_text=f"{label} {price:.4f}",
                annotation_position="top left",
            )

    chart.update_layout(
        title=f"{symbol} Candlestick View",
        template="plotly_dark",
        height=620,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        font={"color": "#e5e7eb"},
    )
    chart.update_xaxes(showgrid=True, gridcolor="#1f2937")
    chart.update_yaxes(showgrid=True, gridcolor="#1f2937", side="right")
    return chart

