from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from .api import fetch_historical_data_detailed
from .backtesting import backtest_strategy
from .charts import build_candlestick_chart
from .config import describe_connection_state, get_settings
from .models import train_price_model, predict_next_price
from .risk_management import build_trade_plan
from .utils import SYMBOLS, add_features, summarize_market_state
from broker.ctrader_fix import CTraderFixExecutor
from broker.paper import PaperTradingExecutor
from broker.interfaces import OrderRequest


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding: 1.1rem 2rem 2rem; max-width: 1800px; }
        [data-testid="stMetric"] { padding: 0.2rem 0.35rem; }
        [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #94a3b8; }
        [data-testid="stMetricValue"] { font-size: 1.25rem; line-height: 1.2; }
        div[data-testid="stVerticalBlock"] > div:has(> div.compact-card) { gap: 0.25rem; }
        .dashboard-title { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.03em; margin: 0; }
        .dashboard-subtitle { color: #94a3b8; font-size: 0.78rem; margin: 0.15rem 0 0; }
        .compact-card { background: #111827; border: 1px solid #243244; border-radius: 10px; padding: 0.72rem 0.85rem; min-height: 74px; }
        .compact-card .label { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; }
        .compact-card .value { color: #f8fafc; font-size: 1.15rem; font-weight: 650; margin-top: 0.22rem; }
        .compact-card .detail { color: #64748b; font-size: 0.7rem; margin-top: 0.18rem; }
        .section-label { color: #cbd5e1; font-size: 0.82rem; font-weight: 650; letter-spacing: 0.02em; margin: 0.55rem 0 0.35rem; }
        .signal-buy { color: #4ade80 !important; }
        .signal-sell { color: #fb7185 !important; }
        .status-pill { display: inline-block; border-radius: 999px; padding: 0.25rem 0.55rem; font-size: 0.7rem; font-weight: 650; background: #172033; color: #cbd5e1; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(label: str, value: str, detail: str = "", value_class: str = "") -> str:
    return (
        f'<div class="compact-card"><div class="label">{label}</div>'
        f'<div class="value {value_class}">{value}</div>'
        f'<div class="detail">{detail}</div></div>'
    )


def _card_row(cards: list[str]) -> None:
    columns = st.columns(len(cards), gap="small")
    for column, card in zip(columns, cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)


def run() -> None:
    st.set_page_config(page_title="AI Trading App", layout="wide")
    _inject_dashboard_styles()
    if "run_requested" not in st.session_state:
        st.session_state.run_requested = False
    if "paper_executor" not in st.session_state:
        st.session_state.paper_executor = PaperTradingExecutor()

    title_col, status_col = st.columns([3, 1])
    with title_col:
        st.markdown('<p class="dashboard-title">AI Trading Terminal</p>', unsafe_allow_html=True)
        st.markdown('<p class="dashboard-subtitle">Deriv market intelligence, forecast signals, and paper execution</p>', unsafe_allow_html=True)
    with status_col:
        st.markdown('<div style="text-align:right"><span class="status-pill">PAPER MODE</span></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls")
        symbol = st.selectbox("Select Symbol", list(SYMBOLS.keys()))
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=0)
        candle_count = st.slider("History Length", min_value=250, max_value=5000, value=1200, step=50)
        risk_percentage = st.slider("Risk Per Trade", min_value=0.005, max_value=0.05, value=0.01, step=0.005)
        account_balance = st.number_input("Account Balance", min_value=100.0, value=10_000.0, step=100.0)
        live_refresh = st.toggle("Live Refresh", value=False)
        refresh_seconds = st.slider("Refresh Seconds", min_value=5, max_value=60, value=15, step=5)
        settings = get_settings()
        connection_state = describe_connection_state(settings)
        st.divider()
        st.caption("Deriv connectivity")
        st.write(f"Public market data: `{ 'Ready' if connection_state['public_data_enabled'] else 'Unavailable' }`")
        st.write(f"Account auth: `{connection_state['auth_status']}`")
        st.write(f"Account mode: `{connection_state['account_mode']}`")
        if st.button("Run Analysis", type="primary"):
            st.session_state.run_requested = True

    if live_refresh and st.session_state.run_requested:
        st_autorefresh(interval=refresh_seconds * 1000, key="market-autorefresh")

    if not st.session_state.run_requested:
        st.info("Choose a symbol and run the analysis to build a forecast, trade plan, and backtest.")
        return

    with st.spinner("Fetching data and training the forecast ensemble..."):
        raw_data, data_status = fetch_historical_data_detailed(symbol, timeframe, candle_count)
        source = data_status.source
        feature_frame = add_features(raw_data)
        model_bundle = train_price_model(raw_data)
        forecast = predict_next_price(model_bundle, raw_data)
        market_state = summarize_market_state(feature_frame)
        trade_plan = build_trade_plan(
            current_price=forecast["current_price"],
            predicted_price=forecast["predicted_price"],
            atr=float(feature_frame["atr"].iloc[-1]),
            account_balance=account_balance,
            risk_percentage=risk_percentage,
            confidence=forecast["confidence"],
        )
        trades, backtest_metrics = backtest_strategy(raw_data, model_bundle, risk_percentage, starting_balance=account_balance)

    _card_row(
        [
            _card("Data Source", source.replace("_", " ").upper(), data_status.message),
            _card("Trend", str(market_state["trend"])),
            _card("Predicted Return", f"{forecast['predicted_return'] * 100:.2f}%"),
            _card("Confidence", f"{forecast['confidence'] * 100:.1f}%"),
        ]
    )

    if source == "deriv_public":
        st.success(data_status.message)
    else:
        st.warning(data_status.message)

    chart_col, side_col = st.columns([2.25, 0.85], gap="medium")
    with chart_col:
        st.markdown('<p class="section-label">LIVE CANDLESTICK CHART</p>', unsafe_allow_html=True)
        st.plotly_chart(
            build_candlestick_chart(feature_frame.tail(180), symbol, trade_plan),
            use_container_width=True,
        )

    with side_col:
        st.markdown('<p class="section-label">SIGNAL & EXECUTION</p>', unsafe_allow_html=True)
        signal_class = "signal-buy" if trade_plan.signal == "BUY" else "signal-sell"
        st.markdown(_card("Signal", trade_plan.signal, f"Risk/reward {trade_plan.risk_reward_ratio:.2f}", signal_class), unsafe_allow_html=True)
        st.markdown(_card("Entry", f"{trade_plan.entry_price:.4f}"), unsafe_allow_html=True)
        st.markdown(_card("Stop / Target", f"{trade_plan.stop_loss:.4f} / {trade_plan.take_profit:.4f}"), unsafe_allow_html=True)
        st.caption(f"Size {trade_plan.position_size_units:.2f} units · Risk {trade_plan.capital_at_risk:.2f}")
        paper_executor = st.session_state.paper_executor
        ctrader_executor = CTraderFixExecutor()
        st.caption(f"Paper executor: {len(paper_executor.orders)} orders")
        st.caption(f"cTrader FIX: {'ready' if ctrader_executor.is_ready() else 'stub'}")

        if st.button("Simulate Paper Trade"):
            paper_result = paper_executor.submit_order(
                OrderRequest(
                    symbol=symbol,
                    side=trade_plan.signal,
                    quantity=trade_plan.position_size_units,
                    price=trade_plan.entry_price,
                    stop_loss=trade_plan.stop_loss,
                    take_profit=trade_plan.take_profit,
                )
            )
            st.success(f"{paper_result.message} Order ID: {paper_result.order_id}")

    st.markdown('<p class="section-label">MARKET SNAPSHOT</p>', unsafe_allow_html=True)
    _card_row(
        [
            _card("Current Price", f"{forecast['current_price']:.4f}"),
            _card("Predicted Price", f"{forecast['predicted_price']:.4f}"),
            _card("RSI (14)", f"{market_state['rsi']:.1f}"),
            _card("ATR %", f"{market_state['atr_pct'] * 100:.2f}%"),
        ]
    )

    left_metrics, right_metrics = st.columns(2, gap="medium")
    with left_metrics:
        st.markdown('<p class="section-label">MODEL QUALITY</p>', unsafe_allow_html=True)
        _card_row(
            [
                _card("MAE", f"{model_bundle.metrics['mae']:.4f}"),
                _card("RMSE", f"{model_bundle.metrics['rmse']:.4f}"),
                _card("Direction", f"{model_bundle.metrics['directional_accuracy'] * 100:.1f}%"),
            ]
        )
    with right_metrics:
        st.markdown('<p class="section-label">BACKTEST SUMMARY</p>', unsafe_allow_html=True)
        _card_row(
            [
                _card("Balance", f"{backtest_metrics['ending_balance']:.2f}"),
                _card("Return", f"{backtest_metrics['total_return_pct']:.2f}%"),
                _card("Win Rate", f"{backtest_metrics['win_rate_pct']:.1f}%"),
                _card("Drawdown", f"{backtest_metrics['max_drawdown_pct']:.1f}%"),
            ]
        )

    with st.expander("Data & trade details"):
        st.dataframe(feature_frame.tail(20), use_container_width=True, height=260)

    if not trades.empty:
        with st.expander("Backtest trades"):
            st.dataframe(trades.tail(20), use_container_width=True, height=260)
    else:
        st.warning("The backtest did not find enough high-confidence setups under the current thresholds.")

    if paper_executor.orders:
        with st.expander("Paper orders"):
            st.dataframe(paper_executor.orders[-10:], use_container_width=True, height=220)

    with st.expander("Future Account Auth Setup"):
        st.write("The app is already prepared for optional authenticated Deriv account access later.")
        st.write("Add `DERIV_APP_ID`, `DERIV_API_TOKEN`, and `DERIV_ACCOUNT_ID` to a local `.env` file when you want to enable account-level features such as balance sync or live execution.")
        st.write("For cTrader FIX preparation, add quote/trade hosts, ports, SenderCompID, and password values to `.env` when you are ready.")


if __name__ == "__main__":
    run()
