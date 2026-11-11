from __future__ import annotations

import time

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd

from .api import fetch_historical_data_detailed, timeframe_to_granularity
from .backtesting import backtest_strategy
from .charts import build_candlestick_chart
from .live_chart import render_live_chart
from .config import describe_connection_state, get_settings
from .deriv_auth import (
    account_id,
    account_label,
    create_login,
    exchange_code,
    fetch_accounts,
    fetch_balance,
    place_rise_fall_contract,
)
from .models import train_price_model, predict_next_price
from .risk_management import build_trade_plan, trade_filter
from .utils import SYMBOLS, add_features, summarize_market_state, symbol_to_api_code
from broker.ctrader_fix import CTraderFixExecutor
from broker.paper import PaperTradingExecutor
from broker.interfaces import OrderRequest


# Training and the walk-forward backtest are the slow parts, so they are cached
# and only redone once this many new candles have arrived (or a setting changes).
RETRAIN_EVERY_CANDLES = 15


def _retrain_anchor(raw_data: pd.DataFrame, timeframe: str) -> str:
    """Cache key that changes once every RETRAIN_EVERY_CANDLES candles."""
    block_seconds = timeframe_to_granularity(timeframe) * RETRAIN_EVERY_CANDLES
    return str(pd.Timestamp(raw_data["timestamp"].iloc[-1]).floor(f"{block_seconds}s"))


# Arguments starting with "_" are not hashed: the data is identified by the
# symbol/timeframe/count/source/anchor arguments instead.
@st.cache_data(max_entries=8, show_spinner=False)
def _cached_model(symbol: str, timeframe: str, candle_count: int, source: str, anchor: str, _raw_data: pd.DataFrame):
    return train_price_model(_raw_data)


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_backtest(
    symbol: str,
    timeframe: str,
    candle_count: int,
    source: str,
    anchor: str,
    risk_percentage: float,
    account_balance: float,
    _raw_data: pd.DataFrame,
):
    return backtest_strategy(_raw_data, risk_percentage, starting_balance=account_balance)


def _handle_deriv_login(settings) -> None:
    """Complete the OAuth callback using PKCE values kept in this Streamlit session."""
    params = st.query_params
    code = params.get("code")
    returned_state = params.get("state")
    if not code and not params.get("error"):
        return
    try:
        if params.get("error"):
            raise ValueError("Deriv sign-in was cancelled or denied.")
        if not returned_state or returned_state != st.session_state.get("deriv_oauth_state"):
            raise ValueError("Deriv sign-in could not be verified. Please start again.")
        if not settings.app_id or not settings.oauth_redirect_uri:
            raise ValueError("Deriv OAuth is not configured for this app.")
        token_data = exchange_code(
            code,
            settings.app_id,
            settings.oauth_redirect_uri,
            st.session_state.get("deriv_oauth_verifier", ""),
        )
        token = token_data["access_token"]
        accounts = fetch_accounts(token)
        if not accounts:
            raise ValueError("No Deriv trading accounts were returned for this login.")
        st.session_state.deriv_access_token = token
        st.session_state.deriv_access_token_expires_at = time.time() + int(token_data.get("expires_in", 0))
        st.session_state.deriv_accounts = accounts
        ids = [account_id(account) for account in accounts if account_id(account)]
        if ids:
            st.session_state.deriv_selected_account = ids[0]
        st.session_state.deriv_oauth_verifier = ""
        st.session_state.deriv_oauth_state = ""
        st.session_state.deriv_login_error = ""
    except Exception as error:
        st.session_state.deriv_login_error = str(error)
        st.session_state.deriv_oauth_verifier = ""
        st.session_state.deriv_oauth_state = ""
    finally:
        st.query_params.clear()


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
        .signal-hold { color: #fbbf24 !important; }
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
    settings = get_settings()
    _handle_deriv_login(settings)
    _inject_dashboard_styles()
    if "run_requested" not in st.session_state:
        st.session_state.run_requested = False
    if "paper_executor" not in st.session_state:
        st.session_state.paper_executor = PaperTradingExecutor()
    pending_balance = st.session_state.pop("deriv_pending_balance_sync", None)
    if pending_balance:
        st.session_state["account_balance"] = float(pending_balance["balance"])
        st.session_state["account_balance_source_account"] = pending_balance["account_id"]

    title_col, status_col = st.columns([3, 1])
    with title_col:
        st.markdown('<p class="dashboard-title">AI Trading Terminal</p>', unsafe_allow_html=True)
        st.markdown('<p class="dashboard-subtitle">Deriv market intelligence, forecast signals, and paper execution</p>', unsafe_allow_html=True)
    with status_col:
        mode_label = "DERIV CONNECTED" if st.session_state.get("deriv_access_token") else "PAPER MODE"
        st.markdown(f'<div style="text-align:right"><span class="status-pill">{mode_label}</span></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Controls")
        symbol = st.selectbox("Select Symbol", list(SYMBOLS.keys()))
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=0)
        candle_count = st.slider("History Length", min_value=250, max_value=5000, value=1200, step=50)
        risk_percentage = st.slider("Risk Per Trade", min_value=0.005, max_value=0.05, value=0.01, step=0.005)
        account_balance = st.number_input("Account Balance", min_value=0.01, value=10_000.0, step=100.0, key="account_balance")
        live_chart = st.toggle(
            "Live chart",
            value=True,
            help="Streams candles from Deriv in your browser, tick by tick. The forecast and signal update on Live Refresh or Run Analysis.",
        )
        live_refresh = st.toggle("Live Refresh", value=False, help="Re-runs the forecast and signal on a timer.")
        refresh_seconds = st.slider("Refresh Seconds", min_value=5, max_value=60, value=15, step=5)
        connection_state = describe_connection_state(settings)
        st.divider()
        st.caption("Deriv account")
        if st.session_state.get("deriv_access_token"):
            token_expired = time.time() >= st.session_state.get("deriv_access_token_expires_at", 0)
            accounts = st.session_state.get("deriv_accounts", [])
            account_ids = [account_id(account) for account in accounts]
            account_ids = [identifier for identifier in account_ids if identifier]
            if account_ids:
                labels = {account_id(account): account_label(account) for account in accounts if account_id(account)}
                selected = st.selectbox(
                    "Signed in as",
                    account_ids,
                    format_func=lambda identifier: labels.get(identifier, identifier),
                    key="deriv_selected_account",
                )
                st.caption(f"Connected Deriv account: {selected}")
                if token_expired:
                    st.warning("Deriv authorization expired. Sign out and sign in again.")
                if st.button("Sync Deriv balance", disabled=token_expired):
                    try:
                        balance = fetch_balance(st.session_state.deriv_access_token, selected)
                        st.session_state.deriv_synced_balance = balance
                        st.session_state.deriv_synced_balance_account = selected
                        st.session_state.deriv_pending_balance_sync = {
                            "account_id": selected,
                            "balance": balance["balance"],
                        }
                        st.rerun()
                    except Exception as error:
                        st.error(f"Balance sync failed: {error}")
                synced_balance = st.session_state.get("deriv_synced_balance")
                if synced_balance and st.session_state.get("deriv_synced_balance_account") == selected:
                    st.metric("Deriv balance", f"{synced_balance['balance']:,.2f} {synced_balance.get('currency', '')}")
            else:
                st.caption("Connected to Deriv")
            if st.button("Sign out of Deriv"):
                for key in ("deriv_access_token", "deriv_access_token_expires_at", "deriv_accounts", "deriv_selected_account", "deriv_synced_balance", "deriv_synced_balance_account"):
                    st.session_state.pop(key, None)
                st.rerun()
        elif settings.app_id and settings.oauth_redirect_uri:
            if "deriv_oauth_state" not in st.session_state or not st.session_state.deriv_oauth_state:
                login_url, state, verifier = create_login(settings.app_id, settings.oauth_redirect_uri)
                st.session_state.deriv_login_url = login_url
                st.session_state.deriv_oauth_state = state
                st.session_state.deriv_oauth_verifier = verifier
            st.link_button("Sign in with Deriv", st.session_state.deriv_login_url)
            st.caption("Each browser session gets its own Deriv authorization and account selection.")
        else:
            st.caption("Deriv sign-in is not configured.")
            st.caption("Set DERIV_APP_ID and DERIV_OAUTH_REDIRECT_URI to enable individual logins.")
        if st.session_state.get("deriv_login_error"):
            st.error(st.session_state.deriv_login_error)
        st.caption(f"Public market data: {'Ready' if connection_state['public_data_enabled'] else 'Unavailable'}")
        if st.button("Run Analysis", type="primary"):
            st.session_state.run_requested = True

    if live_refresh and st.session_state.run_requested:
        st_autorefresh(interval=refresh_seconds * 1000, key="market-autorefresh")

    if not st.session_state.run_requested:
        st.info("Choose a symbol and run the analysis to build a forecast, trade plan, and backtest.")
        return

    with st.spinner("Fetching market data..."):
        raw_data, data_status = fetch_historical_data_detailed(symbol, timeframe, candle_count)
    source = data_status.source
    anchor = _retrain_anchor(raw_data, timeframe)
    with st.spinner("Training the forecast ensemble and running the walk-forward backtest..."):
        model_bundle = _cached_model(symbol, timeframe, candle_count, source, anchor, raw_data)
        trades, backtest_metrics = _cached_backtest(
            symbol, timeframe, candle_count, source, anchor, risk_percentage, account_balance, raw_data
        )
    # Forecasting from the newest candle is cheap, so it always uses fresh data,
    # even when the model itself comes from the cache.
    feature_frame = add_features(raw_data)
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
    should_trade, filter_reason = trade_filter(
        forecast["predicted_return"],
        float(feature_frame["atr_pct"].iloc[-1]),
        forecast["confidence"],
    )
    model_trained_through = model_bundle.training_frame["timestamp"].iloc[-1]

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
        chart_plan = trade_plan if should_trade else None
        if live_chart:
            st.markdown('<p class="section-label">LIVE CANDLESTICK CHART</p>', unsafe_allow_html=True)
            render_live_chart(
                raw_data,
                title=data_status.symbol_code or symbol,
                api_symbol=data_status.api_symbol,
                granularity=timeframe_to_granularity(timeframe),
                ws_url=settings.public_market_ws_url,
                trade_plan=chart_plan,
            )
        else:
            st.markdown('<p class="section-label">CANDLESTICK CHART</p>', unsafe_allow_html=True)
            st.plotly_chart(
                build_candlestick_chart(feature_frame.tail(180), symbol, chart_plan),
                use_container_width=True,
            )
        st.caption(
            f"Model trained on candles up to {model_trained_through:%Y-%m-%d %H:%M} UTC; "
            f"it retrains every {RETRAIN_EVERY_CANDLES} new candles or when settings change."
        )

    with side_col:
        st.markdown('<p class="section-label">SIGNAL & EXECUTION</p>', unsafe_allow_html=True)
        if should_trade:
            signal_class = "signal-buy" if trade_plan.signal == "BUY" else "signal-sell"
            st.markdown(_card("Signal", trade_plan.signal, f"Risk/reward {trade_plan.risk_reward_ratio:.2f}", signal_class), unsafe_allow_html=True)
            st.markdown(_card("Entry", f"{trade_plan.entry_price:.4f}"), unsafe_allow_html=True)
            st.markdown(_card("Stop / Target", f"{trade_plan.stop_loss:.4f} / {trade_plan.take_profit:.4f}"), unsafe_allow_html=True)
            st.caption(f"Size {trade_plan.position_size_units:.2f} units · Risk {trade_plan.capital_at_risk:.2f}")
        else:
            st.markdown(
                _card("Signal", "HOLD", f"Leaning {trade_plan.signal} · {filter_reason}", "signal-hold"),
                unsafe_allow_html=True,
            )
            st.markdown(_card("Current price", f"{trade_plan.entry_price:.4f}", "No trade plan while the signal is HOLD"), unsafe_allow_html=True)
        paper_executor = st.session_state.paper_executor
        ctrader_executor = CTraderFixExecutor()
        st.caption(f"Paper executor: {len(paper_executor.orders)} orders")
        st.caption(f"cTrader FIX: {'ready' if ctrader_executor.is_ready() else 'stub'}")

        if st.button("Simulate Paper Trade", disabled=not should_trade):
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

        if st.session_state.get("deriv_access_token"):
            selected_id = st.session_state.get("deriv_selected_account", "")
            selected_account = next(
                (account for account in st.session_state.get("deriv_accounts", []) if account_id(account) == selected_id),
                {},
            )
            currency = str(selected_account.get("currency") or "USD")
            account_kind = str(selected_account.get("account_type") or selected_account.get("type") or selected_account.get("mode") or "").lower()
            is_real = "real" in account_kind or selected_id.startswith("CR")
            st.divider()
            st.markdown("**Deriv contract execution**")
            st.caption("Places a Rise/Fall options contract using the current signal. Stake is the maximum amount at risk.")
            stake = st.number_input(f"Contract stake ({currency})", min_value=0.35, value=1.0, step=0.25, key="deriv_contract_stake")
            duration_minutes = st.selectbox("Contract duration", [1, 2, 5, 10, 15], index=2, format_func=lambda value: f"{value} minute{'s' if value != 1 else ''}")
            real_confirmed = st.checkbox(
                "I confirm placing this contract on the selected Deriv account",
                key="deriv_trade_confirm",
            )
            if is_real:
                st.warning("The selected account appears to be a real-money account. The stake may be lost.")
            synced = st.session_state.get("deriv_synced_balance")
            balance_is_current = st.session_state.get("deriv_synced_balance_account") == selected_id and synced is not None
            available_balance = float(synced["balance"]) if balance_is_current else 0.0
            if not should_trade:
                st.info("The current signal is HOLD, so contract placement is disabled.")
            elif not balance_is_current:
                st.info("Sync the selected account's balance before placing a contract.")
            elif stake > available_balance:
                st.warning(f"Stake exceeds the synced balance of {available_balance:.2f} {currency}.")
            if st.button(
                "Place Deriv contract",
                type="primary",
                disabled=(
                    not should_trade
                    or not real_confirmed
                    or not balance_is_current
                    or stake > available_balance
                    or time.time() >= st.session_state.get("deriv_access_token_expires_at", 0)
                ),
            ):
                try:
                    result = place_rise_fall_contract(
                        st.session_state.deriv_access_token,
                        selected_id,
                        symbol_to_api_code(symbol),
                        trade_plan.signal,
                        stake,
                        currency,
                        duration_minutes,
                    )
                    contract_id = result.get("contract_id", "unknown")
                    buy_price = result.get("buy_price", stake)
                    st.success(f"Deriv contract purchased. Contract ID: {contract_id}; stake: {buy_price} {currency}.")
                    st.session_state.deriv_last_order = {
                        "contract_id": contract_id,
                        "symbol": symbol,
                        "side": trade_plan.signal,
                        "stake": buy_price,
                        "currency": currency,
                    }
                except Exception as error:
                    st.error(f"Deriv order failed: {error}")

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
        metrics = model_bundle.metrics
        st.markdown(
            f'<p class="section-label">MODEL QUALITY · HELD-OUT TEST ({int(metrics["test_rows"])} bars)</p>',
            unsafe_allow_html=True,
        )
        _card_row(
            [
                _card("MAE", f"{metrics['mae']:.4f}", f"No-change baseline {metrics['baseline_mae']:.4f}"),
                _card("RMSE", f"{metrics['rmse']:.4f}", f"No-change baseline {metrics['baseline_rmse']:.4f}"),
                _card("Direction", f"{metrics['directional_accuracy'] * 100:.1f}%", "Coin flip 50.0%"),
            ]
        )
    with right_metrics:
        st.markdown(
            f'<p class="section-label">WALK-FORWARD BACKTEST ({int(backtest_metrics["out_of_sample_bars"])} unseen bars)</p>',
            unsafe_allow_html=True,
        )
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

    with st.expander("Deriv account connection"):
        st.write("Deriv sign-in is handled through OAuth 2.0 with PKCE. Each browser session receives its own authorization and selects from that user's Deriv accounts.")
        st.write("The app does not ask for Deriv passwords or share account credentials between users.")
        st.write("For cTrader FIX preparation, add quote/trade hosts, ports, SenderCompID, and password values to `.env` when you are ready.")


if __name__ == "__main__":
    run()
