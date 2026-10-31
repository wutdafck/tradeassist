# AI Trading App

This project is a Streamlit-based AI trading prototype with:

- feature-engineered market forecasting
- risk-aware trade sizing
- rule-based stop loss and take profit levels
- historical backtesting
- public Deriv market data support without account login
- a demo-data fallback when Deriv market data is unavailable
- optional secure account-auth scaffolding for later live trading
- candlestick charting with timed refresh
- paper-trading executor plus cTrader FIX integration stubs

## Run locally

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local environment file:

```bash
copy .env.example .env
```

4. Optional:
Leave `DERIV_APP_ID`, `DERIV_API_TOKEN`, and `DERIV_ACCOUNT_ID` empty for now if you only want public market data.

5. Start the app:

```bash
streamlit run main.py
```

The app first queries Deriv's current public WebSocket endpoint and its `active_symbols` endpoint to discover the exact broker symbol code, then requests candle history for that code. If Deriv is unavailable from your machine, it falls back to demo-generated data and displays the actual connection error in the app.

## Credential modes

- Public market data:
  no personal login required. This supports market data, analysis, and backtesting.
- Authenticated account mode:
  add `DERIV_APP_ID`, `DERIV_API_TOKEN`, and `DERIV_ACCOUNT_ID` in `.env` later when you are ready to build account-linked features such as balance sync, portfolio state, and live order execution.

## cTrader FIX preparation

The repo also includes a future execution path for cTrader FIX. Add these values to `.env` when you are ready to wire the real connection:

- `CTRADER_QUOTE_HOST`
- `CTRADER_QUOTE_PORT_SSL`
- `CTRADER_QUOTE_PORT_PLAIN`
- `CTRADER_TRADE_HOST`
- `CTRADER_TRADE_PORT_SSL`
- `CTRADER_TRADE_PORT_PLAIN`
- `CTRADER_SENDER_COMP_ID`
- `CTRADER_TARGET_COMP_ID`
- `CTRADER_PASSWORD`

Right now, cTrader FIX is a stub only. The paper executor is the safe test path.
