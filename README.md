# AI Trading App

This project is a Streamlit-based AI trading prototype with:

- feature-engineered market forecasting (ensemble predicting the next candle's return)
- honest model evaluation against a "price stays the same" baseline
- a HOLD signal when the forecast is too weak to act on
- risk-aware trade sizing
- rule-based stop loss and take profit levels
- walk-forward backtesting on bars the model never trained on
- public Deriv market data support without account login
- a demo-data fallback when Deriv market data is unavailable
- per-browser Deriv OAuth 2.0 sign-in with PKCE and account selection
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

4. Optional Deriv account login: register an OAuth 2.0 app in Deriv, set its redirect URL to the URL where Streamlit is hosted, and configure `DERIV_APP_ID` and `DERIV_OAUTH_REDIRECT_URI` in `.env`. For local use the callback can be `http://localhost:8501`; the URI must exactly match the registered value. Without these values, public market data and paper trading remain available.

5. Start the app:

```bash
streamlit run main.py
```

The app first queries Deriv's current public WebSocket endpoint and its `active_symbols` endpoint to discover the exact broker symbol code, then requests candle history for that code. If Deriv is unavailable from your machine, it falls back to demo-generated data and displays the actual connection error in the app.

## Run with Docker

```bash
docker build -t ai_trading_app .
docker run -p 8501:8501 --env-file .env ai_trading_app
```

`.dockerignore` keeps `.env`, API key files and `.venv` out of the image, so settings are passed in at run time with `--env-file`.

## How the forecast works

- **Features** (`app/utils.py`): returns, EMA distance, volatility, RSI, ATR %, momentum and distance from the 20-bar range. All are scale-free ratios, not raw prices. Features are computed for every candle including the newest one; the forecast is always made from the latest candle.
- **Model** (`app/models.py`): Linear Regression, Random Forest and Gradient Boosting predict the next candle's % return. Data is split in time order: the first 60% fits the models, the next 20% sets each model's ensemble weight (inverse error), and the last 20% is only used to report MAE, RMSE and direction accuracy. The dashboard shows these next to a "price stays the same" baseline and a 50% coin flip; on Deriv's synthetic indices, expect the model to be close to both. The live model is then refit on all labelled data with the same weights.
- **Signal** (`app/risk_management.py`): a forecast only becomes BUY or SELL when the predicted move is at least 0.35× ATR and the signal strength is at least 18%. Otherwise the dashboard shows HOLD, removes the stop/target lines and disables paper and Deriv trades. Signal strength is |predicted return| ÷ ATR %, scaled to 0–100%; it is not a calibrated probability.
- **Backtest** (`app/backtesting.py`): walk-forward. The model is retrained about six times across the history, each time only on candles before the ones it then trades, using the same HOLD rule as the dashboard.
- **Caching** (`app/main.py`): training and the backtest are cached and redone after 15 new candles or when a setting changes. The forecast itself always uses the newest data. Deriv's symbol list is cached for an hour.

## Deriv logins

- Public market data:
  no personal login required. This supports market data, analysis, and backtesting.
- Each user signs in directly at Deriv through OAuth 2.0 Authorization Code + PKCE. The app exchanges the one-time code server-side, verifies the state, lists that user's Deriv accounts, and keeps the access token only in that Streamlit browser session's memory. Deriv does not issue refresh tokens, so users sign in again after expiry. Do not run this on a shared public server without HTTPS and a registered HTTPS callback URL.
- Register only the `trade` scope used by account listing. OAuth app ID and callback URL are configured with `DERIV_APP_ID` and `DERIV_OAUTH_REDIRECT_URI`; no user's password or API token is stored in `.env`.
- The sidebar's **Sync Deriv balance** action reads balance from the selected account's authenticated WebSocket and updates the account balance used by the risk-plan view. **Place Deriv contract** requests a Rise/Fall quote and then buys the contract on that same selected account, after the user confirms the action. Stake is the maximum amount risked; the existing position-size estimate is not reused because it measures underlying units, not an options stake. Real accounts show an additional warning. These actions require an OAuth token with the `trade` scope.

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

## Planned live chart and trade monitoring

- Done: the chart streams live candles by default (`app/live_chart.py`). The browser subscribes to Deriv's public WebSocket and updates the last candle on every tick, about once a second, with EMA 10/30 and entry/stop/target lines. Turn off **Live chart** in the sidebar for the static Plotly chart. The forecast and signal still update only on **Run Analysis** or **Live Refresh**.
- Show active paper trades and authenticated Deriv demo or real trades on the live chart, including entry, stop/target where applicable, current price relative to the position, and trade status/P&L when available.
- After a paper trade is opened, keep the chart streaming and make that position easy to monitor. Live/demo trades should be tracked from the selected authenticated Deriv account.
