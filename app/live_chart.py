"""Live candlestick chart that streams straight from Deriv in the browser.

Streamlit only reruns Python when something triggers it, so a Python-drawn
chart can't move between reruns. This chart is a small HTML/JS component: the
browser opens Deriv's public WebSocket itself, subscribes to candles for the
selected symbol, and updates the last candle on every tick (about once a
second). Python only supplies the starting candles (shown if the live feed
can't connect) and the entry / stop / target lines.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit.components.v1 as components

from .risk_management import TradePlan


LIGHTWEIGHT_CHARTS_URL = (
    "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
)

_TEMPLATE = """
<div id="wrap" style="font-family: system-ui, sans-serif; background:#0f172a; border-radius:8px; padding:8px 8px 0;">
  <div style="display:flex; justify-content:space-between; align-items:center; color:#e5e7eb; font-size:13px; padding:0 4px 6px;">
    <span id="title"></span>
    <span id="status" style="font-size:12px; color:#94a3b8;">Connecting…</span>
  </div>
  <div id="chart"></div>
</div>
<script src="__LIB__"></script>
<script>
(function () {
  const cfg = __CONFIG__;
  const statusEl = document.getElementById("status");
  document.getElementById("title").textContent = cfg.title;

  function setStatus(text, color) {
    statusEl.textContent = text;
    statusEl.style.color = color;
  }

  if (!window.LightweightCharts) {
    setStatus("Chart library failed to load", "#fb7185");
    return;
  }

  const chart = LightweightCharts.createChart(document.getElementById("chart"), {
    height: cfg.height,
    layout: { background: { color: "#0f172a" }, textColor: "#e5e7eb" },
    grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
    timeScale: { timeVisible: true, secondsVisible: cfg.granularity < 60, borderColor: "#243244" },
    rightPriceScale: { borderColor: "#243244" },
    crosshair: { mode: 0 },
  });
  new ResizeObserver(() => chart.applyOptions({ width: document.getElementById("wrap").clientWidth - 16 }))
    .observe(document.getElementById("wrap"));

  const candles = chart.addCandlestickSeries({
    upColor: "#22c55e", downColor: "#ef4444", borderVisible: false,
    wickUpColor: "#22c55e", wickDownColor: "#ef4444",
  });
  const emaFast = chart.addLineSeries({ color: "#60a5fa", lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: "EMA 10" });
  const emaSlow = chart.addLineSeries({ color: "#f59e0b", lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: "EMA 30" });
  cfg.lines.forEach((line) => candles.createPriceLine({
    price: line.price, color: line.color, lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: line.label,
  }));

  let bars = [];
  const MAX_BARS = 1000;

  function ema(values, span) {
    const alpha = 2 / (span + 1);
    let prev;
    return values.map((value, i) => (prev = i === 0 ? value : alpha * value + (1 - alpha) * prev));
  }

  function redrawEmas() {
    const closes = bars.map((bar) => bar.close);
    const fast = ema(closes, 10);
    const slow = ema(closes, 30);
    emaFast.setData(bars.map((bar, i) => ({ time: bar.time, value: fast[i] })));
    emaSlow.setData(bars.map((bar, i) => ({ time: bar.time, value: slow[i] })));
  }

  function setBars(newBars) {
    bars = newBars.slice(-MAX_BARS);
    candles.setData(bars);
    redrawEmas();
  }

  function applyPrecision(pipSize) {
    if (pipSize === undefined || pipSize === null) return;
    const precision = Number(pipSize);
    candles.applyOptions({ priceFormat: { type: "price", precision: precision, minMove: Math.pow(10, -precision) } });
  }

  setBars(cfg.initial);
  chart.timeScale().scrollToRealTime();

  if (!cfg.wsUrl || !cfg.symbol) {
    setStatus("Live feed unavailable · showing loaded candles", "#fbbf24");
    return;
  }

  let socket;
  let retries = 0;
  let pingTimer;

  function connect() {
    setStatus("Connecting…", "#94a3b8");
    socket = new WebSocket(cfg.wsUrl);
    socket.onopen = () => {
      retries = 0;
      socket.send(JSON.stringify({
        ticks_history: cfg.symbol, style: "candles", granularity: cfg.granularity,
        end: "latest", count: cfg.count, adjust_start_time: 1, subscribe: 1,
      }));
      pingTimer = setInterval(() => socket.readyState === 1 && socket.send(JSON.stringify({ ping: 1 })), 30000);
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.error) {
        setStatus("Live feed error: " + message.error.message, "#fb7185");
        return;
      }
      if (message.msg_type === "candles") {
        applyPrecision(message.pip_size);
        setBars(message.candles.map((c) => ({
          time: Number(c.epoch), open: Number(c.open), high: Number(c.high), low: Number(c.low), close: Number(c.close),
        })));
      } else if (message.msg_type === "ohlc") {
        const o = message.ohlc;
        applyPrecision(o.pip_size);
        const bar = {
          time: Number(o.open_time), open: Number(o.open), high: Number(o.high), low: Number(o.low), close: Number(o.close),
        };
        const last = bars[bars.length - 1];
        if (last && last.time === bar.time) {
          bars[bars.length - 1] = bar;
        } else if (!last || bar.time > last.time) {
          bars.push(bar);
          if (bars.length > MAX_BARS) bars.shift();
        } else {
          return;
        }
        candles.update(bar);
        redrawEmas();
        const tickTime = new Date(Number(o.epoch) * 1000).toISOString().substring(11, 19);
        setStatus("● LIVE · " + bar.close.toFixed(Math.max(0, Number(o.pip_size) || 2)) + " · " + tickTime + " UTC", "#4ade80");
      }
    };
    socket.onerror = () => setStatus("Live feed connection error", "#fb7185");
    socket.onclose = () => {
      clearInterval(pingTimer);
      const delay = Math.min(30000, 1000 * Math.pow(2, retries++));
      setStatus("Disconnected · retrying in " + Math.round(delay / 1000) + "s", "#fbbf24");
      setTimeout(connect, delay);
    };
  }

  connect();
})();
</script>
"""


def render_live_chart(
    frame: pd.DataFrame,
    title: str,
    api_symbol: str | None,
    granularity: int,
    ws_url: str,
    trade_plan: TradePlan | None = None,
    height: int = 600,
    count: int = 300,
) -> None:
    recent = frame.tail(count)
    initial = [
        {
            "time": int(pd.Timestamp(row.timestamp).timestamp()),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in recent.itertuples()
    ]
    lines = []
    if trade_plan is not None:
        lines = [
            {"label": "Entry", "price": trade_plan.entry_price, "color": "#e5e7eb"},
            {"label": "Stop", "price": trade_plan.stop_loss, "color": "#ef4444"},
            {"label": "Target", "price": trade_plan.take_profit, "color": "#22c55e"},
        ]
    config = {
        "title": title,
        "symbol": api_symbol,
        "wsUrl": ws_url if api_symbol else "",
        "granularity": granularity,
        "count": count,
        "height": height,
        "initial": initial,
        "lines": lines,
    }
    # "</" is escaped so data can never close the <script> tag early.
    config_json = json.dumps(config).replace("</", "<\\/")
    html = _TEMPLATE.replace("__LIB__", LIGHTWEIGHT_CHARTS_URL).replace("__CONFIG__", config_json)
    components.html(html, height=height + 50)
