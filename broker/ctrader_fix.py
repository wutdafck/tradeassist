from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings

from .interfaces import BrokerExecutor, OrderRequest, OrderResult


@dataclass
class CTraderFixExecutor(BrokerExecutor):
    def get_name(self) -> str:
        return "cTrader FIX"

    def is_ready(self) -> bool:
        return get_settings().ctrader_fix_ready

    def get_status(self) -> dict[str, str | bool]:
        settings = get_settings()
        return {
            "executor": self.get_name(),
            "ready": settings.ctrader_fix_ready,
            "quote_session": "Configured" if settings.ctrader_quote_ready else "Not configured",
            "trade_session": "Configured" if settings.ctrader_trade_ready else "Not configured",
            "target_comp_id": settings.ctrader_target_comp_id or "CSERVER",
        }

    def submit_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            accepted=False,
            order_id="stub",
            status="NOT_IMPLEMENTED",
            message="cTrader FIX execution is not wired yet. This stub defines the future interface only.",
            filled_price=None,
        )

