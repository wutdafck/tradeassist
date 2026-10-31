from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC

from .interfaces import BrokerExecutor, OrderRequest, OrderResult


@dataclass
class PaperTradingExecutor(BrokerExecutor):
    starting_balance: float = 10_000.0
    orders: list[dict] = field(default_factory=list)

    def get_name(self) -> str:
        return "Paper Trading"

    def is_ready(self) -> bool:
        return True

    def get_status(self) -> dict[str, str | bool]:
        return {
            "executor": self.get_name(),
            "ready": True,
            "mode": "SIMULATION",
            "orders_recorded": str(len(self.orders)),
        }

    def submit_order(self, request: OrderRequest) -> OrderResult:
        order_id = f"paper-{len(self.orders) + 1}"
        self.orders.append(
            {
                "order_id": order_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.quantity,
                "price": request.price,
                "stop_loss": request.stop_loss,
                "take_profit": request.take_profit,
            }
        )
        return OrderResult(
            accepted=True,
            order_id=order_id,
            status="FILLED",
            message="Paper trade accepted.",
            filled_price=request.price,
        )

