from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class OrderResult:
    accepted: bool
    order_id: str
    status: str
    message: str
    filled_price: float | None = None


class BrokerExecutor(Protocol):
    def get_name(self) -> str: ...

    def is_ready(self) -> bool: ...

    def get_status(self) -> dict[str, str | bool]: ...

    def submit_order(self, request: OrderRequest) -> OrderResult: ...

