from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class DerivSettings:
    public_market_ws_url: str = os.getenv(
        "DERIV_PUBLIC_WS_URL",
        "wss://api.derivws.com/trading/v1/options/ws/public",
    )
    ws_auth_base_url: str = os.getenv(
        "DERIV_AUTH_BASE_URL",
        "https://api.derivws.com/trading/v1/options/accounts",
    )
    app_id: str = os.getenv("DERIV_APP_ID", "").strip()
    oauth_redirect_uri: str = os.getenv("DERIV_OAUTH_REDIRECT_URI", "").strip()
    api_token: str = os.getenv("DERIV_API_TOKEN", "").strip()
    account_id: str = os.getenv("DERIV_ACCOUNT_ID", "").strip()
    account_mode: str = os.getenv("DERIV_ACCOUNT_MODE", "demo").strip().lower()
    ctrader_quote_host: str = os.getenv("CTRADER_QUOTE_HOST", "").strip()
    ctrader_quote_port_ssl: str = os.getenv("CTRADER_QUOTE_PORT_SSL", "").strip()
    ctrader_quote_port_plain: str = os.getenv("CTRADER_QUOTE_PORT_PLAIN", "").strip()
    ctrader_trade_host: str = os.getenv("CTRADER_TRADE_HOST", "").strip()
    ctrader_trade_port_ssl: str = os.getenv("CTRADER_TRADE_PORT_SSL", "").strip()
    ctrader_trade_port_plain: str = os.getenv("CTRADER_TRADE_PORT_PLAIN", "").strip()
    ctrader_sender_comp_id: str = os.getenv("CTRADER_SENDER_COMP_ID", "").strip()
    ctrader_target_comp_id: str = os.getenv("CTRADER_TARGET_COMP_ID", "CSERVER").strip()
    ctrader_quote_sender_sub_id: str = os.getenv("CTRADER_QUOTE_SENDER_SUB_ID", "QUOTE").strip()
    ctrader_trade_sender_sub_id: str = os.getenv("CTRADER_TRADE_SENDER_SUB_ID", "TRADE").strip()
    ctrader_password: str = os.getenv("CTRADER_PASSWORD", "").strip()

    @property
    def public_data_enabled(self) -> bool:
        return bool(self.public_market_ws_url)

    @property
    def account_auth_ready(self) -> bool:
        return bool(self.app_id and self.oauth_redirect_uri)

    @property
    def ctrader_quote_ready(self) -> bool:
        return bool(
            self.ctrader_quote_host
            and self.ctrader_quote_port_ssl
            and self.ctrader_sender_comp_id
            and self.ctrader_password
        )

    @property
    def ctrader_trade_ready(self) -> bool:
        return bool(
            self.ctrader_trade_host
            and self.ctrader_trade_port_ssl
            and self.ctrader_sender_comp_id
            and self.ctrader_password
        )

    @property
    def ctrader_fix_ready(self) -> bool:
        return self.ctrader_quote_ready and self.ctrader_trade_ready


def get_settings() -> DerivSettings:
    return DerivSettings()


def describe_connection_state(settings: DerivSettings) -> dict[str, str | bool]:
    return {
        "public_data_enabled": settings.public_data_enabled,
        "account_auth_ready": settings.account_auth_ready,
        "account_mode": settings.account_mode.upper(),
        "auth_status": (
            "Configured"
            if settings.account_auth_ready
            else "Not configured"
        ),
        "ctrader_fix_ready": settings.ctrader_fix_ready,
    }
