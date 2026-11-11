from __future__ import annotations

import base64
import hashlib
import secrets
from urllib.parse import urlencode

import requests
from websocket import create_connection
import json
from urllib.parse import quote


AUTH_URL = "https://auth.deriv.com/oauth2/auth"
TOKEN_URL = "https://auth.deriv.com/oauth2/token"
ACCOUNTS_URL = "https://api.derivws.com/trading/v1/options/accounts"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def create_login(client_id: str, redirect_uri: str) -> tuple[str, str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    state = secrets.token_urlsafe(32)
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "trade",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URL}?{urlencode(params)}", state, verifier


def exchange_code(code: str, client_id: str, redirect_uri: str, verifier: str) -> dict:
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
        timeout=20,
    )
    response.raise_for_status()
    token_data = response.json()
    token = token_data.get("access_token")
    if not token:
        raise ValueError("Deriv did not return an access token.")
    return token_data


def fetch_accounts(access_token: str) -> list[dict]:
    response = requests.get(
        ACCOUNTS_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    accounts = body.get("data", body.get("accounts", []))
    if isinstance(accounts, dict):
        accounts = accounts.get("accounts", [])
    if not isinstance(accounts, list):
        raise ValueError("Deriv returned an unexpected account list.")
    return accounts


def account_id(account: dict) -> str:
    return str(account.get("account_id") or account.get("id") or account.get("loginid") or "")


def account_label(account: dict) -> str:
    identifier = account_id(account) or "Account"
    currency = str(account.get("currency") or "")
    mode = str(account.get("account_type") or account.get("type") or account.get("mode") or "")
    parts = [identifier]
    if mode:
        parts.append(mode)
    if currency:
        parts.append(currency)
    return " · ".join(parts)


def _authenticated_ws(access_token: str, selected_account_id: str):
    response = requests.post(
        f"https://api.derivws.com/trading/v1/options/accounts/{quote(selected_account_id, safe='')}/otp",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    ws_url = (body.get("data") or {}).get("url")
    if not ws_url:
        raise ValueError("Deriv did not return an authenticated WebSocket URL.")
    return create_connection(ws_url, timeout=15)


def _send_ws_request(connection, payload: dict, req_id: int) -> dict:
    payload = {**payload, "req_id": req_id}
    connection.send(json.dumps(payload))
    while True:
        message = json.loads(connection.recv())
        if message.get("req_id") != req_id:
            continue
        if "error" in message:
            detail = message["error"].get("message", "Deriv request failed.")
            raise ValueError(detail)
        return message


def fetch_balance(access_token: str, selected_account_id: str) -> dict:
    connection = _authenticated_ws(access_token, selected_account_id)
    try:
        message = _send_ws_request(connection, {"balance": 1}, req_id=1)
        balance = message.get("balance") or {}
        if "balance" not in balance:
            raise ValueError("Deriv returned no balance for this account.")
        return balance
    finally:
        connection.close()


def place_rise_fall_contract(
    access_token: str,
    selected_account_id: str,
    symbol: str,
    side: str,
    stake: float,
    currency: str,
    duration_minutes: int,
) -> dict:
    if side not in {"BUY", "SELL"}:
        raise ValueError("The signal must be BUY or SELL.")
    if stake <= 0:
        raise ValueError("Stake must be greater than zero.")
    if duration_minutes < 1:
        raise ValueError("Contract duration must be at least one minute.")

    connection = _authenticated_ws(access_token, selected_account_id)
    try:
        proposal = _send_ws_request(
            connection,
            {
                "proposal": 1,
                "amount": float(stake),
                "basis": "stake",
                "contract_type": "CALL" if side == "BUY" else "PUT",
                "currency": currency,
                "duration": int(duration_minutes),
                "duration_unit": "m",
                "underlying_symbol": symbol,
            },
            req_id=1,
        )
        details = proposal.get("proposal") or {}
        proposal_id = details.get("id")
        ask_price = details.get("ask_price")
        if not proposal_id or ask_price is None:
            raise ValueError("Deriv returned an incomplete price proposal.")
        result = _send_ws_request(
            connection,
            {"buy": proposal_id, "price": float(ask_price)},
            req_id=2,
        )
        return result.get("buy") or {}
    finally:
        connection.close()
