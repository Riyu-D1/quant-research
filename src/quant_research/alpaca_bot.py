"""Alpaca PAPER trading client for the Phase-4 bot.

HARD GUARD: PaperClient refuses to construct with any non-paper base URL.
There is no code path to the live endpoint — paper_api.alpaca.markets only.
Market orders, fractional via `notional` (buy) and `qty` (sell), long-only.
"""
from __future__ import annotations

import time
import zoneinfo

import pandas as pd
import requests

PAPER_URL = "https://paper-api.alpaca.markets"
MIN_NOTIONAL_USD = 1.0        # Alpaca's notional-order floor
ET = zoneinfo.ZoneInfo("America/New_York")   # the trading-date calendar


class OrderExistsError(Exception):
    """Alpaca rejected a duplicate client_order_id — the order already
    exists, so a re-run is a no-op rather than a double-order."""


def log(msg: str) -> None:
    print(f"[alpaca {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def client_order_id(day, sym: str, side: str) -> str:
    """Deterministic per (signal day, symbol, side) — a rerun submits the
    same id and Alpaca dedupes it (we treat the rejection as 'already
    placed'). Max 48 chars per Alpaca."""
    d = pd.Timestamp(day).strftime("%Y%m%d")
    return f"qr-{d}-{sym}-{side}"[:48]


class PaperClient:
    """Thin sync wrapper over the Alpaca REST API — paper-only by default.

    `allow_live` is the second half of the live-trading interlock (the
    first is `live_trading: true` in the pinned config): only when BOTH
    are set may a non-paper base URL be used. Nothing in this repo sets
    either — flipping live is a deliberate owner action documented in
    docs/BOT_SETUP.md. Checked on construction AND on every call.
    """

    def __init__(self, key: str, secret: str,
                 base_url: str = PAPER_URL,
                 allow_live: bool = False) -> None:
        base_url = base_url.rstrip("/")
        self._allow_live = bool(allow_live)
        if "paper-api" not in base_url and not self._allow_live:
            raise RuntimeError(
                f"REFUSING non-paper base URL {base_url!r} — this bot is "
                "paper-only by hard rule (live needs live_trading:true "
                "config + LIVE_TRADING_ACK secret)")
        self.base = base_url
        self.hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _req(self, method: str, path: str, **kw):
        if "paper-api" not in self.base and not self._allow_live:
            raise RuntimeError("non-paper base URL — request refused")
        r = requests.request(method, f"{self.base}{path}",
                             headers=self.hdr, timeout=30, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0)
            r = requests.request(method, f"{self.base}{path}",
                                 headers=self.hdr, timeout=30, **kw)
        if not r.ok:
            if "client_order_id" in r.text:
                raise OrderExistsError(
                    f"duplicate client_order_id on {path}: {r.text[:200]}")
            raise RuntimeError(f"alpaca {method} {path}: {r.status_code} "
                               f"{r.text[:300]}")
        return r.json() if r.text else {}

    # -- reads ---------------------------------------------------------------
    def clock(self) -> dict:
        """{timestamp, is_open, next_open, next_close} — free endpoint."""
        return self._req("GET", "/v2/clock")

    def account(self) -> dict:
        return self._req("GET", "/v2/account")

    def positions(self) -> dict[str, dict]:
        """{canon_symbol: {qty, market_value, avg_entry_price}}"""
        out = {}
        for p in self._req("GET", "/v2/positions"):
            sym = p["symbol"].replace(".", "-")
            out[sym] = {"qty": float(p["qty"]),
                        "market_value": float(p["market_value"]),
                        "avg_entry_price": float(p["avg_entry_price"])}
        return out

    def orders_since(self, day: pd.Timestamp, status: str = "all") -> list[dict]:
        """Orders submitted on/after `day` (UTC date compare)."""
        ords = self._req("GET", "/v2/orders",
                         params={"status": status, "limit": 500,
                                 "after": day.strftime("%Y-%m-%dT00:00:00Z"),
                                 "direction": "asc"})
        return ords

    def traded_today(self, day: pd.Timestamp) -> list[dict]:
        """Orders submitted on `day`'s US/Eastern calendar date — the
        idempotency anchor.

        `day` is the wall-clock trading date the caller wants guarded
        (run_day passes today-ET; may be naive or tz-aware), NOT the
        signal day. Anchoring on the ET trading date is what makes the
        backup cron slots safe: a rerun later the same ET day sees the
        earlier run's orders and exits cleanly, while yesterday's orders
        — submitted on what is now the signal day — can never mask a
        fresh session. Dead orders don't count."""
        d0 = pd.Timestamp(day).normalize()
        if d0.tzinfo is None:
            d0 = d0.tz_localize(ET)
        ords = self.orders_since(d0 - pd.Timedelta(days=1))
        return [o for o in ords
                if pd.Timestamp(o["submitted_at"]).tz_convert(ET)
                .normalize() == d0
                and o.get("status") not in
                ("canceled", "expired", "rejected")]

    # -- calendar / open-order management -------------------------------------
    def calendar(self, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        """Trading days in [start, end]: [{date, open, close, session_open,
        session_close}] — half-days show an early close."""
        return self._req("GET", "/v2/calendar",
                         params={"start": pd.Timestamp(start)
                                 .strftime("%Y-%m-%d"),
                                 "end": pd.Timestamp(end)
                                 .strftime("%Y-%m-%d")})

    def open_orders(self) -> list[dict]:
        """All open (unfilled, uncancelled) orders — stale-order cleanup."""
        return self._req("GET", "/v2/orders",
                         params={"status": "open", "limit": 500})

    def cancel_order(self, order_id: str) -> None:
        self._req("DELETE", f"/v2/orders/{order_id}")

    def cancel_all(self) -> int:
        """Cancel every open order; returns how many were cancelled."""
        n = 0
        for o in self.open_orders():
            try:
                self.cancel_order(o["id"])
                n += 1
            except Exception as e:  # noqa: BLE001
                log(f"cancel {o.get('id')} failed: {e}")
        return n

    # -- orders (all long-only; no shorts ever) --------------------------------
    def buy_notional(self, sym: str, usd: float,
                     reason: str = "", coid: str | None = None) -> dict | None:
        if usd < MIN_NOTIONAL_USD:
            log(f"skip buy {sym}: ${usd:.2f} < ${MIN_NOTIONAL_USD} minimum")
            return None
        from quant_research.live import to_alpaca
        body = {"symbol": to_alpaca(sym), "notional": round(usd, 2),
                "side": "buy", "type": "market", "time_in_force": "day"}
        if coid:
            body["client_order_id"] = coid
        log(f"order BUY {sym} ${usd:.2f} ({reason})")
        return self._req("POST", "/v2/orders", json=body)

    def sell_qty(self, sym: str, qty: float,
                 reason: str = "", coid: str | None = None) -> dict | None:
        if qty <= 0:
            return None
        from quant_research.live import to_alpaca
        body = {"symbol": to_alpaca(sym), "qty": f"{qty:.6f}".rstrip("0")
                .rstrip("."), "side": "sell", "type": "market",
                "time_in_force": "day"}
        if coid:
            body["client_order_id"] = coid
        log(f"order SELL {sym} {qty} ({reason})")
        return self._req("POST", "/v2/orders", json=body)
