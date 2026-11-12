"""Alpaca PAPER trading client for the Phase-4 bot.

HARD GUARD: PaperClient refuses to construct with any non-paper base URL.
There is no code path to the live endpoint — paper_api.alpaca.markets only.
Market orders, fractional via `notional` (buy) and `qty` (sell), long-only.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

PAPER_URL = "https://paper-api.alpaca.markets"
MIN_NOTIONAL_USD = 1.0        # Alpaca's notional-order floor


def log(msg: str) -> None:
    print(f"[alpaca {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PaperClient:
    """Thin sync wrapper over the Alpaca paper REST API."""

    def __init__(self, key: str, secret: str,
                 base_url: str = PAPER_URL) -> None:
        base_url = base_url.rstrip("/")
        # The one and only guard that matters: refuse anything that isn't
        # the paper endpoint. Checked on construction AND on every call.
        if "paper-api" not in base_url:
            raise RuntimeError(
                f"REFUSING non-paper base URL {base_url!r} — this bot is "
                "paper-only by hard rule")
        self.base = base_url
        self.hdr = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _req(self, method: str, path: str, **kw):
        if "paper-api" not in self.base:              # belt and suspenders
            raise RuntimeError("non-paper base URL — request refused")
        r = requests.request(method, f"{self.base}{path}",
                             headers=self.hdr, timeout=30, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2.0)
            r = requests.request(method, f"{self.base}{path}",
                                 headers=self.hdr, timeout=30, **kw)
        if not r.ok:
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
        """Orders submitted on/after `day` — the idempotency anchor.

        The signal day is the last completed bar; orders are submitted at
        run time, which can be *after* that bar's date (e.g. a morning run
        before today's bar exists). So the guard is `submitted_at >= day`:
        anything placed since the signal day was produced means this day
        already traded. Older orders belong to earlier signal days."""
        ords = self.orders_since(day - pd.Timedelta(days=1))
        d0 = day.normalize()
        return [o for o in ords
                if pd.Timestamp(o["submitted_at"]).tz_localize(None) >= d0
                and o.get("status") not in
                ("canceled", "expired", "rejected")]

    # -- orders (all long-only; no shorts ever) --------------------------------
    def buy_notional(self, sym: str, usd: float,
                     reason: str = "") -> dict | None:
        if usd < MIN_NOTIONAL_USD:
            log(f"skip buy {sym}: ${usd:.2f} < ${MIN_NOTIONAL_USD} minimum")
            return None
        from quant_research.live import to_alpaca
        body = {"symbol": to_alpaca(sym), "notional": round(usd, 2),
                "side": "buy", "type": "market", "time_in_force": "day"}
        log(f"order BUY {sym} ${usd:.2f} ({reason})")
        return self._req("POST", "/v2/orders", json=body)

    def sell_qty(self, sym: str, qty: float,
                 reason: str = "") -> dict | None:
        if qty <= 0:
            return None
        from quant_research.live import to_alpaca
        body = {"symbol": to_alpaca(sym), "qty": f"{qty:.6f}".rstrip("0")
                .rstrip("."), "side": "sell", "type": "market",
                "time_in_force": "day"}
        log(f"order SELL {sym} {qty} ({reason})")
        return self._req("POST", "/v2/orders", json=body)
