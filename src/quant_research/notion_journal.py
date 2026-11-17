"""Notion journal for the Phase-4 paper bot — free-tier REST API.

One "Trade Journal" database, three entry types via the EntryType select:
  trade           one page per submitted order (date, ticker, side, qty,
                  price, signal score, weights, regime, validated, reason)
  day-summary     one page per run (picks, weights, exposure, budget, fx,
                  validated flag) — also the idempotency anchor
  weights-change  written when the monthly signal weights shift vs the
                  stored state (old -> new weights + config diff)

Outcome/Lesson: when a position is sold, the matching open BUY page is
patched with "closed +x.x% over Nd — <reason>" and a short lesson.

The integration needs edit access to the parent page / database — the
owner shares it once (see docs/BOT_SETUP.md). No paid Notion features used.
"""
from __future__ import annotations

import json
import math
import time

import pandas as pd
import requests

API = "https://api.notion.com/v1"
VERSION = "2022-06-28"


def log(msg: str) -> None:
    print(f"[notion {time.strftime('%H:%M:%S')}] {msg}", flush=True)


DB_PROPS = {
    "Ticker": {"title": {}},
    "Date": {"date": {}},
    "Side": {"select": {"options": [
        {"name": "buy"}, {"name": "sell"}, {"name": "info"}]}},
    "Qty": {"number": {"format": "number"}},
    "Price": {"number": {"format": "number"}},
    "Notional": {"number": {"format": "number"}},
    "SignalScore": {"number": {"format": "number"}},
    "SignalWeights": {"rich_text": {}},
    "Regime": {"select": {"options": [
        {"name": "calm"}, {"name": "spy<sma"}, {"name": "vol>q"},
        {"name": "spy<sma+vol>q"}]}},
    "Validated": {"checkbox": {}},
    "EntryType": {"select": {"options": [
        {"name": "trade"}, {"name": "day-summary"},
        {"name": "weights-change"}, {"name": "note"}, {"name": "error"},
        {"name": "weekly-summary"}, {"name": "halt"}]}},
    "Reason": {"rich_text": {}},
    "Outcome": {"rich_text": {}},
    "Lesson": {"rich_text": {}},
}


class NotionJournal:
    def __init__(self, token: str, db_id: str = "",
                 parent_page_id: str = "") -> None:
        if not token:
            raise RuntimeError("NOTION_TOKEN missing")
        self.hdr = {"Authorization": f"Bearer {token}",
                    "Notion-Version": VERSION}
        if db_id:
            self.db = db_id
        elif parent_page_id:
            self.db = self.create_db(parent_page_id)
        else:
            raise RuntimeError(
                "need NOTION_DB_ID or NOTION_PARENT_PAGE_ID — "
                "run scripts/notion_setup.py to create the database")
        self._ensure_entry_types()

    def _ensure_entry_types(self) -> None:
        """Add any EntryType options the db is missing (older dbs predate
        'error'/'weekly-summary'/'halt'). Best-effort — a schema read
        failure must not stop a run."""
        try:
            db = self._req("GET", f"/databases/{self.db}")
            prop = db["properties"].get("EntryType", {})
            have = {o["name"] for o in
                    prop.get("select", {}).get("options", [])}
            want = {o["name"] for o in
                    DB_PROPS["EntryType"]["select"]["options"]}
            missing = want - have
            if missing:
                opts = [{"name": n} for n in sorted(have | missing)]
                self._req("PATCH", f"/databases/{self.db}", json={
                    "properties": {"EntryType": {
                        "select": {"options": opts}}}})
                log(f"EntryType options extended: +{sorted(missing)}")
        except Exception as e:  # noqa: BLE001
            log(f"entry-type ensure failed (non-fatal): {e}")

    # -- plumbing --------------------------------------------------------------
    def _req(self, method: str, path: str, **kw):
        r = requests.request(method, f"{API}{path}", headers=self.hdr,
                             timeout=30, **kw)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2)))
            r = requests.request(method, f"{API}{path}", headers=self.hdr,
                                 timeout=30, **kw)
        if not r.ok:
            raise RuntimeError(f"notion {method} {path}: {r.status_code} "
                               f"{r.text[:300]}")
        return r.json()

    def create_db(self, parent_page_id: str) -> str:
        body = {"parent": {"type": "page_id", "page_id": parent_page_id},
                "title": [{"type": "text",
                           "text": {"content": "Trade Journal"}}],
                "properties": DB_PROPS}
        d = self._req("POST", "/databases", json=body)
        log(f"created Trade Journal db {d['id']} under {parent_page_id}")
        return d["id"]

    # -- writes -----------------------------------------------------------------
    @staticmethod
    def _rt(text: str):
        return [{"type": "text", "text": {"content": str(text)[:2000]}}]

    @staticmethod
    def _num(x, nd: int = 4) -> float:
        """Notion rejects non-JSON floats — NaN is truthy so `x or 0`
        does NOT catch it; normalise every numeric prop to a finite
        value or a real order gets filled and the run dies journalling
        it (happened 2026-11-17: SPY core trade carried score=NaN)."""
        try:
            v = float(x or 0)
        except (TypeError, ValueError):
            v = 0.0
        return round(v if math.isfinite(v) else 0.0, nd)

    def _page(self, props: dict):
        return self._req("POST", "/pages",
                         json={"parent": {"database_id": self.db},
                               "properties": props})

    def log_trade(self, day: pd.Timestamp, t: dict, ctx: dict) -> None:
        self._page({
            "Ticker": {"title": self._rt(t["sym"])},
            "Date": {"date": {"start": day.strftime("%Y-%m-%d")}},
            "Side": {"select": {"name": t["side"]}},
            "Qty": {"number": self._num(t.get("qty"), 6)},
            "Price": {"number": self._num(t.get("price"), 4)},
            "Notional": {"number": self._num(t.get("notional"), 4)},
            "SignalScore": {"number": self._num(t.get("score"), 4)},
            "SignalWeights": {"rich_text": self._rt(ctx["weights_str"])},
            "Regime": {"select": {"name": ctx["regime"]}},
            "Validated": {"checkbox": bool(ctx["validated"])},
            "EntryType": {"select": {"name": "trade"}},
            "Reason": {"rich_text": self._rt(t.get("reason") or "")},
            "Outcome": {"rich_text": self._rt("")},
            "Lesson": {"rich_text": self._rt("")}})

    def log_day_summary(self, day: pd.Timestamp, ctx: dict) -> None:
        state = ctx.get("recon_state")
        self._page({
            "Ticker": {"title": self._rt("— day summary —")},
            "Date": {"date": {"start": day.strftime("%Y-%m-%d")}},
            "Side": {"select": {"name": "info"}},
            "Qty": {"number": 0}, "Price": {"number": 0},
            "Notional": {"number": self._num(ctx["budget_usd"], 2)},
            "SignalScore": {"number": 0},
            "SignalWeights": {"rich_text": self._rt(ctx["weights_str"])},
            "Regime": {"select": {"name": ctx["regime"]}},
            "Validated": {"checkbox": bool(ctx["validated"])},
            "EntryType": {"select": {"name": "day-summary"}},
            "Reason": {"rich_text": self._rt(ctx["summary"])},
            "Outcome": {"rich_text": self._rt(
                "STATE " + json.dumps(state) if state else "")},
            "Lesson": {"rich_text": self._rt(ctx.get("lesson") or "")}})

    def log_error(self, day: pd.Timestamp, message: str,
                  ctx: dict | None = None) -> None:
        """A loud failure row — the workflow still exits non-zero, but the
        journal shows what died without digging through run logs."""
        ctx = ctx or {}
        self._page({
            "Ticker": {"title": self._rt("— ERROR —")},
            "Date": {"date": {"start": day.strftime("%Y-%m-%d")}},
            "Side": {"select": {"name": "info"}},
            "Qty": {"number": 0}, "Price": {"number": 0},
            "Notional": {"number": 0}, "SignalScore": {"number": 0},
            "SignalWeights": {"rich_text": self._rt(
                ctx.get("weights_str") or "")},
            "Regime": {"select": {"name": "calm"}},
            "Validated": {"checkbox": bool(ctx.get("validated"))},
            "EntryType": {"select": {"name": "error"}},
            "Reason": {"rich_text": self._rt(message[:1900])},
            "Outcome": {"rich_text": self._rt("")},
            "Lesson": {"rich_text": self._rt(
                "see the workflow run log for the traceback")}})

    def log_note(self, day: pd.Timestamp, text: str,
                 ctx: dict | None = None) -> None:
        """Anomaly row — rejected orders, partial fills, stale inputs."""
        ctx = ctx or {}
        self._page({
            "Ticker": {"title": self._rt("— note —")},
            "Date": {"date": {"start": day.strftime("%Y-%m-%d")}},
            "Side": {"select": {"name": "info"}},
            "Qty": {"number": 0}, "Price": {"number": 0},
            "Notional": {"number": 0}, "SignalScore": {"number": 0},
            "SignalWeights": {"rich_text": self._rt(
                ctx.get("weights_str") or "")},
            "Regime": {"select": {"name": "calm"}},
            "Validated": {"checkbox": bool(ctx.get("validated"))},
            "EntryType": {"select": {"name": "note"}},
            "Reason": {"rich_text": self._rt(text[:1900])},
            "Outcome": {"rich_text": self._rt("")},
            "Lesson": {"rich_text": self._rt("")}})

    def log_weights_change(self, month: str, old: dict, new: dict,
                           ctx: dict) -> None:
        diff = ", ".join(f"{k}: {old.get(k,0):.2f}->{new.get(k,0):.2f}"
                         for k in sorted(set(old) | set(new)))
        self._page({
            "Ticker": {"title": self._rt("— what changed —")},
            "Date": {"date": {"start": month}},
            "Side": {"select": {"name": "info"}},
            "Qty": {"number": 0}, "Price": {"number": 0},
            "Notional": {"number": 0}, "SignalScore": {"number": 0},
            "SignalWeights": {"rich_text": self._rt(ctx["weights_str"])},
            "Regime": {"select": {"name": ctx["regime"]}},
            "Validated": {"checkbox": bool(ctx["validated"])},
            "EntryType": {"select": {"name": "weights-change"}},
            "Reason": {"rich_text": self._rt(f"monthly reweight: {diff}")},
            "Outcome": {"rich_text": self._rt("")},
            "Lesson": {"rich_text": self._rt("")}})

    def log_weekly_summary(self, day: pd.Timestamp, ctx: dict) -> None:
        """One row per week: equity, return vs SPY, drawdown, turnover,
        hit rate, what changed in weights, lessons."""
        self._page({
            "Ticker": {"title": self._rt("— weekly summary —")},
            "Date": {"date": {"start": day.strftime("%Y-%m-%d")}},
            "Side": {"select": {"name": "info"}},
            "Qty": {"number": 0}, "Price": {"number": 0},
            "Notional": {"number": round(ctx.get("equity_usd", 0.0), 2)},
            "SignalScore": {"number": 0},
            "SignalWeights": {"rich_text": self._rt(
                ctx.get("weights_str") or "")},
            "Regime": {"select": {"name": "calm"}},
            "Validated": {"checkbox": bool(ctx.get("validated"))},
            "EntryType": {"select": {"name": "weekly-summary"}},
            "Reason": {"rich_text": self._rt(ctx["summary"])},
            "Outcome": {"rich_text": self._rt(
                "STATE " + json.dumps(ctx.get("state"))
                if ctx.get("state") else "")},
            "Lesson": {"rich_text": self._rt(ctx.get("lesson") or "")}})

    def log_halt(self, day: pd.Timestamp, reason: str,
                 ctx: dict | None = None) -> None:
        """Kill-switch / circuit-breaker row — unmistakable in the journal."""
        ctx = ctx or {}
        self._page({
            "Ticker": {"title": self._rt("— HALT —")},
            "Date": {"date": {"start": day.strftime("%Y-%m-%d")}},
            "Side": {"select": {"name": "info"}},
            "Qty": {"number": 0}, "Price": {"number": 0},
            "Notional": {"number": 0}, "SignalScore": {"number": 0},
            "SignalWeights": {"rich_text": self._rt(
                ctx.get("weights_str") or "")},
            "Regime": {"select": {"name": "calm"}},
            "Validated": {"checkbox": bool(ctx.get("validated"))},
            "EntryType": {"select": {"name": "halt"}},
            "Reason": {"rich_text": self._rt(reason[:1900])},
            "Outcome": {"rich_text": self._rt("")},
            "Lesson": {"rich_text": self._rt(
                "no orders placed this run")}})

    # -- queries / updates ------------------------------------------------------
    def _query(self, filt: dict) -> list[dict]:
        d = self._req("POST", f"/databases/{self.db}/query",
                      json={"filter": filt, "page_size": 100})
        return d.get("results", [])

    def day_logged(self, day: pd.Timestamp,
                   entry_type: str = "day-summary") -> bool:
        return bool(self._query({"and": [
            {"property": "EntryType", "select": {"equals": entry_type}},
            {"property": "Date", "date": {
                "equals": day.strftime("%Y-%m-%d")}}]}))

    def open_buy(self, sym: str) -> dict | None:
        pages = self._query({"and": [
            {"property": "Ticker", "title": {"equals": sym}},
            {"property": "Side", "select": {"equals": "buy"}},
            {"property": "EntryType", "select": {"equals": "trade"}},
            {"property": "Outcome", "rich_text": {"is_empty": True}}]})
        return max(pages, key=lambda p: p["created_time"]) if pages else None

    def open_buys(self) -> list[str]:
        """Tickers of all still-open BUY trade pages — used to detect
        corporate actions (a position that vanished without a sell)."""
        pages = self._query({"and": [
            {"property": "Side", "select": {"equals": "buy"}},
            {"property": "EntryType", "select": {"equals": "trade"}},
            {"property": "Outcome", "rich_text": {"is_empty": True}}]})
        out = []
        for p in pages:
            t = p["properties"]["Ticker"]["title"]
            txt = "".join(x.get("plain_text", "") for x in t)
            if txt:
                out.append(txt)
        return out

    def day_summaries(self, n: int = 10) -> list[dict]:
        """The last n day-summary pages (newest first) — weekly rollup
        and the equity-peak baseline for the drawdown breaker."""
        d = self._req("POST", f"/databases/{self.db}/query", json={
            "filter": {"property": "EntryType",
                       "select": {"equals": "day-summary"}},
            "sorts": [{"property": "Date", "direction": "descending"}],
            "page_size": max(1, min(n, 100))})
        return d.get("results", [])

    def trades_between(self, start: str, end: str) -> list[dict]:
        """Trade pages with start <= Date <= end — turnover accounting."""
        return self._query({"and": [
            {"property": "EntryType", "select": {"equals": "trade"}},
            {"property": "Date", "date": {"on_or_after": start}},
            {"property": "Date", "date": {"on_or_before": end}}]})

    def trade_pages(self, sym: str, side: str, day: str) -> list[dict]:
        return self._query({"and": [
            {"property": "Ticker", "title": {"equals": sym}},
            {"property": "Side", "select": {"equals": side}},
            {"property": "EntryType", "select": {"equals": "trade"}},
            {"property": "Date", "date": {"equals": day}}]})

    def mark_fill(self, sym: str, day: str, qty: float,
                  price: float) -> None:
        """Reconcile a submitted BUY against its actual Alpaca fill —
        patches Price/Qty/Notional to real values and appends the fill to
        Reason. Outcome stays empty so close_out still matches the page."""
        pages = self.trade_pages(sym, "buy", day)
        if not pages:
            return
        pg = max(pages, key=lambda p: p["created_time"])
        rt = pg["properties"]["Reason"]["rich_text"]
        reason = "".join(t.get("plain_text", "") for t in rt)
        if "filled " in reason:            # already reconciled — idempotent
            return
        self._req("PATCH", f"/pages/{pg['id']}", json={"properties": {
            "Price": {"number": self._num(price, 4)},
            "Qty": {"number": self._num(qty, 6)},
            "Notional": {"number": self._num(qty * price, 4)},
            "Reason": {"rich_text": self._rt(
                f"{reason} | filled {qty:g} @ {price:.2f}")}}})
        log(f"reconciled fill {sym}: {qty:g} @ {price:.2f}")

    def last_summary_state(self) -> dict:
        """The STATE blob from the most recent day-summary page — the
        equity/SPY baseline for the next run's P&L tracking. State lives
        in Notion (not runner disk) so runs are machine-independent."""
        d = self._req("POST", f"/databases/{self.db}/query", json={
            "filter": {"property": "EntryType",
                       "select": {"equals": "day-summary"}},
            "sorts": [{"property": "Date", "direction": "descending"}],
            "page_size": 1})
        res = d.get("results", [])
        if not res:
            return {}
        rt = res[0]["properties"]["Outcome"]["rich_text"]
        txt = "".join(t.get("plain_text", "") for t in rt)
        if not txt.startswith("STATE "):
            return {}
        try:
            return json.loads(txt[len("STATE "):])
        except ValueError:
            return {}

    def close_out(self, sym: str, exit_price: float,
                  reason: str = "") -> None:
        pg = self.open_buy(sym)
        if not pg:
            return
        entry = (pg["properties"]["Price"]["number"] or 0)
        days = (pd.Timestamp.now() -
                pd.Timestamp(pg["properties"]["Date"]["date"]["start"])).days
        pct = (exit_price / entry - 1) * 100 if entry else 0.0
        lesson = ("regime cut — guard fired, not a bad pick" if "regime"
                  in reason else
                  "rank fell out of top-5 — signal decayed" if "rank"
                  in reason else "see signal notes")
        self._req("PATCH", f"/pages/{pg['id']}", json={"properties": {
            "Outcome": {"rich_text": self._rt(
                f"closed {pct:+.1f}% over {days}d @ {exit_price:.2f} — "
                f"{reason}")},
            "Lesson": {"rich_text": self._rt(lesson)}}})
        log(f"close-out {sym}: {pct:+.1f}% over {days}d")
