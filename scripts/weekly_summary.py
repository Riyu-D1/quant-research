"""Weekly Notion summary — one journal row per week.

    .venv/bin/python scripts/weekly_summary.py            # real journal
    .venv/bin/python scripts/weekly_summary.py --dry-run  # prints, no writes

Reads the last ~8 day-summary STATE blobs from the Trade Journal (the
durable state store) plus the week's trade pages, then writes a
`weekly-summary` row: equity (USD+GBP), week return vs SPY, drawdown from
peak, turnover, hit rate, the latest signal weights, and a lessons line.

Env: NOTION_TOKEN + NOTION_DB_ID (or PARENT_PAGE_ID); ALPACA keys only
for the live equity read — without them the last journaled equity is
reused. --dry-run needs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quant_research import live as LV  # noqa: E402
from quant_research import notion_journal as NJ  # noqa: E402


def log(msg: str) -> None:
    print(f"[weekly] {msg}", flush=True)


def _state_of(page: dict) -> dict:
    try:
        rt = page["properties"]["Outcome"]["rich_text"]
        txt = "".join(x.get("plain_text", "") for x in rt)
        if txt.startswith("STATE "):
            return json.loads(txt[6:])
    except Exception:  # noqa: BLE001
        pass
    return {}


def _num(page: dict, prop: str) -> float:
    try:
        return float(page["properties"][prop]["number"] or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _date(page: dict) -> str:
    try:
        return page["properties"]["Date"]["date"]["start"]
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--weeks-state", type=int, default=10,
                   help="day-summaries to scan for baseline/peak")
    args = p.parse_args()

    if args.dry_run:
        ctx = {"equity_usd": 12.70, "weights_str": "mom:0.50, vq:0.50",
               "validated": False,
               "summary": "DRY weekly: eq $12.70 (+1.2% w/w vs SPY +0.4%) "
                          "| dd -2.0% | turnover 0.10 | hit 3/5",
               "lesson": "dry run — replace with a real week's lesson",
               "state": {"equity_usd": 12.70, "cum_bot": 0.27}}
        print(json.dumps(ctx, indent=1))
        return

    j = NJ.NotionJournal(os.environ.get("NOTION_TOKEN", ""),
                         os.environ.get("NOTION_DB_ID", ""),
                         os.environ.get("NOTION_PARENT_PAGE_ID", ""))
    today = pd.Timestamp.now().normalize()
    week_ago = today - pd.Timedelta(days=7)

    # --- equity: Alpaca is truth; fall back to last journaled equity ------
    states = [_state_of(pg) for pg in j.day_summaries(args.weeks_state)]
    states = [s for s in states if s.get("equity_usd")]
    last = states[0] if states else {}
    eq = float(last.get("equity_usd", 0.0))
    try:
        from quant_research.alpaca_bot import PaperClient
        c = PaperClient(os.environ.get("ALPACA_API_KEY", ""),
                        os.environ.get("ALPACA_SECRET_KEY", ""))
        eq = float(c.account().get("portfolio_value") or eq)
    except Exception as e:  # noqa: BLE001
        log(f"alpaca account read failed — using journaled equity: {e}")
    fx = LV.gbpusd()

    # --- week-over-week + drawdown ---------------------------------------
    prev = next((s for s in states
                 if s.get("date", "") <= str(week_ago.date())), {})
    w_bot = eq / float(prev.get("equity_usd", eq)) - 1 if prev else 0.0
    spy_now = float(last.get("spy") or 0.0)
    w_spy = (spy_now / float(prev.get("spy", spy_now)) - 1
             if prev.get("spy") and spy_now else 0.0)
    peak = max([eq] + [float(s.get("peak", s.get("equity_usd", 0)) or 0)
                       for s in states])
    dd = eq / peak - 1 if peak else 0.0

    # --- turnover + hit rate over the week --------------------------------
    trades = j.trades_between(str(week_ago.date()), str(today.date()))
    sells = [_num(t, "Notional") for t in trades
             if (t["properties"]["Side"]["select"] or {}).get("name")
             == "sell"]
    turnover = sum(sells) / eq if eq else 0.0
    week_states = [s for s in states
                   if s.get("date", "") > str(week_ago.date())]
    ups = sum(1 for a, b in zip(week_states, week_states[1:] + [prev])
              if b.get("equity_usd") and
              a["equity_usd"] > b["equity_usd"])
    hit = f"{ups}/{len(week_states)} up days"

    # --- latest signal weights ---------------------------------------------
    wjson = REPO / "state" / "live" / "weights.json"
    wstr = ""
    if wjson.exists():
        w = json.loads(wjson.read_text()).get("weights", {})
        wstr = ", ".join(f"{k}:{v:.2f}" for k, v in w.items())

    validated = bool((REPO / "configs" / "live_signal.yaml").exists() and
                     "validated: true" in
                     (REPO / "configs" / "live_signal.yaml").read_text())
    ctx = {"equity_usd": eq, "weights_str": wstr,
           "validated": validated,
           "summary": (f"week: eq ${eq:.2f} (£{eq / fx:.2f}) "
                       f"{w_bot:+.1%} vs SPY {w_spy:+.1%} | "
                       f"dd {dd:.1%} | turnover {turnover:.2f} | "
                       f"hit {hit} | trades {len(trades)}"),
           "state": {"date": str(today.date()),
                     "equity_usd": round(eq, 4),
                     "equity_gbp": round(eq / fx, 4),
                     "peak": round(peak, 4)},
           "lesson": ""}
    j.log_weekly_summary(today, ctx)
    log(f"weekly summary logged: {ctx['summary']}")


if __name__ == "__main__":
    main()
