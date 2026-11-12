# Paper bot setup — MicroTop5

`scripts/paper_trade.py` runs one trading day end-to-end on an **Alpaca paper
account** and journals every decision to a Notion "Trade Journal" database.
It is armed manually (`workflow_dispatch`) — the daily cron in
`paper_trade.yml` is commented out until you say enable.

**Signal status:** `configs/live_signal.yaml` pins the exact config. It
currently reads `validated: false` — the 2025+ holdout has not been scored
yet, so every run and journal entry is labelled UNVALIDATED. The flag only
flips to `true` if the definitive select + single holdout clear the gate.

## What the bot does each run

1. Pulls ~13 months of daily bars for the universe + SPY from Alpaca's free
   IEX feed (`data.alpaca.markets`, no subscription needed).
2. Checks the market clock and bar freshness — if the newest bar is more
   than a week stale it writes an `error` journal entry and exits non-zero
   rather than trading stale data. Weekends/holidays are fine: the last bar
   is the signal day and the idempotency gate stops re-trades.
3. **Reconciles fills** — reads the last week of Alpaca orders and live
   positions:
   - a filled sell on a now-flat position patches the open BUY page's
     Outcome ("closed +x.x% over Nd") and Lesson using the *actual* fill
     price;
   - filled buys patch their trade page to actual fill price/qty;
   - rejected or partially-filled orders get `note` entries — they never
     crash the run;
   - account equity + SPY close are written as a `STATE` blob on the
     day-summary page, so the next run reports P&L vs SPY without any
     local state (positions come from Alpaca, baselines from Notion —
     never runner disk).
4. Applies the Phase-4 regime guard (SPY below its 200-day SMA or extreme
   trailing volatility → half or zero exposure per `regime` in the config).
5. Computes the pinned signals, weights them (equal or trailing-IC `ewic`),
   and picks the top 5 names.
6. Rebalances the **paper** account toward equal-weight slots sized from a
   £10 budget converted to USD at the live GBPUSD rate; fractional shares,
   long-only, orders under Alpaca's $1 minimum notional are skipped.
7. Sells before buys. If the account already has orders today the run is a
   no-op (idempotent — safe to re-dispatch); reconciliation still runs.
8. Writes each order + a day summary (with the equity-vs-SPY line) + a
   monthly "what changed" page (deduped against the journal — once per
   month even though runner state is ephemeral) to the Trade Journal,
   including the `validated` flag.

## Failure behaviour

Any unhandled error → an `error` entry in the Trade Journal (best effort —
if Notion itself is down only the log shows it) **and** exit code 1, so the
workflow goes red. Expected stops — missing keys, no usable signals, stale
bars — take the same path.

## Secrets — where to get each one

| Secret | Where |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | <https://app.alpaca.markets> → **Paper Trading** toggle (top-left) → "Your API Keys" → Generate New Key. These are the *paper* keys — the bot hard-refuses any non-paper base URL anyway. Repo already has them in the `ALPACA_API_KEY` GitHub Environment. |
| `NOTION_TOKEN` | <https://www.notion.so/profile/integrations> → "New integration" → give it a name, pick the workspace → copy the "Internal Integration Secret" (`ntn_…` / `secret_…`). |
| `NOTION_PARENT_PAGE_ID` | The Notion page you want the journal under: open it → Share → "Copy link" → the id is the 32-hex tail of the URL (`…/My-Page-1a2b3c…`). **Then Share the page with your integration** (Share → invite → pick the integration) or API calls 403. |
| `NOTION_DB_ID` | Not needed up front — `scripts/notion_setup.py` creates the db under the parent page and prints the id; add it afterwards so the bot skips lookup. |
| `LIVE_TRADING_ACK` | **The live-trading interlock. Intentionally unset.** See below. |

### Add them to GitHub

```bash
# Alpaca pair already lives in the environment — only add if rotating:
gh secret set ALPACA_API_KEY   --env ALPACA_API_KEY
gh secret set ALPACA_API_SECRET --env ALPACA_API_KEY

gh secret set NOTION_TOKEN
gh secret set NOTION_PARENT_PAGE_ID
# after notion_setup.py prints a db id:
gh secret set NOTION_DB_ID
```

(`gh secret list --env ALPACA_API_KEY` to check the environment.)

## One-time Notion setup

```bash
NOTION_TOKEN=ntn_xxx NOTION_PARENT_PAGE_ID=1a2b… \
  .venv/bin/python scripts/notion_setup.py
# prints the new database id → gh secret set NOTION_DB_ID
```

The db it creates has: Date, Ticker, Side, Qty, Price, Signal, Weights,
Regime, Validated, Reason, Outcome, Lesson, EntryType (trade / day-summary /
weights-change / note / error). An existing db gains the `error` option
automatically on first use.

## Running it

- **By hand:** Actions → `paper-trade` → Run workflow **on
  `phase4/adaptive`** (the code lives there; master only carries a dispatch
  copy of the workflow). Inputs: `date`, `dry_run`, `fixture`, `config`,
  `job` (`run` or `refresh_snapshot`).
- **Daily:** uncomment the `schedule:` block in
  `.github/workflows/paper_trade.yml` (22:00 UTC Mon–Fri, after the US
  close) — only when you're happy with dry runs. The second commented cron
  (1st of month 09:00 UTC) runs the snapshot refresh instead of a trade
  day; the jobs self-select by cron expression, so both can share the one
  `schedule:` block.
- **Snapshot refresh (manual):** same dispatch with `job=refresh_snapshot`
  — rebuilds `state/live/vq_snapshot.parquet` (and `news_snapshot` when the
  Kaggle sentiment has landed) from the latest artifacts and commits them
  to `phase4/adaptive`. Keeps snapshots inside the 45-day staleness window.
- **Proof without keys:** the `mock_day` CI job runs a full day against
  fake Alpaca/Notion clients on every push (`pytest tests/test_paper_trade.py
  tests/test_live_holdout.py`).
- **Local dry run:**
  ```bash
  .venv/bin/python scripts/paper_trade.py --dry-run
  ```
  Synthetic bars + fake broker/journal — nothing leaves the machine;
  the journal lands in `results/paper_dryrun/<date>.json`.

## The live-trading switch (OFF — never flip without thinking twice)

Three independent gates, all required to reach real money:

1. `live_trading: true` in the config file (currently `false`).
2. The `LIVE_TRADING_ACK` repo secret set to `I_UNDERSTAND_LIVE_RISK`.
3. `ALPACA_BASE_URL` + `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` pointed at the
   live endpoint and live keys (a different env, not the paper one).

To go live an owner would: set the flag in `configs/live_signal.yaml`,
`gh secret set LIVE_TRADING_ACK` with the ack string, and wire the live
base URL + live keys into a *separate* GitHub environment's secrets.
`PaperClient` only accepts a non-paper base URL when `allow_live` is
passed, which `paper_trade.py` sets only when gates 1+2 hold. **None of
this has been done — every shipped path is paper-only.**

## State files (tracked, committed)

- `configs/live_universe.txt` — tradable tickers (qlib canon, `BRK-B` style)
- `state/live/vq_snapshot.parquet` — value/quality scores (refresh via the
  `refresh_snapshot` job)
- `state/live/weights.json` — last weights + month, drives the monthly
  "what changed" page (the journal is the durable copy — runner state is
  not guaranteed to persist)

When a position fully exits, `close_out` patches the matching open BUY page
in Notion with the outcome ("closed +x.x% over Nd — reason") and a lesson —
no separate open-trades file needed.

## The 2025 holdout

`scripts/live_holdout.py` scores this exact pipeline over 2025+ as a
walk-forward (weights fixed through 2024-12-31, monthly re-derivation, same
signal math, same MicroTop5 sim, 10 bps). It runs on Actions via
`phase4-adaptive` → `job=livebot_holdout` — results land in
`results/live_holdout_us.json` + `results/holdout_log.jsonl`. It is a
**re-viewed holdout** (phases 2/3 already saw 2025): never tune on it.
