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
2. Applies the Phase-4 regime guard (SPY below its 200-day SMA or extreme
   trailing volatility → half or zero exposure per `regime` in the config).
3. Computes the pinned signals, weights them (equal or trailing-IC `ewic`),
   and picks the top 5 names.
4. Rebalances the **paper** account toward equal-weight slots sized from a
   £10 budget converted to USD at the live GBPUSD rate; fractional shares,
   long-only, orders under Alpaca's $1 minimum notional are skipped.
5. Sells before buys. If the account already has orders today the run is a
   no-op (idempotent — safe to re-dispatch).
6. Writes each order + a day summary + a monthly "what changed" page (when
   the signal weights move) to the Notion Trade Journal, including the
   `validated` flag.

## Secrets — where to get each one

| Secret | Where |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | <https://app.alpaca.markets> → **Paper Trading** toggle (top-left) → "Your API Keys" → Generate New Key. These are the *paper* keys — the bot hard-refuses any non-paper base URL anyway. Repo already has them in the `ALPACA_API_KEY` GitHub Environment. |
| `NOTION_TOKEN` | <https://www.notion.so/profile/integrations> → "New integration" → give it a name, pick the workspace → copy the "Internal Integration Secret" (`ntn_…` / `secret_…`). |
| `NOTION_PARENT_PAGE_ID` | The Notion page you want the journal under: open it → Share → "Copy link" → the id is the 32-hex tail of the URL (`…/My-Page-1a2b3c…`). **Then Share the page with your integration** (Share → invite → pick the integration) or API calls 403. |
| `NOTION_DB_ID` | Not needed up front — `scripts/notion_setup.py` creates the db under the parent page and prints the id; add it afterwards so the bot skips lookup. |

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
Regime, Validated, Reason, Outcome, Lesson, OrderID, EntryType.

## Running it

- **By hand:** Actions → `paper-trade` → Run workflow → optional `date`,
  `dry_run`, `fixture` inputs.
- **Daily:** uncomment the `schedule:` block in
  `.github/workflows/paper_trade.yml` (22:00 UTC Mon–Fri, after the US
  close) — only when you're happy with dry runs.
- **Proof without keys:** the `mock_day` CI job runs a full day against
  fake Alpaca/Notion clients on every push (`pytest tests/test_paper_trade.py`).
- **Local dry run:**
  ```bash
  ALPACA_API_KEY=… ALPACA_SECRET_KEY=… \
    .venv/bin/python scripts/paper_trade.py --dry-run
  ```
  `--dry-run` still fetches real bars but skips the broker + Notion — the
  journal lands in `results/paper_dryrun/<date>.json`.

## State files (tracked, committed)

- `configs/live_universe.txt` — tradable tickers (qlib canon, `BRK-B` style)
- `state/live/vq_snapshot.parquet` — value/quality scores (refresh monthly
  with `scripts/live_snapshot.py`)
- `state/live/weights.json` — last weights + month, drives the monthly
  "what changed" page

When a position fully exits, `close_out` patches the matching open BUY page
in Notion with the outcome ("closed +x.x% over Nd — reason") and a lesson —
no separate open-trades file needed.
