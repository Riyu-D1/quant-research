# Paper bot setup — Phase-8 core+satellite book

`scripts/paper_trade.py` runs one trading day end-to-end on an **Alpaca
paper account** and journals every decision to a Notion "Trade Journal"
database. It runs on the daily cron (`paper_trade.yml`: orders 15:00 UTC,
reconciliation 22:00 UTC, Mon–Fri — each with off-the-hour backup slots,
since GitHub schedules can lag or drop) and can also be armed manually
with `workflow_dispatch`. GitHub schedules only fire from the default branch
(`master`), so the cron goes live when the phase-8 branch merges.

**Book shape (Phase 8):** the config pins a `core:`/`sleeve:` schema —
a fixed % of the book in an SPY core plus a signal sleeve of up to 8
names (equal or inverse-volatility weights, 20% single-stock cap, 2%
rebalance band, $1 order floor). With no `core:` key the legacy top-N
equal-weight book runs unchanged. On a risk-off regime day the sleeve's
freed share moves into the SPY core (`risk_off: to_core`) — never to a
sideline.

Phase 8 adds four optional mechanisms, each pinned in
`configs/live_signal.yaml` only when the predeclared walk-forward gate
(`configs/experiments/phase8_us.yaml`) selected them:

- `sleeve.sector_max` — cap on sleeve picks per GICS sector
  (`sector_map` points at the CSV; unmapped names are their own sector).
- `sn_<base>` signals — sector-neutralised signal legs: the signal's
  cross-sectional rank is computed within each sector, so the sleeve
  can't concentrate in one hot sector's momentum.
- `sleeve.pick_freeze: monthly` — the sleeve list is fixed at the month's
  first trading day and recomputed only there (deterministic; needs no
  persisted state).
- `sleeve.ic_shrink` / `sleeve.ic_gate` — trailing-EW rank-IC of the mom
  leg scales sleeve exposure toward the core when momentum stops paying
  (`full_at` ⇒ scale 1, decays to `prior_scale`) and can switch the mom
  leg off entirely when its trailing IC ≤ 0. vq is a snapshot signal
  with no live IC history, so it is ungateable by construction — same
  convention as the walk-forward simulator.

Pick candidates must have a *recent* bar (within 7 business days of the
signal day) — delisted names keep serving history at their last close and
must not consume sleeve slots; held names keep their last-known price so
they can still be sold.

**Signal status:** `configs/live_signal.yaml` pins the exact config.
`validated` is whatever the last honest holdout gate produced — it is
`false` until a single fresh-window look clears the predeclared match
gate, and every run/journal entry carries the flag.

**Paper vs live config:** the paper bot defaults to
`configs/paper_signal.yaml` — the identical B50_momvq map plus
`book_size: account_equity`, which sizes targets off the paper account's
full equity (~$100k) so the paper journal shows the strategy's real
moves. `configs/live_signal.yaml` keeps `budget_gbp` sizing so a future
£10 live run still mirrors reality; `book_size: fixed_gbp` is the
default when the key is absent.

## What the bot does each run

1. Pulls ~13 months of daily bars for the universe + SPY from Alpaca's
   free IEX feed (no subscription needed).
2. Checks the market clock and bar freshness — if the newest bar is more
   than a week stale it writes an `error` journal entry and exits
   non-zero rather than trading stale data.
3. **Reconciles fills** — reads the last week of Alpaca orders and live
   positions (Alpaca is the source of truth):
   - a filled sell on a now-flat position patches the open BUY page's
     Outcome ("closed +x.x% over Nd") using the *actual* fill price;
   - filled buys patch their trade page to actual fill price/qty;
   - rejected or partially-filled orders get `note` entries;
   - account equity + SPY close + running equity **peak** + GBP equity go
     into a `STATE` blob on the day-summary page — the durable state the
     next run's breakers and P&L-vs-SPY lines read.
4. **Corporate actions** — if the journal still has an open BUY for a
   symbol that is no longer a position (delist / split-away / symbol
   change), the position is "closed out" in the journal with the reason,
   so the audit trail stays honest.
5. Applies the SPY regime guard (below 200-day SMA or extreme trailing
   vol → reduced or zero exposure per `regime` in the config).
6. Computes the pinned signals, weights them (equal or trailing-IC
   `ewic`), and picks the top-N sleeve names.
7. **Gates, in order, before any order:**
   - `BOT_HALT=1` repo/environment variable → reconcile + journal + a
     `halt` row, **zero orders** (the kill switch);
   - **order window** — Alpaca clock + calendar: orders only during the
     regular session, at least `order_after_open_min` (15) minutes after
     the open; half-day early closes are honoured automatically; never
     extended hours;
   - **idempotency** — if orders already exist for today the run is a
     no-op; plus every order carries a deterministic
     `client_order_id = (date, symbol, side)` so even a race/retry can't
     double-order;
   - **stale-order cleanup** — all still-open orders are cancelled before
     rebalancing;
   - **sanity checks** — priced-universe floor, buy-notional ≤ equity,
     no >50% single-day price jumps among traded names; any failure
     aborts the run with an `error` row and **no orders**;
   - **circuit breakers** — drawdown > 25% from the running equity peak,
     or daily loss > 5%, blocks *new buys* (sells still de-risk);
     `max_position_weight` clips oversized buys; `max_orders` caps the
     run;
   - **PDT guard** — accounts under $25k can't open and close the same
     symbol the same day; the guard drops any trade that would (checked
     against today's orders *and* within the batch, with a hard assert).
8. Sells before buys. Rebalances toward the weight map sized from the
   book — the paper account's full equity under
   `book_size: account_equity`, or a £10 budget converted to USD at the
   live GBPUSD rate under the default `fixed_gbp` (each run records
   the rate used; equity is journaled in both USD and GBP); fractional
   shares, long-only. A rebalance is skipped when the weight gap is under
   2% or the order is under the $1 Alpaca minimum — except outright
   exits, which always trade (full close-outs are exempt from the floor).
9. **Drift check** — actual vs target weights are computed every run and
   logged into the day summary, so missed fills or stale positions are
   visible instead of silent.
10. Writes each order + a day summary (equity-vs-SPY line, gate notes,
    drift) + a monthly "what changed" page to the Trade Journal, and a
    structured JSON run record to `results/paper_run/<date>.json`
    (uploaded as a workflow artifact — every gate decision is auditable
    from the commit SHA in `GITHUB_SHA`).

## Failure behaviour

Any unhandled error → an `error` entry in the Trade Journal (best effort)
**and** exit code 1, so the workflow goes red. Expected stops — missing
keys, no usable signals, stale bars, sanity-check failure — take the same
path.

## Secrets — where to get each one

| Secret | Where |
|---|---|
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | <https://app.alpaca.markets> → **Paper Trading** toggle → "Your API Keys" → Generate. Paper keys — the bot hard-refuses non-paper base URLs anyway. They live in the `ALPACA_API_KEY` GitHub Environment. |
| `NOTION_TOKEN` | <https://www.notion.so/profile/integrations> → "New integration" → copy the internal secret (`ntn_…`). |
| `NOTION_PARENT_PAGE_ID` | Notion page → Share → "Copy link" → 32-hex tail. **Share the page with the integration** or calls 403. |
| `NOTION_DB_ID` | Printed by `scripts/notion_setup.py` once the db exists. |
| `BOT_HALT` | **Variable** (not secret): Settings → Secrets and variables → Actions → **Variables** tab → `BOT_HALT` = `1` to halt. Delete it (or set `0`) to resume. |
| `LIVE_TRADING_ACK` | **The live-trading interlock. Intentionally unset.** See below. |

### Add them to GitHub

```bash
gh secret set ALPACA_API_KEY   --env ALPACA_API_KEY
gh secret set ALPACA_API_SECRET --env ALPACA_API_KEY
gh secret set NOTION_TOKEN
gh secret set NOTION_PARENT_PAGE_ID
gh secret set NOTION_DB_ID          # after notion_setup.py prints one
gh variable set BOT_HALT --body 1   # kill switch ON (delete to resume)
```

## One-time Notion setup

```bash
NOTION_TOKEN=ntn_xxx NOTION_PARENT_PAGE_ID=1a2b… \
  .venv/bin/python scripts/notion_setup.py
# prints the new database id → gh secret set NOTION_DB_ID
```

The db it creates has: Date, Ticker, Side, Qty, Price, Signal, Weights,
Regime, Validated, Reason, Outcome, Lesson, EntryType (trade /
day-summary / weights-change / note / error / weekly-summary / halt).
Existing dbs gain any missing EntryType options automatically on first
use.

## Running it

- **By hand:** Actions → `paper-trade` → Run workflow on the branch that
  owns the config (phase 7: `phase7/us-live`, later `phase6/benchmarks`
  once the PR lands). Inputs: `date`, `dry_run`, `fixture`, `config`,
  `job` (`run` or `refresh_snapshot`).
- **Daily paper (exact clicks, when you're ready):** edit
  `.github/workflows/paper_trade.yml` on the deployed branch — remove the
  leading `# ` on the `schedule:` block (six commented lines, all
  documented below), commit, push. GitHub runs the cron from the repo's
  default branch, so the enable only takes effect once the branch
  carrying it is merged or set default; until then manual dispatch on the
  branch is the way to run it.
  - `0 15 * * 1-5` — **orders** run, lands ~10:00–11:00 ET year-round
    (past the open+15min buffer in both EDT and EST), with backup slots
    at 15:20 / 16:10 / 17:30 / 19:05 UTC — each a no-op once today's
    (ET) orders are in, and all refused inside the last 15 min before
    the close;
  - `0 22 * * 1-5` — **reconcile** run after the close (idempotent: no
    new orders, fills/positions/drift journaled), backed up at 22:20
    and 23:30 UTC;
  - `0 14 * * 6` — **weekly summary** row (equity, week return vs SPY,
    drawdown, turnover, hit rate, weights);
  - `0 10 1 * *` — **monthly reweight** (the adaptive job: trailing-IC
    weights + what-changed entry + gate re-check);
  - `0 9 1 * *` — **snapshot refresh** (rebuilds the vq snapshot, commits
    to the branch);
  - `0 10 * * 1` — **staleness alarm** (fails + journals an `error` row
    if a snapshot is >45 days old).
  Daily paper keeps trading forever until you halt it — the kill switch
  below is the stop.
- **Snapshot refresh (manual):** dispatch `job=refresh_snapshot`.
- **Local dry run:** `.venv/bin/python scripts/paper_trade.py --dry-run`
  — synthetic bars + fake broker/journal, journal lands in
  `results/paper_dryrun/<date>.json`.
- **CI proof:** `mock_day` runs the full suite on every push
  (`tests/test_paper_trade.py`, `test_live_holdout.py`,
  `test_production.py` — incl. the multi-day scenario: enter, hold,
  partial fill, rejection, delist, halt, breaker, holiday).

## Halting the bot (the kill switch)

Set the repo variable `BOT_HALT=1`
(`gh variable set BOT_HALT --body 1`). The next run still reconciles and
journals — but places **no orders** and writes a `halt` row saying why.
Delete the variable (or set `0`) to resume. For an *immediate* stop,
also cancel any open orders in the Alpaca paper dashboard.

## The live-trading switch (OFF — never flip without thinking twice)

`.github/workflows/live_trade.yml` is the real-money scaffold — its cron
is commented out and every run is dispatch-only inside a `live` GitHub
Environment that pauses for an owner approval click.

Five independent gates, **all** required to reach real money:

1. **Environment approval** — Settings → Environments → create `live`,
   add yourself as a required reviewer. Every `live-trade` run then
   pauses until you click Approve.
2. `live_trading: true` in `configs/live_signal.yaml` (currently
   `false`) — flip it only after the paper book has run long enough for
   you to trust it.
3. The `LIVE_TRADING_ACK` secret set to `I_UNDERSTAND_LIVE_RISK`.
4. `ALPACA_BASE_URL` variable pointing at the live endpoint (the
   preflight refuses anything containing `paper`), plus the **live**
   `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` secrets inside the `live`
   environment — the paper keys stay in the `ALPACA_API_KEY` environment
   and are never touched by this job.
5. The workflow's own preflight re-checks 2–4 and exits before any
   network call if one is missing.

`PaperClient` only accepts a non-paper base URL when `allow_live` is
passed, which `paper_trade.py` sets only when gates 2+3 hold. **None of
this has been done — every shipped path is paper-only.**

### Going live with £10 — the exact order of clicks

1. Watch the daily paper run for a while; confirm the journal matches
   Alpaca paper fills.
2. Create the `live` environment (required reviewer = you), put the
   **live** Alpaca keys in its secrets, and set its `ALPACA_BASE_URL`
   variable to `https://api.alpaca.markets`.
3. Add repo secret `LIVE_TRADING_ACK` = `I_UNDERSTAND_LIVE_RISK`.
4. Edit `configs/live_signal.yaml` on the deployed branch: set
   `live_trading: true`. Commit + push.
5. Fund the live Alpaca account with ~£10 of USD.
6. Actions → `live-trade` → Run workflow on the deployed branch →
   **approve** the environment prompt. The run record lands as an
   artifact; fills + journal reconcile as usual.
7. Optional: uncomment the `schedule:` block in `live_trade.yml` for
   daily live runs — each still needs the environment approval.
8. To stop: `gh variable set BOT_HALT --body 1` (next run places no
   orders) and/or delete the `live` environment.

## State files (tracked, committed)

- `configs/live_universe.txt` — tradable tickers (qlib canon).
- `state/live/vq_snapshot.parquet` — value/quality scores (refreshed by
  the `refresh_snapshot` job; >45 days stale = dropped + alarmed).
- `state/live/weights.json` — last weights + month; the journal is the
  durable copy.

## The holdouts

- `job=livebot_holdout` → `results/live_holdout_us.json` — the first
  re-viewed score (close fills, flat 10 bps, today's universe).
- `job=honest_holdout` → `results/honest_holdout_us.json` — the
  production score: point-in-time S&P membership (delisted names
  included), liquidity-bucket half-spread costs + flat-bps sweep
  (0/10/25/50), next-open fills, delist liquidation, and a robustness
  block (per-year OOF, rolling 12-month excess IR vs SPY, top-5/10/20,
  weekly/monthly rebalance, bootstrap CI) — every variant counts in
  `n_trials`. Both append to `results/holdout_log.jsonl`. These windows
  were already viewed in phases 2/3 — **re-viewed holdout, no tuning.**
- Phase 7's single fresh-2026 look failed the match gate
  (`results/phase7_us_holdout.json`) — see
  `docs/PHASE8_DIAGNOSIS.md` for the post-mortem: a point-in-time
  fundamentals/vq bug (later amendments overwrote AND hid the visible
  version) was the dominant measured cause, fixed in
  `src/quant_research/fundamentals.py`.
- `phase8.yml` `job=walkforward` → `results/phase8_us_walkforward.json` —
  the strict expanding-window yearly walk-forward (2017→2026 on the
  honest Alpaca feed — it only starts 2016-01-04; 2012–2016 exists only
  in the survivorship-biased community dump and is supplementary). The
  predeclared gate re-pins `configs/live_signal.yaml` and appends
  `holdout_log.jsonl`. The workflow becomes dispatchable once it lands
  on the default branch.

## Known limitations

- Alpaca's free IEX feed under-covers some OTC/illiquid names; unpriced
  names fall out of scoring (they can't be bought, which is safe).
- `configs/live_universe.txt` contains ~250 delisted tickers (the
  ever-S&P-500 list); they cost a wasted bar fetch each, and the
  recent-bar rule keeps them out of picks.
- The 2% rebalance band is effectively inert on a £10 book: the $1
  minimum order is ~8% of equity, so the floor binds first (measured in
  `results/phase8_diagnosis.json` — band on/off changed $0.00).
- Fractional market DAY orders at ~10:00 ET are *not* open-auction
  prices; the 15-minute buffer avoids the open's worst spreads.
- A symbol that 1:1-renames may journal a corp-action close and re-enter
  under the new ticker — correct but noisy.
- `ewic` weighting falls back to equal whenever a snapshot signal (vq,
  news) is in the mix — they have no live IC history.
- Paper fills are free and instant; the honest-holdout cost model (not
  the paper account) is what makes the economics believable.
