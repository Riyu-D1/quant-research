# US daily data — options & verdict

## Why we're looking

The qlib community `us_data` dump (Yahoo-derived) carries almost no next-day
alpha: raw 5-day momentum IC ≈ **0.004** vs **0.058** on the CN dump (same
probe, `scripts/momentum_probe.py`, Jan–Jun 2019 window). All three GBDT
benchmarks score IC 0.002–0.007 on it. The dump also cannot reproduce a
survivorship-free universe: `instruments/sp500.txt` has date-windowed
constituents but the features only cover symbols still listed at dump time
(Yahoo drops delisted tickers), so names that left the index by delisting are
missing entirely.

## Free candidates (no paid tiers; all need review before use)

| Source | Delisted coverage | Cost | Access | Notes |
|---|---|---|---|---|
| **Stooq** | Partial — keeps many dead tickers, but coverage of older delistings is incomplete | Free, no key | per-symbol CSV download | Best truly-free option; must verify dead-symbol coverage empirically |
| **Alpaca Market Data** | Weak — API mainly serves listed symbols; delisted history is unreliable | Free tier exists | account + API key required (ask user first) | Free tier EOD bars; IEX-only intraday |
| **Tiingo** | Claims delisted retention, free tier: 50 req/hr, 500/day — too slow for a full universe | Free tier | account + key (ask first) | ~30y EOD; rate-limited for bulk builds |
| **Polygon/Massive** | Delisted only on paid tiers | Free tier unusable (5 req/min, 2y) | — | skip |
| **SimFin** | Bulk CSV incl. inactive companies | Free bulk | account required (ask first) | Fundamentals+prices; ~10y history |
| **Alpha Vantage** | No meaningful delisted set | 25 req/day free | — | unusable for a universe |

## Verdict

1. **Stooq** is the only zero-account, zero-cost source — converter ready:
   `scripts/us_data_to_qlib.py` ingests its `<sym>.us.csv` files.
   Downloading a large universe means fetching thousands of per-symbol URLs;
   acceptable (no key) but verify delisted coverage first with a handful of
   known dead names (e.g. Lehman LEH, Bear Stearns BSC, Enron-era names won't
   exist — Stooq's history depth varies).
2. If Stooq coverage proves insufficient, next-best is **Alpaca** or **Tiingo**
   free tiers — both need an account. Ask the user before creating one.
3. The raw momentum IC probe (`momentum_probe.py`) doubles as the acceptance
   test for any new US dump: target mom5 IC ≥ ~0.03 cross-sectionally before
   trusting downstream results.
