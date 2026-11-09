# US daily data — options & verdict

## Why we're looking

The qlib community `us_data` dump (Yahoo-derived) was suspected of carrying
almost no next-day alpha after the GBDT benchmarks scored IC 0.002–0.007.
**Update 2026-11-09:** the probe that produced the "US 0.004 vs CN 0.058"
comparison was misaligned (see "Probe bug" below). Corrected numbers show raw
daily momentum is weak in *both* markets; the US-vs-CN gap is much smaller
than believed and may not be a data defect at all.

## Probe bug (found & fixed 2026-11-09)

qlib `.bin` files start with a float32 `start_index` into the calendar, then
values for `calendar[start + k]`. `momentum_probe.py` read the header as a
price and dropped every instrument whose series didn't span the full
calendar, so cross-sections mixed different real dates. After fixing the
alignment (and a NaN leak into `ics20`):

| universe (Jan–Jun 2019) | mom5 IC | mom20 IC | n |
|---|---|---|---|
| qlib cn_data csi300 | **0.0068** | 0.0238 | 683 |
| qlib us_data sp500 | **0.0098** | 0.0128 | 660 |
| qlib us_data all | −0.0038 | 0.0067 | 3000 |
| ohlcvault us_data S&P500∩ | 0.0043 | 0.0033 | 445 |
| ohlcvault cn csi300∩ | −0.0078 | — | 673 |

Same probe, same window, two independent vendors: raw 5-day momentum has
~±0.01 IC everywhere at daily horizon. The old "0.004 vs 0.058" asymmetry was
mostly the bug. The CN GBDT ICs (~0.05) are still real — that edge comes
from the 158-feature/model stack, not from a data asymmetry the probe sees.

## Source status

| Source | Delisted coverage | Cost | Access | Status 2026-11-09 |
|---|---|---|---|---|
| **Stooq** | Partial (dead tickers kept) | Free | was keyless CSV | **DEAD for scripted access** — JS proof-of-work on all pages since ~May 2026; `q/d/l/` returns `Access denied`/`Odmowa dostępu` even after solving the PoW (`scripts/stooq_download.py` implements it — verified working, endpoint still refuses); `q/l/` 404; bulk `d_us_txt.zip` 404/captcha. pandas-datareader removed its Stooq reader in v0.11.0. Now requires an API key → treat as account-gated. |
| **ohlcvault** | **None for US** — own metadata: Nasdaq SymDir listed-only, `complete:false` | Free | keyless static CDN (`data.ohlcvault.com`), monthly `.json.gz` shards 1972→ | Works (probed 2019-01..07). US: 3168–3301 syms/month; LEH/BSC/WYE/MON/TWX/CELG/RSH all absent. Prices verified vs real AAPL closes. Unadjusted only, `source: akshare_sina`. Listed-only ⇒ fails the delisted requirement; usable as a stopgap universe if we accept survivorship bias. |
| **Alpaca Market Data** | Post-2016 delisted incl. S&P500 dropouts (MON/TWX/CELG/...); pre-2016 (LEH/BSC/WYE) absent — SIP starts 2016 | Free (historical SIP; IEX realtime only) | account + key required | **Pipeline built 2026-11-09** — keys supplied by user as GitHub Environment secrets; `us-alpaca-data` workflow + `scripts/alpaca_us_data.py`. Free-plan caveat: `feed=sip` works only with `end` ≥15 min old — fine for daily bars. |
| **Tiingo** | Claims delisted retention; free tier 50 req/hr, 500/day — too slow for a full universe | Free tier | account + key required | needs user approval |
| **Polygon/Massive** | Delisted only on paid tiers | — | — | skip |
| **SimFin** | Bulk CSV incl. inactive companies | Free bulk | account required | needs user approval |
| **Alpha Vantage** | No meaningful delisted set | 25 req/day | — | unusable |

## Verdict

1. **Stooq is no longer a free, keyless source.** No account was created; do
   not sign up without approval.
2. **The delisted-name requirement is unmet by every zero-account source
   found.** Next real candidates are Alpaca or Tiingo free tiers — both need
   an account. Ask the user.
3. **Reframe:** the corrected probe says the old us_data dump is *not*
   obviously defective — raw daily momentum IC is ~±0.01 on two independent
   sources in both markets. Before paying effort/money for a new US dump,
   consider that the US Phase-1 task (Alpha158, daily, sp500,
   TopkDropoutStrategy) may simply be thin-alpha by design; a different
   label/horizon/universe may matter more than a different vendor.
4. If a new dump is still wanted: `scripts/us_data_to_qlib.py` now writes the
   correct qlib `.bin` layout (start-index header), and the probe is the
   acceptance test — target mom5 IC ≥ ~0.03 on a clean universe (neither
   tested source reaches it on sp500; the old 0.058 CN bar was inflated).

## Alpaca pipeline (built 2026-11-09, user supplied keys)

Flow — `.github/workflows/us-alpaca-data.yml` (manual dispatch,
`environment: ALPACA_API_KEY`):

1. `scripts/alpaca_us_data.py universe` — point-in-time sp500 membership from
   `fja05680/sp500` (1262 windows, 1996→, updated for index changes).
2. `... fetch` — `GET /v2/stocks/bars`, `adjustment=all` + a `raw` pass for
   `close_raw`. Delisted/ex-member symbols are queried with `asof` = index-exit
   date so recycled tickers resolve to the right entity. Watchlist
   (LEH BSC WYE RSH MON TWX CA ESRX LLL CELG AGN RTN XLNX ATVI SGEN SIVB FRC)
   is the coverage gate: zero found → exit 3, report, no paid-plan retry.
   Auth rejection → exit 2, same stop.
3. `scripts/us_data_to_qlib.py` → qlib layout, then `... finalize` rewrites
   `factor.day.bin` as adj/raw (community-dump convention: raw = close/factor)
   and `instruments/sp500.txt` with the membership windows.
4. `momentum_probe.py` Jan–Jun 2019 + `spot_check_prices.py` (Yahoo/Nasdaq
   cross-check of adjusted closes). Dump → `us-qlib-dump` artifact; coverage +
   IC appended to EXPERIMENTS.md via an auto-PR.

US configs now benchmark **SPY** (Alpaca has no ^GSPC index) and use
2016–2026 windows (train →2021, valid 2022, test 2023→). `gpu_run.py` names:
`lightgbm_us xgboost_us doubleensemble_us transformer_us hist_us` on
`data/qlib/us_alpaca`.
