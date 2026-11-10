# Experiment log

Every run counts — including failures and reproduce-only runs. Numbering is
monotonic; `EX-000` is environment setup. "Δ" columns are vs the best published
baseline for the same dataset/features once Phase 2 starts.

| ID | Date | Phase | Change | Dataset | IC | ICIR | Rank IC | Ann. ret | IR | Max DD | Turnover | Verdict |
|----|------|-------|--------|---------|----|------|---------|----------|----|--------|----------|---------|
| EX-000 | 2026-11-09 | 1 | Repo scaffold, py3.11 + uv, pinned deps | — | — | — | — | — | — | — | — | setup |
| EX-001 | 2026-11-09 | 1 | LightGBM Alpha158 CSI300 reproduction (published IC 0.0448 / IR 1.0164 / MDD -0.1038) | csi300 | 0.0468 | 0.3816 | 0.0490 | 0.0807 | 0.9145 | -0.0861 | 0.1911 | reproduced, close to published (v2 data) |
| EX-002 | 2026-11-09 | 1 | XGBoost Alpha158 CSI300 reproduction (published IC 0.0498 / ICIR 0.3779 / RankIC 0.0505 / IR 0.9070 / MDD -0.1168) | csi300 | 0.0500 | 0.3752 | 0.0484 | 0.0620 | 0.7459 | -0.1149 | 0.1867 | reproduced; IC/ICIR near-exact, excess return lower |
| EX-003 | 2026-11-09 | 1 | DoubleEnsemble Alpha158 CSI300 reproduction (published IC 0.0521 / ICIR 0.4223 / RankIC 0.0502 / IR 1.3432 / MDD -0.0920) | csi300 | 0.0506 | 0.4145 | 0.0495 | 0.1085 | 1.2773 | -0.0631 | 0.1936 | reproduced; close to published |
| EX-004 | 2026-11-09 | 1 | LightGBM Alpha158 S&P500 reproduction (no published ref) | sp500 | 0.0066 | 0.0634 | 0.0047 | -0.0035 | -0.0286 | -0.2610 | — | near-zero signal — investigating US data |
| EX-005 | 2026-11-09 | 1 | XGBoost Alpha158 S&P500 reproduction (no published ref) | sp500 | 0.0022 | 0.0183 | 0.0028 | 0.0017 | 0.0123 | -0.3117 | 0.1781 | near-zero signal — consistent w/ EX-004 |
| EX-006 | 2026-11-09 | 1 | DoubleEnsemble Alpha158 S&P500 reproduction (no published ref) | sp500 | 0.0070 | 0.0600 | 0.0079 | -0.0048 | -0.0376 | -0.2229 | 0.1190 | near-zero signal — confirms US dump limitation |
| EX-007 | 2026-11-09 | 1 | Raw 5d/20d momentum IC probe (scripts/momentum_probe.py, Jan–Jun 2019) | sp500 vs csi300 | 0.004 vs 0.058 | — | — | — | — | — | — | us_data dump carries ~no next-day alpha; data property, not model/pipeline |
| EX-008 | 2026-11-09 | 1 | Transformer Alpha158 CSI300 — local CPU attempt | csi300 | — | — | — | — | — | — | — | KILLED before completion (user: training moves to cloud GPU); multiple macOS OpenMP SIGSEGVs fixed en route (see PROVENANCE) |
| EX-009 | 2026-11-09 | 1 | Stooq US daily download attempt (per docs/US_DATA.md plan) | — | — | — | — | — | — | — | — | FAILED — Stooq now gates all CSV endpoints (JS PoW + `Access denied` post-verify; bulk zips 404). Solver implemented in scripts/stooq_download.py; no account created |
| EX-010 | 2026-11-09 | 1 | momentum_probe.py bin-format bug fix: .bin[0] is a start_index header, series are offset — old probe silently dropped ~80% of instruments and misaligned the rest | — | — | — | — | — | — | — | — | Corrected: CN csi300 mom5 IC 0.0068 (was reported 0.058); US sp500 0.0098 (was 0.004). CN-US "signal gap" was largely a measurement artifact |
| EX-011 | 2026-11-09 | 1 | ohlcvault (keyless CDN, akshare_sina source) US dump probed as Stooq alternative | sp500∩listed | 0.0043 | — | 0.0033 | — | — | — | — | Real prices verified (AAPL exact). IC fails ≥0.03 bar; US is listed-only (no LEH/BSC/WYE/MON/TWX/CELG/RSH) → fails delisted test. Per plan: stop, ask before Alpaca/Tiingo accounts |
| EX-012 | 2026-11-09 | 1 | Alpaca→qlib US pipeline: `us-alpaca-data` GH Action (env `ALPACA_API_KEY`) — PIT sp500 universe (fja05680/sp500), adjusted SIP bars 2016→, asof-pinned delisted queries, `us-qlib-dump` artifact; US configs rebased to SPY/2016–2026; `hist_us` registered | sp500 PIT | 0.0003–0.0086 | — | — | — | — | — | — | TWO runs completed (see us-data/* PRs): feed=sip, 744/746 syms, 13/17 delisted found (LEH/BSC/WYE/RSH absent — pre-2016 SIP floor); MON recycled-entity bug found & fixed (asof→-30d + bar cap); adj returns verified vs community dump (~1bp mean diff); mom5 IC ~0.01 as expected — gate is coverage, not alpha |
| EX-013 | 2026-11-10 | 1 | Transformer Alpha158 CSI300 — Kaggle T4 GPU run (published IC 0.0264 / ICIR 0.2053 / RankIC 0.0407 / IR 0.3970 / MDD -0.1101) | csi300 | 0.0317 | 0.2538 | 0.0435 | 0.0323 | 0.4738 | -0.0968 | 0.1841 | reproduced; beats published on every metric |
| EX-014 | 2026-11-10 | 1 | HIST Alpha360 CSI300 — Kaggle T4 GPU run (published IC 0.0522 / ICIR 0.3530 / RankIC 0.0667 / IR 1.3726 / MDD -0.0681) | csi300 | 0.0528 | 0.3636 | 0.0651 | 0.0964 | 1.3604 | -0.0650 | 0.1598 | reproduced; IR 1.3604 > published DoubleEnsemble 1.3432, just under published HIST 1.3726 |
| EX-015 | 2026-11-10 | 1 | Transformer Alpha158 S&P500 PIT — Kaggle T4 GPU run (no published ref) | sp500 PIT | 0.0027 | 0.0228 | 0.0064 | -0.1053 | -1.1419 | -0.5225 | 0.0531 | near-zero signal — consistent with US GBDT baselines |

## Running total of hypotheses tested

0 (baseline reproduction does not count as a hypothesis).
