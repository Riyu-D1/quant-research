# Leaderboard — our runs vs published qlib benchmarks

Backtest: test period 2017-01-01→2020-08-01, TopkDropout(50,5),
open_cost 5bps / close_cost 15bps, deal at close. Returns are
excess vs benchmark *with* cost. Turnover = mean daily portfolio turnover.

## hist_cn  (35.68 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0528 | — |
| icir | 0.3636 | — |
| rank_ic | 0.0651 | — |
| rank_icir | 0.4595 | — |
| annualized_return | 0.0964 | — |
| information_ratio | 1.3604 | — |
| max_drawdown | -0.0650 | — |
| turnover | 0.1598 | — |

## transformer_cn  (14.45 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0317 | — |
| icir | 0.2538 | — |
| rank_ic | 0.0435 | — |
| rank_icir | 0.3626 | — |
| annualized_return | 0.0323 | — |
| information_ratio | 0.4738 | — |
| max_drawdown | -0.0968 | — |
| turnover | 0.1841 | — |

## transformer_us  (17.58 min, 961 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0027 | — |
| icir | 0.0228 | — |
| rank_ic | 0.0064 | — |
| rank_icir | 0.0551 | — |
| annualized_return | -0.1053 | — |
| information_ratio | -1.1419 | — |
| max_drawdown | -0.5225 | — |
| turnover | 0.0531 | — |
