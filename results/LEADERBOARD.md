# Leaderboard — our runs vs published qlib benchmarks

Backtest: test period 2017-01-01→2020-08-01, TopkDropout(50,5),
open_cost 5bps / close_cost 15bps, deal at close. Returns are
excess vs benchmark *with* cost. Turnover = mean daily portfolio turnover.

## lightgbm_alpha158_cn  (191.38 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0468 | 0.0448 |
| icir | 0.3816 | 0.3660 |
| rank_ic | 0.0490 | 0.0469 |
| rank_icir | 0.4067 | 0.3877 |
| annualized_return | 0.0807 | 0.0901 |
| information_ratio | 0.9145 | 1.0164 |
| max_drawdown | -0.0861 | -0.1038 |
| turnover | 0.1911 | — |
