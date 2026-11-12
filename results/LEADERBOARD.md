# Leaderboard — our runs vs published qlib benchmarks

Backtest: test period 2017-01-01→2020-08-01, TopkDropout(50,5),
open_cost 5bps / close_cost 15bps, deal at close. Returns are
excess vs benchmark *with* cost. Turnover = mean daily portfolio turnover.

## doubleensemble_alpha158_cn  (46.43 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0506 | 0.0521 |
| icir | 0.4145 | 0.4223 |
| rank_ic | 0.0495 | 0.0502 |
| rank_icir | 0.4081 | 0.4117 |
| annualized_return | 0.1085 | 0.1158 |
| information_ratio | 1.2773 | 1.3432 |
| max_drawdown | -0.0631 | -0.0920 |
| turnover | 0.1936 | — |

## doubleensemble_alpha158_us  (122.91 min, 901 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0070 | — |
| icir | 0.0600 | — |
| rank_ic | 0.0079 | — |
| rank_icir | 0.0701 | — |
| annualized_return | -0.0048 | — |
| information_ratio | -0.0376 | — |
| max_drawdown | -0.2229 | — |
| turnover | 0.1190 | — |

## hist_alpha360_cn  (35.68 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0528 | 0.0522 |
| icir | 0.3636 | 0.3530 |
| rank_ic | 0.0651 | 0.0667 |
| rank_icir | 0.4595 | 0.4576 |
| annualized_return | 0.0964 | 0.0987 |
| information_ratio | 1.3604 | 1.3726 |
| max_drawdown | -0.0650 | -0.0681 |
| turnover | 0.1598 | — |

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

## lightgbm_alpha158_us  (16.99 min, 901 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0066 | — |
| icir | 0.0634 | — |
| rank_ic | 0.0047 | — |
| rank_icir | 0.0453 | — |
| annualized_return | -0.0035 | — |
| information_ratio | -0.0286 | — |
| max_drawdown | -0.2610 | — |
| turnover | 0.1340 | — |

## transformer_alpha158_cn  (14.45 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0317 | 0.0264 |
| icir | 0.2538 | 0.2053 |
| rank_ic | 0.0435 | 0.0407 |
| rank_icir | 0.3626 | 0.3273 |
| annualized_return | 0.0323 | 0.0273 |
| information_ratio | 0.4738 | 0.3970 |
| max_drawdown | -0.0968 | -0.1101 |
| turnover | 0.1841 | — |

## transformer_alpha158_us  (17.58 min, 961 days)
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

## xgboost_alpha158_cn  (4.97 min, 871 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0500 | 0.0498 |
| icir | 0.3752 | 0.3779 |
| rank_ic | 0.0484 | 0.0505 |
| rank_icir | 0.3917 | 0.4131 |
| annualized_return | 0.0620 | 0.0780 |
| information_ratio | 0.7459 | 0.9070 |
| max_drawdown | -0.1149 | -0.1168 |
| turnover | 0.1867 | — |

## xgboost_alpha158_us  (14.52 min, 901 days)
| metric | ours | published |
|---|---|---|
| ic | 0.0022 | — |
| icir | 0.0183 | — |
| rank_ic | 0.0028 | — |
| rank_icir | 0.0181 | — |
| annualized_return | 0.0017 | — |
| information_ratio | 0.0123 | — |
| max_drawdown | -0.3117 | — |
| turnover | 0.1781 | — |
