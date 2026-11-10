"""Turnover-penalised mean-variance strategy (Phase 2 experiment d).

Replaces TopkDropoutStrategy: each day, take the top `n_stock` names by signal
and solve

    max_w  score' w  -  risk_lambda * w' Sigma w  -  turnover_lambda * ||w - w_prev||_1
    s.t.   sum(w) = degree,  0 <= w <= max_weight

Sigma is a shrunk sample covariance of trailing `lookback` daily returns of the
candidate names, fetched via qlib D. Falls back to TopkDropout-style weights
(equal weight on top-k) if the solve fails.
"""
import cvxpy as cp
import numpy as np
import pandas as pd
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D


class TurnoverPenalizedMVO(WeightStrategyBase):
    def __init__(
        self,
        n_stock: int = 50,
        lookback: int = 60,
        risk_lambda: float = 10.0,
        turnover_lambda: float = 0.02,
        max_weight: float = 0.05,
        degree: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_stock = n_stock
        self.lookback = lookback
        self.risk_lambda = risk_lambda
        self.turnover_lambda = turnover_lambda
        self.max_weight = max_weight
        self.degree = degree
        self._prev_w: dict[str, float] = {}
        self._ret_panel: pd.DataFrame | None = None

    def _returns_panel(self, codes) -> pd.DataFrame:
        """Full-range daily-returns panel for the scored universe, fetched once
        (per-day D.features queries are the bottleneck otherwise)."""
        if self._ret_panel is None:
            df = D.features(list(codes), ["$close / Ref($close, 1) - 1"],
                            start_time="1990-01-01", end_time="2099-01-01")
            self._ret_panel = (df.iloc[:, 0].unstack(level=0)
                               if df is not None and not df.empty
                               else pd.DataFrame())
        return self._ret_panel

    def _cov(self, fetch_codes, codes, end) -> np.ndarray:
        ret = (self._returns_panel(fetch_codes)
               .reindex(columns=list(codes))
               .loc[:pd.Timestamp(end)].tail(self.lookback))
        if len(ret.dropna(how="all")) < 20:
            return np.eye(len(codes))
        cov = np.cov(np.nan_to_num(ret.values, nan=0.0).T)
        cov = np.nan_to_num(cov, nan=0.0)
        cov = (cov + cov.T) / 2
        return cov + 1e-4 * np.eye(len(codes))

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna().sort_values(ascending=False)
        cand = list(score.index[: self.n_stock])
        if not cand:
            return {}
        # `current` is a qlib Position; get_stock_weight_dict excludes cash.
        held = current.get_stock_weight_dict(only_stock=False)
        all_codes = list(dict.fromkeys(cand + [c for c in held if c not in cand]))

        a = score.reindex(all_codes).fillna(score.min()).values
        w_prev = np.array([held.get(c, 0.0) for c in all_codes])
        try:
            sigma = self._cov(list(score.index), all_codes, trade_start_time)
        except Exception:  # noqa: BLE001
            sigma = np.eye(len(all_codes))

        # Feasibility floor: sum(w)=degree needs cap >= degree/len(all_codes).
        cap = max(self.max_weight, self.degree / len(all_codes) * 1.0001)
        w = cp.Variable(len(all_codes))
        t = cp.Variable(len(all_codes))
        prob = cp.Problem(
            cp.Maximize(a @ w - self.risk_lambda * cp.quad_form(w, sigma)
                        - self.turnover_lambda * cp.sum(t)),
            [cp.sum(w) == self.degree, w >= 0, w <= cap,
             t >= w - w_prev, t >= w_prev - w],
        )
        try:
            prob.solve()
        except Exception:  # noqa: BLE001
            w.value = None
        if w.value is None:
            eq = self.degree / len(cand)
            return {c: eq for c in cand}
        out = {c: max(0.0, float(v)) for c, v in zip(all_codes, w.value) if v > 1e-5}
        self._prev_w = out
        return out
