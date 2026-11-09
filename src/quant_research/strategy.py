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

    def _cov(self, codes, end) -> np.ndarray:
        start = pd.Timestamp(end) - pd.Timedelta(days=self.lookback * 2)
        df = D.features(list(codes), ["Ref($close, 1) / $close - 1"],
                        start_time=start, end_time=end)
        df = df.droplevel(0).unstack(level=0).iloc[:, 0].unstack(0) if df is not None else None
        ret = df.T.groupby(level=0).last().T if df is not None else df
        ret = ret.tail(self.lookback)
        if ret is None or len(ret) < 20:
            return np.eye(len(codes))
        cov = np.cov(ret.reindex(columns=list(codes)).fillna(0).values.T)
        return cov + 1e-4 * np.eye(len(codes))

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        score = score.dropna().sort_values(ascending=False)
        cand = list(score.index[: self.n_stock])
        if not cand:
            return {}
        held = {c: v for c, v in getattr(current, "position", {}).items()}
        all_codes = list(dict.fromkeys(cand + [c for c in held if c not in cand]))

        a = score.reindex(all_codes).fillna(score.min()).values
        w_prev = np.array([held.get(c, {}).get("weight", 0.0) for c in all_codes])
        try:
            sigma = self._cov(all_codes, trade_start_time)
        except Exception:
            sigma = np.eye(len(all_codes))

        w = cp.Variable(len(all_codes))
        t = cp.Variable(len(all_codes))
        prob = cp.Problem(
            cp.Maximize(a @ w - self.risk_lambda * cp.quad_form(w, sigma)
                        - self.turnover_lambda * cp.sum(t)),
            [cp.sum(w) == self.degree, w >= 0, w <= self.max_weight,
             t >= w - w_prev, t >= w_prev - w],
        )
        try:
            prob.solve()
        except Exception:
            prob.status = "fail"
        if prob.status != "optimal" or w.value is None:
            eq = self.degree / len(cand)
            return {c: eq for c in cand}
        out = {c: max(0.0, float(v)) for c, v in zip(all_codes, w.value) if v > 1e-5}
        self._prev_w = out
        return out
