"""Phase-6 qlib-API-compatible custom models.

RankLGBModel — LightGBM LambdaRank with one query group per trading day:
the stock-selection problem IS per-day cross-sectional ranking, so this
objective aligns the loss with the topk portfolio construction (plan item
3: "ranking losses (LambdaRank)"). Labels arrive CSRankNormed; they are
binned into per-day decile grades (0..G-1) because lambdarank needs
integer relevance grades; `label_gain` keeps the gains linear so grade 9
is worth 9x grade 0, not 2^9-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model


def _day_groups(idx: pd.MultiIndex) -> np.ndarray:
    """Row-consecutive query sizes for lambdarank `group=`."""
    return idx.get_level_values("datetime").value_counts(sort=False).values


def _to_grades(y: pd.Series, grades: int) -> np.ndarray:
    """Per-day pct-rank of the label -> int grade in [0, grades)."""
    r = y.groupby(level="datetime").rank(pct=True)
    return np.clip((r * grades).astype(int), 0, grades - 1).values


class RankLGBModel(Model):
    """Per-day LambdaRank LightGBM.

    kwargs follow the LGBModel convention; `grades` bins each day's
    label into relevance grades, `eval_at` is the ndcg cut used for
    early stopping (matched to the topk width)."""

    def __init__(self, grades: int = 10, eval_at: int = 50,
                 num_boost_round: int = 2000, early_stop: int = 50,
                 **kwargs):
        self.grades = int(grades)
        self.eval_at = int(eval_at)
        self.num_boost_round = int(num_boost_round)
        self.early_stop = int(early_stop)
        self.params = {"objective": "lambdarank", "verbosity": -1,
                       "label_gain": list(range(self.grades)),
                       "lambdarank_norm": True,
                       "lambdarank_truncation_level": self.eval_at,
                       "eval_at": [self.eval_at], **kwargs}
        self.model = None

    def _dataset(self, df: pd.DataFrame):
        import lightgbm as lgb

        feat = df["feature"].astype(np.float32)
        y = df["label"].iloc[:, 0]
        ds = lgb.Dataset(
            feat, _to_grades(y, self.grades),
            group=_day_groups(df.index), feature_name=list(feat.columns),
            free_raw_data=True)
        return ds

    def fit(self, dataset: DatasetH, reweighter=None):
        import lightgbm as lgb

        dtrain, dvalid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L)
        if dtrain.empty or dvalid.empty:
            raise ValueError("Empty data from dataset")
        train_set = self._dataset(dtrain)
        valid_set = self._dataset(dvalid)
        self.model = lgb.train(
            self.params, train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=[valid_set], valid_names=["valid"],
            callbacks=[lgb.early_stopping(self.early_stop, verbose=False),
                       lgb.log_evaluation(0)])
        return self

    def predict(self, dataset: DatasetH) -> pd.Series:
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature",
                                 data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x_test.astype(np.float32)),
                         index=x_test.index)
