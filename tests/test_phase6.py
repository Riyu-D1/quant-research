"""Phase-6 additions: EW-IC ensemble, signal library, portfolio sim, LGB rank."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research import signals6 as s6
from quant_research import walkforward as wf
from quant_research.models import RankLGBModel, _to_grades

REPO = Path(__file__).resolve().parents[1]


def _idx(days, syms):
    return pd.MultiIndex.from_product([days, syms],
                                      names=["datetime", "instrument"])


@pytest.fixture
def toy():
    days = pd.date_range("2021-01-04", periods=60, freq="B")
    syms = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(0)
    label = pd.Series(rng.normal(0, 0.02, len(days) * len(syms)),
                      index=_idx(days, syms), name="y")
    good = (label + rng.normal(0, 0.01, len(label))).rename("s")
    bad = (-label + rng.normal(0, 0.02, len(label))).rename("s")
    noise = pd.Series(rng.normal(0, 1, len(label)), index=label.index)
    return days, syms, label, {"good": good, "bad": bad, "noise": noise}


def test_ewic_weights_prefers_informative_model(toy):
    _days, _syms, label, preds = toy
    w = wf.ewic_weights(preds, label)
    assert w["good"] > w["bad"] and w["bad"] == pytest.approx(0.0, abs=1e-9)
    assert w.sum() == pytest.approx(1.0)


def test_ewic_rank_average_fold_causal(toy):
    days, _syms, label, preds = toy
    folds = [[str(days[0].date()), str(days[29].date())],
             [str(days[30].date()), str(days[59].date())]]
    out = wf.ewic_rank_average(preds, label, folds)
    # fold 0 = equal weights across the 3 models; fold 1 = fold-0 IC weights
    assert out.index.get_level_values("datetime").max() <= days[59]
    f1 = out[out.index.get_level_values("datetime") >= days[30]]
    assert len(f1) > 0


class _FakeDataset:
    """Minimal DatasetH stand-in for RankLGBModel."""

    def __init__(self, X, y, seg):
        self.X, self.y, self.segments = X, y, seg

    def prepare(self, segment, col_set=None, data_key=None):
        def one(name):
            idx = self.segments[name]
            if isinstance(col_set, (list, tuple)):
                return pd.concat(
                    {"feature": self.X.loc[idx],
                     "label": self.y.loc[idx].to_frame("LABEL0")}, axis=1)
            return self.X.loc[idx] if col_set == "feature" \
                else self.y.loc[idx]
        if isinstance(segment, str):
            return one(segment)
        return [one(s) for s in segment]


def _mk_idx(days_per_sym, syms, start="2020-01-01"):
    days = pd.date_range(start, periods=days_per_sym, freq="B")
    return pd.MultiIndex.from_product([days, syms],
                                      names=["datetime", "instrument"])


def test_rank_lgb_fits_and_predicts():
    idx = _mk_idx(80, [f"S{i}" for i in range(30)])
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(len(idx), 8)), index=idx,
                     columns=[f"f{i}" for i in range(8)])
    alpha = X["f0"] * 0.05
    y = pd.Series(alpha + rng.normal(0, 0.02, len(idx)), index=idx)
    days = idx.get_level_values("datetime").unique().sort_values()
    seg = {"train": idx[idx.get_level_values("datetime") < days[50]],
           "valid": idx[(idx.get_level_values("datetime") >= days[50]) &
                        (idx.get_level_values("datetime") < days[65])],
           "test": idx[idx.get_level_values("datetime") >= days[65]]}
    m = RankLGBModel(grades=5, eval_at=5, num_boost_round=30, early_stop=10,
                     num_leaves=8, min_data_in_leaf=5)
    ds = _FakeDataset(X, y, seg)
    m.fit(ds)
    pred = m.predict(ds)
    assert len(pred) == len(seg["test"])
    # sanity: pred correlates positively with the planted alpha on test
    rho = pred.corr(y.loc[seg["test"]], method="spearman")
    assert rho > 0.2


def test_to_grades_monotone():
    days = pd.date_range("2021-01-01", periods=3, freq="B")
    idx = _idx(days, ["A", "B", "C", "D"])
    y = pd.Series([0.1, -0.2, 0.3, -0.1] * 3, index=idx)
    g = pd.Series(_to_grades(y, 4), index=idx)
    # within each day, higher label -> weakly higher grade
    for _, d in g.groupby(level="datetime"):
        order = y.loc[d.index].rank()
        assert (d.sort_values(key=lambda s: order.loc[s.index])
                .is_monotonic_increasing)
    assert g.min() >= 0 and g.max() <= 3


def test_price_signals_signs():
    days = pd.date_range("2021-01-04", periods=300, freq="B")
    rng = np.random.default_rng(2)
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.015, (300, 5)), axis=0),
        index=days, columns=list("ABCDE"))
    vol = pd.DataFrame(rng.integers(1e5, 1e7, (300, 5)).astype(float),
                       index=days, columns=list("ABCDE"))
    sigs = s6.price_signals(close, vol)
    assert set(sigs) == {"srev5", "rev", "mom6", "mom", "lowvol", "beta",
                         "amihud"}
    # lowvol is negative std -> all values <= 0; srev5 = -5d return
    assert (sigs["lowvol"].dropna(how="all") <= 0).all().all()
    man_srev = -(close / close.shift(5) - 1.0)
    pd.testing.assert_frame_equal(sigs["srev5"], man_srev)


def test_sector_neutral_zero_mean_within_sector():
    days = pd.date_range("2021-01-04", periods=5, freq="B")
    idx = _idx(days, ["A", "B", "C", "D"])
    s = pd.Series([1.0, 2.0, 3.0, 4.0] * 5, index=idx)
    smap = {"A": "X", "B": "X", "C": "Y", "D": "Y"}
    out = s6.sector_neutral(s, smap).rename("v").reset_index()
    out["sec"] = out["instrument"].map(smap)
    means = out.groupby(["datetime", "sec"])["v"].mean()
    assert np.allclose(means, 0.0)


def test_portfolio_respects_eligibility_and_costs():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase6_us", "scripts/phase6_us.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    days = pd.date_range("2021-01-04", periods=30, freq="B")
    syms = ["A", "B", "C", "D"]
    idx = _idx(days, syms)
    # A always best by score, B second; C/D never eligible
    lut = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    score = pd.Series([lut[s] for s in idx.get_level_values("instrument")],
                      index=idx)
    label = pd.Series(0.01, index=idx)
    elig = pd.DataFrame(True, index=days, columns=syms)
    elig[["C", "D"]] = False
    r = mod.portfolio_returns(score, label, elig, topk=1, hold=1,
                              cost_bps=0)
    # only A (eligible, top score) is held -> gross = 0.01 every day
    assert np.allclose(r["gross"].dropna(), 0.01)
    r2 = mod.portfolio_returns(score, label, elig, topk=1, hold=1,
                               cost_bps=50)
    # constant weights -> only the day-0 build-up trades
    assert r2["net"].iloc[0] < r2["gross"].iloc[0]
    assert (r2["net"].iloc[1:] == r2["gross"].iloc[1:]).all()


def test_load_fold_pred_tra_frame_becomes_score_series(tmp_path, monkeypatch):
    """qlib TRA predict() returns a DataFrame (score + per-state score_i +
    label); load_fold_pred must hand downstream code the score Series."""
    import pickle
    monkeypatch.setenv("PHASE2_RESULTS_DIR", str(tmp_path))
    days = pd.date_range("2018-01-02", periods=3, freq="B")
    idx = _idx(days, ["A", "B"])
    df = pd.DataFrame({"score": range(6), "label": range(6),
                       "score_0": range(6), "score_1": range(6),
                       "score_2": range(6)}, index=idx)
    lab = pd.Series(0.01, index=idx)
    p = tmp_path / "oof" / "cn"
    p.mkdir(parents=True)
    with open(p / "tra__fold0.pkl", "wb") as f:
        pickle.dump({"pred": df, "label": lab}, f)
    pred, label = wf.load_fold_pred("cn", "tra", "fold0")
    assert isinstance(pred, pd.Series)
    pd.testing.assert_series_equal(pred, df["score"])
    pd.testing.assert_series_equal(label, lab)


def test_gpu_run_skips_us_models_when_dump_missing(monkeypatch, capsys,
                                                   tmp_path):
    """A missing us-qlib-dump warns + drops *_us models instead of the old
    sys.exit that killed the run before the push cell."""
    import scripts.gpu_run as gr
    missing = tmp_path / "no_such_dump"
    monkeypatch.setattr(gr, "US", missing)
    monkeypatch.setitem(gr.REGISTRY, "tcn_us_wf",
                        ("scripts/x.py", missing, "phase6", "fake us entry"))
    monkeypatch.setattr("sys.argv",
                        ["gpu_run.py", "--models", "tcn_us_wf"])
    gr.main()   # must not raise SystemExit
    out = capsys.readouterr().out
    assert "skipping" in out and "tcn_us_wf" in out


def test_kaggle_notebook_push_plumbing():
    """Pin the fixed push discipline in qlib_gpu_run.ipynb — the empty
    kaggle-wf-2026-11-16 push must not regress."""
    import json
    nb = json.loads((REPO / "notebooks" / "qlib_gpu_run.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"]
             if c["cell_type"] == "code"]
    allsrc = "\n".join(cells)
    # scoped force-add before every commit, incl. the holdout-preds summary
    for path in ("results/oof", "results/phase2_cn_holdout_preds.json",
                 "results/runs"):
        assert f"'{path}'" in allsrc, path
    assert "'add', '-f'" in allsrc
    # periodic + per-model checkpoints, not one final blind commit
    assert "_ckpt_loop" in allsrc and "ckpt(name)" in allsrc
    # final cell fails loudly when a successful run staged nothing
    assert "refusing to push an empty" in allsrc
    # missing us-qlib-dump warns + skips *_us instead of raising
    us_cell = next(c for c in cells if "archive_download_url" in c)
    assert "skipping *_us models" in us_cell
    assert "raise RuntimeError('no us-qlib-dump" not in us_cell


def test_live_new_signal_fns():
    from quant_research import live
    days = pd.date_range("2021-01-04", periods=300, freq="B")
    px = pd.Series(np.linspace(50, 80, 300), index=days)
    assert live.reversal5(px, days[-1]) < 0        # rose -> negative srev
    assert live.momentum6(px, days[-1]) > 0        # uptrend -> positive mom6
    assert live.lowvol(px, days[-1]) < 0           # negated vol
    closes = pd.DataFrame({"A": px, "B": px * 1.01},
                         index=days)
    out = live.bars_signals(closes, days[-3:],
                            ["srev5", "mom6", "lowvol", "beta"])
    assert set(out) == {"srev5", "mom6", "lowvol", "beta"}
