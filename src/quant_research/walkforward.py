"""Phase-2 walk-forward evaluation engine.

Protocol (honest evaluation only):

* The data span is split into K expanding-window folds plus a final,
  untouched holdout. For fold i, ``valid`` is the last `valid_span` of the
  pre-eval period (early stopping / model selection), ``eval`` is the
  out-of-fold (OOF) slice that produces stacked-ensemble training data and
  candidate-comparison metrics.
* The holdout is never used for tuning, stacking weights, or candidate
  selection. It is scored exactly once per final candidate via
  ``--holdout signal@strategy``; every scoring is appended to
  ``results/holdout_log.jsonl`` (auditable in git).
* ``n_trials`` for the deflated Sharpe ratio counts Optuna trials plus every
  signal×strategy candidate evaluated on the OOF region plus every holdout
  scoring, so the reported DSR is conservative.

Artifacts (all under results/, committed to git):
    results/oof/<market>/<model>__<fold>.pkl   per-fold eval-slice preds+label
    results/oof/<market>/<model>__holdout.pkl  holdout-fold preds+label
    results/phase2_<market>.json               full metrics/candidate table
    results/holdout_log.jsonl                  one JSON line per holdout score
"""
from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.backtest import backtest as qlib_backtest
from qlib.config import C
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# specs


@dataclass
class Fold:
    name: str
    train: tuple[str, str]
    valid: tuple[str, str]
    eval: tuple[str, str]


def make_folds(spec: dict) -> tuple[list[Fold], Fold]:
    """Build OOF folds + the holdout fold from a market spec.

    OOF fold: train [start, eval_start - 1y - 1d], valid [eval_start - 1y, -1d],
    eval as listed. Holdout fold: same shape with eval = the holdout window.
    """
    folds = []
    start = spec["data_start"]
    valid_days = int(spec.get("valid_days", 365))
    for i, (e0, e1) in enumerate(spec["eval_folds"]):
        e0t, e1t = pd.Timestamp(e0), pd.Timestamp(e1)
        v1 = e0t - pd.Timedelta(days=1)
        v0 = e0t - pd.Timedelta(days=valid_days)
        t1 = v0 - pd.Timedelta(days=1)
        folds.append(Fold(f"fold{i}", (start, str(t1.date())),
                          (str(v0.date()), str(v1.date())),
                          (str(e0t.date()), str(e1t.date()))))
    h0, h1 = pd.Timestamp(spec["holdout"][0]), pd.Timestamp(spec["holdout"][1])
    hv1 = h0 - pd.Timedelta(days=1)
    hv0 = h0 - pd.Timedelta(days=valid_days)
    ht1 = hv0 - pd.Timedelta(days=1)
    holdout = Fold("holdout", (start, str(ht1.date())),
                   (str(hv0.date()), str(hv1.date())),
                   (str(h0.date()), str(h1.date())))
    return folds, holdout


def load_spec(market_or_spec: str) -> dict:
    """Accepts a market name ('us' -> walkforward_us.yaml) or an explicit
    spec stem/file ('walkforward_us_events' / a .yaml path)."""
    p = Path(market_or_spec)
    if not p.suffix:
        p = REPO / "configs/experiments" / (
            market_or_spec if market_or_spec.startswith("walkforward")
            else f"walkforward_{market_or_spec}")
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    spec = yaml.safe_load(p.read_text())
    spec["_spec_name"] = p.stem
    return spec


# --------------------------------------------------------------------------
# dataset shim — DatasetH-compatible API over one fitted handler


class FoldDataset:
    """Serve one fold's train/valid/eval slices from an initialized handler.

    Mirrors the calls qlib models make on DatasetH:
      fit     -> prepare(["train","valid"], col_set=[...], data_key=DK_L)
      predict -> prepare("test", col_set="feature", data_key=DK_I)

    With ``ts_step`` set, prepares behave like TSDatasetH (returns
    TSDataSampler with the look-back extension) — required by Transformer.
    """

    def __init__(self, handler: DataHandlerLP, segments: dict[str, tuple],
                 ts_step: int | None = None):
        self.handler = handler
        self.ts_step = ts_step
        self.segments = {
            k: slice(pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in segments.items()
        }

    def _fetch(self, slc: slice, col_set, data_key, **kw):
        if self.ts_step is None:
            return self.handler.fetch(slc, col_set=col_set, data_key=data_key, **kw)
        from qlib.data import D
        from qlib.data.dataset import TSDataSampler
        dtype = kw.pop("dtype", "float32")
        kw.pop("flt_col", None)
        cal = D.calendar()
        idx = np.searchsorted(cal, np.datetime64(slc.start))
        ext_start = cal[max(0, idx - self.ts_step)]
        data = self.handler.fetch(slice(pd.Timestamp(ext_start), slc.stop),
                                  col_set=col_set, data_key=data_key, **kw)
        return TSDataSampler(data=data, start=slc.start, end=slc.stop,
                             step_len=self.ts_step, dtype=dtype)

    def prepare(self, segments, col_set="__all", data_key=DataHandlerLP.DK_I, **kw):
        if isinstance(segments, slice):
            return self._fetch(segments, col_set, data_key, **kw)
        single = isinstance(segments, str)
        names = [segments] if single else list(segments)
        out = [self._fetch(self.segments[s], col_set, data_key, **kw) for s in names]
        return out[0] if single else out


# --------------------------------------------------------------------------
# handlers / models


def build_handler(spec: dict, feature_set: str, fold: Fold,
                  label: str | None = None,
                  infer_processors: list | None = None) -> DataHandlerLP:
    """Init a feature handler with processors fitted on this fold's train
    window only (no look-ahead across the eval boundary).

    label: per-model training-label expression override (e.g. a 5-day
    horizon). The eval metric label is always fetched separately from the
    canonical handler (spec["label"]) — see run_folds.
    infer_processors: per-model processor-list override (e.g. the 20-feature
    FilterCol subset used by the published neural-model configs)."""
    hconf = spec["handler"]
    extra = {}
    if feature_set == "alpha158":
        cls = {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"}
    elif feature_set == "alpha158_360":
        cls = {"class": "Alpha158_360", "module_path": "quant_research.handler"}
    elif feature_set == "alpha360":
        cls = {"class": "Alpha360", "module_path": "qlib.contrib.data.handler"}
    elif feature_set in ("alpha158_events", "events"):
        from quant_research import events as ev
        cls = ({"class": "Alpha158Events", "module_path": "quant_research.handler"}
               if feature_set == "alpha158_events"
               else {"class": "EventsOnly", "module_path": "quant_research.handler"})
        extra = {"events_panel": str(REPO / spec.get(
                     "events_panel", "data/events/features_us.parquet")),
                 "event_cols": ev.FEATURE_COLS}
    else:
        raise ValueError(feature_set)
    # Load only up to this fold's eval end (+ buffer for the Ref(-2) label);
    # earlier eval folds don't need post-eval rows — halves peak RSS on
    # memory-tight CI runners and changes nothing the processors see.
    end_time = min(pd.Timestamp(spec["data_end"]),
                   pd.Timestamp(fold.eval[1]) + pd.Timedelta(days=10))
    kwargs = {
        "instruments": spec["instruments"],
        "start_time": spec["data_start"],
        "end_time": str(end_time.date()),
        "fit_start_time": fold.train[0],
        "fit_end_time": fold.train[1],
        "infer_processors": (infer_processors if infer_processors is not None
                             else hconf["infer_processors"]),
        "learn_processors": hconf["learn_processors"],
        "label": [label or spec["label"]],
        **extra,
    }
    return init_instance_by_config({**cls, "kwargs": kwargs})


def rule_pred(spec: dict, signal: str, window: tuple[str, str]) -> pd.Series:
    """Precomputed rule signal -> pred series over one eval window.

    Rule signals are columns of the events panel; NaN = 'no signal' so the
    strategy only holds names the rule flagged. Index is qlib-style
    (datetime, instrument) like model predictions."""
    panel = pd.read_parquet(
        REPO / spec.get("events_panel", "data/events/features_us.parquet"),
        columns=[signal])
    s = panel[signal]
    if s.index.names[0] == "instrument":
        s = s.swaplevel().sort_index()
    s = s.loc[pd.Timestamp(window[0]):pd.Timestamp(window[1])]
    # only flagged (inst,day) rows — absent names are out of the candidate
    # set entirely (qlib topk would otherwise fill slots with NaN-scored
    # arbitrary names, polluting the baseline)
    return s[s > 0].rename("score")


def make_dataset(model_cfg: dict, handler: DataHandlerLP,
                 segments: dict[str, tuple], ts_step: int | None = None):
    """Build the dataset object a model trains/predicts on.

    Default is the FoldDataset shim (DatasetH/TSDatasetH-compatible). A
    model may pin a different qlib dataset via spec `dataset:` (e.g.
    MTSDatasetH for TRA) — its kwargs get the handler + segments injected;
    seq_len/num_states-style params come from the spec."""
    dconf = model_cfg.get("dataset")
    if not dconf:
        return FoldDataset(handler, segments, ts_step=ts_step)
    kw = {**dconf.get("kwargs", {}), "handler": handler,
          "segments": segments}
    return init_instance_by_config({"class": dconf["class"],
                                    "module_path": dconf["module_path"],
                                    "kwargs": kw})


def train_predict(model_cfg: dict, dataset, num_threads: int) -> pd.Series:
    kw = dict(model_cfg.get("kwargs", {}))
    for k in ("num_threads", "nthread", "n_jobs"):
        if k in kw:
            kw[k] = num_threads
    model = init_instance_by_config(
        {"class": model_cfg["class"], "module_path": model_cfg["module_path"],
         "kwargs": kw})
    model.fit(dataset)
    pred = model.predict(dataset)  # segment "test" == eval slice
    return pred


# --------------------------------------------------------------------------
# metrics


def signal_metrics(pred: pd.Series, label: pd.Series) -> dict:
    df = pd.concat({"p": pred, "y": label.reindex(pred.index)}, axis=1).dropna()
    if df.empty:
        return {"IC": None, "ICIR": None, "RankIC": None, "RankICIR": None}
    by_day = df.groupby(level="datetime")
    ic = by_day.apply(lambda g: g["p"].corr(g["y"]))
    ric = by_day.apply(lambda g: g["p"].corr(g["y"], method="spearman"))
    out = {"IC": float(ic.mean()), "ICIR": float(ic.mean() / ic.std()),
           "RankIC": float(ric.mean()), "RankICIR": float(ric.mean() / ric.std())}
    return out


def cs_zscore(pred: pd.Series) -> pd.Series:
    """Per-day cross-sectional z-score (NaN-safe)."""
    g = pred.groupby(level="datetime")
    m, s = g.transform("mean"), g.transform("std")
    return (pred - m) / s.replace(0, np.nan)


def cs_rank(pred: pd.Series) -> pd.Series:
    return pred.groupby(level="datetime").rank(pct=True)


def backtest_signal(pred: pd.Series, spec: dict, strategy: dict,
                    window: tuple[str, str]) -> dict:
    """Run qlib daily backtest on `pred` within window; return metrics incl.
    the daily excess-with-cost return series (for DSR downstream).

    strategy class "MicroTop5" routes to the fractional £10-account sim
    (quant_research.micro) instead of the qlib account model."""
    if strategy.get("class") == "MicroTop5":
        from quant_research import micro
        if spec.get("_closes") is None:
            spec["_closes"] = micro.load_closes(
                REPO / spec["provider_uri"], spec["instruments"])
        kw = strategy.get("kwargs", {})
        px = spec["_closes"].loc[
            pd.Timestamp(window[0]):pd.Timestamp(window[1])]
        p = pred.loc[pd.Timestamp(window[0]):pd.Timestamp(window[1])]
        return micro.micro_backtest(
            p, px, n=kw.get("n", 5), cash_gbp=kw.get("cash_gbp", 10.0),
            gbpusd=kw.get("gbpusd", micro.GBPUSD),
            cost_bps=kw.get("cost_bps", 10.0))
    strategy_cfg = {"class": strategy["class"],
                    "module_path": strategy["module_path"],
                    "kwargs": {**{k: v for k, v in strategy.get("kwargs", {}).items()
                                  if k != "signal"},
                               "signal": pred.to_frame()}}
    pm, _ = qlib_backtest(
        executor={"class": "SimulatorExecutor",
                  "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "day",
                             "generate_portfolio_metrics": True}},
        strategy=strategy_cfg,
        start_time=window[0], end_time=window[1],
        account=spec["account"], benchmark=spec["benchmark"],
        exchange_kwargs=spec["exchange_kwargs"],
    )
    df = pm["1day"][0]
    excess = df["return"] - df["bench"] - df["cost"]
    cum = (1 + excess).cumprod()
    return {
        "annualized_return": float(excess.mean() * 252),
        "information_ratio": float(excess.mean() / excess.std() * math.sqrt(252)),
        "max_drawdown": float((cum / cum.cummax() - 1).min()),
        "turnover": float(df["turnover"].mean()) if "turnover" in df else None,
        "n_days": len(df),
        "daily_excess": excess,
    }


# --------------------------------------------------------------------------
# deflated Sharpe (Bailey & López de Prado 2014)


def deflated_sharpe(daily_ret: pd.Series, n_trials: int,
                    trial_sharpes: list[float]) -> dict:
    """DSR: probability that the candidate's Sharpe exceeds E[max Sharpe]
    of n_trials independent trials with the same variance.

    trial_sharpes: annualized Sharpes of the evaluated candidates (used only
    for their cross-sectional variance, per the paper's sigma-hat)."""
    from scipy.stats import norm
    r = daily_ret.dropna()
    t = len(r)
    sr_d = float(r.mean() / r.std())           # per-period Sharpe
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0               # pandas gives excess kurtosis
    n = max(int(n_trials), 2)
    sig = float(np.std(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 \
        else 1.0 / math.sqrt(t - 1)
    sig_d = sig / math.sqrt(252)               # trial SRs are annualized
    g = 0.5772156649015329                     # Euler–Mascheroni
    emax = sig_d * ((1 - g) * norm.ppf(1 - 1 / n)
                    + g * norm.ppf(1 - 1 / (n * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4 * sr_d * sr_d))
    dsr = float(norm.cdf((sr_d - emax) * math.sqrt(t - 1) / den))
    return {"DSR": dsr, "SR_daily": sr_d, "SR_ann": sr_d * math.sqrt(252),
            "SR_star": emax * math.sqrt(252), "n_trials": n,
            "skew": skew, "kurtosis": kurt, "T": t}


# --------------------------------------------------------------------------
# ensemble builders (fit on OOF only)


def zscore_frame(preds: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame({k: cs_zscore(v) for k, v in preds.items()})


def rank_average(preds: dict[str, pd.Series]) -> pd.Series:
    R = pd.DataFrame({k: cs_rank(v) for k, v in preds.items()})
    return R.mean(axis=1).rename("score")


def ewic_weights(preds: dict[str, pd.Series], label: pd.Series,
                 windows: list[tuple[str, str]] | None = None) -> pd.Series:
    """Floored, L1-normalised mean daily rank-IC per model -> weight row.

    windows=None uses all of `preds` (the holdout-fold case). Otherwise
    each model's weight is its mean daily rank-IC over eval days inside
    `windows` — callers pass the *earlier* folds' windows so the weight
    for fold i never sees fold-i eval data."""
    ics = {}
    for name, p in preds.items():
        use = p
        if windows:
            d = p.index.get_level_values("datetime")
            mask = np.zeros(len(p), dtype=bool)
            for a, b in windows:
                mask |= (d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))
            use = p[mask]
        df = pd.concat({"p": use, "y": label.reindex(use.index)},
                       axis=1).dropna()
        if df.empty:
            continue
        ric = df.groupby(level="datetime").apply(
            lambda g: g["p"].corr(g["y"], method="spearman"))
        ics[name] = float(ric.mean())
    w = pd.Series(ics).clip(lower=0.0)
    return w / w.sum() if w.sum() > 0 else pd.Series(
        1.0 / max(len(preds), 1), index=list(preds))


def ewic_rank_average(preds: dict[str, pd.Series], label: pd.Series,
                      fold_windows: list[list]) -> pd.Series:
    """Fold-causal EW-IC weighted rank average.

    For eval days inside fold i, model weights come from mean daily rank
    IC over folds 0..i-1 only (equal weights on the first fold). The
    holdout-side helper is `ewic_weights` on the full OOF span."""
    R = pd.DataFrame({k: cs_rank(v) for k, v in preds.items()})
    days = R.index.get_level_values("datetime")
    out = []
    for i, (a, b) in enumerate(fold_windows):
        m = (days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))
        if not m.any():
            continue
        w = ewic_weights(preds, label, fold_windows[:i])
        block = R[m].reindex(columns=w.index).fillna(0.5)
        out.append(block.mul(w, axis=1).sum(axis=1))
    return (pd.concat(out).sort_index().rename("score") if out
            else pd.Series(dtype=float, name="score"))


def fit_ridge(preds: dict[str, pd.Series], label: pd.Series, alpha: float):
    from sklearn.linear_model import Ridge
    X = zscore_frame(preds).fillna(0.0)
    y = label.reindex(X.index)
    keep = y.notna()
    meta = Ridge(alpha=alpha)
    meta.fit(X[keep], y[keep])
    meta._signal_order = list(preds)  # column order must survive to apply time
    return meta


def apply_ridge(meta, preds: dict[str, pd.Series]) -> pd.Series:
    X = zscore_frame(preds).reindex(columns=meta._signal_order).fillna(0.0)
    return pd.Series(meta.predict(X), index=X.index, name="score")


# --------------------------------------------------------------------------
# artifact IO


def results_dir() -> Path:
    """Artifact root — overridable for smoke runs so real results stay clean."""
    import os
    return Path(os.environ.get("PHASE2_RESULTS_DIR", REPO / "results"))


def oof_dir(market: str) -> Path:
    d = results_dir() / "oof" / market
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fold_pred(market: str, model: str, fold: str,
                   pred: pd.Series, label: pd.Series) -> Path:
    p = oof_dir(market) / f"{model}__{fold}.pkl"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"pred": pred, "label": label}, f)
    tmp.replace(p)  # atomic — a mid-write checkpoint push can't grab a partial
    return p


def load_fold_pred(market: str, model: str, fold: str):
    p = oof_dir(market) / f"{model}__{fold}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["pred"], d["label"]


def append_holdout_log(row: dict) -> None:
    p = results_dir() / "holdout_log.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def qlib_init(provider_uri: str, region: str) -> None:
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(REPO / "mlruns")
    qlib.init(provider_uri=provider_uri, region=region, exp_manager=exp_manager)


def log(msg: str) -> None:
    print(f"[phase2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)
