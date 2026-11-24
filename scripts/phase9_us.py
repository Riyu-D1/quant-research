#!/usr/bin/env python3
"""Phase-9 US — research + improve B50_momvq with an adaptive model.

    uv run python scripts/phase9_us.py walkforward     # the predeclared grid
    uv run python scripts/phase9_us.py retrain         # weekly live retrain (shadow)
    uv run python scripts/phase9_us.py report          # markdown table from the json

Heavy — GitHub Actions only (.github/workflows/phase9.yml), never the Mac.

walkforward  scores every predeclared candidate in configs/experiments/
             phase9_us.yaml (committed BEFORE the first run) on each gate
             year 2017->dump end on the honest Alpaca PIT feed, with the
             phase-8 sim (next-open fills, bucketed half-spread + 5bps, $1
             floor, 2% band, 20% cap, delist liquidation). Adaptive parts
             (monthly IC weights, monthly purged LightGBM refits, vol/trend
             overlays, drift fallback) are re-estimated inside the walk-
             forward from data <= t only. Writes results/phase9_us_
             walkforward.json and appends results/holdout_log.jsonl. It
             NEVER writes a live/paper config — adoption is a reviewed edit.

retrain      the production half of the adaptive model: recomputes the
             same composites through the latest feed day, the drift monitor
             and the fallback mode, and writes state/phase9/adaptive.json +
             appends state/phase9/drift_log.jsonl. Shadow-only: no config
             reads the file until a candidate is adopted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import live_holdout as LH  # noqa: E402
import phase8_us as P8  # noqa: E402
from quant_research import coresat as CS  # noqa: E402
from quant_research import events as ev  # noqa: E402
from quant_research import fundamentals as F  # noqa: E402
from quant_research import micro as mc  # noqa: E402
from quant_research import signals6 as s6  # noqa: E402
from quant_research import walkforward as wf  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase9_us.yaml"
STATE_DIR = REPO / "state" / "phase9"
MODES = ("adaptive", "fallback_b50", "fallback_spy")


def log(msg: str) -> None:
    print(f"[phase9 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          check=False, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


# ---------------------------------------------------------------------------
# small causal building blocks (unit-tested in tests/test_phase9.py)


def row_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def rank_ic(score: pd.DataFrame, fwd: pd.DataFrame,
            min_n: int = 20) -> pd.Series:
    """Per-day Spearman rank-IC on the common support of score and fwd —
    vectorised P8.daily_ic (same pairs, same > min_n rule)."""
    fwd = fwd.reindex(index=score.index, columns=score.columns)
    m = score.notna() & fwd.notna()
    a = score.where(m).rank(axis=1)
    b = fwd.where(m).rank(axis=1)
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(m.sum(axis=1) > min_n)


def usable(ic: pd.Series, days: pd.DatetimeIndex, h: int) -> pd.Series:
    """IC of signal day d is realised at close d+h+1 (next-open forward);
    shift on the FULL trading grid so the value at t is fully known at t."""
    return ic.reindex(days).shift(h + 1)


def ew_mean(vals: np.ndarray, hl: float) -> tuple[float, int]:
    v = vals[np.isfinite(vals)]
    if not len(v):
        return float("nan"), 0
    w = 0.5 ** (np.arange(len(v))[::-1] / max(hl, 1e-9))
    return float((v * w).sum() / w.sum()), int(len(v))


def ew_trailing(s: pd.Series, hl: float, win: int, min_obs: int) -> pd.Series:
    """EW mean over the trailing `win` rows (NaN-skipping); NaN if fewer
    than min_obs finite values."""
    vals = s.values.astype(float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        m, n = ew_mean(vals[max(0, i - win + 1):i + 1], hl)
        if n >= min_obs:
            out[i] = m
    return pd.Series(out, index=s.index)


def month_starts(days: pd.DatetimeIndex) -> list[int]:
    ym = days.year * 100 + days.month
    return [0] + [i for i in range(1, len(days)) if ym[i] != ym[i - 1]]


def weighted_blend(ranks: dict[str, pd.DataFrame], W: pd.DataFrame,
                   required: list[str]) -> pd.DataFrame:
    """sum_leg W[t,leg] * rank_leg[t] / sum W[t]; a missing non-required
    leg sits at the neutral 0.5; names missing a required leg drop out.
    Equal W over [mom, vq] == P7.blend(panels, [mom, vq]) exactly."""
    legs = [c for c in W.columns if W[c].abs().sum() > 0]
    idx = next(iter(ranks.values())).index
    cols = next(iter(ranks.values())).columns
    acc = pd.DataFrame(0.0, index=idx, columns=cols)
    for leg in legs:
        acc += ranks[leg].reindex(index=idx, columns=cols).fillna(0.5) \
            .mul(W[leg].reindex(idx).fillna(0.0), axis=0)
    tot = W[legs].reindex(idx).fillna(0.0).sum(axis=1)
    comp = acc.div(tot.where(tot > 0), axis=0)
    req = None
    for leg in required:
        ok = ranks[leg].reindex(index=idx, columns=cols).notna()
        req = ok if req is None else (req & ok)
    return comp.where(req) if req is not None else comp


def ic_weight_table(ic_use: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list]:
    """Monthly online factor weights from USABLE (already-lagged) ICs.
    Month m's weights read rows <= m's first day only."""
    days = ic_use.index
    legs = list(ic_use.columns)
    prior = pd.Series({leg: float(cfg["prior"].get(leg, 0.0)) for leg in legs})
    prior = prior / prior.sum()
    W = pd.DataFrame(np.nan, index=days, columns=legs)
    ledger = []
    for p in month_starts(days):
        tail = ic_use.iloc[max(0, p - cfg["ic_days"] + 1):p + 1]
        stats = {leg: ew_mean(tail[leg].values, cfg["halflife_days"])
                 for leg in legs}
        nobs = min(n for _, n in stats.values())
        raw = pd.Series({leg: max(0.0, m) if np.isfinite(m) else 0.0
                         for leg, (m, _) in stats.items()})
        if nobs < cfg["min_obs"] or raw.sum() <= 0:
            w, src = prior, "prior"
        else:
            w = (1 - cfg["lam"]) * prior + cfg["lam"] * raw / raw.sum()
            src = "ic"
        W.iloc[p] = w.values
        ledger.append({"date": str(days[p].date()), "src": src,
                       "nobs": int(nobs),
                       "ic": {k: (round(m, 5) if np.isfinite(m) else None)
                              for k, (m, _) in stats.items()},
                       "w": {k: round(float(v), 4) for k, v in w.items()}})
    return W.ffill(), ledger


# ---------------------------------------------------------------------------
# LightGBM, monthly refit, expanding window, purged + embargoed


def _lgbm(params: dict, n_jobs: int):
    import lightgbm as lgb
    p = dict(params)
    seed = p.pop("seed", 0)
    return lgb.LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1,
                             **p)


def _rows(X, Y, valid, pos: np.ndarray):
    sel = valid[pos] & np.isfinite(Y[pos])
    xs = X[pos][sel]
    ys = Y[pos][sel]
    day_id = np.repeat(pos, valid.shape[1]).reshape(len(pos), -1)[sel]
    return xs, ys, day_id


def _mean_daily_ic(pred, y, day_id) -> float:
    df = pd.DataFrame({"p": pred, "y": y, "d": day_id})
    ics = df.groupby("d")[["p", "y"]].apply(
        lambda g: g["p"].corr(g["y"], method="spearman")
        if len(g) > 20 else np.nan)
    return float(ics.mean()) if ics.notna().any() else -np.inf


def purged_cv(X, Y, valid, train_pos: np.ndarray, cfg: dict, h: int,
              n_jobs: int) -> tuple[dict, list]:
    """Blocked k-fold over the training window with the label horizon
    purged on both sides of each test block and an embargo after it.
    Returns the grid point with the best mean daily rank-IC."""
    k = cfg["cv_folds"]
    gap = h + 1
    blocks = np.array_split(train_pos, k)
    scores = []
    for g in cfg["grid"]:
        params = {**cfg["base"], **g}
        fold_ic = []
        for b in blocks:
            if not len(b):
                continue
            lo, hi = b[0], b[-1]
            tr = train_pos[(train_pos < lo - gap)
                           | (train_pos > hi + gap + cfg["embargo"])]
            if len(tr) < 50:
                continue
            xs, ys, _ = _rows(X, Y, valid, tr)
            xt, yt, dt_ = _rows(X, Y, valid, b)
            if len(xs) < 1000 or len(xt) < 100:
                continue
            m = _lgbm(params, n_jobs).fit(xs, ys)
            fold_ic.append(_mean_daily_ic(m.predict(xt), yt, dt_))
        scores.append({"params": g, "cv_ic": float(np.mean(fold_ic))
                       if fold_ic else None})
    ok = [s for s in scores if s["cv_ic"] is not None]
    best = max(ok, key=lambda s: s["cv_ic"])["params"] if ok else {}
    return {**cfg["base"], **best}, scores


def gbm_composite(feats: dict[str, pd.DataFrame], label: pd.DataFrame,
                  valid: pd.DataFrame, cfg: dict, h: int,
                  since: pd.Timestamp | None = None,
                  n_jobs: int | None = None) -> tuple[pd.DataFrame, list]:
    """Month-m predictions from a model trained ONLY on rows whose label
    closed before m (d <= pos(m) - (h+1) - embargo). NaN rows = no model
    yet (caller substitutes the B50 composite)."""
    days, cols = valid.index, valid.columns
    names = cfg["features"]
    X = np.stack([feats[f].reindex(index=days, columns=cols).values
                  for f in names], axis=-1).astype(np.float32)
    Y = label.reindex(index=days, columns=cols).values.astype(np.float32)
    V = valid.reindex(index=days, columns=cols).fillna(False).values
    out = np.full(V.shape, np.nan, dtype=np.float32)
    n_jobs = n_jobs or os.cpu_count() or 2
    labelled = V & np.isfinite(Y)
    has_lab = labelled.any(axis=1)
    starts = month_starts(days)
    params = dict(cfg["base"])
    ledger = []
    for i, p in enumerate(starts):
        q = starts[i + 1] if i + 1 < len(starts) else len(days)
        if since is not None and days[q - 1] < since:
            continue
        cut = p - (h + 1) - cfg["embargo"]
        if cut < 0:
            continue
        lab_days = np.flatnonzero(has_lab[:cut + 1])
        if len(lab_days) < cfg["min_train_days"]:
            continue
        cv = None
        if days[p].month == 1 and len(lab_days) >= cfg["cv_min_days"]:
            cvpos = lab_days[lab_days % cfg["cv_stride"] == 0]
            params, cv = purged_cv(X, Y, labelled, cvpos, cfg, h, n_jobs)
        tpos = lab_days[lab_days % cfg["stride"] == 0]
        xs, ys, _ = _rows(X, Y, labelled, tpos)
        model = _lgbm(params, n_jobs).fit(xs, ys)
        blk = X[p:q].reshape(-1, X.shape[-1])
        vm = V[p:q].reshape(-1)
        pred = np.full(len(vm), np.nan, dtype=np.float32)
        if vm.any():
            pred[vm] = model.predict(blk[vm])
        out[p:q] = pred.reshape(q - p, -1)
        ledger.append({"month": str(days[p].date()),
                       "train_last_day": str(days[cut].date()),
                       "n_train_days": int(len(tpos)),
                       "n_rows": int(len(ys)),
                       "params": {k: params[k] for k in
                                  ("num_leaves", "n_estimators")},
                       "cv": cv,
                       "importance": dict(zip(
                           names, [int(v) for v in
                                   model.feature_importances_]))})
    return pd.DataFrame(out, index=days, columns=cols), ledger


# ---------------------------------------------------------------------------
# shadow sleeve, overlays, drift machine


def shadow_sleeve(comp: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.Series:
    """Cost-free equal-weight return on day t of the top-n names by the
    composite at close t-1 (priced at t-1). Monitoring proxy only — the
    book itself is always the full micro sim."""
    cols = comp.columns
    C = comp.reindex(index=close.index, columns=cols).values
    P = close.reindex(columns=cols).values
    with np.errstate(divide="ignore", invalid="ignore"):
        R = P[1:] / P[:-1] - 1.0
    out = np.full(len(close.index), np.nan)
    for t in range(1, len(close.index)):
        row = np.where(np.isfinite(P[t - 1]), C[t - 1], np.nan)
        ok = np.flatnonzero(np.isfinite(row))
        if not len(ok):
            continue
        top = ok[np.argsort(-row[ok], kind="stable")[:n]]
        r = R[t - 1][top]
        r = r[np.isfinite(r)]
        out[t] = float(r.mean()) if len(r) else 0.0
    return pd.Series(out, index=close.index)


def vol_exposure(sleeve_r: pd.Series, cfg: dict) -> pd.Series:
    sig = sleeve_r.rolling(cfg["win"], min_periods=cfg["win"]).std() \
        * math.sqrt(252)
    tgt = sig.rolling(cfg["target_win"],
                      min_periods=cfg["target_min_obs"]).median()
    e = (tgt / sig).clip(lower=cfg["floor"], upper=1.0)
    return e.where(tgt.notna() & sig.notna() & (sig > 0), 1.0)


def trend_exposure(spy: pd.Series, days: pd.DatetimeIndex,
                   cfg: dict) -> pd.Series:
    s = spy.sort_index()
    ma = s.rolling(cfg["sma"], min_periods=cfg["sma"] // 2).mean()
    risk = (s.shift(1) < ma.shift(1)).reindex(days).fillna(False)
    return pd.Series(np.where(risk, cfg["risk_off_mult"], 1.0), index=days)


def drift_modes(r_ad: pd.Series, r_b: pd.Series, r_spy: pd.Series,
                ic_ad_use: pd.Series, cfg: dict) -> pd.DataFrame:
    """Fallback state machine (spec.drift). Every input at t is known at
    close t; the mode on row t governs day t's signal (next-open fill)."""
    idx = r_ad.index
    act = (r_ad - r_b).rolling(cfg["act_win"],
                               min_periods=cfg["act_win"]).sum()
    ic = ew_trailing(ic_ad_use.reindex(idx), cfg["ic_hl"], cfg["ic_win"],
                     cfg["ic_min_obs"])
    bxs = (r_b - r_spy.reindex(idx)).rolling(
        cfg["bxs_win"], min_periods=cfg["bxs_win"]).sum()
    mode, prev, held = "adaptive", "adaptive", cfg["min_hold"]
    out = []
    for t in range(len(idx)):
        a, i, b = act.iloc[t], ic.iloc[t], bxs.iloc[t]
        held += 1
        if held >= cfg["min_hold"]:
            bad = (np.isfinite(a) and a < -cfg["act_tol"]) or \
                (np.isfinite(i) and i < 0)
            good = (not np.isfinite(a) or a >= 0) and \
                (not np.isfinite(i) or i > 0)
            if mode != "fallback_spy" and np.isfinite(b) \
                    and b < -cfg["spy_tol"]:
                prev, mode, held = mode, "fallback_spy", 0
            elif mode == "fallback_spy":
                if np.isfinite(b) and b > -cfg["spy_recover"]:
                    mode, held = prev, 0
            elif mode == "adaptive" and bad:
                mode, held = "fallback_b50", 0
            elif mode == "fallback_b50" and good:
                mode, held = "adaptive", 0
        out.append(mode)
    return pd.DataFrame({"mode": out, "act": act.values, "ic": ic.values,
                         "bxs": bxs.values}, index=idx)


# ---------------------------------------------------------------------------
# data + panels


def load_vq_parts(spec: dict, feed: dict) -> dict[str, pd.DataFrame]:
    """Raw vq_bm / vq_ep / vq_roe wide panels (same PIT math as signals6's
    composite) — GBM features only."""
    cal = feed["days"]
    close = feed["close"].reindex(cal)
    f = pd.read_parquet(REPO / spec["fundamentals"])
    f["symbol"] = f["symbol"].map(ev.canon)
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c])
    sh = pd.read_parquet(REPO / spec["shares"])
    sh["filed"] = pd.to_datetime(sh["filed"])
    sh["symbol"] = sh["symbol"].map(ev.canon)
    shr_by = {s: g.rename(columns={"val": "shares"})
              for s, g in sh.groupby("symbol")}
    parts = {c: {} for c in ("vq_bm", "vq_ep", "vq_roe")}
    for s, fs in f.groupby("symbol"):
        if s not in close.columns:
            continue
        vqf = F.vq_features(fs, shr_by.get(s), close[s].values, cal)
        for c in parts:
            parts[c][s] = vqf[c]
    return {c: pd.DataFrame(v, index=cal).replace([np.inf, -np.inf], np.nan)
            for c, v in parts.items()}


def load_news_panel(spec: dict, feed: dict) -> pd.DataFrame | None:
    from quant_research import news as NW
    p = REPO / spec["news_daily"]
    if not p.exists():
        return None
    daily = pd.read_parquet(p)
    days, cols = feed["close"].index, feed["close"].columns
    store = pd.DataFrame(index=pd.MultiIndex.from_product(
        [days, cols], names=["datetime", "instrument"]))
    s = NW.daily_signal(daily, store)
    return s.unstack("instrument").reindex(index=days, columns=cols)


def build_inputs(spec: dict, need_gbm: bool, need_news: bool) -> dict:
    provider = REPO / spec["provider"]
    log("loading honest feed")
    feed = P8.load_feed(provider)
    smap = P8.load_sector_map(spec)
    close = feed["close"]
    days = close.index
    log("panels: mom/vq/lowvol/beta via the phase-8 builders")
    p8spec = {"signals": {"x": ["mom", "vq", "lowvol", "beta"]},
              "fundamentals": spec["fundamentals"]}
    panels = P8.build_panels(p8spec, feed, provider, smap)
    volu = feed["bars"]["volume"]["volume"].unstack("instrument") \
        .reindex(index=days, columns=close.columns)
    ps = s6.price_signals(close, volu)
    for k in ("mom6", "rev", "srev5"):
        panels[k] = ps[k].reindex(days)
    if need_gbm:
        log("vq parts for the GBM features")
        panels.update({k: v.reindex(days) for k, v in
                       load_vq_parts(spec, feed).items()})
    if need_news:
        news = load_news_panel(spec, feed)
        if news is not None:
            panels["news"] = news
            log(f"news panel: {int(news.notna().sum().sum())} cells")
    ranks = {k: row_rank(v.reindex(columns=close.columns))
             for k, v in panels.items()}
    fwd = P8.fwd_wide(close, 1, spec["fwd_h"])
    cmap = mc.liquidity_cost_map(feed["bars"]["close"],
                                 feed["bars"]["volume"],
                                 flat_bps=spec["costs"]["flat_bps"])
    return {"feed": feed, "smap": smap, "panels": panels, "ranks": ranks,
            "fwd": fwd, "cmap": cmap, "days": days}


def composites(spec: dict, inp: dict, kinds: set[str],
               since: pd.Timestamp | None = None) -> tuple[dict, dict]:
    """Every composite the candidates need, computed once over all days."""
    ranks, fwd, days = inp["ranks"], inp["fwd"], inp["days"]
    h = spec["fwd_h"]
    req = spec["required_legs"]
    out, ledgers = {}, {}
    eq = pd.DataFrame({"mom": 0.5, "vq": 0.5}, index=days)
    out["b50"] = weighted_blend(ranks, eq, req)
    if kinds & {"ic", "ens", "ens_news"}:
        legs = spec["signals"]["library"]
        log(f"IC weights over {legs}")
        ic_use = pd.DataFrame({leg: usable(rank_ic(inp["panels"][leg], fwd),
                                           days, h) for leg in legs})
        W, ledgers["ic_weights"] = ic_weight_table(ic_use, spec["ic_weights"])
        out["ic"] = weighted_blend(ranks, W, req)
    if kinds & {"gbm", "ens", "ens_news"}:
        cfg = spec["gbm"]
        priced = inp["feed"]["close"].notna()
        valid = out["b50"].notna() & priced
        label = row_rank(fwd.where(valid))
        log("LightGBM monthly walk-forward refits")
        g, ledgers["gbm"] = gbm_composite(ranks, label, valid, cfg, h,
                                          since=since)
        no_model = g.isna().all(axis=1)
        g = row_rank(g)
        g.loc[no_model] = out["b50"].loc[no_model]
        out["gbm"] = g.where(out["b50"].notna())
        ledgers["gbm_fallback_days"] = int(no_model.sum())
    if kinds & {"ens", "ens_news"}:
        out["ens"] = ((row_rank(out["ic"]) + row_rank(out["gbm"])) / 2) \
            .where(out["b50"].notna())
    if "news" in inp["panels"] and kinds & {"news", "ens_news"}:
        eqn = pd.DataFrame({"mom": 1 / 3, "vq": 1 / 3, "news": 1 / 3},
                           index=days)
        out["news"] = weighted_blend(ranks, eqn, req)
        out["ens_news"] = ((row_rank(out["ic"]) + row_rank(out["gbm"])
                            + ranks["news"].fillna(0.5)) / 3) \
            .where(out["b50"].notna())
    return out, ledgers


def candidate_plan(spec: dict, inp: dict, comps: dict, cand: dict) -> dict:
    """-> {comp, exposure{day: x}, modes} for one candidate (all days)."""
    feed, days = inp["feed"], inp["days"]
    if not cand.get("comp"):
        return {"comp": pd.DataFrame(), "exposure": {}, "modes": None}
    comp = comps[cand["comp"]]
    n = int(cand["n"])
    exp = pd.Series(1.0, index=days)
    modes = None
    ov = cand.get("overlay") or []
    if cand.get("drift"):
        r_ad = shadow_sleeve(comp, feed["close"], n)
        r_b = shadow_sleeve(comps["b50"], feed["close"], n)
        ic_ad = usable(rank_ic(comp, inp["fwd"]), days, spec["fwd_h"])
        modes = drift_modes(r_ad, r_b, feed["spy"].pct_change(), ic_ad,
                            spec["drift"])
        use_b = (modes["mode"] != "adaptive").values
        comp = comp.copy()
        comp.loc[use_b] = comps["b50"].loc[use_b]
        exp = exp.where(modes["mode"] != "fallback_spy", 0.0)
    if "vol" in ov:
        exp = exp * vol_exposure(shadow_sleeve(comp, feed["close"], n),
                                 spec["overlays"]["vol"])
    if "trend" in ov:
        exp = exp * trend_exposure(feed["spy"], days, spec["overlays"]["trend"])
    return {"comp": comp, "exposure": exp.to_dict(), "modes": modes}


# ---------------------------------------------------------------------------
# scoring


def spec8(spec: dict, book: str) -> dict:
    """phase-8 sim spec for P8.daily_targets / P8.run_candidate."""
    return {"account": spec["books"][book], "fill": spec["fill"],
            "delist_liquidate": spec["delist_liquidate"],
            "costs": spec["costs"], "rebalance_band": spec["rebalance_band"],
            "min_order_usd": spec["min_order_usd"],
            "core_symbol": spec["core_symbol"],
            "max_stock_weight": spec["max_stock_weight"]}


def cand8(cand: dict) -> dict:
    # guard 'to_core' only sets P8's risk_off routing: freed sleeve -> SPY
    return {"name": cand["name"], "core": cand["core"],
            "sleeve_signal": cand.get("comp"), "guard": "to_core",
            "sleeve": {"n": int(cand.get("n", 0)),
                       "weighting": cand.get("weighting", "equal")}}


def concentration(targets: dict, close: pd.DataFrame, core: str) -> dict:
    """Gross (pre-cost) sleeve P&L by name from target weights: share of
    the year's sleeve contribution carried by the top-3 names."""
    ds = sorted(targets)
    rets = close.reindex(ds).pct_change()
    contrib: dict[str, float] = {}
    for a, b in zip(ds[:-1], ds[1:]):
        for s, w in targets[a].items():
            if s == core:
                continue
            r = rets.at[b, s] if s in rets.columns else np.nan
            if np.isfinite(r):
                contrib[s] = contrib.get(s, 0.0) + w * r
    tot = sum(contrib.values())
    top = sorted(contrib.items(), key=lambda kv: -kv[1])[:3]
    return {"sleeve_contrib": float(tot),
            "top3": [[s, round(v, 4)] for s, v in top],
            "top3_share": float(sum(v for _, v in top) / tot)
            if tot > 0 else None,
            "n_names": len(contrib)}


def score_candidate(spec: dict, inp: dict, cand: dict, plan: dict,
                    years: list[int]) -> dict:
    feed = inp["feed"]
    book = cand.get("book", "paper")
    s8 = spec8(spec, book)
    c8 = cand8(cand)
    by_year, dailies = {}, []
    for y in years:
        yd = feed["days"][feed["days"].year == y]
        if len(yd) < 20:
            continue
        targets = P8.daily_targets(plan["comp"], feed["close"], yd, c8, s8,
                                   plan["exposure"], inp["smap"])
        met = P8.run_candidate(feed, targets, yd, s8, inp["cmap"], None)
        d = met["daily_excess"].dropna()
        row = P8.wf_metrics(d)
        row.update({"micro_ar": met["annualized_return"],
                    "turnover": met["turnover"],
                    "n_trades": len(met["trades"]),
                    "concentration": concentration(
                        targets, feed["close"], spec["core_symbol"])
                    if cand.get("comp") else None})
        if plan["modes"] is not None:
            vc = plan["modes"]["mode"].reindex(yd).value_counts(
                normalize=True)
            row["mode_frac"] = {m: float(vc.get(m, 0.0)) for m in MODES}
        by_year[y] = row
        dailies.append(d)
        log(f"  {cand['name']:>15} {y}: ar={row['annualized_return']:+.3f}"
            f" ir={row['information_ratio']:+.3f}"
            f" mdd={row['max_drawdown']:+.3f}")
    daily = pd.concat(dailies).sort_index() if dailies \
        else pd.Series(dtype=float)
    return {"name": cand["name"], "core": cand["core"],
            "sleeve_signal": cand.get("comp"), "book": book,
            "by_year": by_year, "wf": P8.wf_metrics(daily), "_daily": daily}


def sr_ann(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std() * math.sqrt(252)) \
        if len(x) > 1 and x.std() > 0 else 0.0


def block_boot_ci(x: pd.Series, block: int = 21, n_boot: int = 2000,
                  seed: int = 0) -> list[float]:
    v = x.dropna().values
    if len(v) < 2 * block:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(v) / block))
    st = rng.integers(0, len(v) - block + 1, size=(n_boot, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]) \
        .reshape(n_boot, -1)[:, :len(v)]
    m = v[idx].mean(axis=1) * 252
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def evaluate(results: dict, spy: pd.Series, base: str, n_trials: int,
             years: list[int], spec: dict, spy_by: dict,
             spy_wf: dict) -> dict:
    """DSR (total / xs-SPY / active-vs-baseline), phase-8 gate, adoption."""
    spy_r = spy.pct_change()
    b = results[base]["_daily"]
    xs = {k: (r["_daily"] - spy_r.reindex(r["_daily"].index)).dropna()
          for k, r in results.items()}
    act = {k: (r["_daily"] - b.reindex(r["_daily"].index)).dropna()
           for k, r in results.items() if k != base}
    # trial-SR dispersion: sleeve candidates only — the 100% SPY row has a
    # ~zero-variance excess series whose SR is numerically meaningless
    strat = {k for k, r in results.items() if r.get("sleeve_signal")}
    tri_tot = [sr_ann(r["_daily"]) for r in results.values()]
    tri_xs = [sr_ann(v) for k, v in xs.items() if k in strat]
    tri_act = [sr_ann(v) for k, v in act.items() if len(v) and k in strat]
    gate = P8.deploy_gate(results, spy_by, spy_wf, years, spec["deploy_gate"])
    out = {}
    for k, r in results.items():
        e = {"dsr_total": wf.deflated_sharpe(r["_daily"], n_trials, tri_tot),
             "xs": {"ar": float(xs[k].mean() * 252), "ir": sr_ann(xs[k])},
             "dsr_xs": wf.deflated_sharpe(xs[k], n_trials, tri_xs)
             if len(xs[k]) > 2 and xs[k].std() > 0 else None,
             "gate": gate["decision"].get(k)}
        if k in act and len(act[k]) > 2 and act[k].std() > 0:
            a = act[k]
            yrs = [y for y in years if y in r["by_year"]
                   and y in results[base]["by_year"]]
            beat = [y for y in yrs if r["by_year"][y]["annualized_return"]
                    > results[base]["by_year"][y]["annualized_return"]]
            e["active"] = {"ar": float(a.mean() * 252), "ir": sr_ann(a),
                           "boot95": block_boot_ci(a),
                           "years_beat": beat,
                           "year_frac": len(beat) / max(len(yrs), 1)}
            e["dsr_active"] = wf.deflated_sharpe(a, n_trials, tri_act)
        out[k] = e
    return {"per": out, "gate": gate}


def adoption(ev_: dict, results: dict, spec: dict) -> dict:
    rule = spec["adoption"]
    base = rule["baseline"]
    bxs = (ev_["per"][base].get("dsr_xs") or {}).get("DSR", 0.0)
    checks = {}
    for k, e in ev_["per"].items():
        if k == base or not results[k].get("sleeve_signal") \
                or results[k].get("book") != "paper":
            continue
        a = e.get("active") or {}
        da = (e.get("dsr_active") or {}).get("DSR", 0.0)
        dx = (e.get("dsr_xs") or {}).get("DSR", 0.0)
        c = {"gate": bool((e.get("gate") or {}).get("passed")),
             "active_pos": a.get("ar", -1) > 0,
             "dsr_active_ok": da >= rule["min_dsr"],
             "years_ok": a.get("year_frac", 0) >= rule["min_year_frac"],
             "xs_dsr_ok": dx >= bxs,
             "dsr_active": da, "dsr_xs": dx}
        c["adopt"] = all(c[x] for x in ("gate", "active_pos",
                                        "dsr_active_ok", "years_ok",
                                        "xs_dsr_ok"))
        checks[k] = c
    winners = [k for k, c in checks.items() if c["adopt"]]
    win = max(winners, key=lambda k: checks[k]["dsr_active"]) \
        if winners else None
    return {"baseline": base, "baseline_dsr_xs": bxs, "checks": checks,
            "adopted": win, "keep_baseline": win is None}


# ---------------------------------------------------------------------------
# commands


LOG_PATH = REPO / "results" / "holdout_log.jsonl"


def _log_rows() -> list[str]:
    return LOG_PATH.read_text().splitlines() if LOG_PATH.exists() else []


def _is_p9(line: str) -> bool:
    try:
        return json.loads(line).get("spec") == "phase9_us.yaml"
    except json.JSONDecodeError:
        return False


def prior_trials_ex_phase9() -> int:
    """P7.prior_trials ignoring phase-9's own rows, so a re-run of the SAME
    predeclared grid neither double-counts itself nor shrinks the count."""
    best = 0
    for line in _log_rows():
        if _is_p9(line):
            continue
        try:
            best = max(best, int(json.loads(line).get("n_trials") or 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return best


def drop_phase9_log_rows() -> None:
    keep = [ln for ln in _log_rows() if not _is_p9(ln)]
    LOG_PATH.write_text("\n".join(keep) + ("\n" if keep else ""))


def cmd_walkforward(spec: dict, out: Path) -> dict:
    t0 = time.time()
    cands = spec["candidates"]
    ncands = spec.get("news_candidates") or []
    kinds = {c["comp"] for c in cands + ncands if c.get("comp")}
    news_path = REPO / spec["news_daily"]
    inp = build_inputs(spec, need_gbm=bool(kinds & {"gbm", "ens",
                                                    "ens_news"}),
                       need_news=news_path.exists())
    comps, ledgers = composites(spec, inp, kinds)
    feed = inp["feed"]
    years = P8.gate_years(feed["days"], spec["gate_start_year"])
    log(f"gate years {years}; {len(cands)} candidates")
    results = {}
    for cand in cands:
        plan = candidate_plan(spec, inp, comps, cand)
        results[cand["name"]] = score_candidate(spec, inp, cand, plan, years)
        if plan["modes"] is not None:
            results[cand["name"]]["mode_switches"] = int(
                (plan["modes"]["mode"] != plan["modes"]["mode"].shift())
                .sum() - 1)
    spy_by = P8.spy_by_year(feed, years)
    spy_wf = LH.spy_metrics(feed["spy"],
                            feed["days"][feed["days"].year >= years[0]])

    news_block = {"scored": False,
                  "reason": f"{spec['news_daily']} absent — FinBERT scores "
                            "not produced (phase9.yml news jobs or Kaggle "
                            "MODELS=\"phase4_news_score\")"}
    nres = {}
    if "news" in comps:
        nlast = inp["panels"]["news"].dropna(how="all").index.max()
        nyears = [y for y in years
                  if spec["news_start_year"] <= y <= nlast.year]
        for cand in ncands:
            plan = candidate_plan(spec, inp, comps, cand)
            nres[cand["name"]] = score_candidate(spec, inp, cand, plan,
                                                 nyears)
        base = spec["adoption"]["baseline"]
        b = results[base]
        nres[base] = {**b, "by_year": {y: v for y, v in b["by_year"].items()
                                       if y in nyears},
                      "_daily": b["_daily"][b["_daily"].index.year
                                            .isin(nyears)]}
        nres[base]["wf"] = P8.wf_metrics(nres[base]["_daily"])
        news_block = {"scored": True, "years": nyears,
                      "last_news_day": str(nlast.date())}

    n_windows = sum(len(r["by_year"]) for r in results.values()) + \
        sum(len(r["by_year"]) for k, r in nres.items()
            if k != spec["adoption"]["baseline"])
    prior = prior_trials_ex_phase9()
    n_trials = prior + n_windows
    log(f"n_trials = {prior} prior + {n_windows} phase-9 windows "
        f"= {n_trials}")
    ev_ = evaluate(results, feed["spy"], spec["adoption"]["baseline"],
                   n_trials, years, spec, spy_by, spy_wf)
    adopt = adoption(ev_, results, spec)
    if nres:
        nspy_by = {y: spy_by[y] for y in news_block["years"]}
        nspy_wf = LH.spy_metrics(
            feed["spy"], feed["days"][feed["days"].year
                                      .isin(news_block["years"])])
        news_block["eval"] = evaluate(nres, feed["spy"],
                                      spec["adoption"]["baseline"],
                                      n_trials, news_block["years"], spec,
                                      nspy_by, nspy_wf)
        news_block["adoption"] = adoption(news_block["eval"], nres, spec)

    # reproduction anchor: B50_micro must equal the phase-8 stored years
    stored = json.loads((REPO / "results/phase8_us_walkforward.json")
                        .read_text())["results"]["B50_momvq"]["by_year"]
    repro = {str(y): {"phase9": r["micro_ar"],
                      "phase8": stored[str(y)]["annualized_return"],
                      "match": abs(r["micro_ar"] - stored[str(y)]
                                   ["annualized_return"]) < 5e-3}
             for y, r in results["B50_micro"]["by_year"].items()
             if str(y) in stored}

    ts = dt.datetime.now().isoformat()
    drop_phase9_log_rows()
    for name, r in {**results, **{k: v for k, v in nres.items()
                                  if k != spec["adoption"]["baseline"]}}.items():
        e = ev_["per"].get(name) or news_block.get("eval", {}) \
            .get("per", {}).get(name, {})
        wf.append_holdout_log({
            "ts": ts, "market": "us", "git_sha": git_sha(),
            "spec": "phase9_us.yaml", "candidate": f"phase9:{name}",
            "re_viewed_holdout": True, "IC": None, "ICIR": None,
            "RankIC": None, "RankICIR": None, **r["wf"],
            "dsr": e.get("dsr_total"), "dsr_xs": e.get("dsr_xs"),
            "dsr_active": e.get("dsr_active"), "n_trials": n_trials})

    def clean(rs):
        return {k: {kk: vv for kk, vv in v.items() if kk != "_daily"}
                for k, v in rs.items()}
    summary = {
        "stage": "phase9_us_walkforward", "git_sha": git_sha(), "ts": ts,
        "spec": str(SPEC_PATH.relative_to(REPO)), "gate_years": years,
        "feed_last_day": str(feed["days"][-1].date()),
        "spy_by_year": spy_by, "spy_wf": spy_wf,
        "results": clean(results), "eval": ev_, "adoption": adopt,
        "news": {**news_block, "results": clean(nres)} if nres
        else news_block,
        "reproduction_b50_micro": {"years": repro, "all_match": all(
            v["match"] for v in repro.values())},
        "ledgers": ledgers, "n_trials": n_trials, "prior_trials": prior,
        "n_phase9_windows": n_windows,
        "runtime_s": round(time.time() - t0, 1)}
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"wrote {out} — adopted={adopt['adopted']} "
        f"({summary['runtime_s']}s)")
    return summary


def cmd_retrain(spec: dict, out_dir: Path, months: int) -> dict:
    """Weekly production retrain (shadow): the predeclared end model
    (AD50_full) recomputed through the latest feed day."""
    cand = next(c for c in spec["candidates"] if c["name"] == "AD50_full")
    since = None
    inp = build_inputs(spec, need_gbm=True, need_news=False)
    days = inp["days"]
    if months:
        since = days[-1] - pd.DateOffset(months=months)
    comps, ledgers = composites(spec, inp, {"ens"}, since=since)
    plan = candidate_plan(spec, inp, comps, cand)
    d = days[-1]
    m = plan["modes"].loc[d]
    exp = float(plan["exposure"].get(d, 1.0))
    n = int(cand["n"])
    row = plan["comp"].loc[d].dropna()
    row = row[inp["feed"]["close"].loc[d].reindex(row.index).notna().values]
    picks = CS.pick_top(row, n)
    b_row = comps["b50"].loc[d].dropna()
    b_row = b_row[inp["feed"]["close"].loc[d].reindex(b_row.index)
                  .notna().values]
    targets = CS.build_targets(picks, cand["core"], exposure=exp,
                               risk_off="to_core",
                               core_sym=spec["core_symbol"],
                               cap=spec["max_stock_weight"])
    gl = ledgers.get("gbm") or []
    state = {
        "asof": str(d.date()), "generated": dt.datetime.now().isoformat(),
        "git_sha": git_sha(), "model": cand,
        "live_effect": "shadow — no live/paper config reads this file "
                       "(adoption rule not met; see docs/PHASE9_IMPROVE.md)",
        "mode": m["mode"],
        "drift": {"act_126": None if not np.isfinite(m["act"])
                  else float(m["act"]),
                  "ic_ew": None if not np.isfinite(m["ic"])
                  else float(m["ic"]),
                  "b50_vs_spy_252": None if not np.isfinite(m["bxs"])
                  else float(m["bxs"])},
        "sleeve_exposure": exp,
        "picks_adaptive_model": picks,
        "picks_b50": CS.pick_top(b_row, n),
        "targets": targets,
        "ic_weights_latest": (ledgers.get("ic_weights") or [None])[-1],
        "gbm_latest": gl[-1] if gl else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adaptive.json").write_text(json.dumps(state, indent=1,
                                                      default=str))
    with (out_dir / "drift_log.jsonl").open("a") as fh:
        fh.write(json.dumps({k: state[k] for k in
                             ("asof", "generated", "mode", "drift",
                              "sleeve_exposure", "picks_adaptive_model",
                              "picks_b50")}, default=str) + "\n")
    log(f"retrain asof {state['asof']}: mode={state['mode']} "
        f"exposure={exp:.2f} picks={picks}")
    return state


def _pct(x):
    return f"{x * 100:+.1f}%" if isinstance(x, (int, float)) \
        and np.isfinite(x) else "—"


def _num(x, f="{:.2f}"):
    return f.format(x) if isinstance(x, (int, float)) \
        and np.isfinite(x) else "—"


def cmd_report(path: Path) -> str:
    d = json.loads(path.read_text())
    per = d["eval"]["per"]
    lines = ["| candidate | book | AR | Sharpe | MDD | xs-SPY AR | "
             "DSR xs | active AR vs B50 | DSR active | yrs>B50 | gate | "
             "adopt |", "|" + "---|" * 12]
    ad = d["adoption"]["checks"]
    for k, r in d["results"].items():
        e = per[k]
        a = e.get("active") or {}
        lines.append(
            f"| {k} | {r['book']} | {_pct(r['wf']['annualized_return'])} | "
            f"{_num(r['wf']['information_ratio'])} | "
            f"{_pct(r['wf']['max_drawdown'])} | {_pct(e['xs']['ar'])} | "
            f"{_num((e.get('dsr_xs') or {}).get('DSR'))} | "
            f"{_pct(a.get('ar'))} | "
            f"{_num((e.get('dsr_active') or {}).get('DSR'))} | "
            f"{len(a.get('years_beat', [])) if a else '—'} | "
            f"{'pass' if (e.get('gate') or {}).get('passed') else 'fail'} | "
            f"{'YES' if ad.get(k, {}).get('adopt') else '—'} |")
    txt = "\n".join(lines)
    print(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["walkforward", "retrain", "report"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--months", type=int, default=18,
                    help="retrain: GBM refits only for the trailing N months "
                         "(drift needs ~12m of shadow history)")
    args = ap.parse_args()
    spec = yaml.safe_load(SPEC_PATH.read_text())
    if args.cmd == "walkforward":
        cmd_walkforward(spec, Path(args.out or
                                   "results/phase9_us_walkforward.json"))
    elif args.cmd == "retrain":
        cmd_retrain(spec, Path(args.out or STATE_DIR), args.months)
    else:
        cmd_report(Path(args.out or "results/phase9_us_walkforward.json"))


if __name__ == "__main__":
    main()
