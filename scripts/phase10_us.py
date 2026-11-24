#!/usr/bin/env python3
"""Phase-10 US — documented multi-asset / factor candidates vs B50_momvq.

    uv run python scripts/phase10_us.py fetch     # ETF bars from Alpaca (needs keys)
    uv run python scripts/phase10_us.py dev       # dev walk-forward, holdout sliced off
    uv run python scripts/phase10_us.py holdout   # frozen finalists, ONCE
    uv run python scripts/phase10_us.py report    # markdown tables from the jsons

Heavy — GitHub Actions only (.github/workflows/phase10.yml), never the Mac.

Spec (predeclared, committed before any run): configs/experiments/
phase10_us.yaml. One simulator for every book: the phase-8 micro sim
trading explicit weight targets on the honest Alpaca feed with the ETF
bars appended (next-open fills, liquidity-bucket half-spread + 5 bps, $1
floor, 2% band, delist liquidation). `dev` slices every input to days
before the holdout start BEFORE any signal is built. `holdout` runs the
dev-chosen finalists once and refuses a second run. Neither command
writes a paper/live config.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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

import phase8_us as P8  # noqa: E402
import phase9_us as P9  # noqa: E402
from quant_research import events as ev  # noqa: E402
from quant_research import micro as mc  # noqa: E402
from quant_research import multiasset as MA  # noqa: E402
from quant_research import walkforward as wf  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase10_us.yaml"
DEV_OUT = REPO / "results" / "phase10_dev.json"
DEV_DAILY = REPO / "results" / "phase10" / "dev_daily.csv"
HOLD_OUT = REPO / "results" / "phase10_holdout.json"
LOG_PATH = REPO / "results" / "holdout_log.jsonl"
SPEC_TAG = "phase10_us.yaml"


def log(msg: str) -> None:
    print(f"[phase10 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False,
                          cwd=REPO, capture_output=True, text=True).stdout.strip()


def load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text())


# ---------------------------------------------------------------------------
# data


def cmd_fetch(spec: dict, out: Path, coverage: Path) -> dict:
    """Daily ETF bars (adjustment=all: split + dividend adjusted) for every
    ETF in the spec, via the same Alpaca client as the S&P dump."""
    import alpaca_us_data as AU
    syms = sorted({s for v in spec["etfs"].values() for s in v} - {"SPY"})
    client = AU.Alpaca(feed="auto")
    rows = []
    for i in range(0, len(syms), 20):
        chunk = syms[i:i + 20]
        data = client.bars(chunk, spec["data_start"], spec["sample_end"], "all")
        for s, bars in data.items():
            for b in bars:
                rows.append({"datetime": pd.Timestamp(b["t"][:10]),
                             "instrument": s, "open": float(b["o"]),
                             "close": float(b["c"]), "volume": float(b["v"])})
    df = pd.DataFrame(rows).sort_values(["instrument", "datetime"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    cov = {"feed": client.feed, "adjustment": "all",
           "start": spec["data_start"], "end": spec["sample_end"],
           "symbols": {s: {"first": str(g["datetime"].min().date()),
                           "last": str(g["datetime"].max().date()),
                           "bars": int(len(g))}
                       for s, g in df.groupby("instrument")},
           "missing": [s for s in syms if s not in set(df["instrument"])]}
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text(json.dumps(cov, indent=1))
    log(f"wrote {len(df)} bars for {df['instrument'].nunique()} ETFs -> {out}"
        f" (missing: {cov['missing']})")
    return cov


def slice_feed(feed: dict, end_excl: pd.Timestamp | None) -> dict:
    """The feed restricted to days < end_excl (None = unchanged)."""
    if end_excl is None:
        return feed
    keep = feed["days"][feed["days"] < end_excl]
    last = keep[-1]
    bars = {f: b[b.index.get_level_values("datetime") <= last]
            for f, b in feed["bars"].items()}
    return {"bars": bars, "spy": feed["spy"].loc[:last],
            "cal": feed["cal"][feed["cal"] <= last], "days": keep,
            "close": feed["close"].loc[:last]}


def load_etf_bars(spec: dict, days: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_parquet(REPO / spec["etf_bars"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"].isin(days) & (df["instrument"] != "SPY")]
    return df


def sim_feed(feed: dict, etf: pd.DataFrame) -> dict:
    """feed with ETF bars appended (like SPY) — the simulator's view only;
    signals for single names are built from the original feed."""
    bars = {}
    for f in ("close", "open", "volume"):
        e = etf.set_index(["datetime", "instrument"])[[f]]
        bars[f] = pd.concat([feed["bars"][f], e]).sort_index()
    close = bars["close"]["close"].unstack("instrument").sort_index()
    return {**feed, "bars": bars, "close": close}


def member_mask(provider: Path, days: pd.DatetimeIndex,
                cols: pd.Index) -> pd.DataFrame:
    """days x symbols: True while the name is an S&P 500 member."""
    inst = ev.load_instruments(provider, "sp500")
    m = pd.DataFrame(False, index=days, columns=cols)
    for r in inst.itertuples():
        if r.instrument in m.columns:
            m.loc[(days >= pd.Timestamp(r.start)) & (days <= pd.Timestamp(r.end)),
                  r.instrument] = True
    return m


def load_sue(spec: dict, days: pd.DatetimeIndex, cols: pd.Index,
             max_age: int) -> pd.DataFrame:
    f = pd.read_parquet(REPO / spec["fundamentals"])
    f = f[f["group"] == "ni"].copy()
    f["symbol"] = f["symbol"].map(ev.canon)
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c])
    f = f[f["filed"] <= days[-1]]                 # nothing filed after the window
    evs = {s: MA.sue_events(g) for s, g in f.groupby("symbol") if s in cols}
    return MA.sue_panel(evs, days, max_age).reindex(columns=cols)


# ---------------------------------------------------------------------------
# inputs, signals, targets


def build(spec: dict, end_excl: pd.Timestamp | None) -> dict:
    provider = REPO / spec["provider"]
    log("loading honest feed")
    feed = slice_feed(P8.load_feed(provider), end_excl)
    days, close = feed["days"], feed["close"]
    log(f"feed {days[0].date()} .. {days[-1].date()} ({len(days)} days)")
    etf = load_etf_bars(spec, days)
    sfeed = sim_feed(feed, etf)
    P = sfeed["close"].reindex(index=days)
    etf_cols = sorted(set(etf["instrument"]) | {"SPY"})
    P = P[etf_cols]
    smap = P8.load_sector_map(spec)
    kinds = {c["kind"] for c in spec["candidates"]}
    need = {k for c in spec["candidates"] for k in c.get("parts", {})}
    kinds |= {c["kind"] for c in spec["candidates"] if c["name"] in need}
    comps = {}
    if kinds & {"b50", "combo", "crash"}:
        log("B50 composite (phase-8 builders, phase-9 blend)")
        p8spec = {"signals": {"x": ["mom", "vq"]},
                  "fundamentals": spec["fundamentals"]}
        panels = P8.build_panels(p8spec, feed, provider, smap)
        ranks = {k: P9.row_rank(v.reindex(columns=close.columns))
                 for k, v in panels.items()}
        eq = pd.DataFrame({"mom": 0.5, "vq": 0.5}, index=days)
        comps["b50"] = P9.weighted_blend(ranks, eq, ["mom", "vq"])
    member = member_mask(provider, days, close.columns)
    if "lowvol" in kinds:
        win = next(c["win"] for c in spec["candidates"] if c["kind"] == "lowvol")
        log("residual-vol panel")
        comps["lowvol"] = -MA.residual_vol(close.drop(columns="SPY"),
                                           feed["spy"], win)
    if "strev" in kinds:
        comps["strev"] = -(close / close.shift(5) - 1.0)
    if "pead" in kinds:
        age = next(c["max_age_days"] for c in spec["candidates"]
                   if c["kind"] == "pead")
        log("SUE panel from quarterly XBRL net income")
        comps["pead"] = load_sue(spec, days, close.columns, age)
        comps["pead_cover"] = int(comps["pead"].notna().sum().sum())
    cmap = mc.liquidity_cost_map(sfeed["bars"]["close"], sfeed["bars"]["volume"],
                                 flat_bps=spec["costs"]["flat_bps"])
    ctx = MA.Ctx(P, spec["est"], spec["cash_symbol"])
    return {"feed": feed, "sfeed": sfeed, "P": P, "ctx": ctx, "smap": smap,
            "comps": comps, "member": member, "cmap": cmap, "days": days}


def s8(spec: dict) -> dict:
    return {"account": spec["books"]["paper"], "fill": spec["fill"],
            "delist_liquidate": spec["delist_liquidate"], "costs": spec["costs"],
            "rebalance_band": spec["rebalance_band"],
            "min_order_usd": spec["min_order_usd"],
            "core_symbol": spec["core_symbol"],
            "max_stock_weight": spec["max_stock_weight"]}


def etf_rule(spec: dict, c: dict):
    u = spec["universes"].get(c.get("universe", ""), [])
    cap = float(c.get("gross_cap", 1.0))
    k = c["kind"]
    if k == "hold":
        return MA.w_hold(c["weights"])
    if k == "tsmom":
        return MA.w_tsmom(u, cap)
    if k == "trend":
        return MA.w_trend(u, cap)
    if k == "gtaa":
        return MA.w_gtaa(u)
    if k == "gem":
        return MA.w_gem()
    if k == "dualmom":
        return MA.w_dualmom(u, int(c["k"]))
    if k == "erc":
        return MA.w_erc(u, bool(c.get("trend_filter")), cap)
    if k == "sector":
        return MA.w_sector(u, int(c["k"]), bool(c.get("abs_filter")))
    raise ValueError(k)


SINGLE = {"b50", "lowvol", "strev", "pead"}


def targets_for(spec: dict, inp: dict, c: dict, days: pd.DatetimeIndex,
                by_name: dict, cache: dict) -> dict:
    """{day: book} for candidate c over `days` (one episode)."""
    key = (c["name"], days[0], days[-1])
    if key in cache:
        return cache[key]
    k = c["kind"]
    sp = s8(spec)
    if k in SINGLE:
        comp = inp["comps"][k if k != "b50" else "b50"]
        if c.get("members") == "pit":
            comp = comp.where(inp["member"].reindex_like(comp).fillna(False))
        if k == "b50":
            cand = P9.cand8({"name": c["name"], "core": 0.5, "comp": "b50",
                             "n": c["n"]})
        else:
            cand = {"name": c["name"], "core": float(c.get("core", 0.0)),
                    "guard": None, "sleeve": {"n": int(c["n"]),
                                              "weighting": "equal"}}
            if k == "lowvol":
                cand["pick_freeze"] = "monthly"
        out = P8.daily_targets(comp, inp["feed"]["close"], days, cand, sp, {},
                               inp["smap"])
    elif k == "combo":
        out = MA.combine([(w, targets_for(spec, inp, by_name[n], days, by_name,
                                          cache))
                          for n, w in c["parts"].items()])
    elif k == "crash":
        base = targets_for(spec, inp, by_name[c["base"]], days, by_name, cache)
        e = MA.crash_exposure(inp["feed"]["spy"], days, c["sma"], c["vol_win"],
                              c["vol_lookback"], c["vol_q"], c["risk_mult"],
                              c["min_hold"])
        out = MA.scale_book(base, e, spec["cash_symbol"])
    else:
        out = MA.monthly_book(inp["ctx"], etf_rule(spec, c), days)
    cache[key] = out
    return out


def tradable_days(inp: dict, weights: dict, days: pd.DatetimeIndex
                  ) -> pd.DatetimeIndex:
    """Days on which every ETF of a fixed-weight book has a close (proxies
    with short histories are scored on their overlapping window only)."""
    P = inp["sfeed"]["close"].reindex(days)
    ok = np.ones(len(days), dtype=bool)
    for s in weights:
        ok &= P[s].notna().values if s in P.columns else False
    return days[ok]


def run_episode(spec: dict, inp: dict, c: dict, days: pd.DatetimeIndex,
                by_name: dict, cache: dict) -> dict | None:
    if c.get("kind") == "hold" or "weights" in c:
        days = tradable_days(inp, c["weights"], days)
        if len(days) < 20:
            return None
        tg = {d: dict(c["weights"]) for d in days}
    else:
        tg = targets_for(spec, inp, c, days, by_name, cache)
    met = P8.run_candidate(inp["sfeed"], tg, days, s8(spec), inp["cmap"], None)
    r = met["daily_excess"].dropna()
    if c.get("leverage_sensitivity"):
        cash_r = inp["P"][spec["cash_symbol"]].pct_change()
        drag = MA.financing_drag(tg, days, cash_r,
                                 spec["leverage_financing_spread"],
                                 spec["cash_symbol"])
        r = r - drag.reindex(r.index).fillna(0.0)
    gross = [sum(v for s, v in b.items() if s != spec["cash_symbol"])
             for b in tg.values()]
    return {"daily": r, "turnover": met["turnover"],
            "n_trades": len(met["trades"]),
            "avg_gross_ex_cash": float(np.mean(gross)) if gross else 0.0}


def score(spec: dict, inp: dict, cands: list[dict],
          episodes: dict[str, pd.DatetimeIndex], by_name: dict) -> dict:
    cache: dict = {}
    rf = inp["P"][spec["cash_symbol"]].pct_change()
    out = {}
    for c in cands:
        by_ep, dailies, tov = {}, [], []
        for ep, days in episodes.items():
            res = run_episode(spec, inp, c, days, by_name, cache)
            if res is None:
                continue
            m = MA.perf(res["daily"], rf)
            m.update({k: res[k] for k in ("turnover", "n_trades",
                                          "avg_gross_ex_cash")})
            by_ep[ep] = m
            dailies.append(res["daily"])
            tov.append(res["turnover"])
            log(f"  {c['name']:>16} {ep}: ar={m['ar']:+.3f} sh={m['sharpe']:+.2f}"
                f" mdd={m['mdd']:+.3f} tov={res['turnover']:.3f}")
        daily = pd.concat(dailies).sort_index() if dailies \
            else pd.Series(dtype=float)
        out[c["name"]] = {"by_episode": by_ep, "daily": daily,
                          "turnover": float(np.mean(tov)) if tov else 0.0}
    return out


def summarise(res: dict, spy_r: pd.Series, rf: pd.Series) -> dict:
    d = res["daily"]
    m = MA.perf(d, rf)
    xs = (d - spy_r.reindex(d.index)).dropna()
    m.update({"xs_spy_ar": float(xs.mean() * 252) if len(xs) else 0.0,
              "turnover": res["turnover"]})
    return m


def dsr(d: pd.Series, rf: pd.Series, n: int, trial_srs: list[float]) -> dict:
    x = (d - rf.reindex(d.index).fillna(0.0)).dropna()
    if len(x) < 3 or x.std() == 0:
        return None
    return wf.deflated_sharpe(x, n, trial_srs)


# ---------------------------------------------------------------------------
# trial accounting


def _rows() -> list[str]:
    return LOG_PATH.read_text().splitlines() if LOG_PATH.exists() else []


def _tag(line: str) -> str | None:
    try:
        return json.loads(line).get("spec")
    except json.JSONDecodeError:
        return None


def prior_trials() -> int:
    """Running total before phase 10 (max n_trials of non-phase-10 rows)."""
    best = 0
    for ln in _rows():
        if _tag(ln) == SPEC_TAG:
            continue
        try:
            best = max(best, int(json.loads(ln).get("n_trials") or 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return best


def write_log(rows: list[dict], stage: str) -> None:
    keep = [ln for ln in _rows()
            if not (_tag(ln) == SPEC_TAG and json.loads(ln).get("stage") == stage)]
    keep += [json.dumps(r, default=str) for r in rows]
    LOG_PATH.write_text("\n".join(keep) + "\n")


def trial_cands(spec: dict) -> list[dict]:
    return list(spec["candidates"])


def eligible(spec: dict, c: dict) -> bool:
    return not (c["name"] in ("A_spy", spec["adoption"]["baseline"])
                or c.get("diagnostic") or c.get("leverage_sensitivity"))


def pick_finalists(spec: dict, summ: dict) -> dict:
    base = spec["adoption"]["baseline"]
    tol = spec["finalists"]["mdd_tol"]
    bmdd = summ[base]["mdd"]
    pool = {c["name"]: summ[c["name"]] for c in spec["candidates"]
            if eligible(spec, c)}
    ok = {k: v for k, v in pool.items() if v["mdd"] >= bmdd - tol}
    ranked = sorted(ok, key=lambda k: -ok[k]["sharpe"])
    return {"rule": f"top {spec['finalists']['k']} eligible by dev Sharpe "
                    f"(excess of BIL) with dev MDD >= B50 dev MDD - {tol}",
            "b50_dev_mdd": bmdd, "eligible_ranked": ranked,
            "excluded_mdd": sorted(set(pool) - set(ok)),
            "finalists": ranked[:spec["finalists"]["k"]]}


# ---------------------------------------------------------------------------
# commands


def dev_episodes(days: pd.DatetimeIndex, start_year: int) -> dict:
    out = {}
    for y in sorted({d.year for d in days if d.year >= start_year}):
        yd = days[days.year == y]
        if len(yd) >= 20:
            out[str(y)] = yd
    return out


def proxy_windows(spec: dict, res: dict, rf: pd.Series, names: list[str]) -> dict:
    """Each proxy vs each named book over the proxy's own (overlapping)
    window — states the window start for short-history ETFs."""
    out = {}
    for p in spec["proxies"]:
        pd_ = res[p["name"]]["daily"]
        if pd_.empty:
            continue
        idx = pd_.index
        row = {"start": str(idx[0].date()), "end": str(idx[-1].date()),
               "proxy": MA.perf(pd_, rf)}
        for n in names:
            d = res[n]["daily"]
            row[n] = MA.perf(d[d.index.isin(idx)], rf)
        out[p["name"]] = row
    return out


def cmd_dev(spec: dict) -> dict:
    t0 = time.time()
    hstart = pd.Timestamp(spec["holdout"]["start"])
    inp = build(spec, end_excl=hstart)
    assert inp["days"][-1] < hstart, "holdout leaked into dev"
    eps = dev_episodes(inp["days"], spec["dev_start_year"])
    by_name = {c["name"]: c for c in spec["candidates"]}
    log(f"dev episodes {list(eps)}; {len(spec['candidates'])} candidates + "
        f"{len(spec['proxies'])} proxies")
    res = score(spec, inp, spec["candidates"], eps, by_name)
    pres = score(spec, inp, spec["proxies"], eps, by_name)
    rf = inp["P"][spec["cash_symbol"]].pct_change()
    spy_r = inp["feed"]["spy"].pct_change()
    n_windows = sum(len(r["by_episode"]) for r in res.values())
    prior = prior_trials()
    n = prior + n_windows
    srs = [MA.perf(r["daily"], rf)["sharpe"] for r in res.values()]
    summ = {}
    for k, r in {**res, **pres}.items():
        summ[k] = summarise(r, spy_r, rf)
        summ[k]["dsr"] = dsr(r["daily"], rf, n, srs) if k in res else None
        summ[k]["by_episode"] = r["by_episode"]
    fin = pick_finalists(spec, summ)
    log(f"n_trials = {prior} prior + {n_windows} dev windows = {n}; "
        f"finalists {fin['finalists']}")
    DEV_DAILY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: r["daily"] for k, r in {**res, **pres}.items()}) \
        .to_csv(DEV_DAILY, float_format="%.10g")
    ts = dt.datetime.now().isoformat()
    write_log([{"ts": ts, "market": "us", "git_sha": git_sha(), "spec": SPEC_TAG,
                "stage": "dev", "candidate": f"phase10:{k}",
                "re_viewed_holdout": False, **{kk: summ[k][kk] for kk in
                                               ("ar", "sharpe", "mdd", "n_days")},
                "dsr": summ[k]["dsr"], "n_trials": n} for k in res], "dev")
    out = {"stage": "phase10_dev", "git_sha": git_sha(), "ts": ts,
           "spec": str(SPEC_PATH.relative_to(REPO)),
           "dev_window": [str(inp["days"][0].date()), str(inp["days"][-1].date())],
           "holdout": spec["holdout"], "episodes": list(eps),
           "prior_trials": prior, "n_dev_windows": n_windows, "n_trials": n,
           "summary": summ, "finalists": fin,
           "proxy_windows": proxy_windows(spec, {**res, **pres}, rf,
                                          [spec["adoption"]["baseline"]]
                                          + fin["finalists"]),
           "pead_cells": inp["comps"].get("pead_cover"),
           "runtime_s": round(time.time() - t0, 1)}
    DEV_OUT.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {DEV_OUT} ({out['runtime_s']}s)")
    return out


def adoption(spec: dict, hs: dict, full: dict, dev: dict, fins: list[str]) -> dict:
    rule = spec["adoption"]
    base = rule["baseline"]
    tol = rule["mdd_tol"]
    proxies = [p["name"] for p in spec["proxies"]]
    bd = (full[base].get("dsr") or {}).get("DSR", 0.0)
    checks = {}
    for k in fins:
        h, f, d = hs[k], full[k], dev[k]
        beaten = [p for p in proxies if p in hs and h["sharpe"] > hs[p]["sharpe"]]
        kd = (f.get("dsr") or {}).get("DSR", 0.0)
        c = {"holdout_sharpe_gt_b50": h["sharpe"] > hs[base]["sharpe"],
             "holdout_sharpe_gt_spy": h["sharpe"] > hs["A_spy"]["sharpe"],
             "holdout_mdd_ok": h["mdd"] >= hs[base]["mdd"] - tol,
             "full_mdd_ok": f["mdd"] >= full[base]["mdd"] - tol,
             "dev_sharpe_ge_b50": d["sharpe"] >= dev[base]["sharpe"],
             "dsr_ok": kd >= rule["min_dsr"] and kd >= bd,
             "proxies_beaten": beaten,
             "proxies_ok": len(beaten) >= rule["min_proxies_beaten"],
             "dsr": kd}
        c["adopt"] = all(c[x] for x in (
            "holdout_sharpe_gt_b50", "holdout_sharpe_gt_spy", "holdout_mdd_ok",
            "full_mdd_ok", "dev_sharpe_ge_b50", "dsr_ok", "proxies_ok"))
        checks[k] = c
    win = [k for k, c in checks.items() if c["adopt"]]
    b_beaten = [p for p in proxies if p in hs
                and hs[base]["sharpe"] > hs[p]["sharpe"]]
    return {"baseline": base, "baseline_dsr": bd, "checks": checks,
            "baseline_proxies_beaten_holdout": b_beaten,
            "adopted": max(win, key=lambda k: checks[k]["dsr"]) if win else None,
            "keep_baseline": not win}


def cmd_holdout(spec: dict) -> dict:
    if HOLD_OUT.exists():
        sys.exit(f"{HOLD_OUT} exists — the holdout is scored ONCE (predeclared)")
    dev = json.loads(DEV_OUT.read_text())
    fins = dev["finalists"]["finalists"]
    t0 = time.time()
    inp = build(spec, end_excl=None)
    h0, h1 = pd.Timestamp(spec["holdout"]["start"]), pd.Timestamp(spec["holdout"]["end"])
    hd = inp["days"][(inp["days"] >= h0) & (inp["days"] <= h1)]
    by_name = {c["name"]: c for c in spec["candidates"]}
    base = spec["adoption"]["baseline"]
    run = [by_name[n] for n in dict.fromkeys(
        ["A_spy", base, "B50_pit"] + fins)]
    log(f"HOLDOUT {hd[0].date()} .. {hd[-1].date()} ({len(hd)} days): "
        f"{[c['name'] for c in run]} + proxies")
    res = score(spec, inp, run, {"holdout": hd}, by_name)
    pres = score(spec, inp, spec["proxies"], {"holdout": hd}, by_name)
    rf = inp["P"][spec["cash_symbol"]].pct_change()
    spy_r = inp["feed"]["spy"].pct_change()
    hwin = sum(len(r["by_episode"]) for r in res.values())
    n = dev["n_trials"] + hwin
    dd = pd.read_csv(DEV_DAILY, index_col=0, parse_dates=True)
    hs, full = {}, {}
    full_daily = {}
    for k, r in {**res, **pres}.items():
        hs[k] = summarise(r, spy_r, rf)
        if k in dd.columns:
            full_daily[k] = pd.concat([dd[k].dropna(), r["daily"]]).sort_index()
    srs = [MA.perf(full_daily[k], rf)["sharpe"] for k in res if k in full_daily]
    srs_xs = [MA.perf(full_daily[k], spy_r)["sharpe"] for k in res
              if k in full_daily and k != "A_spy"]
    for k, d in full_daily.items():
        full[k] = MA.perf(d, rf)
        full[k]["dsr"] = dsr(d, rf, n, srs) if k in res else None
        xs = (d - spy_r.reindex(d.index)).dropna()
        full[k]["dsr_xs_spy"] = dsr(d, spy_r, n, srs_xs) \
            if k in res and k != "A_spy" else None
        full[k]["xs_spy_ar"] = float(xs.mean() * 252)
        if k in res and k != base:
            act = (d - full_daily[base].reindex(d.index)).dropna()
            full[k]["dsr_active_b50"] = wf.deflated_sharpe(act, n, srs) \
                if act.std() > 0 else None
    adopt = adoption(spec, hs, full, dev["summary"], fins)
    ts = dt.datetime.now().isoformat()
    write_log([{"ts": ts, "market": "us", "git_sha": git_sha(), "spec": SPEC_TAG,
                "stage": "holdout", "candidate": f"phase10:{k}",
                "re_viewed_holdout": k in (base, "B50_pit", "A_spy"),
                **{kk: hs[k][kk] for kk in ("ar", "sharpe", "mdd", "n_days")},
                "dsr": full[k].get("dsr"), "n_trials": n} for k in res], "holdout")
    out = {"stage": "phase10_holdout", "git_sha": git_sha(), "ts": ts,
           "holdout_window": [str(hd[0].date()), str(hd[-1].date())],
           "finalists": fins, "n_trials": n, "n_holdout_windows": hwin,
           "holdout": hs, "full_sample": full, "adoption": adopt,
           "runtime_s": round(time.time() - t0, 1)}
    HOLD_OUT.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {HOLD_OUT}: adopted={adopt['adopted']} ({out['runtime_s']}s)")
    return out


# ---------------------------------------------------------------------------
# report


def _p(x, f="{:+.1%}"):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) \
        else f.format(x)


def cmd_report() -> str:
    spec = load_spec()
    dev = json.loads(DEV_OUT.read_text())
    S = dev["summary"]
    lines = [f"Dev {dev['dev_window'][0]} .. {dev['dev_window'][1]}, "
             f"n_trials after dev = {dev['n_trials']}", "",
             "| book | P&L ($100k) | AR | Sharpe | MDD | xs-SPY AR | turnover | DSR |",
             "|---|---|---|---|---|---|---|---|"]
    order = [c["name"] for c in spec["candidates"]] + \
        [p["name"] for p in spec["proxies"]]
    for k in order:
        s = S.get(k)
        if not s:
            continue
        lines.append(
            f"| {k}{' (LEV)' if 'LX' in k else ''} | {_p(s['pnl_usd'], '${:,.0f}')} "
            f"| {_p(s['ar'])} | {_p(s['sharpe'], '{:.2f}')} | {_p(s['mdd'])} "
            f"| {_p(s['xs_spy_ar'])} | {_p(s['turnover'], '{:.3f}')} "
            f"| {_p((s.get('dsr') or {}).get('DSR'), '{:.2f}')} |")
    lines += ["", f"Finalists: {dev['finalists']['finalists']}", ""]
    if HOLD_OUT.exists():
        H = json.loads(HOLD_OUT.read_text())
        lines += [f"Holdout {H['holdout_window'][0]} .. {H['holdout_window'][1]}, "
                  f"n_trials = {H['n_trials']}", "",
                  "| book | P&L ($100k) | AR | Sharpe | MDD | full DSR |",
                  "|---|---|---|---|---|---|"]
        for k, s in H["holdout"].items():
            f = H["full_sample"].get(k, {})
            lines.append(f"| {k} | {_p(s['pnl_usd'], '${:,.0f}')} | {_p(s['ar'])} "
                         f"| {_p(s['sharpe'], '{:.2f}')} | {_p(s['mdd'])} "
                         f"| {_p((f.get('dsr') or {}).get('DSR'), '{:.2f}')} |")
        lines += ["", "Adoption: " + json.dumps(H["adoption"], default=str)]
    txt = "\n".join(lines)
    print(txt)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "dev", "holdout", "report"])
    args = ap.parse_args()
    spec = load_spec()
    if args.cmd == "fetch":
        cmd_fetch(spec, REPO / spec["etf_bars"],
                  REPO / "results" / "phase10" / "etf_coverage.json")
    elif args.cmd == "dev":
        cmd_dev(spec)
    elif args.cmd == "holdout":
        cmd_holdout(spec)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
