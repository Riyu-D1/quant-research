#!/usr/bin/env python3
"""Phase-11 US — honest PIT baseline, then the hybrid LLM + algorithmic bot.

    uv run python scripts/phase11_us.py universe --out build/universe_sp500.csv
    uv run python scripts/phase11_us.py sec        # SEC companyfacts (no key)
    uv run python scripts/phase11_us.py baseline   # honest B50 on dev, holdout sealed

Heavy — GitHub Actions only (.github/workflows/phase11.yml), never the Mac.
Spec (predeclared): configs/experiments/phase11_us.yaml.

universe  pinned fja05680 spells + daily Wikipedia snapshots to sample_end
          -> results/phase11/sp500_membership.csv (+ audit json) and the
          Alpaca universe CSV the dump is rebuilt from.
sec       one companyfacts call per universe CIK -> fundamentals.parquet
          (vq tags + the value/quality tilt tags) and shares_out.parquet.
baseline  A_spy, B50_honest (PIT members) and B50_ever (the old ever-member
          universe) on the dev window; writes every B50_honest pick so
          tests/test_phase11.py can check each one against the membership.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import live_holdout as LH  # noqa: E402
import phase8_us as P8  # noqa: E402
import phase9_us as P9  # noqa: E402
import phase10_us as P10  # noqa: E402
from quant_research import edgar as E  # noqa: E402
from quant_research import events as ev  # noqa: E402
from quant_research import factor_dsl as D  # noqa: E402
from quant_research import fundamentals as F  # noqa: E402
from quant_research import micro as mc  # noqa: E402
from quant_research import multiasset as MA  # noqa: E402
from quant_research import pit  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase11_us.yaml"
OUT = REPO / "results" / "phase11"
AUDIT = OUT / "membership_audit.json"
BASE_OUT = OUT / "baseline_dev.json"
PICKS = OUT / "b50_honest_picks.csv"
LLM_OUT = OUT / "llm_factors.json"
DEV_OUT = OUT / "dev.json"
DEV_DAILY = OUT / "dev_daily.csv"
HOLD_OUT = OUT / "holdout.json"
STATE = REPO / "state" / "phase11" / "hybrid.json"
LOG_PATH = REPO / "results" / "holdout_log.jsonl"
SPEC_TAG = "phase11_us.yaml"
NON_STOCK = {"SPY", "BIL"}


def log(msg: str) -> None:
    print(f"[phase11 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False,
                          cwd=REPO, capture_output=True, text=True).stdout.strip()


def load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text())


def load_membership(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path)
    return w.assign(start=pd.to_datetime(w["start"]), end=pd.to_datetime(w["end"]))


# ---------------------------------------------------------------------------
# universe


def cmd_universe(spec: dict, out_csv: Path) -> dict:
    m = spec["membership"]
    s = requests.Session()
    s.headers["User-Agent"] = pit.UA
    win = pit.fja_windows(m["fja_sha"], s)
    revs = pit.wiki_daily_revisions(m["wiki_from"], spec["sample_end"], s)
    tables = {r: pit.wiki_constituents(r, s) for r in sorted(set(revs.values()))}
    snaps = {d: tables[r] for d, r in revs.items()}
    sizes = {str(d.date()): len(t) for d, t in snaps.items()}
    snaps = {d: t for d, t in snaps.items() if 490 <= len(t) <= 510}
    patched = pit.patch_windows(win, snaps, m["persist"])
    end = pd.Timestamp(spec["sample_end"])
    patched = patched[patched["start"] <= end]
    patched = patched[patched["end"].isna() | (patched["end"] > pd.Timestamp(spec["data_start"]))]
    OUT.mkdir(parents=True, exist_ok=True)
    patched.to_csv(REPO / spec["membership_csv"], index=False, date_format="%Y-%m-%d")

    # audit: pinned file vs the first snapshot, the gap changes, and the
    # current table's "Date added" vs the pinned spell starts (2016+)
    first = snaps[min(snaps)]
    open_pin = set(win.loc[win["end"].isna(), "symbol"])
    last = snaps[max(snaps)]
    starts = win.groupby("symbol")["start"].max()
    da = last.dropna(subset=["date_added"]).set_index("symbol")["date_added"]
    da = da[(da >= "2016-01-01") & (da < pd.Timestamp(m["wiki_from"]))]
    both = da.index.intersection(starts.index)
    mism = {k: [str(starts[k].date()), str(da[k].date())] for k in both
            if starts[k] != da[k]}
    gap = patched[(patched["start"] >= pd.Timestamp(m["wiki_from"]))
                  | (patched["end"] >= pd.Timestamp(m["wiki_from"]))]
    audit = {"fja_sha": m["fja_sha"], "wiki_from": m["wiki_from"],
             "sample_end": spec["sample_end"], "persist": m["persist"],
             "n_revisions_read": len(tables), "snapshot_sizes": sizes,
             "pin_vs_first_snapshot": {
                 "date": str(min(snaps).date()),
                 "pinned_open_not_in_wiki": sorted(open_pin - set(first["symbol"])),
                 "wiki_not_pinned_open": sorted(set(first["symbol"]) - open_pin)},
             "gap_changes": [{"symbol": r.symbol, "start": str(r.start.date()),
                              "end": None if pd.isna(r.end) else str(r.end.date())}
                             for r in gap.itertuples()],
             "date_added_checked": int(len(both)),
             "date_added_mismatch": mism,
             "members_at_end": int(((patched["start"] <= end)
                                    & (patched["end"].isna()
                                       | (patched["end"] > end))).sum()),
             "spells": int(len(patched)),
             "symbols": int(patched["symbol"].nunique())}
    AUDIT.write_text(json.dumps(audit, indent=1))

    # Alpaca universe CSV (alpaca_us_data.py fetch/finalize format)
    u = patched.copy()
    u["symbol"] = u["symbol"].str.replace("-", ".")
    u["in_date"] = u["start"].clip(lower=pd.Timestamp(spec["data_start"])).dt.strftime("%Y-%m-%d")
    u["out_date"] = u["end"].fillna(end).clip(upper=end).dt.strftime("%Y-%m-%d")
    u = u.assign(sector="", sub_industry="", source="fja05680+wikipedia")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    u[["symbol", "in_date", "out_date", "sector", "sub_industry", "source"]] \
        .to_csv(out_csv, index=False)
    log(f"membership: {audit['spells']} spells / {audit['symbols']} symbols, "
        f"{len(audit['gap_changes'])} changes after the pin, "
        f"{audit['members_at_end']} members at {spec['sample_end']}")
    return audit


# ---------------------------------------------------------------------------
# SEC companyfacts


def universe_ciks(spec: dict, ed: E.Edgar) -> pd.DataFrame:
    """(symbol, cik) for every membership symbol: the phase-3 map (dead
    tickers included) plus SEC company_tickers.json for newer names."""
    syms = set(load_membership(REPO / spec["membership_csv"])["symbol"])
    cmap = pd.read_parquet(REPO / spec["cik_map"])[["symbol", "cik"]].dropna()
    cmap["symbol"] = cmap["symbol"].map(ev.canon)
    tick = E.ticker_cik_map(ed)
    tick["symbol"] = tick["ticker"].map(ev.canon)
    extra = tick[tick["symbol"].isin(syms - set(cmap["symbol"]))][["symbol", "cik"]]
    cmap = pd.concat([cmap, extra]).drop_duplicates("symbol")
    return cmap[cmap["symbol"].isin(syms)].reset_index(drop=True)


def cmd_sec(spec: dict) -> None:
    """One companyfacts request per universe CIK -> vq + tilt fundamentals
    and dei shares outstanding (same PIT `filed` semantics as phase 3/4)."""
    ed = E.Edgar()
    cmap = universe_ciks(spec, ed)
    syms = set(load_membership(REPO / spec["membership_csv"])["symbol"])
    funds, shares, failed = [], [], []
    for i, r in enumerate(cmap.itertuples()):
        cik = int(r.cik)
        try:
            d = ed.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
        except Exception as e:  # noqa: BLE001 — one bad issuer must not stop the pull
            failed.append(r.symbol)
            log(f"companyfacts {r.symbol} ({cik}): {e}")
            continue
        f = F.parse_companyfacts(d, F.VQ_GROUPS + F.QV_GROUPS)
        if not f.empty:
            funds.append(f.assign(symbol=r.symbol, cik=cik))
        sh = E.parse_shares(d)
        if not sh.empty:
            shares.append(sh.assign(symbol=r.symbol, cik=cik))
        if i % 100 == 0:
            log(f"companyfacts {i}/{len(cmap)}")
    for path, parts in ((spec["fundamentals"], funds), (spec["shares"], shares)):
        p = REPO / path
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
    cov = {"issuers": int(len(cmap)), "with_facts": len(funds),
           "with_shares": len(shares), "failed": failed,
           "no_cik": sorted(syms - set(cmap["symbol"]))}
    (OUT / "sec_coverage.json").write_text(json.dumps(cov, indent=1))
    log(f"sec: {cov['with_facts']} issuers with facts, {len(failed)} failed, "
        f"{len(cov['no_cik'])} symbols without a CIK")


# ---------------------------------------------------------------------------
# inputs + books


def build_core(spec: dict, end_excl: pd.Timestamp | None) -> dict:
    """Feed (sliced before any signal is built), sim feed with ETF bars,
    PIT member mask, costs and the raw B50 legs."""
    provider = REPO / spec["provider"]
    feed = P10.slice_feed(P8.load_feed(provider), end_excl)
    days, close = feed["days"], feed["close"]
    log(f"feed {days[0].date()} .. {days[-1].date()} ({len(days)} days, "
        f"{close.shape[1]} symbols)")
    sfeed = P10.sim_feed(feed, P10.load_etf_bars(spec, days))
    win = load_membership(REPO / spec["membership_csv"])
    member = pit.member_mask(win, days, close.columns)
    log("raw legs: mom (bars) + vq parts (PIT SEC facts)")
    mom = LH.bars_score(close, "mom").unstack("instrument") \
        .reindex(index=days, columns=close.columns)
    vq_parts = fund_panels(spec, feed, F.vq_features)
    cmap = mc.liquidity_cost_map(sfeed["bars"]["close"], sfeed["bars"]["volume"],
                                 flat_bps=spec["costs"]["flat_bps"])
    return {"feed": feed, "sfeed": sfeed, "days": days, "member": member,
            "win": win, "mom": mom, "vq_parts": vq_parts, "cmap": cmap,
            "smap": P8.load_sector_map(spec),
            "rf": sfeed["close"][spec["cash_symbol"]].pct_change(),
            "spy_r": feed["spy"].pct_change()}


def fund_panels(spec: dict, feed: dict, fn) -> dict[str, pd.DataFrame]:
    """Per-symbol PIT fundamentals (F.vq_features / F.qv_features) ->
    {feature: days x symbols} with inf -> NaN."""
    days, close = feed["days"], feed["close"]
    f = pd.read_parquet(REPO / spec["fundamentals"])
    f["symbol"] = f["symbol"].map(ev.canon)
    for c in ("start", "end", "filed"):
        f[c] = pd.to_datetime(f[c])
    sh = pd.read_parquet(REPO / spec["shares"])
    sh["filed"] = pd.to_datetime(sh["filed"])
    sh["symbol"] = sh["symbol"].map(ev.canon)
    shr = {s: g.rename(columns={"val": "shares"}) for s, g in sh.groupby("symbol")}
    parts: dict[str, dict] = {}
    for s, fs in f.groupby("symbol"):
        if s in close.columns:
            for k, v in fn(fs, shr.get(s), close[s].reindex(days).values, days).items():
                parts.setdefault(k, {})[s] = v
    return {k: pd.DataFrame(v, index=days).reindex(columns=close.columns)
            .replace([np.inf, -np.inf], np.nan) for k, v in parts.items()}


def member_rank(panel: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional pct-rank among the day's index members only."""
    return P9.row_rank(panel.reindex_like(member).where(member))


def mean_rank(panels: list[pd.DataFrame], member: pd.DataFrame) -> pd.DataFrame:
    """NaN-skipping mean of member ranks (signals6's vq rule)."""
    rk = [member_rank(p, member) for p in panels]
    cnt = sum(r.notna().astype(float) for r in rk)
    return (sum(r.fillna(0.0) for r in rk) / cnt.where(cnt > 0)).where(member)


def b50_honest(core: dict) -> pd.DataFrame:
    """B50_momvq composite ranked among PIT members only."""
    m = core["member"]
    ranks = {"mom": member_rank(core["mom"], m),
             "vq": member_rank(mean_rank(list(core["vq_parts"].values()), m), m)}
    eq = pd.DataFrame({"mom": 0.5, "vq": 0.5}, index=core["days"])
    return P9.weighted_blend(ranks, eq, ["mom", "vq"]).where(m)


def b50_ever(spec: dict, core: dict) -> pd.DataFrame:
    """The pre-phase-11 composite: every ever-member ranked (look-ahead)."""
    feed = core["feed"]
    panels = P8.build_panels({"signals": {"x": ["mom", "vq"]},
                              "fundamentals": spec["fundamentals"]},
                             feed, REPO / spec["provider"], core["smap"])
    ranks = {k: P9.row_rank(v.reindex(columns=feed["close"].columns))
             for k, v in panels.items()}
    eq = pd.DataFrame({"mom": 0.5, "vq": 0.5}, index=core["days"])
    return P9.weighted_blend(ranks, eq, ["mom", "vq"])


def sleeve_targets(spec: dict, core: dict, comp: pd.DataFrame, name: str,
                   n: int, core_w: float = 0.5) -> dict:
    cand = P9.cand8({"name": name, "core": core_w, "comp": name, "n": n})
    return P8.daily_targets(comp, core["feed"]["close"], core["days"], cand,
                            P10.s8(spec), {}, core["smap"])


def picks_frame(targets: dict) -> pd.DataFrame:
    return pd.DataFrame([{"date": d, "symbol": s} for d, b in sorted(targets.items())
                         for s in sorted(b) if s not in NON_STOCK])


def run_book(spec: dict, core: dict, targets: dict,
             episodes: dict[str, pd.DatetimeIndex]) -> dict:
    """Each episode starts flat; returns {by_episode, daily, turnover}."""
    by_ep, dailies, tov = {}, [], []
    for ep, days in episodes.items():
        tg = {d: targets[d] for d in days if d in targets}
        met = P8.run_candidate(core["sfeed"], tg, days, P10.s8(spec),
                               core["cmap"], None)
        r = met["daily_excess"].dropna()
        m = MA.perf(r, core["rf"])
        m["turnover"] = met["turnover"]
        by_ep[ep] = m
        dailies.append(r)
        tov.append(met["turnover"])
        log(f"    {ep}: ar={m['ar']:+.3f} sh={m['sharpe']:+.2f} mdd={m['mdd']:+.3f}")
    return {"by_episode": by_ep, "daily": pd.concat(dailies).sort_index(),
            "turnover": float(np.mean(tov))}


def hold_targets(weights: dict, days: pd.DatetimeIndex) -> dict:
    return {d: dict(weights) for d in days}


# ---------------------------------------------------------------------------
# trial accounting


def _rows() -> list[str]:
    return LOG_PATH.read_text().splitlines() if LOG_PATH.exists() else []


def _row(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def prior_trials() -> int:
    """Running total before phase 11 (max n_trials of non-phase-11 rows)."""
    best = 0
    for ln in _rows():
        r = _row(ln)
        if r.get("spec") == SPEC_TAG:
            continue
        try:
            best = max(best, int(r.get("n_trials") or 0))
        except (TypeError, ValueError):
            continue
    return best


def stage_trials(stage: str) -> int:
    """n_trials recorded by an earlier phase-11 stage (0 if not run)."""
    return max([int(_row(ln).get("n_trials") or 0) for ln in _rows()
                if _row(ln).get("spec") == SPEC_TAG
                and _row(ln).get("stage") == stage] or [0])


def write_log(rows: list[dict], stage: str) -> None:
    keep = [ln for ln in _rows()
            if not (_row(ln).get("spec") == SPEC_TAG and _row(ln).get("stage") == stage)]
    LOG_PATH.write_text("\n".join(keep + [json.dumps(r, default=str) for r in rows]) + "\n")


def phase10_dev_sharpes() -> list[float]:
    d = json.loads((REPO / "results" / "phase10_dev.json").read_text())
    names = {c["name"] for c in yaml.safe_load(
        (REPO / "configs/experiments/phase10_us.yaml").read_text())["candidates"]}
    return [v["sharpe"] for k, v in d["summary"].items() if k in names]


# ---------------------------------------------------------------------------
# step 1: baseline


def cmd_baseline(spec: dict) -> dict:
    t0 = time.time()
    hstart = pd.Timestamp(spec["holdout"]["start"])
    core = build_core(spec, end_excl=hstart)
    assert core["days"][-1] < hstart, "holdout leaked into the baseline"
    eps = P10.dev_episodes(core["days"], spec["dev_start_year"])
    comps = {"B50_honest": b50_honest(core), "B50_ever": b50_ever(spec, core)}
    res, picks = {}, {}
    for c in spec["baseline"]:
        log(c["name"])
        if c["kind"] == "hold":
            tg = hold_targets(c["weights"], core["days"])
        else:
            tg = sleeve_targets(spec, core, comps[c["name"]], c["name"], c["n"])
            picks[c["name"]] = picks_frame(tg)
        res[c["name"]] = run_book(spec, core, tg, eps)
    bad = pit.non_member_picks(picks["B50_honest"], core["win"])
    assert bad.empty, f"B50_honest picked non-members:\n{bad.head()}"
    ever_bad = pit.non_member_picks(picks["B50_ever"], core["win"])
    picks["B50_honest"].to_csv(PICKS, index=False, date_format="%Y-%m-%d")

    prior = prior_trials()
    n_windows = sum(len(r["by_episode"]) for r in res.values())
    n = prior + n_windows
    srs = phase10_dev_sharpes() + [MA.perf(r["daily"], core["rf"])["sharpe"]
                                   for r in res.values()]
    srs_xs = [MA.perf(r["daily"], core["spy_r"])["sharpe"]
              for k, r in res.items() if k != "A_spy"]
    summ = {}
    for k, r in res.items():
        s = P10.summarise(r, core["spy_r"], core["rf"])
        s["dsr"] = P10.dsr(r["daily"], core["rf"], n, srs)
        s["dsr_xs_spy"] = None if k == "A_spy" else \
            P10.dsr(r["daily"], core["spy_r"], n, srs_xs + [0.0])
        s["by_episode"] = r["by_episode"]
        summ[k] = s
    ts = dt.datetime.now().isoformat()
    write_log([{"ts": ts, "market": "us", "git_sha": git_sha(), "spec": SPEC_TAG,
                "stage": "baseline", "candidate": f"phase11:{k}",
                "re_viewed_holdout": False,
                **{kk: summ[k][kk] for kk in ("ar", "sharpe", "mdd", "n_days")},
                "dsr": summ[k]["dsr"], "n_trials": n} for k in res], "baseline")
    by_name = ever_bad.groupby("symbol")["date"].agg(["min", "max", "count"])
    out = {"stage": "phase11_baseline", "git_sha": git_sha(), "ts": ts,
           "dev_window": [str(core["days"][0].date()), str(core["days"][-1].date())],
           "holdout_sealed_from": spec["holdout"]["start"], "episodes": list(eps),
           "prior_trials": prior, "n_windows": n_windows, "n_trials": n,
           "summary": summ,
           "b50_honest_non_member_picks": int(len(bad)),
           "b50_ever_non_member_pick_days": int(len(ever_bad)),
           "b50_ever_non_member_names": {
               s: {"first": str(r["min"].date()), "last": str(r["max"].date()),
                   "days": int(r["count"])}
               for s, r in by_name.sort_values("count", ascending=False).iterrows()},
           "runtime_s": round(time.time() - t0, 1)}
    BASE_OUT.write_text(json.dumps(out, indent=1, default=str))
    log(f"n_trials = {prior} + {n_windows} = {n}; B50_ever made {len(ever_bad)} "
        f"non-member pick-days; wrote {BASE_OUT} ({out['runtime_s']}s)")
    return out


# ---------------------------------------------------------------------------
# step 3: hybrid composites (all ranked among PIT members)


def qv_comp(core: dict) -> pd.DataFrame:
    """0.5 rank(EBIT/TEV) + 0.5 quality (mean rank of GP/A, ROA)."""
    m, q = core["member"], core["qv_parts"]
    v = member_rank(q["ebit_tev"], m)
    qual = mean_rank([q["gpa"], q["roa"]], m)
    return member_rank((v + qual) / 2, m)


def lazy_prices(spec: dict, days: pd.DatetimeIndex, cols: pd.Index) -> pd.DataFrame:
    """days x symbols cosine similarity of a firm's latest 10-K risk-factor
    embedding with its previous 10-K, live from the day after filing for
    `valid_days`; NaN where no comparable pair exists."""
    cfg = spec["text"]
    e = pd.read_parquet(REPO / cfg["embeddings"]).sort_values("filed")
    e["filed"] = pd.to_datetime(e["filed"])
    out = pd.DataFrame(np.nan, index=days, columns=cols)
    for cik, g in e.groupby("cik"):
        v = np.stack(g["emb"].values)
        gap = g["filed"].diff().dt.days.values
        ok = (gap >= cfg["min_gap_days"]) & (gap <= cfg["max_gap_days"])
        sims = pd.Series(np.r_[np.nan, (v[1:] * v[:-1]).sum(axis=1)][ok],
                         index=(g["filed"] + pd.Timedelta(days=1)).values[ok])
        if sims.empty:
            continue
        sims = sims[~sims.index.duplicated(keep="last")]
        live = sims.reindex(days, method="ffill")
        seen = pd.Series(sims.index, index=sims.index).reindex(days, method="ffill")
        age = (pd.Series(days, index=days) - seen).dt.days
        live = live.where(age <= cfg["valid_days"])
        for sym in g["symbols"].iloc[-1].split(";"):
            if sym in out.columns:
                out[sym] = live.values
    return out


def llm_env(core: dict) -> dict[str, pd.DataFrame]:
    feed = core["feed"]
    close = feed["close"]
    bars = feed["bars"]
    wide = {f: bars[f][f].unstack("instrument").reindex(index=core["days"],
                                                        columns=close.columns)
            for f in ("open", "volume")}
    vq, qv = core["vq_parts"], core["qv_parts"]
    return {"close": close, "open": wide["open"], "volume": wide["volume"],
            "ret": close.pct_change(fill_method=None), "bm": vq["vq_bm"],
            "ep": vq["vq_ep"], "roe": vq["vq_roe"], **qv}


def screen_factors(spec: dict, core: dict, panels: dict[str, pd.DataFrame],
                   b50: pd.DataFrame, years: list[int]) -> dict:
    """Each January: rank-IC of every LLM factor vs the 21d next-open
    forward return on month-start days of the trailing `years`, using only
    samples whose forward window closed before the decision day."""
    cfg, h = spec["screen"], spec["fwd_h"]
    days, close, m = core["days"], core["feed"]["close"], core["member"]
    ms = P9.month_starts(days)
    fwd = P8.fwd_wide(close, 1, h).iloc[ms].where(m.iloc[ms])
    ic = {k: P9.rank_ic(v.iloc[ms], fwd) for k, v in panels.items()}
    cb = {k: P9.rank_ic(v.iloc[ms], b50.iloc[ms]) for k, v in panels.items()}
    pos = pd.Series(np.arange(len(days)), index=days)
    out = {}
    for y in years:
        yd = days[days.year == y]
        if not len(yd):
            continue
        t = int(pos[yd[0]])
        rows = {}
        for k in panels:
            pk = pos[ic[k].index].values
            use = (pk + h + 1 < t) & (pk >= t - 252 * cfg["years"])
            x, c = ic[k][use].dropna(), cb[k][use].dropna()
            n = len(x)
            tstat = float(x.mean() / x.std(ddof=1) * np.sqrt(n)) if n > 2 and x.std() > 0 else None
            rows[k] = {"n": n, "ic": float(x.mean()) if n else None, "t": tstat,
                       "corr_b50": float(c.mean()) if len(c) else None}
        ok = [k for k, r in rows.items() if r["n"] >= cfg["min_samples"] and r["t"]
              and r["t"] > cfg["min_t"] and abs(r["corr_b50"] or 0) < cfg["max_corr_b50"]]
        acc = sorted(ok, key=lambda k: -rows[k]["t"])[:cfg["max_accepted"]]
        out[y] = {"decided": str(yd[0].date()), "accepted": acc, "stats": rows}
    return out


def lf_comp(b50: pd.DataFrame, panels: dict, screen: dict, core: dict) -> pd.DataFrame:
    """1/2 B50 + 1/2 mean rank of the factors accepted for each day's year."""
    out = b50.copy()
    for y, r in screen.items():
        rows = core["days"].year == y
        if not r["accepted"]:
            continue
        llm = mean_rank([panels[k].loc[rows] for k in r["accepted"]], core["member"].loc[rows])
        ranks = {"b50": P9.row_rank(b50.loc[rows]), "llm": P9.row_rank(llm)}
        eq = pd.DataFrame({"b50": 0.5, "llm": 0.5}, index=b50.index[rows])
        out.loc[rows] = P9.weighted_blend(ranks, eq, ["b50"])
    return out.where(core["member"])


def tx_comp(spec: dict, b50: pd.DataFrame, sim: pd.DataFrame, member: pd.DataFrame
            ) -> pd.DataFrame:
    """B50 with the day's bottom-quintile (most changed) 10-Ks excluded."""
    low = member_rank(sim, member) <= spec["text"]["exclude_quantile"]
    return b50.where(~low)


def build_hybrid(spec: dict, core: dict, factors: list[dict]) -> dict:
    """Every composite the candidates need, over all of core's days."""
    m = core["member"]
    b50 = b50_honest(core)
    qv = qv_comp(core)
    ranks = {"mom": member_rank(core["mom"], m), "qv": qv,
             "vq": member_rank(mean_rank(list(core["vq_parts"].values()), m), m)}
    third = pd.DataFrame({"mom": 1 / 3, "vq": 1 / 3, "qv": 1 / 3}, index=core["days"])
    bq = P9.weighted_blend(ranks, third, ["mom", "vq"]).where(m)
    env = llm_env(core)
    log(f"{len(factors)} LLM factors")
    panels = {f["id"]: member_rank(D.evaluate(f["formula"], env), m) for f in factors}
    years = sorted({d.year for d in core["days"] if d.year >= spec["dev_start_year"]})
    screen = screen_factors(spec, core, panels, b50, years)
    sim = lazy_prices(spec, core["days"], core["feed"]["close"].columns)
    return {"comps": {"b50": b50, "qv": qv, "bq": bq,
                      "lf": lf_comp(b50, panels, screen, core),
                      "tx": tx_comp(spec, b50, sim, m)},
            "screen": screen,
            "tx_cover": float(sim.where(m).notna().sum(axis=1).div(m.sum(axis=1))
                              .loc[lambda x: x.index.year >= spec["dev_start_year"]].mean())}


# ---------------------------------------------------------------------------
# step 3: learning layer


def hedge_weights(shadows: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Month-start Hedge weights over sleeves from trailing shadow log
    returns (rows <= t), half-shrunk to equal weights; held for the month."""
    k = shadows.shape[1]
    lr = np.log1p(shadows.fillna(0.0)).rolling(cfg["lookback"], min_periods=cfg["lookback"]).sum()
    W = pd.DataFrame(np.nan, index=shadows.index, columns=shadows.columns)
    for t in P9.month_starts(shadows.index):
        row = lr.iloc[t]
        if row.notna().all():
            z = np.exp(cfg["eta"] * (row - row.max()))
            W.iloc[t] = cfg["shrink"] / k + (1 - cfg["shrink"]) * z / z.sum()
        else:
            W.iloc[t] = 1.0 / k
    return W.ffill()


def fallback_modes(r_h: pd.Series, r_b: pd.Series, cfg: dict) -> pd.Series:
    """True = fallback to B50: the hybrid shadow trailed the B50 shadow by
    more than `tol` over `win` days; back once the gap is >= 0. Switches
    hold >= min_hold days. Inputs at t are known at close t."""
    act = (r_h - r_b).rolling(cfg["win"], min_periods=cfg["win"]).sum()
    out, state, held = [], False, cfg["min_hold"]
    for a in act.values:
        held += 1
        if held >= cfg["min_hold"] and np.isfinite(a):
            if not state and a < -cfg["tol"]:
                state, held = True, 0
            elif state and a >= 0:
                state, held = False, 0
        out.append(state)
    return pd.Series(out, index=r_h.index)


def meta_plan(spec: dict, core: dict, sleeves: list[str], comp_of: dict,
              n_of: dict) -> dict:
    """Daily sleeve weights for a meta candidate: Hedge, reset to equal in
    stress, 100% B50_honest while the drift monitor is in fallback."""
    cfg = spec["meta"]
    close, days = core["feed"]["close"], core["days"]
    sh = pd.DataFrame({k: P9.shadow_sleeve(comp_of[k], close, n_of[k]) for k in sleeves})
    W = hedge_weights(sh, cfg)
    st = cfg["stress"]
    stress = MA.crash_exposure(core["feed"]["spy"], days, st["sma"], st["vol_win"],
                               st["vol_lookback"], st["vol_q"], 0.0, st["min_hold"]) == 0.0
    W.loc[stress.values] = 1.0 / len(sleeves)
    r_h = (W.shift(1) * sh).sum(axis=1, min_count=1)
    fb = fallback_modes(r_h, sh["B50_honest"], cfg["fallback"])
    used = W.copy()
    used.loc[fb.values] = 0.0
    used.loc[fb.values, "B50_honest"] = 1.0
    return {"weights": used, "stress": stress, "fallback": fb}


def mix_books(books: dict[str, dict], W: pd.DataFrame) -> dict:
    """Per-day weighted sum of sleeve books -> one book."""
    out = {}
    for d in W.index:
        acc: dict[str, float] = {}
        for k, w in W.loc[d].items():
            for s, v in books[k].get(d, {}).items():
                acc[s] = acc.get(s, 0.0) + w * v
        out[d] = {s: v for s, v in acc.items() if v > 1e-12}
    return out


def candidate_targets(spec: dict, core: dict, hy: dict) -> tuple[dict, dict]:
    """{name: {day: book}} for every candidate + meta diagnostics."""
    cands = {c["name"]: c for c in spec["candidates"]}
    tg, diag = {}, {}
    for c in spec["candidates"]:
        if c["kind"] == "hold":
            tg[c["name"]] = hold_targets(c["weights"], core["days"])
        elif c["kind"] == "sleeve":
            log(f"targets {c['name']}")
            tg[c["name"]] = sleeve_targets(spec, core, hy["comps"][c["comp"]], c["name"], c["n"])
    for c in spec["candidates"]:
        if c["kind"] == "meta":
            ks = c["sleeves"]
            plan = meta_plan(spec, core, ks, {k: hy["comps"][cands[k]["comp"]] for k in ks},
                             {k: cands[k]["n"] for k in ks})
            tg[c["name"]] = mix_books({k: tg[k] for k in ks}, plan["weights"])
            diag[c["name"]] = plan
    return tg, diag


def meta_summary(plan: dict, years: list[int]) -> dict:
    W, out = plan["weights"], {}
    for y in years:
        r = W.index.year == y
        if r.any():
            out[str(y)] = {"mean_weights": W.loc[r].mean().round(3).to_dict(),
                           "stress_days": float(plan["stress"].loc[r].mean()),
                           "fallback_days": float(plan["fallback"].loc[r].mean())}
    return out


def score_all(spec: dict, core: dict, tg: dict, names: list[str], eps: dict) -> dict:
    res = {}
    for k in names:
        log(k)
        res[k] = run_book(spec, core, tg[k], eps)
    return res


def score_proxies(spec: dict, core: dict, eps: dict) -> dict:
    res = {}
    for p in spec["proxies"]:
        sub = {e: P10.tradable_days({"sfeed": core["sfeed"]}, p["weights"], d)
               for e, d in eps.items()}
        sub = {e: d for e, d in sub.items() if len(d) >= 20}
        if sub:
            days = pd.DatetimeIndex(sorted(set().union(*sub.values())))
            res[p["name"]] = run_book(spec, core, hold_targets(p["weights"], days), sub)
    return res


def load_factors() -> list[dict]:
    return json.loads(LLM_OUT.read_text())["factors"]


def cmd_dev(spec: dict) -> dict:
    t0 = time.time()
    hstart = pd.Timestamp(spec["holdout"]["start"])
    core = build_core(spec, end_excl=hstart)
    assert core["days"][-1] < hstart, "holdout leaked into dev"
    core["qv_parts"] = fund_panels(spec, core["feed"], F.qv_features)
    factors = load_factors()
    hy = build_hybrid(spec, core, factors)
    eps = P10.dev_episodes(core["days"], spec["dev_start_year"])
    tg, diag = candidate_targets(spec, core, hy)
    names = [c["name"] for c in spec["candidates"]]
    res = score_all(spec, core, tg, names, eps)
    pres = score_proxies(spec, core, eps)
    picks = pd.concat([picks_frame(tg[k]) for k in names if k != "A_spy"])
    bad = pit.non_member_picks(picks.drop_duplicates(), core["win"])
    assert bad.empty, f"non-member picks:\n{bad.head()}"

    prior = stage_trials("baseline")
    assert prior > 0, "run the baseline stage first"
    n_windows = sum(len(r["by_episode"]) for r in res.values())
    n_screen = len(factors) * sum(1 for y in hy["screen"] if str(y) in eps)
    n = prior + n_windows + n_screen
    srs = phase10_dev_sharpes() + [MA.perf(r["daily"], core["rf"])["sharpe"]
                                   for r in res.values()]
    summ = {}
    for k, r in {**res, **pres}.items():
        s = P10.summarise(r, core["spy_r"], core["rf"])
        s["dsr"] = P10.dsr(r["daily"], core["rf"], n, srs) if k in res else None
        s["by_episode"] = r["by_episode"]
        summ[k] = s
    fin = pick_finalists(spec, summ)
    pd.DataFrame({k: r["daily"] for k, r in {**res, **pres}.items()}) \
        .to_csv(DEV_DAILY, float_format="%.10g")
    ts = dt.datetime.now().isoformat()
    write_log([{"ts": ts, "market": "us", "git_sha": git_sha(), "spec": SPEC_TAG,
                "stage": "dev", "candidate": f"phase11:{k}",
                "re_viewed_holdout": False, "llm_dev_contaminated": bool(
                    next(c for c in spec["candidates"] if c["name"] == k).get("llm")),
                **{kk: summ[k][kk] for kk in ("ar", "sharpe", "mdd", "n_days")},
                "dsr": summ[k]["dsr"], "n_trials": n} for k in res], "dev")
    yrs = [int(e) for e in eps]
    out = {"stage": "phase11_dev", "git_sha": git_sha(), "ts": ts,
           "dev_window": [str(core["days"][0].date()), str(core["days"][-1].date())],
           "holdout": spec["holdout"], "episodes": list(eps),
           "prior_trials": prior, "n_dev_windows": n_windows,
           "n_screen_trials": n_screen, "n_trials": n,
           "summary": summ, "finalists": fin,
           "screen": {str(y): v for y, v in hy["screen"].items()},
           "tx_coverage": hy["tx_cover"],
           "meta": {k: meta_summary(p, yrs) for k, p in diag.items()},
           "proxy_windows": P10.proxy_windows(spec, {**res, **pres}, core["rf"],
                                              ["B50_honest"] + fin["finalists"]),
           "runtime_s": round(time.time() - t0, 1)}
    DEV_OUT.write_text(json.dumps(out, indent=1, default=str))
    log(f"n_trials = {prior} + {n_windows} windows + {n_screen} screens = {n}; "
        f"finalists {fin['finalists']}; wrote {DEV_OUT} ({out['runtime_s']}s)")
    return out


def pick_finalists(spec: dict, summ: dict) -> dict:
    cfg, base = spec["finalists"], spec["adoption"]["baseline"]
    pool = [c["name"] for c in spec["candidates"] if c["name"] not in ("A_spy", base)]
    ok = [k for k in pool if summ[k]["mdd"] >= summ[base]["mdd"] - cfg["mdd_tol"]]
    ranked = sorted(ok, key=lambda k: -summ[k]["sharpe"])
    fins = ranked[:cfg["k"]] + [k for k in cfg["always"] if k not in ranked[:cfg["k"]]]
    return {"rule": f"top {cfg['k']} by dev Sharpe with dev MDD >= {base} - "
                    f"{cfg['mdd_tol']}, plus {cfg['always']}",
            "eligible_ranked": ranked, "excluded_mdd": sorted(set(pool) - set(ok)),
            "finalists": fins}


def cmd_holdout(spec: dict) -> dict:
    if HOLD_OUT.exists():
        sys.exit(f"{HOLD_OUT} exists — the holdout is scored ONCE (predeclared)")
    dev = json.loads(DEV_OUT.read_text())
    fins = dev["finalists"]["finalists"]
    base = spec["adoption"]["baseline"]
    t0 = time.time()
    core = build_core(spec, end_excl=None)
    core["qv_parts"] = fund_panels(spec, core["feed"], F.qv_features)
    factors = load_factors()
    hy = build_hybrid(spec, core, factors)
    h0, h1 = pd.Timestamp(spec["holdout"]["start"]), pd.Timestamp(spec["holdout"]["end"])
    hd = core["days"][(core["days"] >= h0) & (core["days"] <= h1)]
    eps = {"holdout": hd}
    tg, diag = candidate_targets(spec, core, hy)
    names = list(dict.fromkeys(["A_spy", base] + fins))
    res = score_all(spec, core, tg, names, eps)
    pres = score_proxies(spec, core, eps)
    picks = pd.concat([picks_frame({d: tg[k][d] for d in hd}) for k in names if k != "A_spy"])
    assert pit.non_member_picks(picks.drop_duplicates(), core["win"]).empty

    new_screens = len(factors) * sum(1 for y in hy["screen"] if y > int(dev["episodes"][-1]))
    hwin = sum(len(r["by_episode"]) for r in res.values())
    n = dev["n_trials"] + hwin + new_screens
    dd = pd.read_csv(DEV_DAILY, index_col=0, parse_dates=True)
    hs, full, fd = {}, {}, {}
    for k, r in {**res, **pres}.items():
        hs[k] = P10.summarise(r, core["spy_r"], core["rf"])
        fd[k] = pd.concat([dd[k].dropna(), r["daily"]]).sort_index()
    srs = phase10_dev_sharpes() + [MA.perf(fd[k], core["rf"])["sharpe"] for k in res]
    srs_xs = [MA.perf(fd[k], core["spy_r"])["sharpe"] for k in res if k != "A_spy"] + [0.0]
    for k, d in fd.items():
        f = MA.perf(d, core["rf"])
        xs = (d - core["spy_r"].reindex(d.index)).dropna()
        f["xs_spy_ar"] = float(xs.mean() * 252)
        if k in res:
            f["dsr"] = P10.dsr(d, core["rf"], n, srs)
            f["dsr_xs_spy"] = None if k == "A_spy" else P10.dsr(d, core["spy_r"], n, srs_xs)
        full[k] = f
    bd = full[base]["dsr"]["DSR"]
    checks = {}
    for k in fins:
        kd = full[k]["dsr"]["DSR"]
        c = {"holdout_sharpe_gt_base": hs[k]["sharpe"] > hs[base]["sharpe"],
             "dsr_ok": kd >= spec["adoption"]["min_dsr"] and kd >= bd, "dsr": kd,
             "proxies_beaten": [p for p in pres if hs[k]["sharpe"] > hs[p]["sharpe"]]}
        c["adopt"] = c["holdout_sharpe_gt_base"] and c["dsr_ok"]
        checks[k] = c
    win = [k for k, c in checks.items() if c["adopt"]]
    hy_name = "HY_full"
    plan = diag[hy_name]
    last = core["days"][-1]
    state = {"asof": str(last.date()), "mode": "fallback_b50" if plan["fallback"].iloc[-1]
             else ("stress_equal" if plan["stress"].iloc[-1] else "hedge"),
             "weights": plan["weights"].iloc[-1].round(4).to_dict(),
             "book": {s: round(v, 4) for s, v in sorted(tg[hy_name][last].items())},
             "llm_factors_live": hy["screen"].get(last.year, {}).get("accepted", []),
             "live_effect": "shadow"}
    ts = dt.datetime.now().isoformat()
    write_log([{"ts": ts, "market": "us", "git_sha": git_sha(), "spec": SPEC_TAG,
                "stage": "holdout", "candidate": f"phase11:{k}",
                "re_viewed_holdout": k in ("A_spy", base),
                **{kk: hs[k][kk] for kk in ("ar", "sharpe", "mdd", "n_days")},
                "dsr": full[k]["dsr"], "n_trials": n} for k in res], "holdout")
    out = {"stage": "phase11_holdout", "git_sha": git_sha(), "ts": ts,
           "holdout_window": [str(hd[0].date()), str(hd[-1].date())],
           "finalists": fins, "n_trials": n, "n_holdout_windows": hwin,
           "n_new_screen_trials": new_screens, "holdout": hs, "full_sample": full,
           "adoption": {"baseline": base, "baseline_dsr": bd, "checks": checks,
                        "adopted": max(win, key=lambda k: checks[k]["dsr"]) if win else None},
           "screen_2026": hy["screen"].get(2026),
           "meta_holdout": {k: meta_summary(p, [2025, 2026]) for k, p in diag.items()},
           "state_at_end": state, "runtime_s": round(time.time() - t0, 1)}
    HOLD_OUT.write_text(json.dumps(out, indent=1, default=str))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1))
    log(f"wrote {HOLD_OUT}: adopted={out['adoption']['adopted']} ({out['runtime_s']}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["universe", "sec", "baseline", "dev", "holdout"])
    ap.add_argument("--out", type=Path, default=REPO / "build" / "universe_sp500.csv")
    args = ap.parse_args()
    spec = load_spec()
    if args.cmd == "universe":
        cmd_universe(spec, args.out)
    elif args.cmd == "sec":
        cmd_sec(spec)
    elif args.cmd == "baseline":
        cmd_baseline(spec)
    elif args.cmd == "dev":
        cmd_dev(spec)
    else:
        cmd_holdout(spec)


if __name__ == "__main__":
    main()
