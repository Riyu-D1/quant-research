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
from quant_research import fundamentals as F  # noqa: E402
from quant_research import micro as mc  # noqa: E402
from quant_research import multiasset as MA  # noqa: E402
from quant_research import pit  # noqa: E402

SPEC_PATH = REPO / "configs" / "experiments" / "phase11_us.yaml"
OUT = REPO / "results" / "phase11"
AUDIT = OUT / "membership_audit.json"
BASE_OUT = OUT / "baseline_dev.json"
PICKS = OUT / "b50_honest_picks.csv"
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


def cmd_sec(spec: dict) -> None:
    """One companyfacts request per universe CIK -> vq + tilt fundamentals
    and dei shares outstanding (same PIT `filed` semantics as phase 3/4)."""
    w = load_membership(REPO / spec["membership_csv"])
    syms = set(w["symbol"])
    cmap = pd.read_parquet(REPO / spec["cik_map"])[["symbol", "cik"]].dropna()
    cmap["symbol"] = cmap["symbol"].map(ev.canon)
    ed = E.Edgar()
    tick = E.ticker_cik_map(ed)
    tick["symbol"] = tick["ticker"].map(ev.canon)
    extra = tick[tick["symbol"].isin(syms - set(cmap["symbol"]))][["symbol", "cik"]]
    cmap = pd.concat([cmap, extra]).drop_duplicates("symbol")
    cmap = cmap[cmap["symbol"].isin(syms)]
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
    vq_parts = P9.load_vq_parts(spec, feed)
    cmap = mc.liquidity_cost_map(sfeed["bars"]["close"], sfeed["bars"]["volume"],
                                 flat_bps=spec["costs"]["flat_bps"])
    return {"feed": feed, "sfeed": sfeed, "days": days, "member": member,
            "win": win, "mom": mom, "vq_parts": vq_parts, "cmap": cmap,
            "smap": P8.load_sector_map(spec),
            "rf": sfeed["close"][spec["cash_symbol"]].pct_change(),
            "spy_r": feed["spy"].pct_change()}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["universe", "sec", "baseline"])
    ap.add_argument("--out", type=Path, default=REPO / "build" / "universe_sp500.csv")
    args = ap.parse_args()
    spec = load_spec()
    if args.cmd == "universe":
        cmd_universe(spec, args.out)
    elif args.cmd == "sec":
        cmd_sec(spec)
    else:
        cmd_baseline(spec)


if __name__ == "__main__":
    main()
