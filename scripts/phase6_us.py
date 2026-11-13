#!/usr/bin/env python3
"""Phase-6 US signal selection + one-shot holdout runner.

select : build every spec'd signal over the PIT S&P500 universe, score
         each signal × portfolio config on the OOF folds (pre-2025 only),
         build the predeclared signal combos (equal / EW-IC rank blends),
         pick the winner by net OOF IR with a per-fold stability filter,
         write results/phase6_us_select.json + the candidate table.

holdout: re-score ONLY the pinned winner on the 2025+ window (labelled
         re-viewed holdout — this window has been looked at before),
         compare against SPY, apply the live gate, and — only if it
         passes — flip configs/live_signal.yaml validated: true.

Honesty rules: n_trials counts every evaluated config including failures;
label is the store's LABEL0 (close[t+2]/close[t+1]-1) which embeds the
next-session fill delay; costs are turnover × cost_bps per side.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quant_research import events as ev
from quant_research import signals6 as s6
from quant_research import walkforward as wf

REPO = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[phase6 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_spec() -> dict:
    p = REPO / "configs" / "experiments" / "phase6_us.yaml"
    return yaml.safe_load(p.read_text())


def load_universe(provider: Path) -> tuple[list[str], pd.DataFrame]:
    inst = ev.load_instruments(provider, "sp500")
    return inst["instrument"].tolist(), inst


def member_mask(cal: pd.DatetimeIndex, inst: pd.DataFrame) -> pd.DataFrame:
    """(date × symbol) bool — was the name in the PIT S&P500 that day?"""
    m = pd.DataFrame(False, index=cal, columns=inst["instrument"])
    for r in inst.itertuples():
        a = pd.Timestamp(r.start) if pd.notna(r.start) else cal[0]
        b = pd.Timestamp(r.end) if pd.notna(r.end) else cal[-1]
        m.loc[(m.index >= a) & (m.index <= b), r.instrument] = True
    return m


def build_label(close: pd.DataFrame) -> pd.Series:
    """LABEL0 = close[t+2]/close[t+1]-1 on the (datetime, instrument) grid."""
    y = close.shift(-2) / close.shift(-1) - 1.0
    s = y.stack()
    s.index.names = ["datetime", "instrument"]
    return s


def portfolio_returns(score: pd.Series, label: pd.Series,
                      eligible: pd.DataFrame, topk: int, hold: int,
                      cost_bps: float) -> pd.DataFrame:
    """Turnover-smoothed top-k: each day the portfolio is the mean of the
    last `hold` days' top-k equal-weight targets (TopkDropout-style).
    Return at day t earns label[t] (already next-session delayed)."""
    df = score.rename("s").to_frame().join(label.rename("y"), how="left")
    df = df.reset_index()
    days = sorted(df["datetime"].unique())
    elig = eligible.reindex(days).fillna(False)
    piv = df.pivot(index="datetime", columns="instrument", values="s")
    piv = piv.reindex(days).where(elig)
    ypiv = df.pivot(index="datetime", columns="instrument", values="y") \
             .reindex(days)
    target = piv.rank(axis=1, ascending=False, pct=True) \
        .le(topk / elig.sum(axis=1).clip(lower=1), axis=0)
    # equal weight among top-k selected
    target = target.div(target.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = target.rolling(hold, min_periods=1).mean()
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gross = (w * ypiv.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()   # day-0 build-up trades too
    cost = turn * cost_bps / 1e4
    return pd.DataFrame({"gross": gross, "net": gross - cost,
                         "turnover": turn}, index=pd.DatetimeIndex(days))


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return {"ann_ret": 0.0, "IR": 0.0, "mdd": 0.0}
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return {"ann_ret": float(r.mean() * 252),
            "IR": float(r.mean() / r.std() * np.sqrt(252)),
            "mdd": mdd}


def by_year(r: pd.Series) -> dict:
    return {str(y): float(v.mean() * 252)
            for y, v in r.groupby(r.index.year)}


def ewic_blend(signals: dict[str, pd.Series], names: list[str],
               label: pd.Series, ref_days: pd.DatetimeIndex,
               apply_days: pd.DatetimeIndex) -> pd.Series:
    """EW rank-IC weights on ref_days, applied to apply_days' ranks."""
    ics = {}
    for n in names:
        s = signals[n]
        df = pd.concat({"p": s, "y": label.reindex(s.index)}, axis=1).dropna()
        df = df[df.index.get_level_values("datetime").isin(ref_days)]
        if df.empty:
            continue
        ics[n] = df.groupby(level="datetime").apply(
            lambda g: g["p"].corr(g["y"], method="spearman")).mean()
    w = pd.Series(ics).clip(lower=0.0)
    w = w / w.sum() if w.sum() > 0 else \
        pd.Series(1.0 / len(names), index=names)
    parts = []
    for n in w.index:
        s = signals[n]
        s = s[s.index.get_level_values("datetime").isin(apply_days)]
        parts.append(wf.cs_rank(s).rename(n))
    if not parts:
        return s6.empty()
    R = pd.concat(parts, axis=1).fillna(0.5)
    return (R * w).sum(axis=1).rename("score")


def eq_blend(signals: dict, names: list[str],
             apply_days: pd.DatetimeIndex) -> pd.Series:
    parts = []
    for n in names:
        s = signals[n]
        s = s[s.index.get_level_values("datetime").isin(apply_days)]
        parts.append(wf.cs_rank(s).rename(n))
    if not parts:
        return s6.empty()
    return pd.concat(parts, axis=1).fillna(0.5).mean(axis=1).rename("score")


def cmd_select(spec: dict, out: Path) -> dict:
    provider = REPO / spec["provider"]
    cal = ev.load_calendar(provider)
    symbols, inst = load_universe(provider)
    log(f"universe {len(symbols)} names, cal {cal[0].date()}..{cal[-1].date()}")
    close, volume = s6.load_panels(provider, symbols, cal)
    elig = member_mask(cal, inst)
    label = build_label(close)
    sector = {}
    smap = REPO / "configs" / "benchmarks" / "assets" / "sp500_sector_map.csv"
    if smap.exists():
        sm = pd.read_csv(smap)
        sector = dict(zip(sm["symbol"].map(ev.canon), sm["sector"]))
    sigs = s6.build_signals(
        provider, symbols, cal, spec["signals"],
        fundamentals_path=REPO / spec.get("fundamentals",
                                          "data/events/fundamentals.parquet"),
        feats_path=REPO / spec.get("event_features",
                                   "data/events/features_us.parquet"),
        news_dir=REPO / spec.get("news_dir", "results"),
        sector_map=sector, close=close, volume=volume)
    oof_days = cal[(cal >= pd.Timestamp(spec["oof"][0])) &
                   (cal <= pd.Timestamp(spec["oof"][1]))]
    fold_days = [(cal[(cal >= pd.Timestamp(a)) & (cal <= pd.Timestamp(b))])
                 for a, b in spec["folds"]]

    rows, n_trials = [], 0
    cand_scores = {n: s for n, s in sigs.items()}
    # ---- combos (predeclared in the spec — count them too) ---------------
    for c in spec.get("combos", []):
        names = [n for n in c["signals"] if n in sigs]
        if not names:
            continue
        if c["weight"] == "ewic":
            # EW-IC weights from the first fold applied forward — fold-causal
            blocks = []
            for i, fd in enumerate(fold_days):
                ref = fold_days[i - 1] if i else fd[:0]
                blocks.append(ewic_blend(sigs, names, label, ref, fd))
            cand_scores[c["name"]] = pd.concat(blocks).sort_index()
        else:
            cand_scores[c["name"]] = eq_blend(
                sigs, names, cal[cal <= pd.Timestamp(spec["oof"][1])])
    log(f"evaluating {len(cand_scores)} signals/combos "
        f"x {len(spec['portfolios'])} portfolios")
    for sname, score in cand_scores.items():
        if score.empty:
            rows.append({"signal": sname, "note": "no coverage"})
            n_trials += len(spec["portfolios"])
            continue
        sm = wf.signal_metrics(
            score[score.index.get_level_values("datetime").isin(oof_days)],
            label)
        for port in spec["portfolios"]:
            n_trials += 1
            r = portfolio_returns(score, label, elig, port["topk"],
                                  port["hold"], port["cost_bps"])
            roof = r.loc[r.index.intersection(oof_days)]
            st = perf_stats(roof["net"])
            fold_irs = [perf_stats(r.loc[r.index.intersection(fd)]["net"])["IR"]
                        for fd in fold_days]
            rows.append({"signal": sname, "portfolio": port["name"],
                         **{k: sm.get(k) for k in
                            ("RankIC", "RankICIR")},
                         **st, "turnover": float(roof["turnover"].mean()),
                         "fold_irs": fold_irs,
                         "pos_folds": int(sum(x > 0 for x in fold_irs)),
                         "by_year": by_year(roof["net"])})
            log(f"  {sname}/{port['name']}: IR {st['IR']:+.3f} "
                f"RankIC {sm.get('RankIC') or 0:+.4f} "
                f"pos_folds {sum(x > 0 for x in fold_irs)}/{len(fold_days)}")
    valid = [r for r in rows
             if r.get("IR") is not None and
             r.get("pos_folds", 0) >= max(2, len(fold_days) // 2 + 1)]
    valid.sort(key=lambda r: r["IR"], reverse=True)
    winner = valid[0] if valid else None
    summary = {"stage": "phase6_us_select", "n_trials": n_trials,
               "winner": winner, "rows": rows,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    log(f"winner: {winner and (winner['signal'], winner['portfolio'], winner['IR'])}")
    return summary


def cmd_holdout(spec: dict, select_path: Path, out: Path) -> dict:
    sel = json.loads(select_path.read_text())
    winner = sel.get("winner")
    if not winner:
        sys.exit("no winner from select — nothing to score on holdout")
    provider = REPO / spec["provider"]
    cal = ev.load_calendar(provider)
    symbols, inst = load_universe(provider)
    close, volume = s6.load_panels(provider, symbols, cal)
    elig = member_mask(cal, inst)
    label = build_label(close)
    sector = {}
    smap = REPO / "configs" / "benchmarks" / "assets" / "sp500_sector_map.csv"
    if smap.exists():
        sm = pd.read_csv(smap)
        sector = dict(zip(sm["symbol"].map(ev.canon), sm["sector"]))
    port = next(p for p in spec["portfolios"]
                if p["name"] == winner["portfolio"])
    combo = next((c for c in spec.get("combos", [])
                  if c["name"] == winner["signal"]), None)
    need = combo["signals"] if combo else [winner["signal"]]
    sigs = s6.build_signals(
        provider, symbols, cal, need,
        fundamentals_path=REPO / spec.get("fundamentals",
                                          "data/events/fundamentals.parquet"),
        feats_path=REPO / spec.get("event_features",
                                   "data/events/features_us.parquet"),
        news_dir=REPO / spec.get("news_dir", "results"),
        sector_map=sector, close=close, volume=volume)
    hdays = cal[(cal >= pd.Timestamp(spec["holdout"][0])) &
                (cal <= pd.Timestamp(spec["holdout"][1]))]
    oof_days = cal[(cal >= pd.Timestamp(spec["oof"][0])) &
                   (cal <= pd.Timestamp(spec["oof"][1]))]
    if combo:
        if combo["weight"] == "ewic":
            score = ewic_blend(sigs, combo["signals"], label, oof_days, hdays)
        else:
            score = eq_blend(sigs, combo["signals"], hdays)
    else:
        score = sigs[winner["signal"]]
    r = portfolio_returns(score, label, elig, port["topk"], port["hold"],
                          port["cost_bps"])
    rh = r.loc[r.index.intersection(hdays)]
    stats = perf_stats(rh["net"])
    spy = ev.load_field(provider, "SPY", cal, "close")
    spy_r = pd.Series(spy, index=cal).pct_change().loc[
        rh.index.intersection(hdays)]
    spy_stats = perf_stats(spy_r)
    rets = rh["net"].dropna()
    trial_sharpes = [float(r_["IR"]) for r_ in sel.get("rows", [])
                     if r_.get("IR") is not None]
    dsr = wf.deflated_sharpe(rets, n_trials=sel.get("n_trials", 1),
                             trial_sharpes=trial_sharpes)
    dsr_val = float(dsr["DSR"])
    gate = {"positive": stats["ann_ret"] > 0,
            "beats_spy_ir": stats["IR"] > spy_stats["IR"],
            "dsr": dsr_val > 0.5}
    summary = {"stage": "phase6_us_holdout (re-viewed)", "winner": winner,
               "n_trials": sel.get("n_trials"), "holdout": stats,
               "spy": spy_stats, "dsr": dsr, "gate": gate,
               "turnover": float(rh["turnover"].mean()),
               "by_year": by_year(rh["net"]),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.write_text(json.dumps(summary, indent=1))
    if all(gate.values()):
        _wire_live(winner, combo, spec)
    else:
        log("gate FAILED — live_signal.yaml left at validated: false")
    log(f"holdout IR {stats['IR']:+.3f} vs SPY {spy_stats['IR']:+.3f} "
        f"DSR {dsr_val:.3f} gate {gate}")
    return summary


_LIVE_COMPUTABLE = {"mom", "rev", "srev5", "mom6", "lowvol", "beta",
                    "vq", "news"}


def _wire_live(winner: dict, combo: dict | None, spec: dict) -> None:
    """Gate passed: pin the winner into configs/live_signal.yaml.

    Only components the live bot can compute are kept; a winner whose
    signals are all snapshot/model-based cannot be wired and is logged
    instead of silently validated."""
    cfg_path = REPO / "configs" / "live_signal.yaml"
    need = combo["signals"] if combo else [winner["signal"]]
    live_sigs = [s for s in need
                 if s.removeprefix("sn_") in _LIVE_COMPUTABLE
                 and not s.startswith("sn_")]
    if not live_sigs:
        log("gate passed but no live-computable signal components — "
            "manual wiring needed; leaving validated: false")
        return
    # targeted line edits — a full yaml round-trip would strip the
    # documented pin rationale and risk comments.
    import re
    text = cfg_path.read_text()
    weighting = "ewic" if combo and combo["weight"] == "ewic" else "equal"
    text = re.sub(r"^signals: .*$", f"signals: {live_sigs}",
                  text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^weighting: .*$", f"weighting: {weighting}",
                  text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^validated: .*$", "validated: true",
                  text, count=1, flags=re.MULTILINE)
    text = ("# Phase-6 pin: {}/{} re-viewed holdout IR {:+.3f}\n".format(
                winner["signal"], winner["portfolio"],
                winner.get("IR") or 0.0) + text)
    cfg_path.write_text(text)
    log(f"live_signal.yaml updated: signals={live_sigs} validated: true")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "holdout"])
    ap.add_argument("--select", default="results/phase6_us_select.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = load_spec()
    if args.cmd == "select":
        cmd_select(spec, Path(args.out or "results/phase6_us_select.json"))
    else:
        cmd_holdout(spec, Path(args.select),
                    Path(args.out or "results/phase6_us_holdout.json"))


if __name__ == "__main__":
    main()
