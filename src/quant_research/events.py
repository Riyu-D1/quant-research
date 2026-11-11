"""Phase-3: disclosure event tables -> daily point-in-time feature panel.

Honesty contract (enforced by tests/test_pit.py):
  a feature at trading day D uses ONLY events whose *information date*
  (EDGAR filing date, House filing date) is a trading day STRICTLY BEFORE D.
  A filing on day D is never visible to the day-D signal; a filing on a
  non-trading day is visible from the next trading day.

Canonical symbol form is the qlib/Alpaca instrument name: uppercase with
'-' class separators (BRK-B). EDGAR trading symbols use the same form;
the universe CSV's '.' form is converted on read.

Event sources -> features (windows are calendar days, trailing):
  insider (Form 3/4/5 bulk):
    ins_buy_usd_90d    Σ open-market purchase $ (code P), 90d
    ins_sell_usd_90d   Σ open-market sale $ (code S) excl. 10b5-1-flagged
                       filings, 90d
    ins_net_usd_90d    buy − sell, 90d
    ins_nbuyers_90d    distinct owners with a P txn, 90d
    ins_nsellers_90d   distinct owners with an S txn (unflagged), 90d
    ins_cluster_30d    distinct P owners, 30d (cluster-buy count)
    ins_role_net_90d   net $ signed P+/S− weighted by role
                       (exec-officer 2.0, officer 1.5, director 1.0,
                        10%-owner 0.75, other 0.5), 90d
    ins_net_mcap_90d   ins_net_usd_90d / market cap (latest-filed shares × close)
    ins_days_since_p   trading days since last P filing (cap 250)
  13F (filing date keyed; per-fund latest-known holdings):
    f13f_nholders      # tracked funds holding the stock (latest-known)
    f13f_nholders_chg  vs 91d ago
    f13f_wsum          Σ portfolio weights across funds (latest-known)
    f13f_wsum_chg      vs 91d ago
    f13f_new_91d       # funds whose holding first appeared ≤91d ago
  congress (House PTRs):
    cong_nbuy_45d      # purchases, 45d
    cong_nsell_45d     # sales, 45d
    cong_net_usd_45d   signed $ at amount-range midpoint, 45d
  rule signals (used directly as trade signals, not model features):
    rule_cluster_hold20/60   a cluster-buy event (ins_cluster_30d≥2) held N td
    rule_ins_buy_hold60      any insider open-market buy in last 60 td
    rule_cong_buy_hold60     any House purchase in last 60 td

LABEL0 = close[t+2]/close[t+1]-1 on the dump's adjusted closes — identical
to the Alpha158 label expression; emitted so the events-only handler can
train without Alpha158DL.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ins_buy_usd_90d", "ins_sell_usd_90d", "ins_net_usd_90d",
    "ins_nbuyers_90d", "ins_nsellers_90d", "ins_cluster_30d",
    "ins_role_net_90d", "ins_net_mcap_90d", "ins_days_since_p",
    "f13f_nholders", "f13f_nholders_chg", "f13f_wsum", "f13f_wsum_chg",
    "f13f_new_91d",
    "cong_nbuy_45d", "cong_nsell_45d", "cong_net_usd_45d",
]
RULE_COLS = ["rule_cluster_hold20", "rule_cluster_hold60",
             "rule_ins_buy_hold60", "rule_cong_buy_hold60"]
ALL_COLS = FEATURE_COLS + RULE_COLS


def canon(sym: str) -> str:
    """Any vendor form -> canonical qlib name: upper, '-' class separator."""
    return str(sym).upper().strip().replace(".", "-").replace("_", "-") \
        .replace("/", "-")


# ---------------------------------------------------------------------------
# bin reading (same layout as momentum_probe.py)


def load_calendar(data_dir: Path) -> pd.DatetimeIndex:
    cal = pd.read_csv(data_dir / "calendars/day.txt", header=None,
                      names=["d"], sep=r"\s+")["d"]
    return pd.DatetimeIndex(pd.to_datetime(cal))


def load_close(data_dir: Path, sym: str, cal: pd.DatetimeIndex) -> np.ndarray:
    f = data_dir / "features" / str(sym).lower() / "close.day.bin"
    full = np.full(len(cal), np.nan, dtype=np.float64)
    if f.exists():
        a = np.fromfile(f, dtype=np.float32)
        if len(a) > 1:
            s = int(a[0])
            vals = a[1:]
            e = min(s + len(vals), len(cal))
            if e > s:
                full[s:e] = vals[: e - s]
    return full


def load_instruments(data_dir: Path, name: str) -> pd.DataFrame:
    rows = []
    for line in (data_dir / "instruments" / f"{name}.txt").read_text().splitlines():
        if not line.strip():
            continue
        sym, a, b = line.split("\t")
        rows.append({"instrument": canon(sym), "start": a, "end": b})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# name normalization for issuer/asset matching


_STOP = re.compile(
    r"\b(INC|CORP(ORATION)?|CO|COMPANY|LLC|L\.L\.C|LTD|L\.P\.|LP|PLC|"
    r"HLDGS|HOLDINGS|GROUP|COM|CL|CLASS|NEW|REGISTERED|SHS|"
    r"ORD(INARY)?\s*(SHS|SHARES)?|NV|SA|AG|SE|DELAWARE|DEL|THE)\b")


def norm_name(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"[.,&'/\-()]", " ", s)
    s = _STOP.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def class_letter(s: str) -> str | None:
    m = re.search(r"\bCL(?:ASS)?\s*[- ]?([ABC])\b", str(s).upper()) or \
        re.search(r"\bSER(?:IES)?\s*[- ]?([ABC])\b", str(s).upper())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# rolling "was the event known within the last W calendar days" machinery
#
# An event with info_date fd is visible on trading days d with
# fd < d <= fd + W.  searchsorted(side='right') -> first day strictly after fd.


def _window_sum(days: pd.DatetimeIndex,
                contrib: pd.DataFrame, wdays: int) -> np.ndarray:
    """contrib: (info_date, value) -> per-day trailing-window sum."""
    if contrib.empty:
        return np.zeros(len(days))
    d64 = days.values.astype("datetime64[D]")
    fd = contrib["info_date"].values.astype("datetime64[D]")
    v = contrib["value"].values.astype(float)
    s = np.searchsorted(d64, fd, side="right")
    e = np.searchsorted(d64, fd + np.timedelta64(wdays, "D"), side="right")
    diff = np.zeros(len(days) + 1)
    np.add.at(diff, s, v)
    np.add.at(diff, e, -v)
    return np.cumsum(diff)[: len(days)]


def _window_count_distinct(days: pd.DatetimeIndex,
                           ev: pd.DataFrame, wdays: int) -> np.ndarray:
    """Distinct-owner count: ev has (info_date, owner); each owner covers the
    merged union of their [fd+1, fd+W) intervals."""
    if ev.empty:
        return np.zeros(len(days))
    d64 = days.values.astype("datetime64[D]")
    diff = np.zeros(len(days) + 1)
    for _, g in ev.groupby("owner"):
        fds = np.sort(g["info_date"].values.astype("datetime64[D]"))
        s = np.searchsorted(d64, fds, side="right")
        e = np.searchsorted(d64, fds + np.timedelta64(wdays, "D"),
                            side="right")
        cur_s, cur_e = s[0], e[0]
        for i in range(1, len(s)):
            if s[i] <= cur_e:
                cur_e = max(cur_e, e[i])
            else:
                diff[cur_s] += 1
                diff[cur_e] -= 1
                cur_s, cur_e = s[i], e[i]
        diff[cur_s] += 1
        diff[cur_e] -= 1
    return np.cumsum(diff)[: len(days)]


def _days_since(days: pd.DatetimeIndex, info_dates) -> np.ndarray:
    """Trading days since the latest *visible* event (info_date < D); 250
    when none is visible yet."""
    if len(info_dates) == 0:
        return np.full(len(days), 250.0)
    d64 = days.values.astype("datetime64[D]")
    fd = np.sort(pd.to_datetime(info_dates).values.astype("datetime64[D]"))
    first_seen = np.searchsorted(d64, fd, side="right")   # day idx of fd+1
    marker = np.full(len(days), -1.0)
    marker[np.clip(first_seen, 0, len(days) - 1)] = first_seen
    anchor = np.maximum.accumulate(marker)
    dist = np.where(anchor >= 0, np.arange(len(days)) - anchor, 250)
    return np.minimum(dist.astype(float), 250.0)


# ---------------------------------------------------------------------------
# per-source builders


def role_weight(role, title) -> float:
    role = str(role or "")
    title = str(title or "").upper()
    w = 0.5
    if "TenPercentOwner" in role:
        w = 0.75
    if "Director" in role:
        w = max(w, 1.0)
    if "Officer" in role:
        w = max(w, 2.0 if re.search(r"CEO|CFO|CHIEF|PRESIDENT|CHAIRMAN|COO",
                                  title) else 1.5)
    return w


def insider_features(ins: pd.DataFrame, days: pd.DatetimeIndex,
                     shares: pd.DataFrame | None,
                     close: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if len(ins):
        ins = ins.assign(
            filing_date=pd.to_datetime(ins["filing_date"],
                                       errors="coerce")
        ).dropna(subset=["filing_date"])
    if ins.empty:
        out = {c: np.zeros(len(days)) for c in FEATURE_COLS if c.startswith("ins_")}
        out["ins_days_since_p"] = np.full(len(days), 250.0)
        out["ins_net_mcap_90d"] = np.zeros(len(days))
        return out

    fd = ins["filing_date"]
    code = ins["code"].astype(str).str.upper()
    usd = pd.to_numeric(ins["usd"], errors="coerce").fillna(0.0)
    p = (code == "P")
    s_unflag = (code == "S") & (~ins["is10b5_filing"].fillna(False).astype(bool))

    def contrib(mask, val):
        return pd.DataFrame({"info_date": pd.to_datetime(fd[mask]),
                             "value": np.asarray(val)[np.asarray(mask)]})

    out["ins_buy_usd_90d"] = _window_sum(days, contrib(p, usd), 90)
    out["ins_sell_usd_90d"] = _window_sum(days, contrib(s_unflag, usd), 90)
    out["ins_net_usd_90d"] = out["ins_buy_usd_90d"] - out["ins_sell_usd_90d"]
    evp = ins.loc[p].rename(columns={"filing_date": "info_date"})
    evs = ins.loc[s_unflag].rename(columns={"filing_date": "info_date"})
    out["ins_nbuyers_90d"] = _window_count_distinct(days, evp, 90)
    out["ins_nsellers_90d"] = _window_count_distinct(days, evs, 90)
    out["ins_cluster_30d"] = _window_count_distinct(days, evp, 30)

    signed = usd.where(p, -usd.where(s_unflag, 0.0))
    rw = np.array([role_weight(r, t) for r, t in
                   zip(ins["role"], ins["title"])])
    out["ins_role_net_90d"] = _window_sum(
        days, contrib((p | s_unflag), signed.values * rw), 90)

    out["ins_net_mcap_90d"] = np.full(len(days), np.nan)
    if shares is not None and not shares.empty:
        d64 = days.values.astype("datetime64[D]")
        sh = shares.dropna(subset=["filed", "shares"])
        filed = pd.to_datetime(sh["filed"]).values.astype("datetime64[D]")
        order = np.argsort(filed)
        filed, vals = filed[order], sh["shares"].values.astype(float)[order]
        pos = np.searchsorted(filed, d64, side="left") - 1    # filed < D
        so = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
        so = pd.Series(so).ffill().values
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ins_net_mcap_90d"] = out["ins_net_usd_90d"] / (so * close)

    out["ins_days_since_p"] = _days_since(days, fd[p])
    return out


def f13f_features(f13: pd.DataFrame, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """f13: one symbol's holding rows (fund, filing_date, weight)."""
    out: dict[str, np.ndarray] = {}
    cols = ["f13f_nholders", "f13f_nholders_chg", "f13f_wsum",
            "f13f_wsum_chg", "f13f_new_91d"]
    d64 = days.values.astype("datetime64[D]")
    if f13.empty:
        return {c: np.zeros(len(days)) for c in cols}
    f13 = f13.assign(
        filing_date=pd.to_datetime(f13["filing_date"], errors="coerce")
    ).dropna(subset=["filing_date"])
    if f13.empty:
        return {c: np.zeros(len(days)) for c in cols}

    mat = {}
    for fund, g in f13.groupby("fund"):
        g = g.sort_values("filing_date")
        fdates = pd.to_datetime(g["filing_date"]).values.astype("datetime64[D]")
        w = g["weight"].values.astype(float)
        pos = np.searchsorted(fdates, d64, side="left") - 1    # filing < D
        series = np.where(pos >= 0, w[np.clip(pos, 0, len(w) - 1)], np.nan)
        mat[fund] = pd.Series(series).ffill().fillna(0.0).values
    W = pd.DataFrame(mat)
    holders = (W > 1e-12).sum(axis=1).values.astype(float)
    wsum = W.sum(axis=1).values
    out["f13f_nholders"] = holders
    out["f13f_wsum"] = wsum
    lag = np.searchsorted(d64, d64 - np.timedelta64(91, "D"), side="left")
    lag = np.clip(lag, 0, len(days) - 1)
    out["f13f_nholders_chg"] = holders - holders[lag]
    out["f13f_wsum_chg"] = wsum - wsum[lag]

    # "new" = first time a fund's holding appears, or reappears after a
    # >200d gap in being reported
    news = np.zeros(len(days) + 1)
    for _, g in f13.groupby("fund"):
        fd = np.sort(pd.to_datetime(g["filing_date"])
                     .values.astype("datetime64[D]"))
        if len(fd) == 0:
            continue
        is_new = np.ones(len(fd), dtype=bool)
        if len(fd) > 1:
            is_new[1:] = ((fd[1:] - fd[:-1]) / np.timedelta64(1, "D")) > 200
        s = np.searchsorted(d64, fd[is_new], side="right")
        e = np.searchsorted(d64, fd[is_new] + np.timedelta64(91, "D"),
                            side="right")
        np.add.at(news, s, 1.0)
        np.add.at(news, e, -1.0)
    out["f13f_new_91d"] = np.cumsum(news)[: len(days)]
    return out


def congress_features(cong: pd.DataFrame, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    cols = ["cong_nbuy_45d", "cong_nsell_45d", "cong_net_usd_45d"]
    if cong.empty:
        return {c: np.zeros(len(days)) for c in cols}
    cong = cong.assign(
        filing_date=pd.to_datetime(cong["filing_date"], errors="coerce")
    ).dropna(subset=["filing_date"])
    if cong.empty:
        return {c: np.zeros(len(days)) for c in cols}
    fd = cong["filing_date"]
    mid = ((pd.to_numeric(cong["amount_lo"], errors="coerce")
            + pd.to_numeric(cong["amount_hi"], errors="coerce"))
           / 2.0).fillna(0.0)
    tt = cong["txn_type"].astype(str).str.upper()
    is_buy = tt.str.startswith("P")
    is_sell = tt.str.startswith("S")

    def c(mask, v):
        return pd.DataFrame({"info_date": fd[mask],
                             "value": np.asarray(v)[np.asarray(mask.values)]})

    out = {
        "cong_nbuy_45d": _window_sum(
            days, c(is_buy, np.ones(len(cong))), 45),
        "cong_nsell_45d": _window_sum(
            days, c(is_sell, np.ones(len(cong))), 45),
    }
    signed = mid.where(is_buy, -mid.where(is_sell, 0.0))
    out["cong_net_usd_45d"] = _window_sum(
        days, c(is_buy | is_sell, signed), 45)
    return out


# ---------------------------------------------------------------------------
# panel assembly


def _rule_signals(out: dict[str, np.ndarray], days) -> dict[str, np.ndarray]:
    def hold(mask_active: np.ndarray, n: int) -> np.ndarray:
        ev = np.where(mask_active & ~np.r_[False, mask_active[:-1]])[0]
        if len(ev) == 0:
            return np.zeros(len(days))
        diff = np.zeros(len(days) + 1)
        np.add.at(diff, ev, 1.0)
        np.add.at(diff, np.clip(ev + n, 0, len(days)), -1.0)
        return (np.cumsum(diff)[: len(days)] > 0).astype(float)

    out["rule_cluster_hold20"] = hold(out["ins_cluster_30d"] >= 2, 20)
    out["rule_cluster_hold60"] = hold(out["ins_cluster_30d"] >= 2, 60)
    out["rule_ins_buy_hold60"] = hold(out["ins_buy_usd_90d"] > 0, 60)
    out["rule_cong_buy_hold60"] = hold(out["cong_nbuy_45d"] > 0, 60)
    return out


_EMPTY_F13 = pd.DataFrame(columns=["fund", "filing_date", "weight"])
_EMPTY_CON = pd.DataFrame(columns=["filing_date", "amount_lo", "amount_hi",
                                   "txn_type"])


def build_panel(data_dir: Path, instruments_name: str,
                events_dir: Path, out_path: Path) -> pd.DataFrame:
    cal = load_calendar(data_dir)
    inst = load_instruments(data_dir, instruments_name)

    ins = _read(events_dir / "insider_events.parquet")
    f13 = _read(events_dir / "f13f_events.parquet")
    shr = _read(events_dir / "shares_out.parquet")
    con = _read(events_dir / "congress_events.parquet")
    cmap = _read(events_dir / "cik_map.parquet")

    if not ins.empty:
        ins["symbol"] = ins["symbol"].map(canon)
    if not shr.empty:
        shr["symbol"] = shr["symbol"].map(canon)
        if "shares" not in shr.columns and "val" in shr.columns:
            shr = shr.rename(columns={"val": "shares"})
    if not cmap.empty:
        cmap["symbol"] = cmap["symbol"].map(canon)

    shr_by_sym = {s: g for s, g in shr.groupby("symbol")} if not shr.empty else {}
    name_map = _issuer_name_map(ins, cmap)
    con_by_sym = _congress_by_symbol(con, inst, name_map)
    f13_by_sym = _f13f_by_symbol(f13, inst, name_map)
    ins_by_sym = {s: g for s, g in ins.groupby("symbol")} if not ins.empty else {}

    frames = []
    for sym, g in inst.groupby("instrument"):
        close = load_close(data_dir, sym, cal)
        feats = insider_features(
            ins_by_sym.get(sym, ins.iloc[0:0] if not ins.empty else ins),
            cal, shr_by_sym.get(sym), close)
        feats.update(f13f_features(f13_by_sym.get(sym, _EMPTY_F13), cal))
        feats.update(congress_features(con_by_sym.get(sym, _EMPTY_CON), cal))
        feats = _rule_signals(feats, cal)

        label = np.full(len(cal), np.nan)
        label[:-2] = close[2:] / close[1:-1] - 1
        df = pd.DataFrame({"datetime": cal, **feats, "LABEL0": label})
        mask = np.zeros(len(cal), dtype=bool)
        for r in g.itertuples():
            mask |= ((cal >= pd.Timestamp(r.start)) &
                     (cal <= pd.Timestamp(r.end)))
        df = df[mask]
        df["instrument"] = sym
        frames.append(df)
    panel = (pd.concat(frames, ignore_index=True)
             .set_index(["instrument", "datetime"]).sort_index())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    return panel


def _read(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _issuer_name_map(ins: pd.DataFrame, cmap: pd.DataFrame) -> dict[str, set[str]]:
    """normalized issuer/company name -> {universe symbols}."""
    m: dict[str, set[str]] = {}
    if not ins.empty and "issuer_name" in ins.columns:
        for sym, nm in (ins[["symbol", "issuer_name"]].dropna()
                        .drop_duplicates().itertuples(index=False)):
            m.setdefault(norm_name(nm), set()).add(sym)
    if not cmap.empty and "issuer" in cmap.columns:
        for sym, nm in (cmap[["symbol", "issuer"]].dropna()
                        .drop_duplicates().itertuples(index=False)):
            m.setdefault(norm_name(nm), set()).add(sym)
    return m


def _congress_by_symbol(con: pd.DataFrame, inst: pd.DataFrame,
                        name_map: dict) -> dict:
    if con.empty:
        return {}
    con = con.copy()
    con["ticker"] = con["ticker"].map(lambda t: canon(t) if pd.notna(t) else None)
    uni = set(inst["instrument"])

    def resolve(r):
        if r["ticker"] in uni:
            return r["ticker"]
        n = norm_name(r["asset"])
        if len(n) < 5:
            return None
        cands = set()
        for k, syms in name_map.items():
            if len(k) >= 5 and (k in n or n in k):
                cands |= syms
        return next(iter(cands)) if len(cands) == 1 else None

    con["symbol"] = con.apply(resolve, axis=1)
    con = con.dropna(subset=["symbol"])
    return {s: g for s, g in con.groupby("symbol")}


def _f13f_by_symbol(f13: pd.DataFrame, inst: pd.DataFrame,
                    name_map: dict) -> dict:
    """Map (issuer, class) -> universe symbol via normalized issuer names +
    class letters."""
    if f13.empty:
        return {}
    f13 = f13.copy()
    uni_syms = set(inst["instrument"])
    key_map: dict[str, set[str]] = {}
    for s in uni_syms:
        key_map.setdefault(norm_name(s), set()).add(s)

    # significant-token -> name_map keys containing it (fast substring
    # fallback: only compare keys that share a token, not the whole map)
    from collections import defaultdict
    tok2keys: dict[str, set[str]] = defaultdict(set)
    for k in name_map:
        for t in k.split():
            if len(t) >= 3:
                tok2keys[t].add(k)

    # explicit class-letter -> preferred symbol for multi-class issuers
    _CLASS_PREF = {
        ("A", "GOOGL"), ("C", "GOOG"),
        ("B", "BRK-B"), ("A", "BRK-A"),
        ("A", "FOXA"), ("B", "FOX"),
        ("A", "NWSA"), ("B", "NWS"),
        ("A", "BF-A"), ("B", "BF-B"),
        ("A", "HEI-A"),
        ("A", "LEN"), ("B", "LEN-B"),
        ("A", "CWEN-A"), ("C", "CWEN"),
        ("A", "UAA"), ("C", "UA"),
    }

    memo: dict[tuple, str | None] = {}

    def resolve_one(issuer: str, cls: str) -> str | None:
        n = norm_name(issuer)
        cands = set(name_map.get(n, set())) | set(key_map.get(n, set()))
        if not cands and len(n) >= 5:
            keys: set[str] = set()
            for t in n.split():
                keys |= tok2keys.get(t, set())
            for k in keys:
                if len(k) >= 5 and (k in n or n in k):
                    cands |= name_map[k]
        if not cands:
            return None
        if len(cands) == 1:
            return next(iter(cands))
        cl = class_letter(issuer + " " + str(cls))
        pref = {s for c, s in _CLASS_PREF if c == cl and s in cands}
        return next(iter(pref)) if pref else sorted(cands)[0]

    def memo_resolve(p: tuple) -> str | None:
        if p not in memo:
            memo[p] = resolve_one(*p)
        return memo[p]

    issuers = f13["issuer"].fillna("")
    classes = (f13["class"] if "class" in f13.columns else
               pd.Series("", index=f13.index))
    f13["symbol"] = [
        memo_resolve(p) for p in zip(issuers, classes.fillna(""))
    ]
    f13 = f13.dropna(subset=["symbol"])
    return {s: g for s, g in f13.groupby("symbol")}
