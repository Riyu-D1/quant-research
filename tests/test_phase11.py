"""Phase-11 tests — the PIT-membership backtest fix and the sealed holdout.

Synthetic except for the committed-results checks, which read the files
the Actions runs commit (results/phase11/) and skip until they exist.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import phase11_us as P11  # noqa: E402
from quant_research import pit  # noqa: E402

SPEC = yaml.safe_load((REPO / "configs/experiments/phase11_us.yaml").read_text())
T = pd.Timestamp


def _win(rows):
    return pd.DataFrame(rows, columns=["symbol", "start", "end"]).assign(
        start=lambda d: pd.to_datetime(d["start"]),
        end=lambda d: pd.to_datetime(d["end"]))


# ---------------------------------------------------------------------------
# membership windows


def test_member_mask_end_is_exclusive_and_open_spells_run_to_the_end():
    days = pd.bdate_range("2022-06-06", periods=6)        # Mon 6 .. Mon 13
    w = _win([("FB", "2016-01-04", "2022-06-09"), ("META", "2022-06-09", None),
              ("APP", "2022-06-13", None)])
    m = pit.member_mask(w, days, pd.Index(["FB", "META", "APP", "ZZZ"]))
    assert m["FB"].tolist() == [True, True, True, False, False, False]
    assert m["META"].tolist() == [False, False, False, True, True, True]
    assert m["APP"].tolist() == [False] * 5 + [True]
    assert not m["ZZZ"].any()
    assert not (m["FB"] & m["META"]).any()        # a rename is never double-counted


def test_debounce_ignores_short_flips():
    on = np.array([True, True, False, True, True, True, False, False, False])
    assert pit.debounce(on, 3).tolist() == [True] * 6 + [False] * 3


def test_patch_windows_applies_persistent_wikipedia_changes():
    w = _win([("AAA", "2016-01-04", None), ("BBB", "2016-01-04", None),
              ("OLD", "2016-01-04", "2020-01-02")])
    days = pd.date_range("2026-09-01", periods=10)
    snaps = {}
    for i, d in enumerate(days):
        syms = ["AAA", "BBB"] if i < 4 else ["AAA", "NEW"]
        if i == 1:
            syms = ["AAA"]                     # one-day vandal edit: ignored
        added = [None, None] if i < 4 else [None, "2026-09-04"]
        snaps[d] = pd.DataFrame({"symbol": syms,
                                 "date_added": pd.to_datetime(added[:len(syms)])})
    out = pit.patch_windows(w, snaps, persist=3).set_index("symbol")
    assert pd.isna(out.loc["AAA", "end"])
    assert out.loc["BBB", "end"] == days[4]                 # left on day 5
    assert out.loc["NEW", "start"] == T("2026-09-04")       # its Date added
    assert pd.isna(out.loc["NEW", "end"])
    assert out.loc["OLD", "end"] == T("2020-01-02")         # history untouched


# ---------------------------------------------------------------------------
# the honest composite can only pick members


def _core(n_days=320, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"S{i}" for i in range(12)] + ["LATE", "SPY"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, len(cols))),
                                          axis=0), index=days, columns=cols)
    join = days[200]
    w = _win([(c, "2016-01-04", None) for c in cols[:12]] + [("LATE", join, None)])
    member = pit.member_mask(w, days, close.columns)
    mom = pd.DataFrame(rng.normal(size=close.shape), index=days, columns=cols)
    mom["LATE"] = 99.0                                 # best score every day
    parts = {k: pd.DataFrame(rng.normal(size=close.shape), index=days, columns=cols)
             .assign(LATE=99.0) for k in ("vq_bm", "vq_ep", "vq_roe")}
    core = {"feed": {"close": close}, "days": days, "member": member, "win": w,
            "mom": mom, "vq_parts": parts, "smap": {}}
    return core, join


def test_honest_b50_never_picks_before_index_inclusion():
    core, join = _core()
    comp = P11.b50_honest(core)
    picks = P11.picks_frame(P11.sleeve_targets(SPEC, core, comp, "B50_honest", 6))
    assert pit.non_member_picks(picks, core["win"]).empty
    late = picks.loc[picks["symbol"] == "LATE", "date"]
    assert late.min() == join                    # picked from the join day, not before


def test_ever_member_ranking_is_caught_by_the_check():
    core, join = _core()
    leaky = P11.b50_honest({**core, "member": core["member"] | True})
    picks = P11.picks_frame(P11.sleeve_targets(SPEC, core, leaky, "leaky", 6))
    bad = pit.non_member_picks(picks, core["win"])
    assert not bad.empty and set(bad["symbol"]) == {"LATE"}
    assert bad["date"].max() < join


def test_member_rank_ignores_non_members():
    days = pd.bdate_range("2024-01-01", periods=2)
    panel = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]}, index=days)
    member = pd.DataFrame({"A": [True, True], "B": [True, True], "C": [False, True]},
                          index=days)
    r = P11.member_rank(panel, member)
    assert r.iloc[0].tolist()[:2] == [0.5, 1.0] and np.isnan(r.iloc[0]["C"])
    assert r.iloc[1].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


# ---------------------------------------------------------------------------
# spec + committed results


def test_spec_seals_a_twelve_month_holdout_at_the_sample_end():
    h = SPEC["holdout"]
    assert h["end"] == SPEC["sample_end"]
    assert (T(h["end"]) - T(h["start"])).days in range(363, 367)
    assert any(c["name"] == "B50_honest" and c["members"] == "pit"
               for c in SPEC["baseline"])


def _committed(path):
    p = REPO / path
    if not p.exists():
        pytest.skip(f"{path} not committed yet (written by the Actions run)")
    return p


def test_every_committed_b50_honest_pick_was_a_member_on_its_signal_day():
    picks = pd.read_csv(_committed("results/phase11/b50_honest_picks.csv"),
                        parse_dates=["date"])
    win = P11.load_membership(_committed(SPEC["membership_csv"]))
    assert len(picks) > 1000
    bad = pit.non_member_picks(picks, win)
    assert bad.empty, f"non-member picks:\n{bad.head(10)}"


def test_committed_baseline_never_saw_the_holdout():
    d = json.loads(_committed("results/phase11/baseline_dev.json").read_text())
    assert T(d["dev_window"][1]) < T(SPEC["holdout"]["start"])
    assert d["b50_honest_non_member_picks"] == 0
    assert d["n_trials"] == d["prior_trials"] + d["n_windows"]
    assert d["prior_trials"] >= 769
