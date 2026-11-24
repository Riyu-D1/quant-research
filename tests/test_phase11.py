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


def _core(n_days=320, seed=0, k=12):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"S{i}" for i in range(k)] + ["LATE", "SPY"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, len(cols))),
                                          axis=0), index=days, columns=cols)
    join = days[200]
    w = _win([(c, "2016-01-04", None) for c in cols[:k]] + [("LATE", join, None)])
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


# ---------------------------------------------------------------------------
# step 3: LLM formula language — whitelist + causality

from quant_research import factor_dsl as D  # noqa: E402


def _env(n=300, k=8, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n)
    cols = [f"S{i}" for i in range(k)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, k)), axis=0),
                         index=days, columns=cols)
    env = {f: pd.DataFrame(rng.normal(size=(n, k)), index=days, columns=cols)
           for f in D.FIELDS}
    env["close"] = close
    env["ret"] = close.pct_change()
    return env


@pytest.mark.parametrize("bad", [
    "__import__('os').system('x')", "close.shift(-1)", "delay(close, 0)",
    "delay(close, -5)", "ts_mean(close, 500)", "ts_mean(close, n)", "foo(close)",
    "lambda: 1", "close[1]", "1 + 2", "ts_mean(close, 2.5)", "future_ret",
    " + ".join(["close"] * 20)])
def test_formula_language_rejects_anything_outside_the_whitelist(bad):
    with pytest.raises(D.FormulaError):
        D.parse(bad)


@pytest.mark.parametrize("expr", [
    "-ts_std(ret, 21) / ts_mean(volume, 5)", "rank(ebit_tev) + rank(gpa)",
    "delta(roe, 252)", "ts_corr(ret, volume, 63)", "ts_rank(close, 126) - delay(ret, 1)",
    "sign(ep) * log(abs(bm) + 1)"])
def test_every_formula_is_causal(expr):
    env = _env()
    t = 200
    a = D.evaluate(expr, env)
    shocked = {k: v.copy() for k, v in env.items()}
    for v in shocked.values():
        v.iloc[t + 1:] = v.iloc[t + 1:] * 7.0 + 3.0
    b = D.evaluate(expr, shocked)
    pd.testing.assert_frame_equal(a.iloc[:t + 1], b.iloc[:t + 1])


def test_canonical_dedupes_formatting():
    assert D.canonical("ts_mean( close ,5 )") == D.canonical("ts_mean(close, 5)")


def test_llm_output_parser_and_selection():
    import phase11_llm as L
    raw = [{"theme": "t", "text": "1. cheap | rank(ebit_tev) | value\n"
            "dup | rank( ebit_tev ) | same\nbad | close.shift(-1) | peek\n"
            "chatter without pipes\nq | `gpa - delay(gpa, 252)` | improving"}]
    keep, rej = L.select(raw, max_factors=10)
    assert [k["formula"] for k in keep] == ["rank(ebit_tev)", "gpa - delay(gpa, 252)"]
    assert keep[0]["name"] == "cheap" and keep[0]["id"] == "L01"
    assert {r["reason"] for r in rej} >= {"duplicate"}
    assert len(rej) == 2


# ---------------------------------------------------------------------------
# step 3: algorithmic screen, text panel, learning layer


def test_screen_only_uses_forward_returns_closed_before_january():
    core, _ = _core(n_days=700, k=40)
    days = core["days"]
    close = core["feed"]["close"]
    fwd = close.shift(-22) / close.shift(-1) - 1
    good = fwd.rank(axis=1, pct=True)              # a perfect (look-ahead) factor
    spec = {**SPEC, "screen": {**SPEC["screen"], "min_samples": 3}}
    y = days[-1].year
    out = P11.screen_factors(spec, core, {"F": good.where(core["member"])},
                             P11.b50_honest(core), [y])
    st = out[y]["stats"]["F"]
    t = list(days).index(days[days.year == y][0])
    ms = [i for i in P9_month_starts(days) if i + spec["fwd_h"] + 1 < t]
    assert st["n"] == len([i for i in ms if i >= t - 252 * spec["screen"]["years"]])
    # corrupting every price after the decision day leaves the screen unchanged
    shocked = close.copy()
    shocked.iloc[t:] *= np.random.default_rng(5).uniform(0.5, 2.0, shocked.iloc[t:].shape)
    core2 = {**core, "feed": {"close": shocked}}
    out2 = P11.screen_factors(spec, core2, {"F": good.where(core["member"])},
                              P11.b50_honest(core), [y])
    assert out2[y]["stats"]["F"]["ic"] == pytest.approx(st["ic"])


def P9_month_starts(days):
    import phase9_us as P9
    return P9.month_starts(days)


def test_hedge_weights_move_toward_the_better_sleeve_and_hold_monthly():
    days = pd.bdate_range("2021-01-01", periods=400)
    sh = pd.DataFrame({"B50_honest": 0.0005, "X": 0.002}, index=days)
    cfg = SPEC["meta"]
    W = P11.hedge_weights(sh, cfg)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert (W.iloc[:cfg["lookback"]] == 0.5).all().all()          # prior before data
    late = W.iloc[-1]
    assert late["X"] > 0.5 and late["X"] <= 1 - cfg["shrink"] / 2 + 1e-9
    ms = set(P9_month_starts(days))
    changes = [i for i in range(1, len(W)) if not W.iloc[i].equals(W.iloc[i - 1])]
    assert set(changes) <= ms


def test_fallback_switches_to_b50_after_a_lag_and_recovers():
    days = pd.bdate_range("2021-01-01", periods=600)
    r_b = pd.Series(0.0, index=days)
    r_h = pd.Series(0.0, index=days)
    r_h.iloc[150:250] = -0.002                    # hybrid lags by 20%
    r_h.iloc[350:500] = 0.003                     # then recovers
    fb = P11.fallback_modes(r_h, r_b, SPEC["meta"]["fallback"])
    assert not fb.iloc[:150].any()
    first = fb.idxmax()
    assert days[150] < first < days[250]
    assert fb.loc[first:].iloc[:SPEC["meta"]["fallback"]["min_hold"]].all()
    assert not fb.iloc[-1]


def test_mix_books_weights_sum_and_merge_names():
    d = pd.Timestamp("2024-01-02")
    books = {"A": {d: {"SPY": 0.5, "X": 0.5}}, "B": {d: {"SPY": 0.5, "Y": 0.25, "X": 0.25}}}
    W = pd.DataFrame({"A": [0.25], "B": [0.75]}, index=[d])
    out = P11.mix_books(books, W)[d]
    assert out == pytest.approx({"SPY": 0.5, "X": 0.3125, "Y": 0.1875})


def test_lazy_prices_panel_is_live_only_after_filing(tmp_path):
    days = pd.bdate_range("2020-01-01", "2022-12-30")
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.6, 0.8], dtype=np.float32)
    e = pd.DataFrame({"cik": [1, 1], "symbols": ["AAA;AAA-OLD", "AAA;AAA-OLD"],
                      "accession": ["a", "b"],
                      "filed": pd.to_datetime(["2020-02-20", "2021-02-22"]),
                      "emb": [v1, v2]})
    path = tmp_path / "emb.parquet"
    e.to_parquet(path)
    spec = {"text": {**SPEC["text"], "embeddings": str(path)}}
    P = P11.lazy_prices(spec, days, pd.Index(["AAA", "AAA-OLD", "BBB"]))
    assert P.loc[:"2021-02-22", "AAA"].isna().all()           # not on the filing day
    assert P.loc["2021-02-23", "AAA"] == pytest.approx(0.6)
    assert P.loc["2021-02-23", "AAA-OLD"] == pytest.approx(0.6)
    assert P.loc["2022-03-31":, "AAA"].isna().all()            # expired after 400 days
    assert P["BBB"].isna().all()


def test_tx_excludes_the_most_changed_filers_only():
    days = pd.bdate_range("2024-01-01", periods=1)
    cols = [f"S{i}" for i in range(10)]
    b50 = pd.DataFrame([np.linspace(0.1, 1.0, 10)], index=days, columns=cols)
    sim = pd.DataFrame([np.r_[np.linspace(0.5, 0.95, 9), np.nan]], index=days, columns=cols)
    member = pd.DataFrame(True, index=days, columns=cols)
    tx = P11.tx_comp(SPEC, b50, sim, member)
    assert tx.iloc[0].isna().tolist() == [True] + [False] * 9


def test_spec_part2_predeclares_candidates_and_rules():
    names = [c["name"] for c in SPEC["candidates"]]
    assert names[:2] == ["A_spy", "B50_honest"] and "HY_full" in names
    assert SPEC["adoption"] == {"baseline": "B50_honest", "min_dsr": 0.95}
    assert SPEC["text"]["model"].endswith("20161231")       # cutoff before 2017
    for c in SPEC["candidates"]:
        if c["kind"] == "meta":
            assert c["sleeves"][0] == "B50_honest"
