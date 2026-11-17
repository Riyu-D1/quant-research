"""Phase-4 feature store builder — one heavy pass, reused by every sim.

Builds data/features4/store_us.parquet: the Alpha158 raw feature matrix +
Phase-3 event features + price/fundamental signal columns + LABEL0, on the
(instrument, datetime) grid restricted to each symbol's S&P500 membership
window (same mask as the Phase-3 panel).

    uv run python scripts/phase4_build.py --market us

Heavy: the full-span Alpha158DL evaluation peaks ~7-8GB — it runs in the
phase4 `store` workflow job (with swap), never on the owner's Mac. Monthly
retrains afterwards read this parquet and cost seconds each.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "features4"


def log(msg: str) -> None:
    print(f"[phase4-build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False,
                          cwd=REPO, capture_output=True, text=True).stdout.strip()


def build_store(provider: Path, instruments: str, events_panel: Path,
                fundamentals: Path | None, out: Path) -> dict:
    from qlib.utils import init_instance_by_config

    from quant_research import events as ev
    from quant_research import fundamentals as F
    from quant_research import walkforward as wf

    spec = wf.load_spec("us")
    wf.qlib_init(str(provider), spec["region"])
    cal = ev.load_calendar(provider)
    inst = ev.load_instruments(provider, instruments)

    # ---- Alpha158 raw matrix (no processors — monotone-invariant GBDTs) ----
    t0 = time.time()
    handler = init_instance_by_config({
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {"instruments": spec["instruments"],
                   "start_time": spec["data_start"],
                   "end_time": spec["data_end"],
                   "infer_processors": [], "learn_processors": [],
                   "label": [spec["label"]]}})
    feats = handler.fetch(col_set="feature", data_key="infer")
    label = handler.fetch(col_set="label", data_key="infer").iloc[:, 0]
    a158_cols = list(feats.columns)
    log(f"alpha158 fetched: {feats.shape} ({time.time()-t0:.0f}s)")

    feats = feats.swaplevel(0, 1).sort_index()
    label = label.swaplevel(0, 1).sort_index()

    # ---- restrict to membership windows (same mask as events.build_panel) --
    mask = pd.Series(False, index=feats.index)
    inst_days = feats.index.get_level_values("datetime")
    for r in inst.itertuples():
        m = (feats.index.get_level_values("instrument") == r.instrument) & \
            (inst_days >= pd.Timestamp(r.start)) & \
            (inst_days <= pd.Timestamp(r.end))
        mask |= m
    feats, label = feats[mask], label[mask]
    log(f"membership mask: {len(feats)} rows kept")

    store = feats.astype(np.float32)
    store["LABEL0"] = label.astype(np.float32)

    # ---- Phase-3 event features --------------------------------------------
    panel = pd.read_parquet(events_panel)
    ev_cols = [c for c in ev.FEATURE_COLS + ev.RULE_COLS if c in panel.columns]
    for c in ev_cols:
        store[c] = panel[c].reindex(store.index).astype(np.float32)
    log(f"joined {len(ev_cols)} event cols")

    # ---- closes + momentum/reversal ----------------------------------------
    closes = {}
    for r in inst.itertuples():
        closes[r.instrument] = ev.load_close(provider, r.instrument, cal)
    close_df = pd.DataFrame(closes, index=cal).sort_index()
    c_long = close_df.stack().rename("close")
    c_long.index.names = ["datetime", "instrument"]
    c_long = c_long.swaplevel(0, 1).sort_index()
    store["close"] = c_long.reindex(store.index).astype(np.float32)

    g = store.groupby(level="instrument")["close"]
    store["mom_12_1"] = (g.shift(21) / g.shift(252) - 1).astype(np.float32)
    store["rev_1m"] = -(g.shift(0) / g.shift(21) - 1).astype(np.float32)
    log("mom/rev computed")

    # ---- value/quality ------------------------------------------------------
    vq_cols = ["vq_bm", "vq_ep", "vq_roe"]
    shares_path = REPO / "data" / "events" / "shares_out.parquet"
    if fundamentals is not None and fundamentals.exists():
        facts = pd.read_parquet(fundamentals)
        facts["filed"] = pd.to_datetime(facts["filed"])
        facts["symbol"] = facts["symbol"].map(ev.canon)
        shares = (pd.read_parquet(shares_path) if shares_path.exists()
                  else pd.DataFrame())
        shr_by = {s: gg.rename(columns={"val": "shares"}) for s, gg in
                  shares.assign(symbol=shares["symbol"].map(ev.canon))
                        .groupby("symbol")} if not shares.empty else {}
        f_by = {s: gg for s, gg in facts.groupby("symbol")}
        vq_frames = {c: {} for c in vq_cols}
        filled = 0
        for r in inst.itertuples():
            f = f_by.get(r.instrument)
            vq = (F.vq_features(f, shr_by.get(r.instrument),
                                ev.load_close(provider, r.instrument, cal),
                                cal)
                  if f is not None else
                  {c: np.full(len(cal), np.nan) for c in vq_cols})
            for c in vq_cols:
                vq_frames[c][r.instrument] = vq[c]
            filled += f is not None
        for c in vq_cols:
            m = pd.DataFrame(vq_frames[c], index=cal)
            s = m.stack()
            s.index.names = ["datetime", "instrument"]
            store[c] = s.swaplevel(0, 1).reindex(store.index) \
                          .astype(np.float32)
        log(f"vq factors computed for {filled} symbols")
    else:
        for col in vq_cols:
            store[col] = np.float32(np.nan)
        log("fundamentals.parquet absent — vq cols all-NaN (signal will be "
            "dead weight in the adaptive combiner)")

    # ---- write --------------------------------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(out)
    meta = {"git_sha": git_sha(), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": str(provider), "instruments": instruments,
            "rows": len(store), "a158_cols": a158_cols, "event_cols": ev_cols,
            "vq_cols": vq_cols, "signal_cols": ["mom_12_1", "rev_1m"],
            "label": "LABEL0",
            "span": [str(store.index.get_level_values('datetime').min().date()),
                     str(store.index.get_level_values('datetime').max().date())]}
    (out.parent / (out.stem + ".manifest.json")).write_text(
        json.dumps(meta, indent=1))
    log(f"store -> {out} ({out.stat().st_size/1e9:.2f} GB, {len(store)} rows)")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="us", choices=["us"])
    p.add_argument("--provider", default=None)
    p.add_argument("--instruments", default="sp500")
    p.add_argument("--events-panel", default=None)
    p.add_argument("--fundamentals", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    spec_provider = REPO / "data" / "qlib" / "us_alpaca"
    provider = Path(args.provider) if args.provider else spec_provider
    events = Path(args.events_panel) if args.events_panel else \
        REPO / "data" / "events" / "features_us.parquet"
    fundamentals = (Path(args.fundamentals) if args.fundamentals
                    else REPO / "data" / "events" / "fundamentals.parquet")
    out = Path(args.out) if args.out else OUT_DIR / "store_us.parquet"
    build_store(provider, args.instruments, events,
                fundamentals if fundamentals.exists() else None, out)


if __name__ == "__main__":
    main()
