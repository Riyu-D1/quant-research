"""qlib DataLoader for the Phase-3 events feature panel.

Loads a parquet of (datetime, instrument)-indexed event features and wraps
them in the (feature|label) column MultiIndex qlib handlers expect, so
fetch(col_set="feature"/"label") behaves exactly like Alpha158DL rows.

    EventsDL(path="data/events/features_us.parquet",
             feature_cols=[...], with_label=True)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.data.dataset.loader import DataLoader


def _resolve(instruments) -> list[str] | None:
    """'sp500' -> code list via the initialized provider; lists pass through."""
    if instruments is None or isinstance(instruments, (list, tuple)):
        return list(instruments) if instruments is not None else None
    if isinstance(instruments, str):
        from qlib.data import D
        return list(D.list_instruments(D.instruments(instruments),
                                       as_list=True))
    if isinstance(instruments, dict):  # market dict {market, filter_pipe,...}
        from qlib.data import D
        return list(D.list_instruments(instruments, as_list=True))
    return None


class EventsDL(DataLoader):
    """Static event-feature loader; panel is read once and cached."""

    def __init__(self, path, feature_cols=None, with_label: bool = True):
        self.path = str(path)
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.with_label = with_label
        self._data = None

    def _maybe_load(self) -> None:
        if self._data is None:
            df = pd.read_parquet(self.path)
            # panel may be (instrument, datetime) or (datetime, instrument)
            if df.index.names[0] != "datetime":
                df = df.swaplevel().sort_index()
            cols = self.feature_cols or [
                c for c in df.columns if c != "LABEL0"]
            take = {g: df[cols] for g in ["feature"]}
            if self.with_label:
                take["label"] = df[["LABEL0"]]
            self._data = pd.concat(take, axis=1)

    def load(self, instruments=None, start_time=None,
             end_time=None) -> pd.DataFrame:
        self._maybe_load()
        codes = _resolve(instruments)
        df = self._data
        if codes is not None:
            have = df.index.get_level_values("instrument")
            df = df[have.isin(set(codes))]
        if start_time is not None or end_time is not None:
            dt = df.index.get_level_values("datetime")
            lo = pd.Timestamp(start_time) if start_time else dt.min()
            hi = pd.Timestamp(end_time) if end_time else dt.max()
            df = df[(dt >= lo) & (dt <= hi)]
        return df


class EventsOnlyDL(EventsDL):
    """Convenience: feature cols restricted to the event features + LABEL0."""
