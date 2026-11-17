"""Alpha158+Alpha360 combined handler (Phase 2 experiment a).

The stock qlib Alpha158 handler hardcodes its data_loader, so a combined
feature set can't be injected via YAML kwargs. This subclass of DataHandlerLP
mirrors Alpha158's processor setup but loads through a NestedDataLoader that
left-joins Alpha158DL (158 engineered cols + label) and Alpha360DL (360 raw
normalized OHLCV cols), giving ~518 feature columns.

Phase 3 adds two more handlers on the disclosure-event panel
(src/quant_research/events.py builds it; loader.py serves it):
  Alpha158Events  Alpha158DL + EventsDL(event features) joined left
  EventsOnly      EventsDL(event features + LABEL0) — no price features
"""
from pathlib import Path

import pandas as pd
from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP

_DEFAULT_LABEL = (["Ref($close, -2) / Ref($close, -1) - 1"], ["LABEL0"])


class Alpha158_360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        label=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors or [], fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors or [], fit_start_time, fit_end_time)
        label_cfg = label if label is not None else list(_DEFAULT_LABEL[0])
        data_loader = {
            "class": "qlib.data.dataset.loader.NestedDataLoader",
            "kwargs": {
                "join": "left",
                "dataloader_l": [
                    {"class": "qlib.contrib.data.loader.Alpha158DL",
                     "kwargs": {"config": {"label": (label_cfg, ["LABEL0"])}}},
                    {"class": "qlib.contrib.data.loader.Alpha360DL"},
                ],
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )


def _event_loader(panel: str | Path, with_label: bool,
                  feature_cols=None) -> dict:
    return {
        "class": "EventsDL",
        "module_path": "quant_research.loader",
        "kwargs": {"path": str(panel), "with_label": with_label,
                   "feature_cols": feature_cols},
    }


class Alpha158Events(DataHandlerLP):
    """Alpha158 + disclosure-event features (Phase 3 test b)."""

    def __init__(self, instruments="csi300", start_time=None, end_time=None,
                 freq="day", infer_processors=None, learn_processors=None,
                 fit_start_time=None, fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A, label=None,
                 events_panel="data/events/features_us.parquet",
                 event_cols=None, **kwargs):
        infer_processors = check_transform_proc(
            infer_processors or [], fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(
            learn_processors or [], fit_start_time, fit_end_time)
        label_cfg = label if label is not None else list(_DEFAULT_LABEL[0])
        data_loader = {
            "class": "qlib.data.dataset.loader.NestedDataLoader",
            "kwargs": {
                "join": "left",
                "dataloader_l": [
                    {"class": "qlib.contrib.data.loader.Alpha158DL",
                     "kwargs": {"config": {"label": (label_cfg, ["LABEL0"])}}},
                    _event_loader(events_panel, with_label=False,
                                  feature_cols=event_cols),
                ],
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )


class EventsOnly(DataHandlerLP):
    """Disclosure-event features only (Phase 3 test a)."""

    def __init__(self, instruments="csi300", start_time=None, end_time=None,
                 freq="day", infer_processors=None, learn_processors=None,
                 fit_start_time=None, fit_end_time=None,
                 process_type=DataHandlerLP.PTYPE_A, label=None,
                 events_panel="data/events/features_us.parquet",
                 event_cols=None, **kwargs):
        infer_processors = check_transform_proc(
            infer_processors or [], fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(
            learn_processors or [], fit_start_time, fit_end_time)
        data_loader = _event_loader(events_panel, with_label=True,
                                    feature_cols=event_cols)
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )
