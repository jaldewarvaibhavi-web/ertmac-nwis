"""
engine.py - loads everything once (NWISContext) and runs the full NWIS workflow for one bit position.

  WHAT IS HAPPENING NOW?     current stream up to the bit              (Phase 6)
  WHERE AM I NOW?            depth + formation + metres below its top
  WHAT HAPPENED BEFORE?      relevant offset wells + their events      (Phases 3-5)
  WERE CONDITIONS SIMILAR?   relevance score + parameter/geology comparison (Phase 7)
  WHAT IS THE RISK?          evidence + ML (where validated) + live anomaly (Phase 8)
  WHY?                       evidence table with sources               (Phase 10, explainability.py)
Used by the dashboard, the API and the demo, so all three always show the same answer.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src import anomaly
from src.event_engine import align_events
from src.ingestion import load_tables
from src.preprocessing import available_params, fit_scaler, formation_profiles, model_view, position
from src.utils import CURRENT_PARAMS, DB_PATH


class NWISContext:
    def __init__(self, db_path=DB_PATH):
        t = load_tables(db_path)
        self.tables = t
        self.wells = t["wells"]
        self.tops = t["formation_tops"]
        self.geology = t["geology"]
        self.events = t["events"]
        self.documents = t["documents"]
        self.qc = t["qc_issues"]
        params = model_view(t["historical_parameters"])
        self.forge_params = params[params.data_origin == "REAL-PUBLIC"].reset_index(drop=True)
        self.hist_params = params[params.data_origin == "SYNTHETIC"].reset_index(drop=True)
        self.current = model_view(t["current_stream"])
        cols = available_params(self.hist_params, CURRENT_PARAMS)
        self.scaler = fit_scaler(self.hist_params, cols)
        self.offset_profiles = formation_profiles(self.hist_params, self.geology, self.scaler)
        self.active_ids = sorted(self.current.well_id.unique())
        self._aligned: dict = {}
        self._anomaly: dict = {}

    # ---------------------------------------------------------------- streams
    def stream(self, well_id: str) -> pd.DataFrame:
        if well_id in self.active_ids:
            return self.current[self.current.well_id == well_id]
        if well_id == "FORGE-58-32":
            return self.forge_params
        return self.hist_params[self.hist_params.well_id == well_id]

    def stream_for(self, well_id: str, depth: float) -> pd.DataFrame:
        s = self.stream(well_id)
        return s[s.depth <= depth]

    def current_row(self, well_id: str, depth: float, window_m: float = 10) -> pd.Series:
        """State at the bit = mean of the last `window_m` metres (smooths sensor noise)."""
        s = self.stream_for(well_id, depth)
        last = s[s.depth > depth - window_m]
        row = last.mean(numeric_only=True)
        for c in ["well_id", "formation", "timestamp", "torque_unit"]:
            if c in s and len(s):
                row[c] = s.iloc[-1][c]
        return row

    def position(self, well_id: str, depth: float):
        return position(self.tops[self.tops.well_id == well_id], depth)

    # ---------------------------------------------------------------- cached analyses
    def aligned_events(self, active_id: str) -> pd.DataFrame:
        if active_id not in self._aligned:
            self._aligned[active_id] = align_events(self.events, self.tops, active_id)
        return self._aligned[active_id]

    def anomalies(self, well_id: str):
        """Detector run once over the whole stream; callers reveal it up to the bit (it is causal)."""
        if well_id not in self._anomaly:
            s = self.stream(well_id)
            if well_id == "FORGE-58-32":
                self._anomaly[well_id] = anomaly.analyse_spec(s, well_id)          # fitted on itself
            else:
                model = anomaly.fit_on_spec([g for _, g in self.hist_params.groupby("well_id")])
                self._anomaly[well_id] = anomaly.analyse_spec(s, well_id, model)   # fitted on offsets
        return self._anomaly[well_id]


@lru_cache(maxsize=2)
def get_context(db_path: str = str(DB_PATH)) -> NWISContext:
    return NWISContext(db_path)
