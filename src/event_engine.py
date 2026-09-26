"""
event_engine.py - Phase 5: depth-aware, formation-aware historical event correlation.

WHAT:  every historical event in the relevant offset wells is expressed in the ACTIVE well's
       coordinates:
            aligned_depth = active top of that formation + (event depth - offset top of that formation)
       then compared with the bit:
            AT CURRENT DEPTH  |aligned_depth - bit| <= depth tolerance (configurable, default +-25 m)
            AHEAD             bit + tolerance < aligned_depth <= bit + look-ahead
            PASSED            above the bit
WHY:   the same rock layer sits at different depths in different wells, so raw depth alone
       mis-matches events; the raw depth difference is still shown for transparency.
       Example: W-102 lost mud 44 m below the Barail top at 2,654 m; in ACTIVE-01 the Barail top is
       at 2,610 m, so that zone is at ~2,654 m aligned.
OUTPUT: table of events with raw depth, aligned depth, distance to bit, status, relevance of
        the well and the full provenance (source document, extraction method).
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing import formation_top
from src.utils import get_config


def align_events(events: pd.DataFrame, tops: pd.DataFrame, active_id: str) -> pd.DataFrame:
    tops_a = tops[tops.well_id == active_id]
    rows = []
    for e in events.itertuples():
        if pd.isna(e.depth):
            continue
        tops_o = tops[tops.well_id == e.well_id]
        form = e.formation if isinstance(e.formation, str) else None
        top_o = formation_top(tops_o, form) if form else None
        top_a = formation_top(tops_a, form) if form else None
        if top_o is not None and top_a is not None:
            below = e.depth - top_o
            aligned, method = top_a + below, "formation-aligned"
        else:
            below, aligned, method = None, e.depth, "raw depth (formation not in active well)"
        d = e._asdict()
        d.pop("Index", None)
        d.update(m_below_top=below, aligned_depth=aligned, alignment=method)
        rows.append(d)
    return pd.DataFrame(rows)


def correlate(aligned: pd.DataFrame, relevance: pd.DataFrame, current_depth: float,
              cfg: dict | None = None, only_selected: bool = True) -> pd.DataFrame:
    """Classify offset events relative to the bit and attach each well's relevance."""
    cfg = get_config(cfg)
    if aligned.empty or relevance.empty:
        return pd.DataFrame()
    rel = relevance[relevance.selected] if only_selected else relevance
    df = aligned.merge(rel[["well_id", "distance_km", "relevance_score"]], on="well_id", how="inner")
    tol, look = cfg["depth_tolerance_m"], cfg["lookahead_m"]
    df["delta_to_bit_m"] = df.aligned_depth - current_depth
    df["raw_delta_m"] = df.depth - current_depth
    df["status"] = "PASSED"
    df.loc[df.delta_to_bit_m.abs() <= tol, "status"] = "AT CURRENT DEPTH"
    df.loc[(df.delta_to_bit_m > tol) & (df.delta_to_bit_m <= look), "status"] = "AHEAD"
    order = {"AT CURRENT DEPTH": 0, "AHEAD": 1, "PASSED": 2}
    df["_o"] = df.status.map(order)
    df = df.sort_values(["_o", "delta_to_bit_m", "relevance_score"], ascending=[True, True, False])
    return df.drop(columns="_o").reset_index(drop=True)
