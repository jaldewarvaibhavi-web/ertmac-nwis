"""
preprocessing.py - turn validated tables into model-ready views.

WHAT:  (1) a "model view" of parameter tables: flagged values blanked, duplicates removed,
           depth sorted - the stored data itself is never changed;
       (2) formation helpers: which formation is at a depth, how far below its top;
       (3) per-well, per-formation "profiles" (average standardised parameters + geology)
           used by the relevance score;
       (4) the StandardScaler used for all parameter comparisons.
WHY:   comparing wells only makes sense on clean, comparable numbers, and in formation
       coordinates rather than raw depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import CURRENT_PARAMS


# ------------------------------------------------------------------ clean model view
def model_view(df: pd.DataFrame) -> pd.DataFrame:
    """Blank values flagged out-of-range / non-numeric, drop flagged duplicates, sort by depth."""
    out = df.copy()
    if "qc_flags" in out and out.qc_flags.astype(bool).any():
        flags = out.qc_flags.fillna("")
        for col in [c for c in out.columns if c in CURRENT_PARAMS + ["pit_volume", "flow_out_pct", "hookload",
                                                                     "porosity", "caliper", "depth"]]:
            bad = flags.str.contains(fr"(?:^|;){col}:(?:out_of_range|non_numeric)", regex=True)
            out.loc[bad, col] = np.nan
        out = out[~flags.str.contains(r"(?:^|;)duplicate(?:;|$)", regex=True)]
    sort_cols = [c for c in ["well_id", "depth"] if c in out]
    return out.sort_values(sort_cols).reset_index(drop=True)


def available_params(df: pd.DataFrame, cols=CURRENT_PARAMS) -> list[str]:
    """Parameters that actually have data - unavailable ones are excluded automatically."""
    return [c for c in cols if c in df and df[c].notna().any()]


# ------------------------------------------------------------------ formation helpers
def formation_at(tops_w: pd.DataFrame, depth: float) -> str | None:
    t = tops_w[tops_w.top_depth_m <= depth]
    return None if t.empty else t.sort_values("top_depth_m").iloc[-1].formation


def formation_top(tops_w: pd.DataFrame, formation: str) -> float | None:
    t = tops_w[tops_w.formation == formation]
    return None if t.empty else float(t.top_depth_m.iloc[0])


def position(tops_w: pd.DataFrame, depth: float) -> tuple[str | None, float | None]:
    """(formation, metres below its top) - the 'formation coordinate' of a depth."""
    f = formation_at(tops_w, depth)
    if f is None:
        return None, None
    return f, depth - formation_top(tops_w, f)


def formations_ahead(tops_w: pd.DataFrame, depth: float, lookahead: float) -> list[str]:
    """Current formation plus any formation whose top lies within the look-ahead window."""
    cur = formation_at(tops_w, depth)
    ahead = tops_w[(tops_w.top_depth_m > depth) & (tops_w.top_depth_m <= depth + lookahead)].formation.tolist()
    return [f for f in [cur] + ahead if f]


# ------------------------------------------------------------------ scaler + profiles
def fit_scaler(params: pd.DataFrame, cols: list[str]) -> StandardScaler:
    """StandardScaler on historical offset data, so ROP (m/hr) and SPP (psi) weigh the same."""
    X = params[cols].astype(float)
    sc = StandardScaler().fit(X.fillna(X.median()))
    sc.nwis_cols = cols
    return sc


def scaled(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    cols = scaler.nwis_cols
    X = df[cols].astype(float)
    Z = (X - scaler.mean_) / scaler.scale_
    return Z


def formation_profiles(params: pd.DataFrame, geology: pd.DataFrame, scaler: StandardScaler,
                       max_depth: dict | None = None) -> pd.DataFrame:
    """Mean standardised parameters + mean porosity / caliper per (well, formation).
    max_depth: {well_id: depth} limits a well to what has been drilled (the active well)."""
    p = params.copy()
    g = geology.copy() if geology is not None else pd.DataFrame(columns=["well_id", "depth", "formation"])
    if max_depth:
        for w, d in max_depth.items():
            p = p[(p.well_id != w) | (p.depth <= d)]
            g = g[(g.well_id != w) | (g.depth <= d)]
    Z = scaled(p, scaler)
    Z[["well_id", "formation"]] = p[["well_id", "formation"]]
    prof = Z.groupby(["well_id", "formation"]).mean(numeric_only=True)
    geo_cols = [c for c in ["porosity", "caliper"] if c in g and g[c].notna().any()]
    if geo_cols:
        prof = prof.join(g.groupby(["well_id", "formation"])[geo_cols].mean(), how="left")
    return prof
