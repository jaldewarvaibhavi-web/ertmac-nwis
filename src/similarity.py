"""
similarity.py - how alike are two wells / two drilling situations?  (all scores 0-1)

METHOD (documented as the spec asks, section 11):
  * Parameters are standardised with a StandardScaler fitted on the historical offset data,
    so every parameter counts equally regardless of units.
  * Parameter similarity  = 1 / (1 + d / sqrt(k))
        d = Euclidean distance between the two standardised vectors, k = number of parameters
        used (only parameters available in BOTH wells). 1 = identical, 0.5 = on average ~1 standard
        deviation apart per parameter. Transparent and bounded.
  * Formation similarity  = mean of (share of the current + upcoming formations that the offset
        well also penetrated) and (thickness similarity of the current formation).
  * Depth similarity      = structural similarity of the current formation top
        exp(-|top_active - top_offset| / 150 m), halved if the offset never reached the
        equivalent position.
  * Geological similarity = 1 - mean(|delta porosity| / 0.10, |delta caliper| / 2 in) over shared,
        already-drilled formations. If porosity/caliper are missing -> FALLBACK: similarity of the
        formation sequence (clearly reported as fallback).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.preprocessing import formation_top, formations_ahead
from src.utils import clip01


def param_similarity(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    """Euclidean similarity of two standardised vectors, using only shared, non-missing values."""
    mask = a.notna() & b.notna()
    k = int(mask.sum())
    if k == 0:
        return float("nan"), 0
    d = float(np.sqrt(((a[mask] - b[mask]) ** 2).sum()))
    return 1.0 / (1.0 + d / math.sqrt(k)), k


def formation_score(tops_a: pd.DataFrame, tops_o: pd.DataFrame, depth: float, lookahead: float) -> tuple[float, str]:
    wanted = formations_ahead(tops_a, depth, lookahead)
    if not wanted:
        return float("nan"), "active formation unknown"
    have = set(tops_o.formation)
    presence = sum(f in have for f in wanted) / len(wanted)
    cur = wanted[0]
    ta, to = tops_a[tops_a.formation == cur], tops_o[tops_o.formation == cur]
    if to.empty:
        return presence * 0.5, f"offset never penetrated {cur}"
    thk_a = float(ta.bottom_depth_m.iloc[0] - ta.top_depth_m.iloc[0])
    thk_o = float(to.bottom_depth_m.iloc[0] - to.top_depth_m.iloc[0])
    thick = 1 - abs(thk_a - thk_o) / max(thk_a, thk_o, 1)
    return clip01((presence + thick) / 2), f"shares {sum(f in have for f in wanted)}/{len(wanted)} of {', '.join(wanted)}; {cur} thickness {thk_o:.0f} vs {thk_a:.0f} m"


def depth_score(tops_a: pd.DataFrame, tops_o: pd.DataFrame, depth: float, td_offset: float,
                formation: str | None) -> tuple[float, str]:
    if formation is None:
        return float("nan"), "active formation unknown"
    top_a, top_o = formation_top(tops_a, formation), formation_top(tops_o, formation)
    if top_o is None:
        return 0.0, f"{formation} absent in offset"
    equiv = top_o + (depth - top_a)                       # same position in the offset well
    s = math.exp(-abs(top_a - top_o) / 150.0)
    reached = td_offset >= equiv
    if not reached:
        s *= 0.5
    return clip01(s), (f"{formation} top {top_o:.0f} m vs {top_a:.0f} m (Δ {top_o - top_a:+.0f} m); "
                       f"equivalent depth {equiv:.0f} m {'reached' if reached else 'NOT reached'}")


def geology_score(prof: pd.DataFrame, active_id: str, offset_id: str, shared: list[str],
                  tops_a: pd.DataFrame, tops_o: pd.DataFrame) -> tuple[float, str]:
    geo_cols = [c for c in ["porosity", "caliper"] if c in prof.columns]
    diffs = []
    for f in shared:
        if (active_id, f) in prof.index and (offset_id, f) in prof.index:
            a, o = prof.loc[(active_id, f)], prof.loc[(offset_id, f)]
            if "porosity" in geo_cols and pd.notna(a.porosity) and pd.notna(o.porosity):
                diffs.append(min(1.0, abs(a.porosity - o.porosity) / 0.10))
            if "caliper" in geo_cols and pd.notna(a.caliper) and pd.notna(o.caliper):
                diffs.append(min(1.0, abs(a.caliper - o.caliper) / 2.0))
    if diffs:
        return clip01(1 - float(np.mean(diffs))), f"porosity/caliper compared over {len(shared)} formation(s)"
    # fallback: formation sequence similarity (order of formations)
    seq_a = tops_a.sort_values("top_depth_m").formation.tolist()
    seq_o = tops_o.sort_values("top_depth_m").formation.tolist()
    if not seq_a or not seq_o:
        return float("nan"), "Not available"
    common = [f for f in seq_a if f in seq_o]
    order_ok = common == [f for f in seq_o if f in seq_a]
    s = len(common) / len(set(seq_a) | set(seq_o)) * (1.0 if order_ok else 0.7)
    return clip01(s), "FALLBACK: formation-sequence similarity (porosity/caliper not available)"


def compare_now(current: pd.Series, offset_rows: pd.DataFrame, scaler) -> dict:
    """Current state (standardised) vs an offset well at the same formation position.
    Returns similarity and per-parameter differences in standard deviations (for 'Why?')."""
    cols = scaler.nwis_cols
    if offset_rows.empty:
        return dict(similarity=float("nan"), n_params=0, diffs={})
    cur = pd.to_numeric(pd.Series([current.get(c, np.nan) for c in cols], index=cols), errors="coerce")
    a = pd.Series((cur.values - scaler.mean_) / scaler.scale_, index=cols)
    b = pd.Series((offset_rows[cols].astype(float).mean().values - scaler.mean_) / scaler.scale_, index=cols)
    sim, k = param_similarity(a, b)
    diffs = {c: round(float(a[c] - b[c]), 2) for c in cols if pd.notna(a[c]) and pd.notna(b[c])}
    return dict(similarity=sim, n_params=k, diffs=diffs)
