"""
geo_matching.py - Phase 3 & 4: which offset wells matter for the active well RIGHT NOW?

WHAT:  1. radius search (Haversine distance)          -> candidate wells
       2. five-part Offset Relevance Score (0-1)       -> ranking + explanation
            relevance = 30% spatial + 20% formation + 20% depth + 20% parameter + 10% geology
          (prototype weights only - configurable, NOT scientifically validated).
          Components that cannot be computed are reported as "Not available" and the
          remaining weights are re-normalised - no fake values.
WHY:   the nearest well is not always the most useful one: a well 4 km away that drilled the
       same formation at the same structural depth with similar parameters can matter more
       than one 1 km away that never reached it.
OUTPUT: one row per offset with every component score and a plain-language reason.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing import formation_at, formation_profiles
from src.similarity import depth_score, formation_score, geology_score, param_similarity
from src.utils import clip01, get_config, haversine_km

COMPONENTS = ["spatial", "formation", "depth", "parameter", "geology"]


def nearby_wells(wells: pd.DataFrame, active_id: str, radius_km: float) -> pd.DataFrame:
    act = wells.loc[wells.well_id == active_id].iloc[0]
    off = wells[wells.well_id != active_id].copy()
    off["distance_km"] = [round(haversine_km(act.latitude, act.longitude, r.latitude, r.longitude), 2)
                          for r in off.itertuples()]
    off["in_radius"] = off.distance_km <= radius_km
    return off.sort_values("distance_km").reset_index(drop=True)


def relevance(ctx, active_id: str, current_depth: float, cfg: dict | None = None) -> pd.DataFrame:
    """Rank offset wells inside the radius. `ctx` is an NWISContext (see engine.py)."""
    cfg = get_config(cfg)
    w = cfg["weights"]
    wells = ctx.wells
    near = nearby_wells(wells, active_id, cfg["radius_km"])
    near = near[near.in_radius]
    tops_a = ctx.tops[ctx.tops.well_id == active_id]
    cur_form = formation_at(tops_a, current_depth)

    # active profile only from what has been drilled so far
    active_params = ctx.stream_for(active_id, current_depth)
    active_geo = ctx.geology[(ctx.geology.well_id == active_id) & (ctx.geology.depth <= current_depth)]
    prof_a = formation_profiles(active_params, active_geo, ctx.scaler)
    prof = pd.concat([ctx.offset_profiles, prof_a])
    drilled = [f for f in tops_a.sort_values("top_depth_m").formation if f in prof_a.index.get_level_values(1)]

    rows = []
    for o in near.itertuples():
        tops_o = ctx.tops[ctx.tops.well_id == o.well_id]
        s_sp = clip01(1 - o.distance_km / cfg["radius_km"])
        s_fm, why_fm = formation_score(tops_a, tops_o, current_depth, cfg["lookahead_m"])
        s_dp, why_dp = depth_score(tops_a, tops_o, current_depth, o.total_depth, cur_form)
        shared = [f for f in drilled if f in set(tops_o.formation)]
        sims, ks = [], []
        for f in shared[-3:]:                               # most recent formations matter most
            if (o.well_id, f) in prof.index and (active_id, f) in prof.index:
                cols = ctx.scaler.nwis_cols
                sim, k = param_similarity(prof.loc[(active_id, f), cols], prof.loc[(o.well_id, f), cols])
                if k:
                    sims.append(sim)
                    ks.append(k)
        s_pm = float(sum(sims) / len(sims)) if sims else float("nan")
        why_pm = (f"{max(ks)} parameters over {len(sims)} shared drilled formation(s)" if sims
                  else "Not available (no shared drilled formation / parameters)")
        s_ge, why_ge = geology_score(prof, active_id, o.well_id, shared[-3:], tops_a, tops_o)

        scores = dict(spatial=s_sp, formation=s_fm, depth=s_dp, parameter=s_pm, geology=s_ge)
        avail = {k: v for k, v in scores.items() if pd.notna(v)}
        wsum = sum(w[k] for k in avail)
        rel = sum(w[k] * v for k, v in avail.items()) / wsum if wsum else float("nan")
        rows.append(dict(
            well_id=o.well_id, distance_km=o.distance_km, total_depth=o.total_depth,
            drilling_year=getattr(o, "drilling_year", None), data_origin=getattr(o, "data_origin", None),
            relevance_score=round(rel, 3),
            **{f"{k}_score": (round(v, 3) if pd.notna(v) else None) for k, v in scores.items()},
            missing_components=", ".join(k for k in COMPONENTS if k not in avail) or "-",
            why=(f"{o.distance_km} km away | {why_fm} | {why_dp} | parameters: {why_pm} | geology: {why_ge}"),
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    out["selected"] = (out.index < cfg["max_offsets"]) & (out.relevance_score >= cfg["relevance_threshold"])
    return out
