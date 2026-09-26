"""
anomaly.py - live drilling anomaly detector (current behaviour vs THIS well's own normal)
--------------------------------------------------------------------------------------
Complements the offset-well comparison: it answers "is something unusual happening now?"
and is the only analysis possible on the real FORGE log (no offset wells, no labels).
One engine for any well - the REAL Utah FORGE 58-32 log or our synthetic wells.

Idea: judge every reading against THIS well's own recent "normal"
(rolling baseline of the previous ~30 m), not against fixed numbers.
Only past readings are used (causal), so it behaves exactly like it would
on a live eRTMAC stream.

  Rule detectors (explainable drilling logic)
    - suspected mud loss    : flow-out below normal and/or pit volume falling
    - suspected influx/kick : flow-out above normal / pit gain + ROP jump
    - torque anomaly        : torque well above normal while rotating (stuck pipe / hole cleaning)
    - pressure anomaly      : standpipe pressure far from what the pump rate predicts
    - drilling inefficiency : ROP per unit WOB collapses (bit wear / balling / hard rock)
  Machine learning
    - Isolation Forest on the same "vs normal" features -> anomaly score 0-1

Signals that overlap are merged into ONE incident with a confidence level:
    High   = 2+ independent signals AND the ML model agrees
    Medium = 2+ signals OR 1 signal + ML
    Low    = single signal only (logged, not shown as an alert)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["flow_out_dev", "torque_ratio", "spp_ratio", "pit_change_3m", "rop_eff_ratio", "rop_ratio"]

EVENT_INFO = {
    "flag_kick": ("Suspected influx (kick)", "kick",
                  "Flow out above normal / pit gain - possible influx. Flow check immediately."),
    "flag_loss": ("Suspected mud loss", "mud_loss",
                  "Flow out below normal / pit volume falling - possible lost circulation."),
    "flag_pressure": ("Standpipe pressure anomaly", "pressure",
                      "Pressure far from expected for this pump rate - pack-off, washout, loss or BHA change."),
    "flag_torque": ("Torque anomaly", "torque_spike",
                    "Torque well above normal while rotating - stuck-pipe / poor hole cleaning risk."),
    "flag_rop": ("Drilling inefficiency", "bit_problem",
                 "ROP per WOB collapsed - bit wear/balling or harder formation."),
}
PRIORITY = list(EVENT_INFO)


def standardise(df):
    """Map the columns of different sources onto one set of names."""
    d = df.copy()
    if "torque_surface_psi" in d:
        d["torque"] = d["torque_surface_psi"]
    elif "torque_kftlbs" in d:
        d["torque"] = d["torque_kftlbs"]
    if "flow_out_pct" not in d:
        d["flow_out_pct"] = np.nan                     # synthetic data has no flow-out sensor
    return d.sort_values("depth_md_m").reset_index(drop=True)


def _rows(d, metres):
    step = d.depth_md_m.diff().median()
    return max(1, int(round(metres / (step if step and step > 0 else 1))))


def add_features(df):
    d = standardise(df)
    base_n, short_n = _rows(d, 30), _rows(d, 3)

    # "normal" is measured only within the current formation: a new rock type
    # starts a new baseline (harder rock drilling slower is not a problem)
    if "formation" in d and d.formation.notna().any() and (d.formation != "Unknown").any():
        block = (d.formation != d.formation.shift()).cumsum()
    else:
        block = pd.Series(0, index=d.index)

    def base(s, valid, n=None):
        n = n or base_n
        return (s.where(valid).groupby(block)
                 .transform(lambda x: x.rolling(n, min_periods=max(5, n // 6)).median().shift(1).ffill()))

    q = d.flow_rate_gpm.rolling(base_n, min_periods=5).median().shift(1).bfill()
    d["valid"] = (d.flow_rate_gpm > 0.75 * q) & (d.flow_rate_gpm > 200) & (d.wob_klbs > 2)
    v = d.valid

    for c in ["flow_out_pct", "torque", "spp_psi", "rop_m_hr", "flow_rate_gpm"]:
        d[c + "_base"] = base(d[c], v)
    d["flow_out_dev"] = d.flow_out_pct - d.flow_out_pct_base
    d["torque_ratio"] = d.torque / d.torque_base
    expected_spp = d.spp_psi_base * (d.flow_rate_gpm / d.flow_rate_gpm_base) ** 1.8
    d["spp_ratio"] = d.spp_psi / expected_spp
    d["pit_change_3m"] = d.pit_volume_bbl - d.pit_volume_bbl.shift(short_n)
    d["rop_ratio"] = d.rop_m_hr.rolling(short_n).median() / d.rop_m_hr_base
    eff = d.rop_m_hr / d.wob_klbs.clip(lower=1)
    eff_med = eff.where(v).rolling(_rows(d, 5), min_periods=2).median()
    # compare with the BEST recent performance in this formation (catches slow bit wear)
    best = (eff_med.groupby(block)
            .transform(lambda x: x.rolling(_rows(d, 60), min_periods=_rows(d, 10)).quantile(0.8).shift(1)))
    d["rop_eff_ratio"] = eff_med / best
    d["torque_cv"] = (d.torque.rolling(_rows(d, 5)).std() / d.torque.rolling(_rows(d, 5)).mean())

    # how noisy is this well's pit? (real rigs move mud around -> bigger threshold)
    pc = d.loc[v, "pit_change_3m"].dropna()
    sigma = 1.4826 * (pc - pc.median()).abs().median() if len(pc) else 1.0
    d.attrs["pit_threshold"] = float(max(5.0, 4 * sigma))
    d.attrs["persist_n"] = _rows(d, 2)
    # adaptive torque limits: 3 robust standard deviations above this well's normal
    rot = v & (d.rpm > 20)
    tr = d.loc[rot, "torque_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    cv = d.loc[rot, "torque_cv"].dropna()
    rs = lambda x: 1.4826 * (x - x.median()).abs().median() if len(x) else 0.1
    d.attrs["torque_ratio_thr"] = float(max(1.2, 1 + 3.5 * rs(tr)))
    d.attrs["torque_cv_thr"] = float(max(0.12, cv.median() + 4 * rs(cv))) if len(cv) else 0.3
    return d


def _persistent(flag, window):
    """Trailing window: flag must hold for most of the last ~2 m (no look-ahead)."""
    need = max(1, int(np.ceil(window * 0.7)))
    return flag.astype(int).rolling(window, min_periods=1).sum() >= need


def add_flags(d):
    d = d.copy()
    v, w, pit_thr = d.valid, d.attrs["persist_n"], d.attrs["pit_threshold"]
    has_flow_out = d.flow_out_pct.notna().any()
    rotating = d.rpm > 20
    torque_ok = d.torque_base > 0.2 * d.loc[v, "torque"].median()   # ignore start-up baseline

    if has_flow_out:
        loss = (d.flow_out_dev < -15) | ((d.pit_change_3m < -pit_thr) & (d.flow_out_dev < -5))
        kick = (d.flow_out_dev > 12) & (d.pit_change_3m > pit_thr / 2)
    else:
        loss = d.pit_change_3m < -pit_thr
        kick = (d.pit_change_3m > pit_thr) | ((d.pit_change_3m > pit_thr / 2) & (d.rop_ratio > 1.4))
    d["flag_loss"] = _persistent(v & loss, w)
    d["flag_kick"] = _persistent(v & kick, max(1, w // 2))
    high = d.torque_ratio > d.attrs["torque_ratio_thr"]
    erratic = (d.torque_cv > d.attrs["torque_cv_thr"]) & (d.torque > d.torque_base)
    d["flag_torque"] = _persistent(v & rotating & torque_ok & (high | erratic), w)
    d["flag_pressure"] = _persistent(v & ((d.spp_ratio < 0.8) | (d.spp_ratio > 1.2)), w)
    d["flag_rop"] = _persistent(v & (d.rop_eff_ratio < 0.6), _rows(d, 5))
    return d


def fit_anomaly_model(train_df):
    """Fit Isolation Forest on feature rows of one or more wells (already featurised)."""
    X = train_df.loc[train_df.valid, FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)  # flow_out_dev is NaN where no sensor -> neutral 0
    model = IsolationForest(n_estimators=300, contamination=0.03, random_state=0).fit(X)
    raw = -model.score_samples(X)
    model.nwis_range = (float(raw.min()), float(raw.max()))
    return model


def add_anomaly_score(d, model):
    d = d.copy()
    X = d.loc[d.valid, FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    d["anomaly_score"] = np.nan
    d["anomaly_flag"] = False
    if len(X):
        lo, hi = model.nwis_range
        raw = -model.score_samples(X)
        d.loc[X.index, "anomaly_score"] = np.clip((raw - lo) / (hi - lo), 0, 1).round(3)
        d.loc[X.index, "anomaly_flag"] = model.predict(X) == -1
    return d


def _evidence(flag, g, has_flow_out):
    if flag == "flag_loss":
        if has_flow_out:
            return (f"flow out {g.flow_out_pct.min():.0f}% vs normal {g.flow_out_pct_base.median():.0f}%, "
                    f"pit {g.pit_change_3m.min():+.0f} bbl/3 m")
        return f"pit volume {g.pit_change_3m.min():+.0f} bbl over 3 m"
    if flag == "flag_kick":
        txt = f"pit +{g.pit_change_3m.max():.0f} bbl/3 m"
        if has_flow_out:
            txt = f"flow out {g.flow_out_pct.max():.0f}% vs normal {g.flow_out_pct_base.median():.0f}%, " + txt
        return txt + f", ROP x{g.rop_ratio.max():.1f}"
    if flag == "flag_torque":
        return f"torque up to {g.torque.max():.0f} vs normal {g.torque_base.median():.0f} ({g.torque_ratio.max():.1f}x)"
    if flag == "flag_pressure":
        r = g.spp_ratio
        worst = r.min() if (1 - r.min()) > (r.max() - 1) else r.max()
        return f"SPP {worst:.0%} of expected for {g.flow_rate_gpm.median():.0f} gpm"
    return f"ROP/WOB down to {g.rop_eff_ratio.min():.0%} of normal"


def incidents(d, well_id, merge_gap_m=2.0, min_len_m=0.9):
    """Merge overlapping flags into incidents (one row each)."""
    has_flow_out = d.flow_out_pct.notna().any()
    idx = d.index[d[PRIORITY].any(axis=1) & d.valid]
    rows = []
    if len(idx) == 0:
        return pd.DataFrame(columns=["well_id", "top_m", "bottom_m", "event", "event_type",
                                     "signals", "evidence", "confidence"])
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if d.depth_md_m[i] - d.depth_md_m[prev] > merge_gap_m:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))

    for a, b in groups:
        g = d.loc[a:b]
        g = g[g.valid]
        top, bot = g.depth_md_m.min(), g.depth_md_m.max()
        if bot - top < min_len_m and len(g) < 2:
            continue
        signals = [f for f in PRIORITY if g[f].any()]
        primary = signals[0]
        ml = bool(g.anomaly_flag.any()) if "anomaly_flag" in g else False
        n = len(signals)
        conf = "High" if (n >= 2 and ml) else ("Medium" if (n >= 2 or ml) else "Low")
        rows.append(dict(
            well_id=well_id, top_m=round(top, 1), bottom_m=round(bot, 1),
            length_m=round(bot - top, 1),
            formation=g.formation.mode().iloc[0] if "formation" in g and g.formation.notna().any() else "Unknown",
            event=EVENT_INFO[primary][0], event_type=EVENT_INFO[primary][1],
            signals=" + ".join(EVENT_INFO[f][0] for f in signals),
            evidence="; ".join(_evidence(f, g[g[f]], has_flow_out) for f in signals),
            meaning=EVENT_INFO[primary][2],
            max_anomaly_score=round(float(g.anomaly_score.max()), 2) if "anomaly_score" in g else None,
            ml_agrees="Yes" if ml else "No", confidence=conf))
    return pd.DataFrame(rows)


def analyse(df, well_id, model=None):
    """Full pipeline for one well. If no model is given, fit on this well itself."""
    d = add_flags(add_features(df))
    model = model or fit_anomaly_model(d)
    d = add_anomaly_score(d, model)
    return d, incidents(d, well_id)


# ---------------------------------------------------------------- spec-schema adapter
SPEC_TO_DETECTOR = {"depth": "depth_md_m", "ROP": "rop_m_hr", "WOB": "wob_klbs", "RPM": "rpm",
                    "SPP": "spp_psi", "flow_rate": "flow_rate_gpm", "pit_volume": "pit_volume_bbl",
                    "mud_weight": "mud_weight_ppg"}
DETECTOR_TO_EVENT = {"mud_loss": "Mud Loss", "kick": "Kick", "torque_spike": "Torque Spike",
                     "pressure": "Pressure/Gas Anomaly", "bit_problem": "Bit Problem"}


def to_detector(df):
    d = df.rename(columns=SPEC_TO_DETECTOR).copy()
    unit = df["torque_unit"].iloc[0] if "torque_unit" in df and len(df) else "kft-lbs"
    d["torque_surface_psi" if unit == "psi" else "torque_kftlbs"] = df["torque"]
    d = d.drop(columns=["torque"], errors="ignore")
    if "formation" in d:
        d["formation"] = d["formation"].fillna("Unknown")
    return d


def analyse_spec(df, well_id, model=None):
    """Run the detector on a spec-schema frame; incidents get the canonical event type."""
    d, inc = analyse(to_detector(df), well_id, model)
    if len(inc):
        inc["event_type_canonical"] = inc.event_type.map(DETECTOR_TO_EVENT).fillna("Other")
    return d, inc


def fit_on_spec(frames):
    feats = pd.concat([add_flags(add_features(to_detector(g))) for g in frames])
    return fit_anomaly_model(feats)
