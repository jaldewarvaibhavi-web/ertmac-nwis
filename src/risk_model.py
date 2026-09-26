"""
risk_model.py - Phase 8 & 9: supervised risk model + leave-one-well-out validation.

WHAT:  For each event type WITH ENOUGH LABELS, a Random Forest estimates:
          "probability that this event type starts within the next <look-ahead> m"
       from features available at the bit:
          * offset evidence  - spatially weighted share of OTHER wells whose (formation-aligned)
                               event of this type lies between (bit - tolerance) and (bit + look-ahead)
          * formation (one-hot) and metres below the formation top
          * geology at the bit (porosity, caliper)           [synthetic]
          * drilling parameters at the bit (only those available)
       Event types below the thresholds get NO model: "Insufficient labeled data for reliable ML training."
VALIDATION (spec section 13): leave-one-well-out. For each held-out well, the model is trained on the
       other wells AND the offset-evidence features of the training wells are rebuilt without the
       held-out well's events -> the held-out well is truly unseen ("can knowledge from previous
       wells generalise to another well?").
METRICS: precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC, false-positive / false-negative
       rate, plus an event-level "warned in advance" rate, compared with an evidence-only baseline.
       All results are labelled "Prototype validation only - not representative of field deployment accuracy."
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.utils import (FORMATION_ORDER, INSUFFICIENT_ML, MODELS_DIR, VALIDATION_DISCLAIMER, get_config,
                       haversine_km)

STEP_M = 5


def slug(t: str) -> str:
    return t.lower().replace("/", "_").replace(" ", "_")


def eligible_types(events: pd.DataFrame, cfg: dict) -> dict[str, str]:
    """event_type -> 'trained' or the reason it is not."""
    out = {}
    g = events.groupby("event_type").agg(n=("event_id", "size"), w=("well_id", "nunique"))
    for t, r in g.iterrows():
        if r.n >= cfg["min_events_for_ml"] and r.w >= cfg["min_wells_for_ml"]:
            out[t] = "eligible"
        else:
            out[t] = f"{INSUFFICIENT_ML} ({int(r.n)} events in {int(r.w)} wells)"
    return out


# ------------------------------------------------------------------ features
def _aligned_event_depths(target: str, tops: pd.DataFrame, events: pd.DataFrame, etype: str):
    """Depths (in target-well coordinates) of events of `etype` in OTHER wells."""
    tt = tops[tops.well_id == target].set_index("formation").top_depth_m
    ev = events[(events.event_type == etype) & (events.well_id != target)]
    out = []
    for e in ev.itertuples():
        to = tops[(tops.well_id == e.well_id) & (tops.formation == e.formation)].top_depth_m
        if e.formation in tt.index and len(to):
            out.append((e.well_id, float(tt[e.formation] + (e.depth - to.iloc[0]))))
    return out


def evidence_feature(target: str, depths: np.ndarray, ctx_wells: pd.DataFrame, tops: pd.DataFrame,
                     events: pd.DataFrame, etype: str, cfg: dict, pool: set) -> np.ndarray:
    """Spatially weighted share of pool wells with an aligned event in [d - tol, d + lookahead]."""
    tw = ctx_wells.set_index("well_id")
    lat, lon = tw.loc[target, "latitude"], tw.loc[target, "longitude"]
    weights = {}
    for w in pool:
        if w == target:
            continue
        d = haversine_km(lat, lon, tw.loc[w, "latitude"], tw.loc[w, "longitude"])
        if d <= cfg["radius_km"]:
            weights[w] = 1 - d / cfg["radius_km"]
    if not weights:
        return np.zeros(len(depths))
    ev = events[events.well_id.isin(weights)]
    hits = np.zeros(len(depths))
    for w, ad in _aligned_event_depths(target, tops, ev, etype):
        inside = (ad >= depths - cfg["depth_tolerance_m"]) & (ad <= depths + cfg["lookahead_m"])
        hits += inside * weights[w]
    # count each well once
    hits = np.minimum(hits, sum(weights.values()))
    return hits / sum(weights.values())


def base_rows(params: pd.DataFrame, geology: pd.DataFrame, tops: pd.DataFrame, well: str,
              param_cols: list[str], step: int = STEP_M, max_depth: float | None = None) -> pd.DataFrame:
    p = params[params.well_id == well]
    if max_depth is not None:
        p = p[p.depth <= max_depth]
    p = p[(p.depth % step) < 1e-9] if step > 1 else p
    rows = p[["well_id", "depth", "formation"] + param_cols].copy()
    g = geology[geology.well_id == well].sort_values("depth")
    if len(g):
        rows = pd.merge_asof(rows.sort_values("depth"), g[["depth", "porosity", "caliper"]],
                             on="depth", direction="nearest")
    else:
        rows["porosity"], rows["caliper"] = np.nan, np.nan
    tw = tops[tops.well_id == well].set_index("formation").top_depth_m
    rows["m_below_top"] = rows.depth - rows.formation.map(tw)
    for f in FORMATION_ORDER:
        rows[f"fm_{f}"] = (rows.formation == f).astype(int)
    return rows.reset_index(drop=True)


def labels(rows: pd.DataFrame, events: pd.DataFrame, etype: str, cfg: dict) -> np.ndarray:
    ev = events[(events.event_type == etype)]
    y = np.zeros(len(rows), dtype=int)
    for w, idx in rows.groupby("well_id").groups.items():
        starts = ev[ev.well_id == w].depth.values
        d = rows.loc[idx, "depth"].values[:, None]
        y[np.asarray(list(idx))] = ((starts[None, :] >= d - 10) & (starts[None, :] <= d + cfg["lookahead_m"])).any(1)
    return y


def feature_cols(param_cols):
    return ["evidence", "m_below_top", "porosity", "caliper"] + [f"fm_{f}" for f in FORMATION_ORDER] + param_cols


# ------------------------------------------------------------------ validation + training
def _metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    return dict(precision=round(precision_score(y, pred, zero_division=0), 3),
                recall=round(recall_score(y, pred, zero_division=0), 3),
                f1=round(f1_score(y, pred, zero_division=0), 3),
                roc_auc=round(roc_auc_score(y, p), 3) if both else None,
                pr_auc=round(average_precision_score(y, p), 3) if both else None,
                false_positive_rate=round(fp / max(1, fp + tn), 3),
                false_negative_rate=round(fn / max(1, fn + tp), 3),
                confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                positives=int(y.sum()), rows=int(len(y)))


def _event_warned(rows, prob, events, etype, cfg, thr=0.5):
    """Share of held-out events for which the model crossed the threshold before the bit got there."""
    ev = events[events.event_type == etype]
    warned, leads, n = 0, [], 0
    for e in ev.itertuples():
        r = rows[(rows.well_id == e.well_id)]
        if r.empty:
            continue
        n += 1
        window = r[(r.depth >= e.depth - cfg["lookahead_m"]) & (r.depth <= e.depth)]
        hit = window[prob[window.index] >= thr]
        if len(hit):
            warned += 1
            leads.append(float(e.depth - hit.depth.min()))
    return dict(events=n, warned_in_advance=warned, median_lead_m=float(np.median(leads)) if leads else None)


def train_and_validate(ctx, cfg: dict | None = None, save: bool = True, verbose: bool = True) -> dict:
    cfg = get_config(cfg)
    events = ctx.events
    hist_wells = sorted(ctx.hist_params.well_id.unique())
    param_cols = ctx.scaler.nwis_cols
    status = eligible_types(events, cfg)
    report = {"disclaimer": VALIDATION_DISCLAIMER, "method": "Random Forest, leave-one-well-out",
              "lookahead_m": cfg["lookahead_m"], "depth_tolerance_m": cfg["depth_tolerance_m"],
              "radius_km": cfg["radius_km"], "types": {}}
    base = {w: base_rows(ctx.hist_params, ctx.geology, ctx.tops, w, param_cols) for w in hist_wells}
    fcols = feature_cols(param_cols)

    for etype, st in status.items():
        if st != "eligible":
            report["types"][etype] = {"status": st}
            continue
        all_rows, all_prob, all_base = [], [], []
        for held in hist_wells:
            train_wells = [w for w in hist_wells if w != held]
            pool = set(train_wells)                               # held-out well's events are invisible
            ev_train = events[events.well_id.isin(pool)]
            tr = []
            for w in train_wells:
                r = base[w].copy()
                r["evidence"] = evidence_feature(w, r.depth.values, ctx.wells, ctx.tops, ev_train, etype, cfg, pool)
                tr.append(r)
            tr = pd.concat(tr, ignore_index=True)
            y_tr = labels(tr, ev_train, etype, cfg)
            te = base[held].copy()
            te["evidence"] = evidence_feature(held, te.depth.values, ctx.wells, ctx.tops, ev_train, etype, cfg, pool)
            if y_tr.sum() == 0:
                continue
            model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                                           random_state=0, n_jobs=-1)
            model.fit(tr[fcols].fillna(-1), y_tr)
            all_prob.append(model.predict_proba(te[fcols].fillna(-1))[:, 1])
            all_base.append((te.evidence.values >= 0.25).astype(float))
            all_rows.append(te)
        rows = pd.concat(all_rows, ignore_index=True)
        prob, basep = np.concatenate(all_prob), np.concatenate(all_base)
        y = labels(rows, events, etype, cfg)
        res = {"status": "trained", "n_events": int((events.event_type == etype).sum()),
               "n_wells_with_event": int(events[events.event_type == etype].well_id.nunique()),
               "model": _metrics(y, prob), "baseline_evidence_only": _metrics(y, basep),
               "event_level": _event_warned(rows, prob, events, etype, cfg)}
        # final model on all historical wells (evidence from all others)
        full = []
        for w in hist_wells:
            r = base[w].copy()
            r["evidence"] = evidence_feature(w, r.depth.values, ctx.wells, ctx.tops, events, etype, cfg, set(hist_wells))
            full.append(r)
        full = pd.concat(full, ignore_index=True)
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                       random_state=0, n_jobs=-1).fit(full[fcols].fillna(-1),
                                                                      labels(full, events, etype, cfg))
        res["top_features"] = dict(sorted(zip(fcols, model.feature_importances_.round(3)),
                                          key=lambda x: -x[1])[:6])
        if save:
            MODELS_DIR.mkdir(exist_ok=True)
            joblib.dump({"model": model, "features": fcols, "param_cols": param_cols, "cfg": cfg},
                        MODELS_DIR / f"risk_{slug(etype)}.joblib")
        report["types"][etype] = res
        if verbose:
            m, b = res["model"], res["baseline_evidence_only"]
            print(f"{etype:<22} model: P {m['precision']} R {m['recall']} F1 {m['f1']} ROC {m['roc_auc']} "
                  f"PR {m['pr_auc']} FPR {m['false_positive_rate']} | baseline F1 {b['f1']} | "
                  f"events warned {res['event_level']['warned_in_advance']}/{res['event_level']['events']}")
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        (MODELS_DIR / "risk_metrics.json").write_text(json.dumps(report, indent=2))
    if verbose:
        for t, r in report["types"].items():
            if r["status"] != "trained":
                print(f"{t:<22} {r['status']}")
        print(VALIDATION_DISCLAIMER)
    return report


# ------------------------------------------------------------------ inference
_CACHE: dict = {}


def load_model(etype: str):
    path = MODELS_DIR / f"risk_{slug(etype)}.joblib"
    if not path.exists():
        return None
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def predict_now(ctx, active_id: str, depth: float, etype: str, cfg: dict | None = None) -> float | None:
    """ML score for the bit position, or None when no model exists for this event type."""
    cfg = get_config(cfg)
    bundle = load_model(etype)
    if bundle is None:
        return None
    stream = ctx.stream_for(active_id, depth)
    if stream.empty:
        return None
    row = base_rows(stream, ctx.geology[ctx.geology.depth <= depth], ctx.tops, active_id,
                    bundle["param_cols"], step=1).tail(1)
    if row.empty:
        return None
    pool = set(ctx.hist_params.well_id.unique())
    row["evidence"] = evidence_feature(active_id, row.depth.values, ctx.wells, ctx.tops, ctx.events, etype, cfg, pool)
    for c in bundle["features"]:
        if c not in row:
            row[c] = np.nan
    return float(bundle["model"].predict_proba(row[bundle["features"]].fillna(-1))[:, 1][0])
