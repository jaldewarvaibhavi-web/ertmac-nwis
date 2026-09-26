"""
explainability.py - Phase 10: risk indication + "Why?" for every alert.

WHAT:  for the bit position, builds one risk assessment per event type from four transparent parts
         1. OFFSET EVIDENCE   relevance-weighted share of the selected offset wells whose event of this
                              type lies AT the current depth (+-tolerance) or AHEAD (within look-ahead),
                              after formation alignment; closer zones count more.
         2. CONDITION SIMILARITY  current parameters (last 10 m) vs each supporting offset well's
                              parameters just BEFORE its event (StandardScaler + Euclidean similarity),
                              and porosity/caliper at the same formation position.
         3. LIVE ANOMALY      the current well's own behaviour (anomaly.py) shows a matching signature now.
         4. ML SCORE          only for event types whose model PASSED leave-one-well-out validation
                              (PR-AUC >= 0.3 and F1 >= 0.4). Otherwise not used and the reason is shown.
       Combined score -> Low / Moderate / Elevated *risk indication*. Never "will occur".
WHY:   the spec requires that every alert can answer "Why did you generate this alert?" with
       evidence and sources, and that the engineer stays in charge.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import risk_model
from src.event_engine import correlate
from src.geo_matching import relevance
from src.preprocessing import formation_top
from src.similarity import compare_now
from src.utils import DECISION_NOTE, MODELS_DIR, get_config, risk_label

LIVE_MATCH = {  # live anomaly type -> historical event types it supports
    "Mud Loss": {"Mud Loss"}, "Kick": {"Kick"}, "Bit Problem": {"Bit Problem"},
    "Torque Spike": {"Torque Spike", "Stuck Pipe", "Wellbore Instability"},
    "Pressure/Gas Anomaly": {"Pressure/Gas Anomaly", "Stuck Pipe", "Wellbore Instability", "Mud Loss"},
}
ML_GATE = {"pr_auc": 0.3, "f1": 0.4}


def ml_status() -> dict:
    """event type -> dict(usable: bool, reason: str, metrics)"""
    path = MODELS_DIR / "risk_metrics.json"
    if not path.exists():
        return {}
    rep = json.loads(path.read_text())
    out = {}
    for t, r in rep["types"].items():
        if r["status"] != "trained":
            out[t] = dict(usable=False, reason=r["status"])
            continue
        m = r["model"]
        ok = (m["pr_auc"] or 0) >= ML_GATE["pr_auc"] and m["f1"] >= ML_GATE["f1"]
        out[t] = dict(usable=ok, metrics=m, event_level=r["event_level"],
                      reason=("passed leave-one-well-out validation" if ok else
                              f"model trained but FAILED validation (F1 {m['f1']}, PR-AUC {m['pr_auc']}) - not used"))
    return out


def _precursor_similarity(ctx, current, well_id, event_depth, window=20):
    rows = ctx.hist_params[(ctx.hist_params.well_id == well_id) &
                           (ctx.hist_params.depth >= event_depth - window) & (ctx.hist_params.depth < event_depth)]
    return compare_now(current, rows, ctx.scaler)


def _geology_similarity(ctx, active_id, depth, well_id, aligned_offset_depth, window=10):
    ga = ctx.geology[(ctx.geology.well_id == active_id) & (ctx.geology.depth.between(depth - window, depth))]
    go = ctx.geology[(ctx.geology.well_id == well_id) &
                     (ctx.geology.depth.between(aligned_offset_depth - window, aligned_offset_depth + window))]
    if ga.empty or go.empty or ga.porosity.isna().all():
        return np.nan, "Not available"
    dp = abs(ga.porosity.mean() - go.porosity.mean())
    dc = abs(ga.caliper.mean() - go.caliper.mean())
    s = 1 - np.mean([min(1, dp / 0.10), min(1, dc / 2.0)])
    return float(s), f"porosity {ga.porosity.mean():.2f} vs {go.porosity.mean():.2f}, caliper {ga.caliper.mean():.1f} vs {go.caliper.mean():.1f} in"


def assess(ctx, active_id: str, depth: float, cfg: dict | None = None) -> dict:
    """Everything the dashboard / API needs for one bit position."""
    cfg = get_config(cfg)
    tops_a = ctx.tops[ctx.tops.well_id == active_id]
    formation, below = ctx.position(active_id, depth)
    current = ctx.current_row(active_id, depth)
    rel = relevance(ctx, active_id, depth, cfg) if active_id in ctx.active_ids else pd.DataFrame()
    corr = correlate(ctx.aligned_events(active_id), rel, depth, cfg) if len(rel) else pd.DataFrame()
    _, inc_all = ctx.anomalies(active_id)
    live = inc_all[(inc_all.top_m <= depth) & (inc_all.bottom_m >= depth - 15) & (inc_all.confidence != "Low")] \
        if len(inc_all) else pd.DataFrame()
    mls = ml_status()

    selected = rel[rel.selected] if len(rel) else rel
    rel_total = float(selected.relevance_score.sum()) if len(selected) else 0.0
    near = corr[corr.status.isin(["AT CURRENT DEPTH", "AHEAD"])] if len(corr) else corr
    types = set(near.event_type) if len(near) else set()
    live_types = set(live.event_type_canonical) if len(live) else set()
    types |= {t for lt in live_types for t in LIVE_MATCH.get(lt, {lt}) if t in types or lt == t}
    types |= live_types

    risks = []
    for t in sorted(types):
        ev = near[near.event_type == t] if len(near) else pd.DataFrame()
        support = []
        for e in ev.itertuples():
            prec = _precursor_similarity(ctx, current, e.well_id, e.depth)
            top_o = formation_top(ctx.tops[ctx.tops.well_id == e.well_id], formation) if formation else None
            top_a = formation_top(tops_a, formation) if formation else None
            geo_s, geo_txt = (_geology_similarity(ctx, active_id, depth, e.well_id, top_o + (depth - top_a))
                              if top_o is not None and top_a is not None else (np.nan, "Not available"))
            proximity = 1.0 if e.status == "AT CURRENT DEPTH" else 1 - 0.5 * (e.delta_to_bit_m / cfg["lookahead_m"])
            support.append(dict(
                well_id=e.well_id, distance_km=e.distance_km, relevance_score=e.relevance_score,
                event_type=e.event_type, event_depth_m=e.depth, aligned_depth_m=round(e.aligned_depth, 1),
                m_below_formation_top=None if pd.isna(e.m_below_top) else round(e.m_below_top, 1),
                delta_to_bit_m=round(e.delta_to_bit_m, 1), status=e.status, formation=e.formation,
                severity=e.severity, description=e.description, mitigation=e.mitigation, outcome=e.outcome,
                npt_hours=e.npt_hours, source_document=e.source_document, extraction_method=e.extraction_method,
                parameter_similarity=None if pd.isna(prec["similarity"]) else round(prec["similarity"], 3),
                parameter_differences_sd=prec["diffs"], geology_similarity=None if pd.isna(geo_s) else round(geo_s, 3),
                geology_note=geo_txt, proximity=round(proximity, 3)))
        sup = pd.DataFrame(support)
        wells_with = sup.drop_duplicates("well_id") if len(sup) else sup
        share = float(wells_with.relevance_score.sum() / rel_total) if (len(sup) and rel_total) else 0.0
        psim = float(np.nanmean(sup.parameter_similarity.astype(float))) if len(sup) and sup.parameter_similarity.notna().any() else np.nan
        gsim = float(np.nanmean(sup.geology_similarity.astype(float))) if len(sup) and sup.geology_similarity.notna().any() else np.nan
        prox = float(sup.proximity.max()) if len(sup) else 0.0
        evidence_score = share * prox * (0.6 + 0.4 * (psim if not np.isnan(psim) else 0.5))
        live_hit = live[live.event_type_canonical.map(lambda lt: t in LIVE_MATCH.get(lt, {lt}))] if len(live) else live
        if len(live_hit):
            evidence_score = min(1.0, max(evidence_score, 0.45) + 0.25)
        ml = mls.get(t, {})
        # ML only for wells of the field it was trained on (never applied to the FORGE geothermal well)
        ml_prob = (risk_model.predict_now(ctx, active_id, depth, t, cfg)
                   if ml.get("usable") and active_id in ctx.active_ids else None)
        score = 0.5 * evidence_score + 0.5 * ml_prob if ml_prob is not None else evidence_score
        level = risk_label(score)

        why = []
        if formation:
            why.append(f"Bit at {depth:,.0f} m in {formation} ({below:.0f} m below its top).")
        if len(wells_with):
            why.append(f"{len(wells_with)} of {len(selected)} selected relevant offset wells had {t} at a comparable "
                       f"formation position (relevance-weighted share {share:.0%}).")
            at = sup[sup.status == "AT CURRENT DEPTH"]
            if len(at):
                why.append(f"{len(at)} of these events {'lies' if len(at) == 1 else 'lie'} within "
                           f"±{cfg['depth_tolerance_m']:.0f} m of the bit after formation alignment.")
            ah = sup[sup.status == "AHEAD"]
            if len(ah):
                why.append(f"{len(ah)} {'lies' if len(ah) == 1 else 'lie'} ahead of the bit, nearest "
                           f"{ah.delta_to_bit_m.min():.0f} m below it.")
        if not np.isnan(psim):
            why.append(f"Current drilling parameters vs those offset wells just before their events: similarity {psim:.2f} (0-1).")
        if not np.isnan(gsim):
            why.append(f"Geology at the same formation position: similarity {gsim:.2f} (synthetic porosity/caliper).")
        if len(live_hit):
            why.append("Current well's own data shows a matching live signature: " +
                       "; ".join(f"{r.event} ({r.confidence}): {r.evidence}" for r in live_hit.itertuples()))
        if ml_prob is not None:
            why.append(f"ML model (validated leave-one-well-out) score {ml_prob:.2f} — not a calibrated probability.")
        elif ml.get("usable"):
            why.append("ML: not applied - the model was trained on the synthetic field, not on this well.")
        elif ml:
            why.append(f"ML: {ml.get('reason')}")
        else:
            why.append("ML: no model for this event type.")

        risks.append(dict(
            risk_type=t, level=level, score=round(score, 3),
            headline=f"{level.upper()} {t.upper()} RISK INDICATION",
            statement=(f"{level} {t.lower()} risk indication based on similarity to historical offset-well conditions."
                       if len(wells_with) else f"{level} {t.lower()} risk indication based on the current well's own behaviour."),
            components=dict(offset_evidence=round(share, 3), proximity=round(prox, 3),
                            parameter_similarity=None if np.isnan(psim) else round(psim, 3),
                            geology_similarity=None if np.isnan(gsim) else round(gsim, 3),
                            live_anomaly=bool(len(live_hit)), ml_score=None if ml_prob is None else round(ml_prob, 3),
                            evidence_score=round(evidence_score, 3)),
            ml_note=("not applied to this well (model trained on the synthetic field only)"
                     if ml.get("usable") and ml_prob is None else ml.get("reason", "no model for this event type")),
            why=why, evidence=support,
            historical_mitigations=_mitigations(sup), decision_note=DECISION_NOTE,
        ))
    risks.sort(key=lambda r: -r["score"])
    alerts = [r for r in risks if r["level"] in ("Elevated", "Moderate")]
    return dict(
        well_id=active_id, depth=float(depth), formation=formation, m_below_top=below,
        timestamp=str(current.get("timestamp", "")), current=current, relevance=rel, correlated=corr,
        live_incidents=live, risks=risks, alerts=alerts, config=cfg)


def _mitigations(sup: pd.DataFrame) -> list[dict]:
    """What was done historically (not an instruction): action, outcome, count, sources."""
    if sup is None or sup.empty:
        return []
    g = sup.groupby(["mitigation", "outcome"]).agg(times=("well_id", "size"),
                                                   wells=("well_id", lambda s: ", ".join(sorted(set(s)))),
                                                   sources=("source_document", lambda s: ", ".join(sorted(set(s)))))
    return g.reset_index().sort_values("times", ascending=False).to_dict("records")


def evidence_payload(alert: dict, state: dict) -> dict:
    """Structured, minimal evidence handed to Claude - Claude may only explain THIS."""
    return dict(
        current_depth_m=state["depth"], formation=state["formation"],
        metres_below_formation_top=None if state["m_below_top"] is None else round(state["m_below_top"], 1),
        risk=alert["risk_type"], risk_level=alert["level"], score=alert["score"], components=alert["components"],
        ml_note=alert["ml_note"],
        relevant_wells=[{k: e[k] for k in ("well_id", "distance_km", "relevance_score", "event_type", "event_depth_m",
                                            "aligned_depth_m", "delta_to_bit_m", "status", "severity", "mitigation",
                                            "outcome", "source_document", "parameter_similarity")}
                        for e in alert["evidence"]],
        historical_mitigations=alert["historical_mitigations"],
        why=alert["why"],
    )


def explain_text(alert: dict) -> str:
    """Deterministic explanation (used when Claude is not configured, and as Claude's reference)."""
    lines = [f"{alert['headline']}", alert["statement"], "", "Why:"]
    lines += [f"- {w}" for w in alert["why"]]
    if alert["evidence"]:
        lines += ["", "Supporting historical evidence:"]
        for e in alert["evidence"]:
            lines.append(f"- {e['well_id']} ({e['distance_km']} km, relevance {e['relevance_score']:.2f}): "
                         f"{e['event_type']} at {e['event_depth_m']:.0f} m ({e['severity']}), aligned to "
                         f"{e['aligned_depth_m']:.0f} m in this well [{e['status']}]. "
                         f"Historically: {e['mitigation']} -> {e['outcome']}. Source: {e['source_document']}")
    lines += ["", alert["decision_note"]]
    return "\n".join(lines)
