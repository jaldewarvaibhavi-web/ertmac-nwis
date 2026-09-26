"""
services.py - the logic behind every API endpoint, as plain Python functions.
Kept separate from FastAPI so it can be unit-tested without a web server.
All return JSON-safe dicts/lists (NaN -> null).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import document_intelligence as di
from src.engine import get_context
from src.event_engine import correlate
from src.explainability import assess, evidence_payload, explain_text
from src.geo_matching import nearby_wells, relevance
from src.rag import Retriever, ask
from src.utils import DECISION_NOTE, SYNTHETIC_BANNER, get_config

_RETRIEVER: Retriever | None = None


_READY = False


def _ctx():
    global _READY
    if not _READY:
        from src.bootstrap import ensure_ready
        ensure_ready(verbose=False)
        _READY = True
    return get_context()


def _retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(_ctx().documents)
    return _RETRIEVER


def jsonable(x):
    if isinstance(x, pd.DataFrame):
        return [jsonable(r) for r in x.to_dict("records")]
    if isinstance(x, pd.Series):
        return jsonable(x.to_dict())
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if (math.isnan(x) or math.isinf(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def _check_well(well_id):
    if well_id not in set(_ctx().wells.well_id):
        raise KeyError(f"Unknown well_id '{well_id}'")


def health():
    c = _ctx()
    return {"status": "ok", "wells": int(len(c.wells)), "events": int(len(c.events)),
            "documents": int(len(c.documents)), "active_wells": c.active_ids, "data_label": SYNTHETIC_BANNER}


def list_wells():
    return jsonable(_ctx().wells.drop(columns=["qc_flags"], errors="ignore"))


def get_well(well_id):
    _check_well(well_id)
    c = _ctx()
    w = c.wells[c.wells.well_id == well_id].drop(columns=["qc_flags"], errors="ignore").iloc[0]
    tops = c.tops[c.tops.well_id == well_id][["formation", "top_depth_m", "bottom_depth_m"]]
    return jsonable({"well": w, "formation_tops": tops, "n_events": int((c.events.well_id == well_id).sum())})


def nearby(well_id, radius_km):
    _check_well(well_id)
    n = nearby_wells(_ctx().wells, well_id, radius_km)
    return jsonable(n[n.in_radius][["well_id", "distance_km", "total_depth", "drilling_year", "data_origin"]])


def events(well_id=None):
    e = _ctx().events.drop(columns=["qc_flags"], errors="ignore")
    if well_id:
        _check_well(well_id)
        e = e[e.well_id == well_id]
    return jsonable(e)


def current(well_id, depth=None, limit=200):
    _check_well(well_id)
    s = _ctx().stream(well_id)
    if depth is not None:
        s = s[s.depth <= depth]
    return jsonable({"label": "Synthetic eRTMAC-like current drilling stream" if well_id.startswith("ACTIVE")
                     else "Replay of a drilling log (not eRTMAC)",
                     "rows": s.tail(limit).drop(columns=["qc_flags"], errors="ignore")})


def match_offsets(well_id, depth, config=None):
    _check_well(well_id)
    cfg = get_config(config)
    rel = relevance(_ctx(), well_id, depth, cfg)
    corr = correlate(_ctx().aligned_events(well_id), rel, depth, cfg) if len(rel) else pd.DataFrame()
    return jsonable({"config": cfg, "weights_note": "prototype weights only - not scientifically validated",
                     "offsets": rel, "events_near_bit": corr[corr.status != "PASSED"] if len(corr) else []})


def risk_predict(well_id, depth, config=None):
    _check_well(well_id)
    st = assess(_ctx(), well_id, depth, config)
    risks = [{k: v for k, v in r.items() if k != "evidence"} | {"n_supporting_events": len(r["evidence"])}
             for r in st["risks"]]
    return jsonable({"well_id": well_id, "depth": depth, "formation": st["formation"], "timestamp": st["timestamp"],
                     "risks": risks, "decision_note": DECISION_NOTE})


def explain_alert(well_id, depth, risk_type=None, use_claude=True, config=None):
    _check_well(well_id)
    st = assess(_ctx(), well_id, depth, config)
    alert = next((r for r in st["risks"] if r["risk_type"] == risk_type), None) if risk_type else \
        (st["alerts"][0] if st["alerts"] else None)
    if alert is None:
        return {"explanation": "Insufficient historical evidence. No matching risk indication at this depth.",
                "mode": "deterministic", "evidence": None}
    res = ask(f"Why is this {alert['risk_type']} risk indication being shown?", st, _retriever(), use_claude)
    return jsonable({"explanation": res["answer"], "mode": res["mode"], "sources": res["sources"],
                     "evidence": evidence_payload(alert, st), "deterministic_explanation": explain_text(alert),
                     "grounding_warnings": res["grounding_warnings"]})


def documents_ingest(filename, content: bytes):
    """Extract events from an uploaded report. Not written to the database (review first)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".txt", ".pdf"):
        raise ValueError("Only .txt and .pdf reports are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        path = Path(tmp.name)
    text, used_ocr = di.read_report(path)
    rows = di.extract_events(text, filename, _ctx().tops, used_ocr)
    spec = di.to_spec(rows)
    return jsonable({"filename": filename, "extraction_method": "OCR" if used_ocr else "text",
                     "events": spec if len(spec) else [], "n_characters": len(text),
                     "note": "Extracted events are returned for engineer review; they are not auto-inserted."})


def documents_search(query, k=5):
    hits = _retriever().search(query, k=k)
    return jsonable({"backend": _retriever().backend,
                     "results": hits[["filename", "well_id", "score", "text"]] if len(hits) else []})
