import json
import re

import numpy as np

from tests.helpers import ctx
from src.explainability import assess, explain_text
from src.risk_model import eligible_types, evidence_feature
from src.utils import DECISION_NOTE, INSUFFICIENT_ML, MODELS_DIR, VALIDATION_DISCLAIMER, get_config

FORBIDDEN = re.compile(r"\bwill (occur|happen)\b|\bexpected to occur\b", re.I)


def test_insufficient_types_get_no_model():
    st = eligible_types(ctx().events, get_config())
    assert st["Kick"].startswith(INSUFFICIENT_ML)
    assert st["Mud Loss"] == "eligible"


def test_leave_one_well_out_evidence_ignores_heldout_events():
    """Evidence for a training well must not change when the held-out well's events are removed."""
    c, cfg = ctx(), get_config()
    held = "W-102"
    pool = set(c.hist_params.well_id.unique()) - {held}
    depths = np.arange(2400, 2800, 10.0)
    with_held = evidence_feature("W-113", depths, c.wells, c.tops, c.events, "Mud Loss", cfg, pool)
    without = evidence_feature("W-113", depths, c.wells, c.tops, c.events[c.events.well_id != held],
                               "Mud Loss", cfg, pool)
    assert np.allclose(with_held, without)


def test_validation_report_is_labelled():
    rep = json.loads((MODELS_DIR / "risk_metrics.json").read_text())
    assert rep["disclaimer"] == VALIDATION_DISCLAIMER and rep["method"].endswith("leave-one-well-out")


def test_demo_alert_at_barail_losses():
    st = assess(ctx(), "ACTIVE-01", 2690)
    a = st["alerts"][0]
    assert a["risk_type"] == "Mud Loss" and a["level"] == "Elevated"
    assert "risk indication" in a["statement"]
    assert all(e["source_document"] for e in a["evidence"])       # provenance on every fact


def test_no_certainty_language_and_decision_note():
    for d in (2450, 2690, 2940):
        for a in assess(ctx(), "ACTIVE-01", d)["alerts"]:
            text = explain_text(a)
            assert not FORBIDDEN.search(text)
            assert DECISION_NOTE in text


def test_no_alert_before_the_zone():
    assert assess(ctx(), "ACTIVE-01", 2300)["alerts"] == []


def test_forge_missing_parameters_stay_missing_and_no_ml():
    st = assess(ctx(), "FORGE-58-32", 2082)
    assert np.isnan(st["current"]["mud_weight"])                   # not invented
    assert all(r["components"]["ml_score"] is None for r in st["risks"])
    assert st["relevance"].empty                                    # no offset wells for the real well
